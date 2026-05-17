from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Callable, Any
from multi_agent.tracing.recorder import Recorder
from multi_agent.tracing.ulid_gen import fresh_run_id
from multi_agent.schemas.events import RunStarted, RunFinished
from multi_agent.providers.base import LLMProvider
from multi_agent.agents.base import BaseAgent, AgentInput


async def run_query(
    *,
    query: str,
    agent_factory: Callable[[LLMProvider, Recorder], BaseAgent],
    provider: LLMProvider,
    runs_root: Path,
    config: dict[str, Any] | None = None,
    session_id: str | None = None,
    memory_store=None,
    turn_indexer=None,
    compaction_provider: LLMProvider | None = None,
    compaction_model: str | None = None,
    agent_input_extra: dict[str, Any] | None = None,
    extra_agents_invoked: list[str] | None = None,
) -> dict:
    """Top-level entry. Guarantees a RunFinished event regardless of outcome.

    Returns a small dict {run_id, status, final_answer?}.

    Optional params:
        session_id: if provided alongside memory_store, appends a Turn and
                    updates StickyContext.linked_runs after a successful run.
        memory_store: a MarkdownMemoryStore instance (or compatible duck-type).
        compaction_provider: if set (together with compaction_model), triggers
                    maybe_compact() after write_sticky() per spec §5.4.1.
                    Intentionally separate from `provider` so a cheaper model
                    can be used for compaction.
        compaction_model: model name passed to maybe_compact().
    """
    started_at = datetime.now()
    run_id = fresh_run_id()
    run_dir = Path(runs_root) / run_id
    recorder = Recorder(run_id=run_id, run_dir=run_dir)
    recorder.set_meta(query=query, config=(config or {}))

    final_answer: str | None = None
    status = "ok"
    error: str | None = None
    agent: BaseAgent | None = None

    try:
        recorder.emit(RunStarted(
            event_id=recorder.fresh_event_id(), run_id=run_id,
            timestamp=recorder.now(), parent_id=None,
            query=query, config=(config or {}),
        ))
        agent = agent_factory(provider, recorder)
        # Phase 6f: 允许 caller 注入额外 payload (例如 prefetched_evidences for fast-path)
        payload = {"query": query}
        if agent_input_extra:
            payload.update(agent_input_extra)
        output = await agent.run(AgentInput(payload=payload))
        final_answer = output.payload.model_dump_json()
    except Exception as e:
        status = "error"
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        # Spec §5.4.2: persist WorkingMemory snapshot to artifacts/ alongside trace.
        # Best-effort: ignore failures (agent may not exist, or WM may be empty).
        if agent is not None:
            wm = getattr(agent, "working_memory", None)
            if wm is not None:
                try:
                    artifacts_dir = run_dir / "artifacts"
                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    (artifacts_dir / "working_memory.json").write_text(
                        wm.model_dump_json(indent=2), encoding="utf-8",
                    )
                except Exception:
                    pass
        try:
            recorder.emit(RunFinished(
                event_id=recorder.fresh_event_id(), run_id=run_id,
                timestamp=recorder.now(), parent_id=None,
                status=status, final_answer=final_answer, error=error,
            ))
        finally:
            recorder.close()

    # Expose evidence_pool from the agent's WorkingMemory (if any) so callers
    # like ExperimentRunner can pass it to LLM judges. Empty list when the
    # agent has no WorkingMemory (e.g. Receptionist).
    evidence_pool: list[dict] = []
    if agent is not None:
        wm = getattr(agent, "working_memory", None)
        if wm is not None:
            evidence_pool = [ev.model_dump() for ev in getattr(wm, "retrieved_evidence", [])]

    result = {
        "run_id": run_id,
        "status": status,
        "final_answer": final_answer,
        "evidence_pool": evidence_pool,
    }

    # Memory integration: persist Turn and update StickyContext after success.
    if session_id and memory_store is not None and status == "ok":
        from multi_agent.schemas.memory import StickyContext, Turn, CitedArticle
        sticky = memory_store.read_sticky(session_id) or StickyContext(session_id=session_id)
        if run_id not in sticky.linked_runs:
            sticky.linked_runs.append(run_id)
        existing_turns = memory_store.recent_turns(session_id, n=999)
        next_turn_no = max((t.turn for t in existing_turns), default=0) + 1

        # Phase 6m: 从 final_answer JSON + events.jsonl 抽更多 Turn 字段
        # 之前 Turn 只填 6 个字段, 其他 (answer_mode / citations / total_tokens) 默认.
        parsed_payload: dict = {}
        if final_answer:
            try:
                import json as _json
                p = _json.loads(final_answer)
                if isinstance(p, dict):
                    parsed_payload = p
            except (ValueError, TypeError):
                pass

        # answer_mode (consultation / clarification / 默认)
        turn_answer_mode = parsed_payload.get("mode") or "evidence_grounded"

        # turn_citations (跟 sticky.cited_articles 同步, 但 turn 只记本轮的)
        turn_citations: list[CitedArticle] = []
        for cit in (parsed_payload.get("citations") or []):
            if not isinstance(cit, dict):
                continue
            law = (cit.get("law_short") or "").strip()
            art = (cit.get("article_no") or "").strip()
            if law and art:
                turn_citations.append(CitedArticle(
                    law=law, article=art, from_turn=next_turn_no,
                ))

        # total_tokens: 从 events.jsonl 累加 LLMResponded.usage (best-effort)
        turn_total_tokens = 0
        try:
            import json as _json
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = _json.loads(line)
                if e.get("event_type") == "LLMResponded":
                    u = e.get("usage") or {}
                    turn_total_tokens += int(u.get("input_tokens", 0) or 0)
                    turn_total_tokens += int(u.get("output_tokens", 0) or 0)
        except Exception:
            pass

        # agents_invoked: 当前 run 的 agent + 调用方传入的额外 agent (例如 Receptionist 在 chat.py 层跑)
        agents = []
        if extra_agents_invoked:
            agents.extend(extra_agents_invoked)
        if agent is not None and agent.name not in agents:
            agents.append(agent.name)

        turn = Turn(
            turn=next_turn_no,
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(),
            question=query,
            final_answer=final_answer or "",
            answer_mode=turn_answer_mode,
            agents_invoked=agents,
            total_tokens=turn_total_tokens,
            citations=turn_citations,
        )
        memory_store.append_turn(session_id, turn)

        # Phase 6k: 从 Lawyer 输出 JSON 抽 citations 更新 sticky 累积字段.
        # 之前 sticky.cited_articles / mentioned_laws / last_law_name 永远空,
        # Phase 6h 注入上下文给后续 turn 因此失效. 现在补齐.
        if final_answer:
            try:
                import json as _json
                payload = _json.loads(final_answer)
                citations = payload.get("citations") or []
                # 已有 (law, article) 集合, 去重
                seen = {(c.law, c.article) for c in sticky.cited_articles}
                for cit in citations:
                    if not isinstance(cit, dict):
                        continue
                    law = (cit.get("law_short") or "").strip()
                    art = (cit.get("article_no") or "").strip()
                    if not law or not art:
                        continue
                    if (law, art) in seen:
                        continue
                    seen.add((law, art))
                    sticky.cited_articles.append(CitedArticle(
                        law=law, article=art, from_turn=next_turn_no,
                    ))
                # mentioned_laws: 去重保序
                for cit in citations:
                    if not isinstance(cit, dict):
                        continue
                    law = (cit.get("law_short") or "").strip()
                    if law and law not in sticky.mentioned_laws:
                        sticky.mentioned_laws.append(law)
                # last_law_name: 取本次 citations 的第一条 (Lawyer 通常按重要性排)
                for cit in citations:
                    if isinstance(cit, dict) and cit.get("law_short"):
                        sticky.last_law_name = cit["law_short"]
                        break
            except (ValueError, TypeError, KeyError):
                pass  # final_answer 不是 JSON / 不是 LawyerOutput shape → 跳过
        memory_store.write_sticky(sticky)

        # Phase 6e: 索引到 ma_user_history 在后台跑, 不阻塞返回
        # (bge-m3 encode + Qdrant upsert 约 1-3 秒, 用户不应该等)
        if turn_indexer is not None:
            import asyncio
            async def _bg_index():
                try:
                    await turn_indexer.index_turn(session_id=session_id, turn=turn)
                except Exception:
                    # Best-effort: 索引失败不影响主流程, 也不抛
                    pass
            # fire-and-forget; 在 Python 3.11+ 上是 detached task
            # 用 ensure_future 兼容更广; loop.create_task 也行
            asyncio.ensure_future(_bg_index())

        # Spec §5.4.1: auto-compact when session exceeds threshold.
        # Only runs when caller opts in with both compaction_provider + compaction_model.
        if compaction_provider is not None and compaction_model is not None:
            from multi_agent.memory.compaction import maybe_compact
            await maybe_compact(
                session_id,
                memory_store,
                provider=compaction_provider,
                model=compaction_model,
            )

    return result

# Phase 5 · Harness Runtime + LangGraph + ConversationManager + ContextCompactor

## 依赖

- Phase 1–4 全部完成。
- Phase 4 的 ContextComposer 已能在 `NullCompactor` 下跑通。

## 本阶段交付物

1. `src/legal_rag/harness/state.py`（`LegalRAGState`，多轮 + 压缩字段齐全）。
2. **`src/legal_rag/harness/context_compactor.py`**（Claude-Code 风格的工作上下文压缩器，**runtime 层组件**，不是 agent，不进 graph）。
3. `src/legal_rag/prompts/context_compactor.md`。
4. `src/legal_rag/harness/runtime.py`（`HarnessRuntime` ＝ `ConversationManager`，注入真 ContextCompactor）。
5. `src/legal_rag/harness/tracing.py`（`JsonlTracer`，按 `run_id + session_id` 双索引，`compaction` 写专属事件类型）。
6. `src/legal_rag/harness/validators.py`（含 Citation Checker，对 `cumulative_evidence + digest.pinned_evidence_ids` 友好）。
7. `src/legal_rag/harness/policies.py`（脱敏正则）。
8. `src/legal_rag/harness/tool_registry.py`。
9. `src/legal_rag/graph/routing.py`。
10. `src/legal_rag/graph/legal_rag_graph.py`（**没有 compactor 节点**）。
11. `src/legal_rag/memory/session_store.py`：`SessionStore` 抽象 + `InMemorySessionStore`（Phase 6 加 SQLite 实现）。
12. `scripts/chat.py` + `scripts/ask.py`。
13. `tests/test_context_compactor.py`、`tests/test_graph_routes.py`、`tests/test_conversation.py`。

---

## 1. ContextCompactor（核心）

`src/legal_rag/harness/context_compactor.py`：

```python
from __future__ import annotations
import time, uuid, json
from pathlib import Path
from pydantic import BaseModel, Field
from legal_rag.providers.base import LLMProvider, LLMMessage
from legal_rag.config import settings
from legal_rag.schemas import WorkingContextDigest

class CompactorOutput(BaseModel):
    """LLM 输出结构（直接对应 WorkingContextDigest 的可写字段）。"""
    user_facts: list[str] = Field(default_factory=list)
    intake_summary: str = ""
    retrieval_summary: str = ""
    evidence_summary: str = ""
    answer_summary: str = ""
    reviewer_observations: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)
    pinned_evidence_ids: list[str] = Field(default_factory=list)
    dropped_evidence_ids: list[str] = Field(default_factory=list)


class ContextCompactor:
    """
    Claude Code 风格的工作上下文压缩器。

    特点：
      - 不是 agent，没有 graph 节点；
      - 由 ContextComposer 在装配 messages 前透明触发；
      - 一次性收集 LegalRAGState 中"全部工作过程产物"，让 LLM 输出结构化 digest；
      - 把 digest 写回 state["working_context_digest"]，之后所有 LLM 调用都带这份 system 前缀；
      - 把已淘汰的 evidence 从 cumulative_evidence 删掉，user_facts 合并进 sticky_intake.pinned_facts；
      - 所有触发都写入 trace 的 'harness.compaction' 事件。
    """

    def __init__(self, llm: LLMProvider, prompts_dir: Path, tracer=None):
        self.llm = llm
        self.tracer = tracer
        self._prompt_tpl = (prompts_dir / "context_compactor.md").read_text("utf-8")

    def maybe_compact(self, state: dict, *, estimated_tokens: int, agent_name: str) -> dict:
        """ContextComposer 调入口。返回 LegalRAGState patch；不需要压缩时返回 {}.

        触发条件（任一满足）：
          1) estimated_tokens >= settings.session_compact_trigger_tokens
          2) len(history_messages)//2 >= settings.session_compact_trigger_turns
          3) 当前 LLM 模型上下文窗口的 70% 被吃满（防御性）
        """
        if not self._should_compact(state, estimated_tokens):
            return {}
        return self._compact(state, triggered_by="budget", agent_name=agent_name,
                             token_before=estimated_tokens)

    def force_compact(self, state: dict) -> dict:
        """运维 / API 触发。"""
        return self._compact(state, triggered_by="manual", agent_name="manual",
                             token_before=self.llm.estimate_tokens(self._snapshot_text(state)))

    # -------- internals --------
    def _should_compact(self, state, estimated):
        if estimated >= settings.session_compact_trigger_tokens:
            return True
        n_turns = len(state.get("history_messages", [])) // 2
        if n_turns >= settings.session_compact_trigger_turns:
            return True
        ctx = self.llm.context_window
        if estimated >= int(ctx * 0.7):
            return True
        return False

    def _compact(self, state, *, triggered_by, agent_name, token_before) -> dict:
        t0 = time.perf_counter()
        snapshot = self._collect_snapshot(state)        # 把全部工作产物结构化成一段文本
        messages = self._build_messages(snapshot)
        try:
            out = self.llm.chat_json(messages, CompactorOutput, max_tokens=1500)
        except Exception:
            out = self._fallback(snapshot, state)

        # 装 WorkingContextDigest
        digest = WorkingContextDigest(
            digest_id="dig_" + uuid.uuid4().hex[:10],
            until_run_id=state.get("run_id"),
            until_turn_id=state.get("turn_id", 0),
            triggered_by=triggered_by,
            user_facts=out.user_facts,
            intake_summary=out.intake_summary,
            retrieval_summary=out.retrieval_summary,
            evidence_summary=out.evidence_summary,
            answer_summary=out.answer_summary,
            reviewer_observations=out.reviewer_observations,
            open_issues=out.open_issues,
            pinned_evidence_ids=out.pinned_evidence_ids,
            dropped_evidence_ids=out.dropped_evidence_ids,
            token_estimate_before=token_before,
            token_estimate_after=self.llm.estimate_tokens(out.model_dump_json()),
            created_at=time.time(),
        )

        # cumulative_evidence 清理
        cumulative = dict(state.get("cumulative_evidence", {}))
        for eid in out.dropped_evidence_ids:
            cumulative.pop(eid, None)

        # 把 user_facts 合并进 sticky_intake.pinned_facts
        sticky = dict(state.get("sticky_intake", {}))
        merged = list(dict.fromkeys(sticky.get("pinned_facts", []) + out.user_facts))
        sticky["pinned_facts"] = merged

        # 截断 history：digest 已涵盖旧轮，仅保留最近 keep_recent
        keep = settings.session_keep_recent_turns * 2
        history = state.get("history_messages", [])
        new_history = history[-keep:] if len(history) > keep else list(history)

        # trace
        latency_ms = (time.perf_counter() - t0) * 1000
        events = list(state.get("compaction_events", []))
        evt = {
            "ts": time.time(),
            "kind": "harness.compaction",
            "triggered_by": triggered_by,
            "by_agent": agent_name,
            "token_before": digest.token_estimate_before,
            "token_after": digest.token_estimate_after,
            "saved_ratio": 1.0 - (digest.token_estimate_after / max(1, digest.token_estimate_before)),
            "dropped_evidence_ids": out.dropped_evidence_ids,
            "pinned_evidence_ids": out.pinned_evidence_ids,
            "latency_ms": latency_ms,
        }
        events.append(evt)
        if self.tracer:
            self.tracer.event(state.get("run_id", "?"), state.get("session_id", "?"),
                              "harness.compaction", latency_ms=latency_ms,
                              output=evt, token_estimate=digest.token_estimate_after)

        return {
            "working_context_digest": digest.model_dump(),
            "cumulative_evidence": cumulative,
            "sticky_intake": sticky,
            "history_messages": new_history,
            "compaction_events": events,
        }

    def _collect_snapshot(self, state) -> str:
        """把工作过程的全部产物拼成给 LLM 的输入文本。"""
        sections = []
        if state.get("history_messages"):
            sections.append("[历史对话]\n" + "\n".join(
                f"({m['role']}) {m['content']}" for m in state["history_messages"]
            ))
        if state.get("intake_result"):
            sections.append("[Intake 结果]\n" + json.dumps(state["intake_result"], ensure_ascii=False))
        if state.get("sticky_intake"):
            sections.append("[Sticky Intake]\n" + json.dumps(state["sticky_intake"], ensure_ascii=False))
        if state.get("plan"):
            sections.append("[研究计划]\n- " + "\n- ".join(state["plan"]))
        if state.get("failed_queries"):
            sections.append("[失败 query 累积]\n- " + "\n- ".join(state["failed_queries"]))
        if state.get("retrieved_docs"):
            sections.append("[本 turn 已检索 evidence]\n" + json.dumps(
                [{"id": e["evidence_id"],
                  "law": e.get("metadata", {}).get("law_name"),
                  "art": e.get("metadata", {}).get("article_number"),
                  "snippet": e["text"][:160]} for e in state["retrieved_docs"]],
                ensure_ascii=False,
            ))
        if state.get("cumulative_evidence"):
            sections.append("[历史 cumulative evidence]\n" + json.dumps(
                [{"id": eid,
                  "law": ev.get("metadata", {}).get("law_name"),
                  "art": ev.get("metadata", {}).get("article_number"),
                  "snippet": ev["text"][:160]} for eid, ev in state["cumulative_evidence"].items()],
                ensure_ascii=False,
            ))
        if state.get("evidence_assessments"):
            sections.append("[Evidence 评估]\n" + json.dumps(state["evidence_assessments"], ensure_ascii=False))
        if state.get("draft_answer"):
            sections.append("[当前草稿]\n" + state["draft_answer"][:1200])
        if state.get("review_comments"):
            sections.append("[Reviewer 意见]\n- " + "\n- ".join(state["review_comments"]))
        return "\n\n".join(sections)

    def _snapshot_text(self, state) -> str:
        return self._collect_snapshot(state)

    def _build_messages(self, snapshot: str) -> list[LLMMessage]:
        sys = (
            "你是法律研究系统的工作上下文压缩器。"
            "把整段会话工作过程压缩成结构化 JSON，让后续 LLM 调用看到压缩后仍能继续工作。"
            "禁止编造未在输入中出现的事实/法条/引用。"
            "禁止改写法律结论。"
            "输出 JSON 字段严格匹配 CompactorOutput schema，不要 markdown 包裹。"
        )
        user = self._prompt_tpl.replace("{{snapshot}}", snapshot)
        return [LLMMessage(role="system", content=sys), LLMMessage(role="user", content=user)]

    def _fallback(self, snapshot: str, state) -> CompactorOutput:
        """LLM 解析失败时的规则兜底。"""
        return CompactorOutput(
            user_facts=state.get("sticky_intake", {}).get("pinned_facts", []),
            intake_summary=json.dumps(state.get("sticky_intake", {}), ensure_ascii=False)[:200],
            retrieval_summary=";".join(state.get("failed_queries", []))[:300],
            evidence_summary=";".join(list(state.get("cumulative_evidence", {}).keys())[:8]),
            answer_summary=(state.get("draft_answer") or "")[:300],
            reviewer_observations=state.get("review_comments", [])[:5],
            open_issues=[],
            pinned_evidence_ids=list(state.get("cumulative_evidence", {}).keys())[:8],
            dropped_evidence_ids=[],
        )
```

---

## 2. context_compactor.md（prompt）

`src/legal_rag/prompts/context_compactor.md`：

```text
你正在为法律研究系统压缩工作上下文。

下面是这次会话的工作过程快照（包括用户对话、intake 结论、研究计划、失败 query、检索证据、评估结果、草稿答案、reviewer 意见）：

----- BEGIN SNAPSHOT -----
{{snapshot}}
----- END SNAPSHOT -----

请输出 JSON，严格匹配以下 schema：

{
  "user_facts":            ["用户已在对话中确认的、对法律分析重要的事实，每条 <= 60 字"],
  "intake_summary":        "<= 100 字，已识别的法律领域 / 任务类型 / 风险等级",
  "retrieval_summary":     "<= 200 字，已尝试过哪些 query，命中了什么主题，哪些方向是 dead-end",
  "evidence_summary":      "<= 250 字，关键 evidence 的角色（哪个支持何种结论 / 哪个用作反驳）",
  "answer_summary":        "<= 200 字，当前草稿/已发布回答的主旨",
  "reviewer_observations": ["反复出现的反方意见 / 必须照顾的限制条件"],
  "open_issues":           ["仍未答清楚的子问题"],
  "pinned_evidence_ids":   ["必须保留在 cumulative_evidence 中的 evidence_id，<= 10"],
  "dropped_evidence_ids":  ["可以从 cumulative_evidence 淘汰的 evidence_id（已被否定 / 与当前话题无关）"]
}

约束：
- 不引入未出现在 SNAPSHOT 中的法条 / 案例 / 事实。
- pinned + dropped 的并集 ⊆ SNAPSHOT 中出现过的 evidence_id。
- 如果用户已明显切换话题，dropped 应包括旧话题相关 evidence。
- 总输出长度 <= 1500 token。
```

---

## 3. JsonlTracer

```python
import json, time
from pathlib import Path

class JsonlTracer:
    def __init__(self, log_dir: Path):
        self.runs = Path(log_dir) / "runs"
        self.sessions = Path(log_dir) / "sessions"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.sessions.mkdir(parents=True, exist_ok=True)

    def event(self, run_id: str, session_id: str, node: str, *,
              latency_ms: float, output: dict, error: str | None = None,
              token_estimate: int | None = None):
        rec = {"ts": time.time(), "run_id": run_id, "session_id": session_id,
               "node": node, "latency_ms": latency_ms, "token_estimate": token_estimate,
               "output": output, "error": error}
        with (self.runs / f"{run_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with (self.sessions / f"{session_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
```

`harness.compaction` 事件由 ContextCompactor 直接调 `tracer.event(...)` 写入。

---

## 4. Validators

```python
ALLOWED_AGENTS = {"statute_agent", "case_agent", "contract_agent"}

def check_citations(state) -> list[str]:
    pool = {e["evidence_id"]: e for e in state.get("retrieved_docs", [])}
    pool.update(state.get("cumulative_evidence", {}))
    pinned = set(state.get("working_context_digest", {}).get("pinned_evidence_ids", []))
    # pinned 中 id 在 pool 必须存在；否则忽略（说明 cumulative 已被压缩裁剪）
    errors: list[str] = []
    for c in state.get("citations", []):
        eid = c.get("evidence_id")
        if eid not in pool:
            if eid in pinned:
                errors.append(f"citation {eid} 在 digest 中标记为 pinned 但 evidence 已被裁剪——压缩策略 bug")
            else:
                errors.append(f"citation {eid} 不在 evidence pool")
            continue
        text = pool[eid]["text"]
        if c.get("quote") and c["quote"] not in text:
            errors.append(f"citation {eid} 的 quote 不是 evidence.text 子串")
        meta = pool[eid].get("metadata", {})
        if c.get("law_name") and c["law_name"] != meta.get("law_name"):
            errors.append(f"citation {eid} law_name 不一致")
        if c.get("article_number") and c["article_number"] != meta.get("article_number"):
            errors.append(f"citation {eid} article_number 不一致")
    return errors
```

---

## 5. Routing

```python
# graph/routing.py
def decide_route(state):
    intake = state.get("intake_result", {})
    route = []
    if intake.get("needs_statute"):  route.append("statute_agent")
    if intake.get("needs_case"):     route.append("case_agent")
    if intake.get("needs_contract"): route.append("contract_agent")
    if not route: route.append("statute_agent")
    return route

def is_clarification(state) -> bool:
    return bool(state.get("is_clarification_turn"))

def should_retry_retrieval(state) -> bool:
    if state.get("retrieval_retry_count", 0) >= state.get("max_retrieval_retry", 2):
        return False
    return state.get("evidence_score", 0.0) < 0.75

def should_revise_answer(state) -> bool:
    if state.get("answer_revision_count", 0) >= state.get("max_answer_revision", 1):
        return False
    if state.get("reviewer_score", 1.0) >= 0.75 and not state.get("review_comments"):
        return False
    return True
```

> 没有 `need_compactor`，因为压缩对 graph 透明。

---

## 6. Graph

```python
# graph/legal_rag_graph.py
from langgraph.graph import StateGraph, END
from . import routing
from legal_rag.agents import (
    intake_agent, memory_retriever, planner_agent,
    statute_agent, case_agent, contract_agent,
    evidence_checker, query_rewriter, answer_agent, reviewer_agent, memory_agent,
)

def build_legal_rag_graph(deps) -> "CompiledGraph":
    g = StateGraph(LegalRAGState)
    g.add_node("intake",                lambda s: intake_agent.run(s, deps))
    g.add_node("memory_retrieve",       lambda s: memory_retriever.run(s, deps))
    g.add_node("planner",               lambda s: planner_agent.run(s, deps))
    g.add_node("retrieve",              lambda s: _run_retrieval(s, deps))
    g.add_node("rerank",                lambda s: _run_rerank(s, deps))
    g.add_node("evidence_check",        lambda s: evidence_checker.run(s, deps))
    g.add_node("rewrite",               lambda s: query_rewriter.run(s, deps))
    g.add_node("answer",                lambda s: answer_agent.run(s, deps))
    g.add_node("review",                lambda s: reviewer_agent.run(s, deps))
    g.add_node("finalize",              lambda s: _finalize(s, deps))
    g.add_node("finalize_clarification",lambda s: _finalize_clarification(s, deps))
    g.add_node("memory_write",          lambda s: memory_agent.run(s, deps))

    g.set_entry_point("intake")
    g.add_conditional_edges(
        "intake",
        lambda s: "clarify" if routing.is_clarification(s) else "memory_retrieve",
        {"clarify": "finalize_clarification", "memory_retrieve": "memory_retrieve"},
    )
    g.add_edge("memory_retrieve", "planner")
    g.add_edge("planner", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "evidence_check")
    g.add_conditional_edges(
        "evidence_check",
        lambda s: "rewrite" if routing.should_retry_retrieval(s) else "answer",
        {"rewrite": "rewrite", "answer": "answer"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", "review")
    g.add_conditional_edges(
        "review",
        lambda s: "answer" if routing.should_revise_answer(s) else "finalize",
        {"answer": "answer", "finalize": "finalize"},
    )
    g.add_edge("finalize", "memory_write")
    g.add_edge("finalize_clarification", "memory_write")
    g.add_edge("memory_write", END)
    return g.compile()
```

> Graph 节点列表与 Phase 4 完全一致；compaction 由 agent 内部的 `composer.compose()` 透明触发，trace 里以 `harness.compaction` 事件出现，但不在 graph 节点名中。

---

## 7. SessionStore（Phase 5 内存版）

```python
# memory/session_store.py
from abc import ABC, abstractmethod
from legal_rag.schemas import ConversationState, TurnRecord, WorkingContextDigest

class SessionStore(ABC):
    @abstractmethod
    def create(self, user_id: str | None = None) -> str: ...
    @abstractmethod
    def load(self, session_id: str) -> ConversationState: ...
    @abstractmethod
    def save(self, conv: ConversationState) -> None: ...
    @abstractmethod
    def append_turn(self, session_id: str, turn: TurnRecord) -> None: ...
    @abstractmethod
    def append_digest(self, session_id: str, digest: WorkingContextDigest) -> None: ...
    @abstractmethod
    def merge_evidence(self, session_id: str, retrieved: list[dict]) -> tuple[list[dict], dict[str, str]]: ...
    @abstractmethod
    def close(self, session_id: str) -> None: ...
    @abstractmethod
    def gc(self, older_than_days: int = 7) -> int: ...

class InMemorySessionStore(SessionStore):
    def __init__(self):
        self._sessions: dict[str, ConversationState] = {}
        self._ev_seq: dict[str, int] = {}
        self._chunk_to_eid: dict[str, dict[str, str]] = {}
    ...
```

---

## 8. HarnessRuntime ＝ ConversationManager

```python
# harness/runtime.py
import time, uuid
from pathlib import Path
from legal_rag.config import settings
from legal_rag.providers.factory import get_llm_provider
from legal_rag.indexes.bm25_index import BM25Index
from legal_rag.indexes.dense_index import DenseIndex
from legal_rag.indexes.hybrid_retriever import HybridRetriever
from legal_rag.indexes.reranker import Reranker
from legal_rag.agents._context_composer import ContextComposer, PromptLibrary
from legal_rag.agents._deps import AgentDeps
from legal_rag.graph.legal_rag_graph import build_legal_rag_graph
from legal_rag.memory.session_store import InMemorySessionStore, SessionStore
from legal_rag.schemas import ConversationState, TurnRecord, WorkingContextDigest
from .context_compactor import ContextCompactor
from .tracing import JsonlTracer

class HarnessRuntime:
    """既是单 turn 执行器（run_turn），也是 ConversationManager（start/run/close）。"""

    def __init__(self, store: SessionStore | None = None):
        bm25 = BM25Index(); bm25.load(Path(settings.index_dir) / "bm25")
        dense = DenseIndex(); dense.load(Path(settings.index_dir) / "faiss")
        llm = get_llm_provider()
        self.tracer = JsonlTracer(Path(settings.log_dir))
        compactor = ContextCompactor(llm, Path("src/legal_rag/prompts"), tracer=self.tracer)
        composer = ContextComposer(llm, PromptLibrary(Path("src/legal_rag/prompts")), compactor=compactor)
        self.deps = AgentDeps(
            llm=llm, composer=composer, compactor=compactor,
            retriever=HybridRetriever(bm25, dense), reranker=Reranker(),
        )
        self.graph = build_legal_rag_graph(self.deps)
        self.store: SessionStore = store or InMemorySessionStore()

    # -------- 会话级 API --------
    def start_session(self, user_id: str | None = None) -> str:
        return self.store.create(user_id)

    def close_session(self, session_id: str) -> None:
        self.store.close(session_id)

    def force_compact(self, session_id: str) -> WorkingContextDigest:
        """运维 / API /sessions/{id}/compact 入口。"""
        conv = self.store.load(session_id)
        # 构造一个最小 state（含 history + cumulative + sticky）调 compactor.force_compact
        state = self._snapshot_state_for_force(conv)
        patch = self.deps.compactor.force_compact(state)
        digest = WorkingContextDigest(**patch["working_context_digest"])
        conv.digests.append(digest)
        conv.cumulative_evidence = patch["cumulative_evidence"]
        conv.sticky_intake = type(conv.sticky_intake)(**patch["sticky_intake"])
        conv.digest_until_turn = digest.until_turn_id
        self.store.save(conv)
        return digest

    def run_turn(self, session_id: str, user_input: str,
                 jurisdiction: str | None = None,
                 options: dict | None = None) -> dict:
        conv = self.store.load(session_id)
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        state = self._assemble_state(conv, run_id, user_input, jurisdiction, options or {})

        try:
            final = self.graph.invoke(state)
        except Exception as e:
            final = {**state, "errors": [repr(e)]}
            self.tracer.event(run_id, session_id, "runtime",
                              latency_ms=0, output={}, error=repr(e))
        self._merge_back(conv, run_id, user_input, final)
        self.store.save(conv)
        return self._build_response(conv, final, run_id)

    def run_oneshot(self, user_input: str, jurisdiction: str | None = None) -> dict:
        sid = self.start_session()
        try:
            return self.run_turn(sid, user_input, jurisdiction)
        finally:
            self.close_session(sid)

    # -------- 实现细节 --------
    def _assemble_state(self, conv, run_id, user_input, jurisdiction, options):
        history = self._history_messages(conv)
        digest = conv.digests[-1].model_dump() if conv.digests else None
        return {
            "run_id": run_id,
            "session_id": conv.session_id,
            "turn_id": len(conv.turns) + 1,
            "user_query": user_input,
            "jurisdiction": jurisdiction or settings.default_jurisdiction,
            "history_messages": history,
            "sticky_intake": conv.sticky_intake.model_dump(),
            "cumulative_evidence": dict(conv.cumulative_evidence),
            "working_context_digest": digest,            # 注入当前生效 digest
            "compaction_events": [],
            "failed_queries": self._collect_failed(conv),
            "max_retrieval_retry": options.get("max_retrieval_retry", settings.max_retrieval_retry),
            "max_answer_revision": options.get("max_answer_revision", settings.max_answer_revision),
            "retrieval_retry_count": 0,
            "answer_revision_count": 0,
            "errors": [],
        }

    def _history_messages(self, conv):
        # digest 不进 history_messages（ContextComposer 单独渲染）；
        # 这里只放还没被 digest 涵盖的 turn 原文。
        cutoff = conv.digest_until_turn
        msgs = []
        for t in conv.turns:
            if t.turn_id <= cutoff:
                continue
            msgs.append({"role": t.role, "content": t.content})
        return msgs

    def _merge_back(self, conv, run_id, user_input, final):
        # 1) evidence remap
        retrieved = final.get("retrieved_docs", [])
        rewritten, remap = self.store.merge_evidence(conv.session_id, retrieved)
        citations = []
        for c in final.get("citations", []):
            cc = dict(c); cc["evidence_id"] = remap.get(cc["evidence_id"], cc["evidence_id"])
            citations.append(cc)
        for ev in rewritten:
            conv.cumulative_evidence[ev["evidence_id"]] = ev

        # 2) digest（如果本 turn 内发生过压缩）
        d = final.get("working_context_digest")
        if d and (not conv.digests or conv.digests[-1].digest_id != d["digest_id"]):
            conv.digests.append(WorkingContextDigest(**d))
            conv.digest_until_turn = d["until_turn_id"]
        # cumulative 与 sticky 也已经在压缩 patch 中应用过；这里同步
        conv.cumulative_evidence = final.get("cumulative_evidence", conv.cumulative_evidence)
        sticky = final.get("sticky_intake", {})
        if sticky:
            from legal_rag.schemas import StickyIntake
            conv.sticky_intake = StickyIntake(**sticky)

        # 3) 追加 turn
        now = time.time()
        conv.turns.append(TurnRecord(
            turn_id=len(conv.turns)+1, run_id=run_id, role="user",
            content=user_input, created_at=now,
        ))
        kind = "clarification" if final.get("is_clarification_turn") else "answer"
        assistant_text = final.get("final_answer") or final.get("clarification_text") or ""
        conv.turns.append(TurnRecord(
            turn_id=len(conv.turns)+1, run_id=run_id, role="assistant",
            content=assistant_text, kind=kind,
            citations=[c for c in citations],
            evidence_ids=[c["evidence_id"] for c in citations],
            created_at=now,
        ))

        conv.status = "awaiting_user" if final.get("is_clarification_turn") else "active"
        conv.last_active_at = now

    def _build_response(self, conv, final, run_id):
        comp_evts = final.get("compaction_events", [])
        return {
            "session_id": conv.session_id,
            "turn_id": len(conv.turns) // 2,
            "run_id": run_id,
            "kind": "clarification" if final.get("is_clarification_turn") else "answer",
            "answer": final.get("final_answer"),
            "clarification": final.get("clarification_text"),
            "citations": [c for c in final.get("citations", [])],
            "evidence_score": final.get("evidence_score"),
            "citation_score": final.get("citation_score"),
            "groundedness_score": final.get("groundedness_score"),
            "reviewer_score": final.get("reviewer_score"),
            "retrieval_retry_count": final.get("retrieval_retry_count", 0),
            "answer_revision_count": final.get("answer_revision_count", 0),
            "session_status": conv.status,
            "compactions_in_turn": len(comp_evts),
            "digest_token_after": (conv.digests[-1].token_estimate_after if conv.digests else None),
        }

    def _collect_failed(self, conv) -> list[str]:
        # 把所有 assistant turn 中曾经记录的失败 query 汇总
        out: list[str] = []
        for t in conv.turns:
            if t.kind == "answer" and t.role == "assistant":
                # 这里实际上需要从 trace / RunLog 取；MVP 简化为空
                pass
        return out

    def _snapshot_state_for_force(self, conv) -> dict:
        """force_compact 用：拼一个最小 state 喂给 compactor。"""
        history = [{"role": t.role, "content": t.content} for t in conv.turns]
        return {
            "session_id": conv.session_id,
            "run_id": "manual",
            "turn_id": len(conv.turns) // 2,
            "history_messages": history,
            "sticky_intake": conv.sticky_intake.model_dump(),
            "cumulative_evidence": dict(conv.cumulative_evidence),
            "working_context_digest": conv.digests[-1].model_dump() if conv.digests else None,
            "compaction_events": [],
        }
```

---

## 9. CLI

```python
# scripts/chat.py
import typer, json
from legal_rag.harness.runtime import HarnessRuntime

app = typer.Typer()

@app.command()
def session(jurisdiction: str = "CN"):
    rt = HarnessRuntime()
    sid = rt.start_session()
    typer.echo(f"session_id={sid}")
    while True:
        try: line = input("user> ").strip()
        except EOFError: break
        if not line: continue
        if line == "/compact":
            d = rt.force_compact(sid)
            typer.echo(f"[compacted] before={d.token_estimate_before} after={d.token_estimate_after}")
            continue
        if line in (":q", "exit"):
            rt.close_session(sid); break
        resp = rt.run_turn(sid, line, jurisdiction=jurisdiction)
        typer.echo(json.dumps(resp, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    app()
```

---

## 端到端验收

### 验收命令

```bash
# 1. mock 单 turn
EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock RERANKER_PROVIDER=noop \
  python scripts/ask.py --query "公司单方面解除劳动合同是否合法？"

# 2. 多轮，含 /compact
EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock RERANKER_PROVIDER=noop \
  printf '公司单方解除劳动合同合法吗？\n严重违反规章制度，没补偿\n那条法条具体写了什么？\n/compact\n继续：还有别的相关法条吗？\n:q\n' \
  | python scripts/chat.py

# 3. 看 trace
ls logs/runs/ logs/sessions/
grep harness.compaction logs/sessions/*.jsonl

pytest -q tests/test_context_compactor.py tests/test_graph_routes.py tests/test_conversation.py
```

### 验收通过条件

- 单 turn `ask.py` 输出含 9 个字段 + `compactions_in_turn`、`digest_token_after`。
- 多轮 `chat.py`：
  - 第 1 轮 `kind=="clarification"`、`session_status=="awaiting_user"`；
  - 第 2 轮 `kind=="answer"`；
  - `/compact` 命令立即返回 `before/after`，`after < before`；
  - 后续 turn 的 trace 里 ContextComposer 装配的 system 第二条是 digest。
- 当人为构造一个 8 轮历史 → 第 9 轮的 trace 里出现至少一个 `harness.compaction` 事件，且 `output.saved_ratio > 0`。
- ContextCompactor 在 LLM JSON 解析失败时走 `_fallback`，不抛异常。
- `tests/test_context_compactor.py` 覆盖：
  - `_should_compact` 三种触发路径（budget / turn_count / 70% ctx）；
  - `_collect_snapshot` 包含 history + intake + plan + failed_queries + retrieved + cumulative + assessments + draft + review；
  - LLM 成功与失败两条路径；
  - `dropped_evidence_ids` 真的从 cumulative 中删除；
  - `user_facts` 合并进 sticky_intake.pinned_facts 不重复；
  - history_messages 被截断到 keep_recent。
- Citation Checker 仍接受 cumulative 中的 evidence_id；pinned 中已被裁剪的 id 报"压缩策略 bug"。

---

## Codex Prompt

```text
基于 Phase 1–4，实现 Phase 5：Harness Runtime + LangGraph + ConversationManager + ContextCompactor。

按 PLAN/07_PHASE5_HARNESS_GRAPH.md 实现：

1. src/legal_rag/harness/{state,runtime,context_compactor,tracing,validators,policies,tool_registry}.py
2. src/legal_rag/prompts/context_compactor.md
3. src/legal_rag/graph/{routing,legal_rag_graph}.py
4. src/legal_rag/memory/session_store.py（SessionStore 抽象 + InMemorySessionStore）
5. scripts/chat.py + scripts/ask.py
6. tests/test_context_compactor.py + test_graph_routes.py + test_conversation.py

强约束：
1. ContextCompactor 在 harness 层；不进 graph；通过 ContextComposer 透明触发。
2. Graph 节点与 Phase 4 一致：没有 compactor 节点、没有 compactor_check 节点。
3. _should_compact：三条触发 (budget / turn_count / 70% ctx)，任一满足即压。
4. _collect_snapshot 必须收集：history_messages / intake_result / sticky_intake / plan / failed_queries / retrieved_docs / cumulative_evidence / evidence_assessments / draft_answer / review_comments。
5. 压缩 patch 必须：写 working_context_digest / 删 dropped_evidence_ids / 合并 user_facts 到 sticky_intake.pinned_facts / 截断 history_messages 到 keep_recent / 写 compaction_events / 通过 tracer 写 harness.compaction 事件。
6. ContextComposer 装配时若 state["working_context_digest"] 非空，把它渲染成第二条 system message（紧跟主 system）。
7. force_compact API：不依赖 graph，直接调 compactor.force_compact，把结果写入 SessionStore。
8. /compact CLI 命令调 force_compact。
9. JsonlTracer 双索引；compaction 事件 node="harness.compaction"。
10. Validators.check_citations 同时考虑 retrieved_docs + cumulative_evidence；digest.pinned_evidence_ids 中却已不在 pool 的 id 报"压缩策略 bug"。
11. ConversationManager._history_messages 不重复返回已被 digest 涵盖的旧 turn（按 conv.digest_until_turn 过滤）。

测试：
- tests/test_context_compactor.py：≥10 个用例，覆盖 §端到端验收 中所有 ContextCompactor 行为。
- tests/test_graph_routes.py：decide_route / should_retry_retrieval / should_revise_answer 各 4 个边界。
- tests/test_conversation.py：clarification 流程、代词引用复用 cumulative、8 轮触发自动压缩、/compact 强制触发、close 后再 run_turn 报 410 类异常。

不要实现 SQLite memory（保留 InMemorySessionStore 即可，Phase 6 再换）。

验收：
  EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock RERANKER_PROVIDER=noop python scripts/ask.py --query "测试"
  EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock RERANKER_PROVIDER=noop \
    printf 'A\nB\nC\nD\nE\nF\nG\nH\nI\n:q\n' | python scripts/chat.py    # 9 轮触发自动 compaction
  pytest -q tests/test_context_compactor.py tests/test_graph_routes.py tests/test_conversation.py
  grep harness.compaction logs/sessions/*.jsonl
```

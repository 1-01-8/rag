#!/usr/bin/env python
"""交互式 CLI — 直接问问题, 看完整答复 + Supervisor 审核 + 记忆.

用法 (最简单):
    python scripts/chat.py

进入 REPL 后输入问题, 回车提交; 输入 'exit' 或 Ctrl-D 退出.

可选参数:
    --statutes-collection ma_statutes      # 已建好的索引名 (默认临时构建 composite)
    --statutes-sparse <path>               # 配套 sparse json
    --session-id <id>                      # 持续多轮共享记忆
    --memory-root <dir>                    # 记忆目录 (默认 ./memory_store)
    --no-supervisor                        # 跳过审核员加速
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from multi_agent.schemas.document import Document, Chunk
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.tools.retrievers.turn_indexer import TurnIndexer
from multi_agent.tools.retrievers.history_search import HistorySearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.agents.supervisor import SupervisorAgent
from multi_agent.orchestration.supervised import run_with_supervisor
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.runner import run_query


# 内置 composite seed corpus — 覆盖民事/交通常见法条
_SEED_CHUNKS = [
    ("民法典-510", "民法典", "510",
     "当事人就合同补充内容没有约定的,按照合同相关条款或者交易习惯确定。"),
    ("民法典-563", "民法典", "563",
     "有下列情形之一的,当事人可以解除合同:法律规定的其他情形。"),
    ("民法典-577", "民法典", "577",
     "当事人一方不履行合同义务的,应当承担继续履行、采取补救措施或者赔偿损失等违约责任。"),
    ("民法典-584", "民法典", "584",
     "造成对方损失的,损失赔偿额应当相当于因违约所造成的损失,包括合同履行后可以获得的利益。"),
    ("民法典-703", "民法典", "703",
     "租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。"),
    ("民法典-707", "民法典", "707",
     "租赁期限六个月以上的,应当采用书面形式。当事人未采用书面形式,无法确定租赁期限的,视为不定期租赁。"),
    ("民法典-1165", "民法典", "1165",
     "行为人因过错侵害他人民事权益造成损害的,应当承担侵权责任。"),
    ("民法典-1184", "民法典", "1184",
     "侵害他人财产的,财产损失按照损失发生时的市场价格或者其他合理方式计算。"),
    ("民法典-1188", "民法典", "1188",
     "无民事行为能力人、限制民事行为能力人造成他人损害的,由监护人承担侵权责任。"),
    ("道路交通安全法-76", "道路交通安全法", "76",
     "机动车之间发生交通事故的,由有过错的一方承担赔偿责任;双方都有过错的,按照各自过错的比例分担责任。"),
    ("反家庭暴力法-23", "反家庭暴力法", "23",
     "当事人因遭受家庭暴力或者面临家庭暴力的现实危险,向人民法院申请人身安全保护令的,人民法院应当受理。"),
]


def _build_seed_index(name: str, sparse_path: Path) -> None:
    docs = [Document(
        law_name="composite_seed", law_short="composite", source_path="composite",
        chunks=[Chunk(doc_id=d, law_name="composite", law_short=ls,
                     article_no=an, text=t) for (d, ls, an, t) in _SEED_CHUNKS],
    )]
    build_index(documents=docs, collection_name=name,
                sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder())


def _print_answer(result: dict, *, with_supervisor: bool) -> None:
    """漂亮打印 lawyer 答复 + supervisor verdict."""
    final = result.get("lawyer_result", {}).get("final_answer") if with_supervisor \
        else result.get("final_answer")
    try:
        lo = json.loads(final or "{}")
    except Exception:
        lo = {}

    print("\n" + "=" * 70)
    print("📋 答复")
    print("=" * 70)
    print(lo.get("primary_answer", "(无核心结论)"))

    cits = lo.get("citations") or []
    if cits:
        print("\n📖 引用法条:")
        for c in cits:
            print(f"  • 《{c.get('law_short','?')}》第{c.get('article_no','?')}条")
            ex = (c.get("excerpt") or "").strip()
            if ex:
                print(f"      \"{ex[:80]}{'...' if len(ex) > 80 else ''}\"")

    five = lo.get("five_section") or {}
    if five:
        print("\n📑 详细分析:")
        for key, label in [
            ("dispute_analysis", "争议分析"),
            ("applicable_laws", "适用法规"),
            ("similar_cases", "相似类案"),
            ("remedy_suggestions", "维权建议"),
            ("risk_assessment", "风险评估"),
        ]:
            v = (five.get(key) or "").strip()
            if v:
                print(f"\n  【{label}】")
                # 缩进每行
                for line in v.split("\n"):
                    print(f"    {line}")

    if with_supervisor and "supervisor_verdict" in result:
        v = result["supervisor_verdict"]
        verdict = v.get("verdict", "?")
        emoji = {"pass": "✅", "revise": "⚠️", "reject": "❌"}.get(verdict, "?")
        print(f"\n{emoji} Supervisor: {verdict}  (confidence {v.get('confidence', 0):.2f})")
        issues = v.get("issues") or []
        for i in issues:
            print(f"     - {i}")

    print("=" * 70)


async def chat_loop(args) -> int:
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    # 索引: 已有 / 临时构造
    cleanup_coll = None
    if args.statutes_collection and args.statutes_sparse:
        coll = args.statutes_collection
        sparse = Path(args.statutes_sparse)
        print(f"使用现有索引: {coll}")
    else:
        coll = f"chat_seed_{uuid.uuid4().hex[:6]}"
        sparse = runs_root / f"{coll}_sparse.json"
        print(f"构建临时种子索引 {coll} ({len(_SEED_CHUNKS)} 条法条)... ", end="", flush=True)
        _build_seed_index(coll, sparse)
        cleanup_coll = coll
        print("ok")

    # Memory
    store = MarkdownMemoryStore(root=Path(args.memory_root))
    session_id = args.session_id or f"chat_{uuid.uuid4().hex[:6]}"
    encoder = DenseEncoder()
    history_coll = "ma_user_history_chat"
    turn_indexer = TurnIndexer(collection_name=history_coll, dense_encoder=encoder)
    history_search = HistorySearchTool(
        collection_name=history_coll, dense_encoder=encoder,
        default_session_id=session_id,
    )

    provider = OpenAICompatibleProvider()
    statute_search = StatuteSearchTool(collection_name=coll, sparse_artifact_path=sparse)

    print(f"会话 session_id = {session_id}")
    print(f"记忆目录: {Path(args.memory_root).resolve()}")
    print(f"Supervisor: {'开启' if not args.no_supervisor else '关闭'}")

    # 检测非交互 TTY — 常见原因: `conda run` 默认捕获 stdin
    if not sys.stdin.isatty():
        print("\n❌ 检测到 stdin 不是 TTY — 这通常是 `conda run` 捕获 stdin 导致.")
        print("请改用以下任一方式启动:")
        print("  1) conda activate qwen35  &&  python scripts/chat.py")
        print("  2) conda run -n qwen35 --live-stream python scripts/chat.py")
        print("  3) conda run -n qwen35 --no-capture-output python scripts/chat.py")
        if cleanup_coll:
            drop_collection(cleanup_coll)
        return 1

    print("\n输入你的问题 (输入 'exit' / 'quit' / 'q' 或 Ctrl-D 退出):")

    turn = 0
    try:
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question or question.lower() in {"exit", "quit", "q"}:
                break
            turn += 1
            print(f"\n[Turn {turn}] 处理中...", flush=True)

            tools_for_lawyer = [statute_search]
            if turn > 1:
                tools_for_lawyer.append(history_search)

            try:
                if args.no_supervisor:
                    result = await run_query(
                        query=question,
                        agent_factory=lambda p, r: LawyerAgent(
                            name="lawyer", role="advisor", provider=p, recorder=r,
                            tools=tools_for_lawyer,
                            model="qwen3.5-9b", specialty=args.specialty,
                            max_steps=6, max_tool_calls=8, max_pre_tool_rejections=2,
                        ),
                        provider=provider, runs_root=runs_root,
                        session_id=session_id, memory_store=store,
                        turn_indexer=turn_indexer,
                    )
                    _print_answer(result, with_supervisor=False)
                else:
                    result = await run_with_supervisor(
                        query=question,
                        lawyer_factory=lambda p, r: LawyerAgent(
                            name="lawyer", role="advisor", provider=p, recorder=r,
                            tools=tools_for_lawyer,
                            model="qwen3.5-9b", specialty=args.specialty,
                            max_steps=6, max_tool_calls=8, max_pre_tool_rejections=2,
                        ),
                        supervisor_factory=lambda p, r: SupervisorAgent(
                            name="supervisor", role="qa", provider=p, recorder=r,
                            model="qwen3.5-9b", max_steps=3, max_pre_tool_rejections=5,
                        ),
                        lawyer_provider=provider, supervisor_provider=provider,
                        runs_root=runs_root,
                    )
                    _print_answer(result, with_supervisor=True)
                    # 持久化 turn (run_with_supervisor 没接 memory; 单独写)
                    from datetime import datetime
                    from multi_agent.schemas.memory import Turn, StickyContext
                    sticky = store.read_sticky(session_id) or StickyContext(session_id=session_id)
                    if result["lawyer_run_id"] not in sticky.linked_runs:
                        sticky.linked_runs.append(result["lawyer_run_id"])
                    existing = store.recent_turns(session_id, n=999)
                    next_no = max((t.turn for t in existing), default=0) + 1
                    final_answer = result["lawyer_result"].get("final_answer") or ""
                    t = Turn(turn=next_no, run_id=result["lawyer_run_id"],
                            started_at=datetime.now(), finished_at=datetime.now(),
                            question=question, final_answer=final_answer,
                            agents_invoked=["lawyer", "supervisor"])
                    store.append_turn(session_id, t)
                    await turn_indexer.index_turn(session_id=session_id, turn=t)
                    store.write_sticky(sticky)
            except Exception as e:
                print(f"\n❌ 错误: {type(e).__name__}: {e}")
                continue
    finally:
        if cleanup_coll:
            drop_collection(cleanup_coll)
            print(f"\n清理临时索引 {cleanup_coll}")
        print(f"\n会话 {session_id} 持久化到 {Path(args.memory_root).resolve() / 'sessions' / session_id}")
        print("再见 👋")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-agent legal RAG 交互式 CLI")
    p.add_argument("--statutes-collection", default=None,
                   help="已建好的 Qdrant 法条集合名 (省略则临时构建 11 条种子索引)")
    p.add_argument("--statutes-sparse", default=None,
                   help="配套 sparse json 路径")
    p.add_argument("--session-id", default=None,
                   help="会话 ID; 同 id 可继续上次对话")
    p.add_argument("--memory-root", default="memory_store_chat",
                   help="记忆目录 (默认 ./memory_store_chat)")
    p.add_argument("--runs-root", default="runs",
                   help="trace 目录 (默认 ./runs)")
    p.add_argument("--specialty", default="民事",
                   help="律师专业方向 民事/交通/婚姻/房产/劳动/通用 (默认 民事)")
    p.add_argument("--no-supervisor", action="store_true",
                   help="跳过审核员加速 (但失去引用真实性校验)")
    args = p.parse_args()
    return asyncio.run(chat_loop(args))


if __name__ == "__main__":
    sys.exit(main())

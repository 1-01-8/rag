# 01 · 整体架构与核心数据结构

> 所有 Phase 文档都默认 Codex 已经读过本文。
> 多轮对话与上下文压缩从本文起就是一等公民，不存在"先单 turn 再加多 turn"的两阶段。

---

## 1. Harness 设计哲学

Harness 在本项目中指**包裹 LLM/Agent 的完整工程环境**：

```text
User Request (within a Session)
  ↓
Harness Runtime (= ConversationManager)
  ├── SessionStore                  (持久化会话状态)
  ├── ContextComposer               (per-agent 拼 messages，含 token budget 与裁剪)
  │     └── ContextCompactor        (Claude Code 风格的工作上下文压缩；
  │                                   ContextComposer 装配前估算 → 超阈值透明触发；
  │                                   产出 WorkingContextDigest 作为后续所有 LLM 调用的 system 前缀)
  ├── State Manager                 (LegalRAGState：本 turn 的工作内存)
  ├── Tool Registry                 (各 agent 只能调用白名单工具)
  ├── Agent Graph Executor          (LangGraph，对压缩透明)
  ├── Validator / Guardrail         (含 Citation Checker)
  ├── Observability Logger          (JSONL trace，按 run_id 与 session_id 双索引；compaction 写专属事件)
  ├── Evaluation Harness            (回归门禁)
  └── Memory Updater                (写 SessionStore 与长期记忆)
```

> **ContextCompactor 不是 agent，没有 graph 节点。** 它由 ContextComposer 在装配 messages 前根据 token 估算自动调起，对 graph 与 agent 透明。等价于 Claude Code 的 `/compact` —— 把整个 session 的工作过程（用户对话 + agent 中间产物 + 检索/证据/草稿/复核记录）一次性折叠成结构化 digest，digest 之后会作为所有 LLM 调用的固定 system 前缀。

法律 RAG 比通用 RAG 更需要 Harness，因为典型失败模式不是模型不会写，而是：

- 引用不相关法条 / 编造法条；
- 忽略法条时效；
- 把案例事实错误类比；
- 检索不充分却强行下结论；
- 没区分「法条依据」与「类案参考」；
- 缺少不确定性提示；
- 用户上传的合同未与法律对齐；
- 错误回答无法复盘；
- 多轮里反复问同一个 missing_info、同一个 evidence 反复检索、token 爆炸。

Harness 的作用：把这些问题变成**可检查、可阻断、可复盘**的工程状态。

---

## 2. 顶层目录结构

```text
legal-research-agent/
├── README.md
├── PLAN/                              # 把 00–11 这套文档放这里
├── pyproject.toml
├── .env.example
├── .gitignore
├── data/
│   ├── raw/{statutes,cases,contracts}/
│   ├── processed/
│   ├── indexes/{bm25,faiss}/
│   └── eval/{queries.jsonl,golden_answers.jsonl,multiturn.jsonl}
├── src/legal_rag/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── app.py
│   ├── providers/                     # 见 02_MODEL_PROVIDERS.md
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── embedding_local.py
│   │   ├── embedding_siliconflow.py
│   │   ├── reranker_local.py
│   │   ├── reranker_siliconflow.py
│   │   ├── llm_local.py
│   │   └── llm_siliconflow.py
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── cleaners.py
│   │   ├── chunkers.py
│   │   ├── metadata_extractor.py
│   │   └── pipeline.py
│   ├── indexes/
│   │   ├── bm25_index.py
│   │   ├── dense_index.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── agents/
│   │   ├── _deps.py
│   │   ├── _context_composer.py       # 上下文拼装 + 预算；内部委托 harness/context_compactor
│   │   ├── intake_agent.py            # 含 sticky/clarification
│   │   ├── memory_retriever.py
│   │   ├── planner_agent.py
│   │   ├── statute_agent.py
│   │   ├── case_agent.py
│   │   ├── contract_agent.py
│   │   ├── evidence_checker.py
│   │   ├── query_rewriter.py
│   │   ├── answer_agent.py            # 含 cumulative_evidence 引用
│   │   ├── reviewer_agent.py
│   │   └── memory_agent.py
│   ├── harness/
│   │   ├── runtime.py                 # ConversationManager
│   │   ├── state.py
│   │   ├── context_compactor.py       # ★ Claude-Code-style 工作上下文压缩
│   │   ├── tool_registry.py
│   │   ├── validators.py
│   │   ├── policies.py
│   │   ├── tracing.py
│   │   └── errors.py
│   ├── graph/
│   │   ├── legal_rag_graph.py
│   │   └── routing.py
│   ├── memory/
│   │   ├── db.py
│   │   ├── models.py                  # 长期 4 张表 + 会话 3 张表
│   │   ├── query_memory.py
│   │   ├── evidence_memory.py
│   │   ├── route_memory.py
│   │   ├── prompt_memory.py
│   │   └── session_store.py           # 会话持久化
│   ├── prompts/
│   │   ├── intake.md
│   │   ├── planner.md
│   │   ├── query_rewrite.md
│   │   ├── evidence_check.md
│   │   ├── answer.md
│   │   ├── reviewer.md
│   │   └── context_compactor.md       # ★ 工作上下文压缩 prompt
│   └── eval/
│       ├── metrics.py
│       ├── run_eval.py
│       ├── retrieval_eval.py
│       ├── answer_eval.py
│       ├── multiturn_eval.py
│       └── regression_gates.py
├── tests/
│   ├── test_providers.py
│   ├── test_chunker.py
│   ├── test_metadata.py
│   ├── test_retrieval.py
│   ├── test_context_composer.py
│   ├── test_context_compactor.py
│   ├── test_agents.py
│   ├── test_graph_routes.py
│   ├── test_conversation.py
│   ├── test_memory.py
│   └── test_eval_gates.py
└── scripts/
    ├── check_providers.py
    ├── ingest_docs.py
    ├── build_indexes.py
    ├── retrieve.py
    ├── chat.py                        # 多轮 CLI（替代 ask.py）
    ├── ask.py                         # 单 turn 语法糖
    ├── run_eval.py
    └── reset_demo_db.py
```

---

## 3. 核心数据结构

### 3.1 DocumentChunk

```python
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    source_type: Literal["statute", "case", "contract", "article", "unknown"]

    jurisdiction: Optional[str] = None
    law_name: Optional[str] = None
    article_number: Optional[str] = None
    article_number_raw: Optional[str] = None
    chapter: Optional[str] = None
    effective_date: Optional[str] = None
    valid_status: Optional[Literal["valid", "amended", "repealed", "unknown"]] = None

    case_name: Optional[str] = None
    court: Optional[str] = None
    trial_level: Optional[str] = None
    cause_of_action: Optional[str] = None

    contract_section: Optional[str] = None
    clause_type: Optional[str] = None

    keywords: list[str] = Field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    extra: dict[str, Any] = Field(default_factory=dict)
```

### 3.2 RetrievedEvidence

```python
class RetrievedEvidence(BaseModel):
    evidence_id: str            # **session 内全局唯一**，由 SessionStore 分配/复用
    chunk_id: str
    text: str
    source_type: str
    source_path: str
    score_bm25: float | None = None
    score_dense: float | None = None
    score_hybrid: float | None = None
    score_rerank: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

`evidence_id` 规则：

- 第一次见到某 `chunk_id` → 分配 `ev_<session_id_short>_<seq>`；
- 后续 turn 再次检索到同 `chunk_id` → 复用旧 `evidence_id`；
- 这样代词引用（"那条法条"）能跨 turn 复用。

### 3.3 EvidenceAssessment

```python
class EvidenceAssessment(BaseModel):
    evidence_id: str
    relevance: Literal["high", "medium", "low"]
    support: Literal["full", "partial", "none"]
    freshness: Literal["valid", "outdated", "unknown"]
    citation_ready: bool
    problem: str | None = None
```

### 3.4 Citation

```python
class Citation(BaseModel):
    evidence_id: str
    source_type: str
    law_name: str | None = None
    article_number: str | None = None
    case_name: str | None = None
    quote: str                  # 必须是 evidence.text 的子串
    span: tuple[int, int] | None = None
```

### 3.5 会话相关 schema（Phase 1 就要落地）

```python
class TurnRecord(BaseModel):
    turn_id: int                          # session 内 1 起递增
    run_id: str                           # 对应 LegalRAGState.run_id
    role: Literal["user", "assistant", "system"]
    content: str                          # user 是脱敏后原文；assistant 是 final_answer / clarification
    citations: list[Citation] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    kind: Literal["answer", "clarification", "compaction_marker"] = "answer"
    created_at: float                     # epoch seconds

class StickyIntake(BaseModel):
    """会话级 intake：除非用户明显切换话题，否则后续 turn 复用。"""
    legal_domain: str | None = None
    task_type: str | None = None
    jurisdiction: str | None = None
    risk_level: str | None = None
    pinned_facts: list[str] = Field(default_factory=list)
    open_missing_info: list[str] = Field(default_factory=list)

class WorkingContextDigest(BaseModel):
    """Claude-Code 风格的工作上下文压缩产物。
    把"agent 工作过程的全部中间产物 + 用户对话"压成一份结构化摘要，
    之后所有 LLM 调用都把它作为固定 system 前缀。"""
    digest_id: str
    until_run_id: str | None = None       # 最后一次被涵盖的 run
    until_turn_id: int = 0
    triggered_by: Literal["budget", "turn_count", "manual"] = "budget"

    # 各类工作产物的浓缩
    user_facts: list[str] = Field(default_factory=list)        # 用户已确认事实
    intake_summary: str = ""                                   # 已识别的领域 / 任务 / 风险等级
    retrieval_summary: str = ""                                # 已尝试过哪些 query、哪些命中、哪些 dead-end
    evidence_summary: str = ""                                 # 关键 evidence_id 与作用（与 pinned_evidence_ids 对齐）
    answer_summary: str = ""                                   # 历次草稿/终稿主旨
    reviewer_observations: list[str] = Field(default_factory=list)  # 反复出现的反方意见
    open_issues: list[str] = Field(default_factory=list)       # 未答清的子问题
    pinned_evidence_ids: list[str] = Field(default_factory=list)    # cumulative_evidence 中必须保留的
    dropped_evidence_ids: list[str] = Field(default_factory=list)   # 已淘汰的

    token_estimate_before: int = 0
    token_estimate_after: int = 0
    created_at: float = 0.0

class ConversationState(BaseModel):
    session_id: str
    user_id: str | None = None
    turns: list[TurnRecord] = Field(default_factory=list)
    sticky_intake: StickyIntake = Field(default_factory=StickyIntake)
    cumulative_evidence: dict[str, dict[str, Any]] = Field(default_factory=dict)
    digests: list[WorkingContextDigest] = Field(default_factory=list)
    """工作上下文压缩历史；最近一条是当前生效的 digest，作为所有 LLM 调用的 system 前缀。"""
    digest_until_turn: int = 0
    created_at: float = 0.0
    last_active_at: float = 0.0
    status: Literal["active", "awaiting_user", "closed"] = "active"
```

### 3.6 LegalRAGState（单 turn 工作内存）

```python
from typing import Any, TypedDict

class LegalRAGState(TypedDict, total=False):
    # ===== 会话上下文 =====
    session_id: str
    turn_id: int
    history_messages: list[dict[str, str]]        # 经 ContextComposer 截断/含摘要后的消息
    sticky_intake: dict[str, Any]                 # StickyIntake.model_dump()
    cumulative_evidence: dict[str, dict[str, Any]]

    # ===== 本 turn 输入 =====
    run_id: str
    user_query: str
    jurisdiction: str | None
    legal_domain: str | None
    task_type: str | None

    # ===== Intake / Plan =====
    intake_result: dict[str, Any]
    is_clarification_turn: bool                   # True → 不进检索/answer，直接出 clarification
    clarification_text: str | None
    memory_hints: list[dict[str, Any]]
    plan: list[str]
    route: list[str]

    # ===== 检索循环 =====
    rewritten_queries: list[str]
    failed_queries: list[str]                     # session 级累积，避免跨 turn 重复试错
    retrieved_docs: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]
    evidence_assessments: list[dict[str, Any]]
    evidence_score: float
    evidence_gaps: list[str]
    should_retry: bool
    rewrite_hint: str | None
    retrieval_retry_count: int
    max_retrieval_retry: int                      # 默认 2

    # ===== 生成 + 复核 =====
    draft_answer: str | None
    citations: list[dict[str, Any]]
    review_comments: list[str]
    citation_score: float
    groundedness_score: float
    reviewer_score: float                         # = min(citation_score, groundedness_score)
    answer_revision_count: int
    max_answer_revision: int                      # 默认 1
    final_answer: str | None

    # ===== 工作上下文压缩 =====
    working_context_digest: dict[str, Any] | None # 当前生效的 WorkingContextDigest（透明注入到每次 LLM system 前缀）
    compaction_events: list[dict[str, Any]]       # 本 turn 内发生的 compaction 元数据（trace 用）

    # ===== 收尾 =====
    memory_updates: list[dict[str, Any]]
    errors: list[str]
```

**关键不变量**：

- `retrieval_retry_count` 与 `answer_revision_count` 必须分开。
- `cumulative_evidence` ∪ `retrieved_docs` 是 Citation Checker 的合法引用池。
- `evidence_id` 在 session 内全局唯一，由 SessionStore 分配。
- `failed_queries` 是 session 级，不在 turn 结束时重置。

---

## 4. Harness 节点契约

每个 agent 节点必须满足：

```text
输入：LegalRAGState
输出：LegalRAGState 的 dict patch（只更新自己负责的 key）
禁止：直接修改外部数据库（仅 Memory Agent / Compactor 的 session 写回除外）
禁止：访问未在 tool_registry 注册的工具
禁止：自行拼 LLMMessage 列表（必须经 ContextComposer）
必须：通过 Pydantic 校验后才写入 state
必须：写 trace
```

---

## 5. Graph 流程

> **注意：graph 里没有 compaction 节点。** 上下文压缩由 `ContextComposer` 在装配 messages 前透明触发，对 graph 与 agent 完全不可见。

```text
START
  ↓
intake_agent                              # 复用 sticky_intake；判断 clarification / 话题切换
  ↓
[is_clarification_turn?]
  ├── yes → finalize_clarification → memory_write → END
  └── no  → continue
  ↓
memory_retriever
  ↓
planner_agent
  ↓
route_decider                             # graph/routing.py
  ↓
retrieval_agents (并行: statute / case / contract)
  ↓
reranker (可选)
  ↓
evidence_checker
  ↓
[retrieval_retry?]
  ├── yes → query_rewriter → retrieval_agents → evidence_checker
  └── no  → answer_agent
  ↓
reviewer_agent
  ↓
[answer_revision?]
  ├── yes → answer_agent (revision 模式)
  └── no  → finalizer (拼接免责声明 + Citation Checker)
  ↓
memory_agent                              # 写 run_log + 追加 turn 到 SessionStore
  ↓
END
```

任意 agent 调 `deps.composer.compose(...)` 时：

```text
ContextComposer.compose(...)
  ├── 估算 (system_prefix + digest + history + evidence + user) tokens
  ├── if estimated > SESSION_COMPACT_TRIGGER_TOKENS:
  │      └── ContextCompactor.compact(state)        # 同步调一次 LLM
  │            ├── 收集 state 的全部工作产物：
  │            │     - turns (用户/助手历史)
  │            │     - intake_result / sticky_intake
  │            │     - planner.plan / search_queries
  │            │     - retrieval 历史 (failed_queries + 旧 retrieved_docs)
  │            │     - cumulative_evidence
  │            │     - 历次 evidence_assessments
  │            │     - 历次 draft_answer / review_comments
  │            ├── 调 LLM 输出 WorkingContextDigest
  │            ├── 把 digest 写入 state["working_context_digest"]
  │            ├── 从 cumulative_evidence 删除 dropped_evidence_ids
  │            ├── 把 user_facts 合并进 sticky_intake.pinned_facts
  │            └── 把 compaction event 加入 state["compaction_events"]
  └── 用 state 的 (digest 作 system 前缀 + 最近 N 轮原文 + 当前 evidence + user) 装配 messages
```

判定函数（实现见 PHASE5）：

```python
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

阈值与 PHASE4 Evidence Checker 的语义保持一致：

```text
evidence_score >= 0.75 → 直接生成
0.50 <= evidence_score < 0.75 → 优先 retry，超出 max 后进入"谨慎"模式生成
evidence_score < 0.50 → 优先 retry，超出 max 后走"证据不足型回答"模板
```

---

## 6. Tool 权限白名单

```text
intake_agent:           {}
memory_retriever:       {memory_read}
planner_agent:          {memory_read}
statute_agent:          {bm25_search, dense_search, hybrid_search, rerank}
case_agent:             {bm25_search, dense_search, hybrid_search, rerank}
contract_agent:         {bm25_search, dense_search, hybrid_search}
evidence_checker:       {}
query_rewriter:         {memory_read}
answer_agent:           {}
reviewer_agent:         {}
memory_agent:           {memory_read, memory_write, session_write}
```

ContextCompactor 在 harness 层运行，不属于 agent，不申请 tool 权限；它只读 LegalRAGState 并写回 `working_context_digest`。`session_write` 仅 `memory_agent` 与 `ConversationManager` 直接持有。

# Phase 6 · Memory（会话存储 + 长期记忆 + 自进化）

## 依赖

- Phase 5：HarnessRuntime + LangGraph + InMemorySessionStore 已端到端可用。
- Phase 4：`memory_retriever` 与 `memory_agent` 在 Phase 4 是桩，本阶段替换为真实现。

## 本阶段交付物

1. `src/legal_rag/memory/db.py`（SQLAlchemy engine / session）。
2. `src/legal_rag/memory/models.py`：
   - 长期 4 张表：`QueryMemory / EvidenceMemory / RouteMemory / RunLog`
   - 会话 4 张表：`Session / SessionTurn / SessionEvidence / SessionDigest`
3. `src/legal_rag/memory/{query_memory,evidence_memory,route_memory,prompt_memory}.py`（长期记忆 reader/writer）。
4. `src/legal_rag/memory/session_store.py`：新增 `SqliteSessionStore`（继承 Phase 5 的抽象）。
5. 替换 `src/legal_rag/agents/memory_retriever.py` 与 `memory_agent.py` 为真实现。
6. 修改 `HarnessRuntime`：默认 store=SqliteSessionStore；`close_session` 触发 session-level memory write。
7. `scripts/reset_demo_db.py`（含 `--gc-sessions --older-than-days N`）。
8. `tests/test_memory.py`、补充 `tests/test_conversation.py` 的 SQLite round-trip 用例。

> 设计原则：长期表（QueryMemory 等）与会话表（Session 等）共用同一 SQLite 文件、同一 `Base`，但 reader/writer 严格分离：MemoryReader 不读 session 表，SessionStore 不读长期表。

---

## 1. 表结构

### 1.1 长期记忆（4 张）

```python
class QueryMemory(Base):
    __tablename__ = "query_memory"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_type: Mapped[str | None] = mapped_column(String(64))
    legal_domain: Mapped[str | None] = mapped_column(String(64))
    original_query_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    pattern: Mapped[str | None] = mapped_column(Text)
    rewritten_query: Mapped[str | None] = mapped_column(Text)
    success_score: Mapped[float | None] = mapped_column(Float)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    pattern_embedding: Mapped[bytes | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

class EvidenceMemory(Base):
    __tablename__ = "evidence_memory"
    id: Mapped[int] = mapped_column(primary_key=True)
    legal_domain: Mapped[str | None] = mapped_column(String(64))
    task_type: Mapped[str | None] = mapped_column(String(64))
    chunk_id: Mapped[str] = mapped_column(String(128), index=True)
    law_name: Mapped[str | None] = mapped_column(String(128))
    article_number: Mapped[str | None] = mapped_column(String(32))
    usefulness_score: Mapped[float | None] = mapped_column(Float)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

class RouteMemory(Base):
    __tablename__ = "route_memory"
    id: Mapped[int] = mapped_column(primary_key=True)
    query_pattern: Mapped[str | None] = mapped_column(Text)
    task_type: Mapped[str | None] = mapped_column(String(64))
    legal_domain: Mapped[str | None] = mapped_column(String(64))
    route_json: Mapped[str | None] = mapped_column(Text)
    success_score: Mapped[float | None] = mapped_column(Float)
    avg_retry_count: Mapped[float | None] = mapped_column(Float)
    pattern_embedding: Mapped[bytes | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

class RunLog(Base):
    __tablename__ = "run_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    user_query_hash: Mapped[str | None] = mapped_column(String(64))
    task_type: Mapped[str | None] = mapped_column(String(64))
    legal_domain: Mapped[str | None] = mapped_column(String(64))
    evidence_score: Mapped[float | None] = mapped_column(Float)
    citation_score: Mapped[float | None] = mapped_column(Float)
    groundedness_score: Mapped[float | None] = mapped_column(Float)
    reviewer_score: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    retrieval_retry_count: Mapped[int] = mapped_column(Integer, default=0)
    answer_revision_count: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

### 1.2 会话存储（4 张）

```python
class Session(Base):
    __tablename__ = "session"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    sticky_intake_json: Mapped[str | None] = mapped_column(Text)
    summary_until_turn: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_active_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

class SessionTurn(Base):
    __tablename__ = "session_turn"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    turn_id: Mapped[int] = mapped_column(Integer)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(16), default="answer")
    content_redacted: Mapped[str] = mapped_column(Text)         # 写库前必脱敏
    citations_json: Mapped[str | None] = mapped_column(Text)
    evidence_ids_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

class SessionEvidence(Base):
    __tablename__ = "session_evidence"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    evidence_id: Mapped[str] = mapped_column(String(64))
    chunk_id: Mapped[str] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text)                     # contract: 仅占位
    metadata_json: Mapped[str | None] = mapped_column(Text)
    seq: Mapped[int] = mapped_column(Integer, default=0)        # session 内分配序号

class SessionDigest(Base):
    """工作上下文压缩历史（Claude-Code 风格 /compact 的产物）。"""
    __tablename__ = "session_digest"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    digest_id: Mapped[str] = mapped_column(String(64), unique=True)
    until_run_id: Mapped[str | None] = mapped_column(String(64))
    until_turn_id: Mapped[int] = mapped_column(Integer, default=0)
    triggered_by: Mapped[str] = mapped_column(String(16), default="budget")  # budget|turn_count|manual
    user_facts_json: Mapped[str | None] = mapped_column(Text)
    intake_summary: Mapped[str | None] = mapped_column(Text)
    retrieval_summary: Mapped[str | None] = mapped_column(Text)
    evidence_summary: Mapped[str | None] = mapped_column(Text)
    answer_summary: Mapped[str | None] = mapped_column(Text)
    reviewer_observations_json: Mapped[str | None] = mapped_column(Text)
    open_issues_json: Mapped[str | None] = mapped_column(Text)
    pinned_evidence_ids_json: Mapped[str | None] = mapped_column(Text)
    dropped_evidence_ids_json: Mapped[str | None] = mapped_column(Text)
    token_estimate_before: Mapped[int] = mapped_column(Integer, default=0)
    token_estimate_after: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

---

## 2. SqliteSessionStore

继承 Phase 5 的 `SessionStore` 抽象，把 `InMemorySessionStore` 中的 dict 替换成 SQLite 操作。

要点：

- `merge_evidence`：先查 `SessionEvidence` 中是否已有同 (session_id, chunk_id)；有则复用 evidence_id，无则按 `seq` 自增分配 `ev_<sid>_<seq>`。
- `append_turn`：`content_redacted = harness/policies.redact(turn.content)`。
- `save`：把 `ConversationState.digests` 中尚未入库的 `WorkingContextDigest`（按 `digest_id` 去重）插入 `SessionDigest`。
- `append_digest(session_id, digest)`：写入一条 `SessionDigest`；ContextCompactor 触发后由 ConversationManager 调用，避免等到 turn 结束才落库。
- `gc(older_than_days)`：删除 `last_active_at < now - N days` 的 session 与级联 turn / evidence / digest。

合同隐私：

```python
def _maybe_redact_evidence_text(metadata: dict, text: str) -> str:
    if metadata.get("source_type") == "contract":
        return "<REDACTED:contract>"
    return text
```

---

## 3. memory_score 公式

```python
def compute_memory_score(success_score, created_at, legal_domain_match) -> float:
    days = max(0.0, (datetime.utcnow() - created_at).total_seconds() / 86400)
    recency = math.exp(-days / 30.0)
    domain  = 1.0 if legal_domain_match else 0.5
    return success_score * recency * domain
```

仅 `memory_score >= 0.5` 进入 hints。

---

## 4. memory_retriever 真实现

```python
def run(state, deps) -> dict:
    if deps.memory_read is None: return {"memory_hints": []}
    legal_domain = state.get("legal_domain") or state.get("intake_result", {}).get("legal_domain")
    task_type    = state.get("task_type")    or state.get("intake_result", {}).get("task_type")
    query = state["user_query"]
    hints: list[dict] = []
    hints += deps.memory_read.find_route_hints(query, task_type, legal_domain, top_k=3)
    hints += deps.memory_read.find_evidence_hints(legal_domain, task_type, top_k=5)
    hints += deps.memory_read.find_query_hints(query, task_type, legal_domain, top_k=3)
    hints = [h for h in hints if h["score"] >= 0.5]
    return {"memory_hints": hints}
```

`find_*` 内部用 `providers.factory.get_embedding_provider()` 计算 query 向量与 `pattern_embedding` 的余弦相似度，结合 `compute_memory_score`。

---

## 5. memory_agent：分两层

```python
def run(state, deps) -> dict:
    """每个 turn 必调，写 RunLog；其余表的写入延迟到 close_session。"""
    if deps.memory_write is None: return {"memory_updates": []}
    deps.memory_write.write_run_log(state)
    return {"memory_updates": [{"type": "run_log"}]}


# HarnessRuntime.close_session 内调用：
def write_session_memory(conv: ConversationState, deps: AgentDeps) -> None:
    """session 关闭时，把整段会话聚合为长期记忆。"""
    if deps.memory_write is None: return
    runs = deps.memory_write.list_runs_in_session(conv.session_id)
    if not runs: return
    avg_reviewer = sum(r.reviewer_score or 0 for r in runs) / len(runs)
    success = avg_reviewer >= 0.75 and not any(r.success == 0 and r.reviewer_score is None for r in runs)
    if success:
        deps.memory_write.write_query_memory_session(conv, success_score=avg_reviewer)
        deps.memory_write.write_route_memory_session(conv, success_score=avg_reviewer)
        deps.memory_write.write_evidence_memory_session(conv, success_score=avg_reviewer)
    else:
        deps.memory_write.write_query_memory_failure(conv)
```

`HarnessRuntime.close_session`：

```python
def close_session(self, session_id: str) -> None:
    conv = self.store.load(session_id)
    write_session_memory(conv, self.deps)
    self.store.close(session_id)
```

---

## 6. 隐私

- 所有 `original_query` 写库前 `sha256`。
- `pattern` 仅在 `success=True` 时由 LLM 生成抽象短语。
- `EvidenceMemory` 不写 `source_type=contract` 的 chunk。
- `SessionTurn.content_redacted` 必经 `harness/policies.redact()`。
- `SessionEvidence.text` 在 contract 时仅占位。

---

## 端到端验收

### 验收命令

```bash
python scripts/reset_demo_db.py        # 建表

# 跑两段相似 session
EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock RERANKER_PROVIDER=noop \
  printf '公司违法解除劳动合同怎么办？\n严重违反规章制度被开除\n:q\n' \
  | python scripts/chat.py

EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock RERANKER_PROVIDER=noop \
  printf '被公司单方面辞退是否违法？\n没违反规章制度\n:q\n' \
  | python scripts/chat.py

# 看 trace
ls logs/sessions/                              # 应有 2 个 session jsonl
sqlite3 legal_rag_memory.sqlite3 "SELECT count(*) FROM session;"            # 2
sqlite3 legal_rag_memory.sqlite3 "SELECT count(*) FROM session_turn;"       # ≥4
sqlite3 legal_rag_memory.sqlite3 "SELECT count(*) FROM run_log;"            # ≥4
sqlite3 legal_rag_memory.sqlite3 "SELECT count(*) FROM query_memory;"       # ≥0（mock LLM 时可能 0；真模型应 ≥2）

# session GC
python scripts/reset_demo_db.py --gc-sessions --older-than-days 0           # 立刻清

pytest -q tests/test_memory.py
```

### 验收通过条件

- 两段 session 关闭后 `session.status == 'closed'`。
- `run_log` 每 turn 都有一条。
- `query_memory` / `route_memory` / `evidence_memory` 仅在 `avg_reviewer_score >= 0.75` 时新增（mock LLM 默认 reviewer_score=0，故走 failure 分支；用真模型则应当看到正向条目）。
- `SessionEvidence.text` 在 contract 来源时为 `<REDACTED:contract>`。
- 没有任何明文用户 query 出现在 `query_memory.original_query_hash` 字段（必须是 64 位 hex）。
- `gc(--older-than-days=0)` 删除两段 session 后，`session_turn` / `session_evidence` / `session_digest` 也级联清空。
- 单测覆盖：
  - `compute_memory_score`：成功 + 今天 + 同领域 ≈ 1.0；30 天前 ≈ 0.37；
  - `merge_evidence`：同 chunk_id 复用旧 evidence_id；
  - reviewer_score=0.6 的 run 不写正向 query_memory；
  - SqliteSessionStore round-trip 与 InMemorySessionStore 行为一致（共享一组 contract test）。

---

## Codex Prompt

```text
基于 Phase 1–5，实现 Phase 6：Memory（会话存储 + 长期记忆）。

按 PLAN/08_PHASE6_MEMORY.md 实现：

1. src/legal_rag/memory/db.py（engine + Base + init_db）
2. src/legal_rag/memory/models.py（4 张长期表 + 4 张会话表，按 PLAN §1）
3. src/legal_rag/memory/{query_memory,evidence_memory,route_memory,prompt_memory}.py
4. src/legal_rag/memory/session_store.py 中新增 SqliteSessionStore，继承 Phase 5 的抽象
5. 把 agents/memory_retriever.py 与 agents/memory_agent.py 替换为真实现（PLAN §4 §5）
6. HarnessRuntime 默认 store=SqliteSessionStore；close_session 调 write_session_memory
7. scripts/reset_demo_db.py（init + --gc-sessions --older-than-days N）
8. tests/test_memory.py，并补 tests/test_conversation.py 的 SQLite round-trip

要求：
- 长期表与会话表共用同一 Base 与 SQLite 文件；reader/writer 严格分离。
- merge_evidence：维护 (session_id, chunk_id) → evidence_id 的 SQLite 索引；同 chunk_id 复用。
- 写库前隐私：original_query → sha256；contract 不写 EvidenceMemory；contract 的 SessionEvidence.text = "<REDACTED:contract>"；SessionTurn.content_redacted 必经 redact()。
- memory_agent：每 turn 写 run_log；session-level 长期记忆在 close_session 触发，依据 avg(reviewer_score) >= 0.75。
- gc(older_than_days)：级联删除 session / session_turn / session_evidence / session_digest。
- append_digest 必须在 ContextCompactor 触发后立刻入库（通过 ConversationManager），不要等到 turn 结束。
- SqliteSessionStore 必须通过与 InMemorySessionStore 同一组 contract test。

不要修改 graph 结构（保持 Phase 5 的节点顺序）。

验收：
  python scripts/reset_demo_db.py
  EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock python scripts/chat.py 跑两次
  pytest -q tests/test_memory.py tests/test_conversation.py
  sqlite3 legal_rag_memory.sqlite3 "SELECT count(*) FROM run_log;"  # ≥ turn 总数
```

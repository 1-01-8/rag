# Phase 8 · FastAPI 暴露（含 `/sessions*` 主路由）

## 依赖

- Phase 1–6 全部完成。
- 推荐 Phase 7 完成后再上线。

## 本阶段交付物

1. `src/legal_rag/app.py`
2. `src/legal_rag/api/`：路由、依赖注入、schema。
3. `tests/test_api.py`。

---

## 1. 路由

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST`   | `/documents/upload` | 上传 PDF/TXT/MD，触发 ingestion |
| `POST`   | `/indexes/build` | 重建 BM25 + Dense 索引 |
| `POST`   | `/sessions` | 创建会话，返回 `session_id` |
| `GET`    | `/sessions/{id}` | 查会话状态（不返回原文，仅 turn 数 / status / 最近压缩元数据） |
| `DELETE` | `/sessions/{id}` | 关闭并触发 session-level memory write；`?purge=true` 立即物理删除 |
| `POST`   | `/sessions/{id}/messages` | 发送 user 消息，触发一轮 graph |
| `GET`    | `/sessions/{id}/messages` | 拉取本会话脱敏后的 messages |
| `POST`   | `/sessions/{id}/compact` | 显式触发一次工作上下文压缩（ContextCompactor） |
| `POST`   | `/ask` | 单 turn 临时 session 语法糖（内部 start→run→close） |
| `GET`    | `/runs/{run_id}` | 查 trace |
| `GET`    | `/healthz` | provider ping |

> `/sessions*` 是主路由；`/ask` 是退化情况。

---

## 2. /sessions

```python
class CreateSessionRequest(BaseModel):
    user_id: str | None = None

class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: float

@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest, rt: HarnessRuntime = Depends(get_runtime)):
    sid = rt.start_session(user_id=req.user_id)
    return CreateSessionResponse(session_id=sid, created_at=time.time())
```

### 关闭

```python
@router.delete("/sessions/{sid}")
def close_session(sid: str, purge: bool = False, rt: HarnessRuntime = Depends(get_runtime)):
    if purge:
        rt.store.purge(sid)            # SqliteSessionStore 提供的方法，物理删 4 张会话表
    rt.close_session(sid)
    return {"session_id": sid, "status": "closed", "purged": purge}
```

---

## 3. /sessions/{id}/messages

```python
class MessageRequest(BaseModel):
    content: str
    options: AskOptions = Field(default_factory=AskOptions)

class AskOptions(BaseModel):
    use_reranker: bool | None = None
    max_retrieval_retry: int | None = None
    max_answer_revision: int | None = None
    return_trace: bool = False

class MessageResponse(BaseModel):
    session_id: str
    turn_id: int
    run_id: str
    kind: Literal["answer", "clarification"]
    answer: str | None
    clarification: str | None
    citations: list[Citation]
    evidence_score: float | None
    citation_score: float | None
    groundedness_score: float | None
    reviewer_score: float | None
    retrieval_retry_count: int
    answer_revision_count: int
    session_status: Literal["active", "awaiting_user", "closed"]
    compactions_in_turn: int          # 本 turn 透明触发的 compaction 次数（≥0）
    digest_token_after: int | None    # 当前生效 digest 的 token 估算
    trace_path: str | None = None

@router.post("/sessions/{sid}/messages", response_model=MessageResponse)
def post_message(sid: str, req: MessageRequest, rt: HarnessRuntime = Depends(get_runtime)):
    try:
        resp = rt.run_turn(sid, req.content, options=req.options.model_dump())
    except ConversationClosedError:
        raise HTTPException(status_code=410, detail="session closed")
    if req.options.return_trace:
        resp["trace_path"] = f"logs/runs/{resp['run_id']}.jsonl"
    return MessageResponse(**resp)
```

### 拉取历史（脱敏）

```python
@router.get("/sessions/{sid}/messages")
def list_messages(sid: str, rt: HarnessRuntime = Depends(get_runtime)):
    conv = rt.store.load(sid)
    return {
        "session_id": sid,
        "status": conv.status,
        "messages": [
            {"turn_id": t.turn_id, "role": t.role, "kind": t.kind,
             "content": t.content, "evidence_ids": t.evidence_ids}
            for t in conv.turns
        ],
        "digests": [
            {"digest_id": d.digest_id, "until_turn_id": d.until_turn_id,
             "triggered_by": d.triggered_by,
             "token_before": d.token_estimate_before, "token_after": d.token_estimate_after}
            for d in conv.digests
        ],
    }
```

---

## 4. /sessions/{id}/compact

显式触发一次工作上下文压缩（Claude Code 风格 `/compact`，用于运维或前端"立即压缩"按钮）：

```python
@router.post("/sessions/{sid}/compact")
def force_compact(sid: str, rt: HarnessRuntime = Depends(get_runtime)):
    digest = rt.force_compact(sid)
    return {
        "session_id": sid,
        "digest_id": digest.digest_id,
        "until_turn_id": digest.until_turn_id,
        "triggered_by": "manual",
        "token_before": digest.token_estimate_before,
        "token_after": digest.token_estimate_after,
        "saved_ratio": 1.0 - (digest.token_estimate_after / max(1, digest.token_estimate_before)),
        "pinned_evidence_ids": digest.pinned_evidence_ids,
        "dropped_evidence_ids": digest.dropped_evidence_ids,
    }
```

`HarnessRuntime.force_compact` 实现：直接调 `harness/context_compactor.py:ContextCompactor.force_compact`，把返回的 `WorkingContextDigest` 写入 `SessionDigest` 表。无 graph、无 agent，纯 runtime 操作。

---

## 5. /ask（单 turn 语法糖）

```python
class AskRequest(BaseModel):
    query: str
    jurisdiction: str = "CN"
    options: AskOptions = Field(default_factory=AskOptions)

@router.post("/ask", response_model=MessageResponse)
def ask(req: AskRequest, rt: HarnessRuntime = Depends(get_runtime)):
    return MessageResponse(**rt.run_oneshot(req.query, req.jurisdiction))
```

---

## 6. /documents/upload, /indexes/build, /runs, /healthz

与 Phase 5 / 7 设计一致。`/healthz` 容错：单 provider 失败时整体 200 status=degraded。

---

## 7. 应用入口

```python
# src/legal_rag/app.py
from fastapi import FastAPI
from .api.documents import router as doc_router
from .api.indexes import router as idx_router
from .api.sessions import router as sess_router
from .api.ask import router as ask_router
from .api.runs import router as run_router
from .api.health import router as health_router

def create_app() -> FastAPI:
    app = FastAPI(title="LegalResearch-Agent")
    app.include_router(doc_router, prefix="/documents")
    app.include_router(idx_router, prefix="/indexes")
    app.include_router(sess_router)        # 含 /sessions/* 全部
    app.include_router(ask_router)
    app.include_router(run_router, prefix="/runs")
    app.include_router(health_router)
    return app

app = create_app()
```

---

## 端到端验收

### 验收命令（6 步多轮流程）

```bash
EMBEDDING_PROVIDER=siliconflow LLM_PROVIDER=siliconflow USE_RERANKER=true \
  uvicorn legal_rag.app:app --reload --port 8080 &

# 1. 健康
curl -s localhost:8080/healthz | jq

# 2. 上传 + 建索引（如未做）
curl -s -F file=@data/raw/statutes/中华人民共和国劳动合同法.txt \
       -F source_type=statute -F jurisdiction=CN \
       localhost:8080/documents/upload | jq
curl -s -X POST localhost:8080/indexes/build | jq

# 3. 创建 session
SID=$(curl -s -X POST localhost:8080/sessions \
        -H 'Content-Type: application/json' -d '{}' | jq -r .session_id)

# 4. 第一轮：触发 clarification
curl -s -X POST localhost:8080/sessions/$SID/messages \
  -H 'Content-Type: application/json' \
  -d '{"content":"公司单方解除劳动合同合法吗？"}' | jq

# 5. 第二轮：补事实，得正式答
curl -s -X POST localhost:8080/sessions/$SID/messages \
  -H 'Content-Type: application/json' \
  -d '{"content":"我严重违反规章制度被开除，没补偿"}' | jq

# 6. 第三轮：代词引用，复用 cumulative_evidence
curl -s -X POST localhost:8080/sessions/$SID/messages \
  -H 'Content-Type: application/json' \
  -d '{"content":"那条法条具体写了什么？"}' | jq

# 7. 显式压缩（可选）
curl -s -X POST localhost:8080/sessions/$SID/compact | jq

# 8. 关闭并触发 session-level memory
curl -s -X DELETE localhost:8080/sessions/$SID | jq
sqlite3 legal_rag_memory.sqlite3 "SELECT count(*) FROM run_log WHERE session_id='$SID';"

pytest -q tests/test_api.py
```

### 验收通过条件

- 全部端点 200。
- 第 4 步 `kind=="clarification"`、`session_status=="awaiting_user"`。
- 第 5 步 `kind=="answer"`，`citations` 非空。
- 第 6 步 `citations[].evidence_id` 至少有一个等于第 5 步出现过的 `evidence_id`（cumulative 复用）。
- 第 7 步返回 `token_before > token_after` 且 `saved_ratio > 0`；后续问答 `digest_token_after` 字段非空。
- 第 8 步关闭后 `run_log` 含本 session 全部 turn。
- `/healthz` 故意配错 key 时返回 200 但 `status=degraded`。
- API 测试覆盖：mock provider 全流程；`?purge=true` 物理删除 4 张会话表。

---

## Codex Prompt

```text
基于 Phase 1–7，实现 Phase 8：FastAPI（含 /sessions*）。

按 PLAN/10_PHASE8_API.md 实现：

1. src/legal_rag/app.py
2. src/legal_rag/api/{documents,indexes,sessions,ask,runs,health}.py
3. tests/test_api.py

要求：
- HarnessRuntime 单例 (lru_cache)，通过 Depends 注入。
- /sessions/{id}/messages 内部调 rt.run_turn；ConversationClosedError → 410。
- /sessions/{id}/compact 调 rt.force_compact；返回 token_before / token_after。
- DELETE /sessions/{id}?purge=true 物理删除 4 张会话表 (session/session_turn/session_evidence/session_digest)（rt.store.purge）。
- /ask 调 rt.run_oneshot，与 /sessions/{id}/messages 共用 MessageResponse 模型。
- TestClient 全流程用 mock provider；上传 fixtures 小 txt → build → 6 步多轮 → close。
- /healthz 容错：单 provider 失败时整体 200，status=degraded。

不要新增检索/agent 功能。

验收：
  EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock RERANKER_PROVIDER=noop pytest -q tests/test_api.py
  EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock RERANKER_PROVIDER=noop \
    uvicorn legal_rag.app:app --port 8080 &
  sleep 2
  bash docs/smoke_six_steps.sh   # 文档 §端到端验收 中的 6 步 curl
```

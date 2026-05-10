# LegalRAG MVP Rescoping and Quality Gates

本文档是对原 8 阶段开发计划的收敛修订，目标是把 LegalRAG 从“完整系统蓝图”调整为可持续验证的产品开发路线：

- 降低 MVP 范围，先完成法条问答垂直切片。
- 提前评估，在检索和回答阶段就建立质量门禁。
- 强化法律数据版本治理，默认只引用当前有效、来源可追溯的法律文本。
- 前置用户隔离，session、evidence、memory 从第一版 API 起按 user/tenant 校验。
- 确保开发过程中可以端到端测试不同组件，并快速定位影响质量的问题模块。

## 1. 调整后的 MVP 路线

### MVP-0: Statute QA Vertical Slice

目标：证明系统能稳定找对法条、拒绝无依据结论、生成带真实引用的可用答案。

范围：

- 仅支持 `statute` 法条库。
- 仅支持 `CN` jurisdiction。
- Provider 支持 `mock`、`local`、`siliconflow` 三类配置，但验收必须能用 `mock` 离线跑通。
- 检索链路只做 `BM25 + dense + hybrid fusion + metadata filter`。
- Agent 只保留 `intake`、`statute_retriever`、`citation_checker`、`answer_finalizer`。
- API 只保留 `/ask` 和最小 `/sessions/{session_id}/messages`。
- Memory、case、contract、ContextCompactor、复杂 LangGraph routing 默认不进入 MVP-0。

MVP-0 必须回答的问题：

- 用户问题是否被正确解析为法律检索意图？
- 系统是否能召回必须出现的法条？
- answer 是否只引用 evidence pool 中的真实文本子串？
- 当 evidence 不足时，系统是否明确说“不足以判断”？
- 哪个组件对质量下降贡献最大？

### MVP-1: Native Multi-turn

目标：支持追问、澄清和历史 evidence 复用。

新增：

- session store。
- sticky intake state。
- cumulative evidence pool。
- `/sessions/{session_id}/messages` 正式化。
- 多轮评估：`evidence_reuse_rate`、`clarification_precision`、`sticky_intake_consistency`。

仍不新增：

- long-term memory。
- case/contract 专用 agent。
- compaction 默认开启。

### MVP-2: Contract and Case Expansion

目标：扩展到合同和案例，但不改变 MVP-0 已验证的引用约束。

新增：

- `contract` chunker。
- `case` chunker。
- `contract_retriever`。
- `case_retriever`。
- 合同脱敏策略和私有语料隔离。

### MVP-3: Harness Hardening

目标：提升复杂问题稳定性和可观测性。

新增：

- LangGraph orchestration。
- reviewer。
- answer revision。
- tracing。
- regression gates。

### MVP-4: Memory and Compaction

目标：优化长会话和复用低风险经验。

新增：

- SQLite long-term memory。
- ContextCompactor。
- `/sessions/{session_id}/compact`。
- memory pollution eval。
- compaction fidelity eval。

Memory 在 MVP-4 前只能作为实验 feature flag，不允许影响法律结论。

## 2. Phase 调整规则

原 Phase 不删除，但执行顺序和验收门槛调整如下：

| 原 Phase | 调整后策略 |
|---|---|
| Phase 1 Provider + Schema | 保留，增加 model capability discovery 和 auth context schema。 |
| Phase 2 Ingestion | 先只实现 statute；增加 Legal Corpus Governance metadata。 |
| Phase 3 Index/Retrieval | 完成后立即运行 Retrieval Eval Lite，不通过不得进入 answer agent。 |
| Phase 4 Agents | 只实现 MVP-0 四个 agent；完成后立即运行 Answer Eval Lite。 |
| Phase 5 Harness Graph | MVP-0 可先用简单 orchestrator；LangGraph 延后到 MVP-3。 |
| Phase 6 Memory | 延后到 MVP-4；MVP-0/1 只允许 session-scoped evidence。 |
| Phase 7 Evaluation | 拆成 Eval Lite + full regression；Eval Lite 前移到 Phase 3/4。 |
| Phase 8 API | API 提前到 MVP-0，但必须带 user/session 隔离。 |

## 3. 提前评估设计

### 3.1 Retrieval Eval Lite

进入条件：Phase 3 完成后立即运行。

数据规模：

- MVP-0 至少 20 条 statute 问题。
- 每条样本必须包含 `query`、`must_include_law`、`must_include_article`、`jurisdiction`、`expected_valid_status`。

核心指标：

| 指标 | MVP-0 门槛 | 用途 |
|---|---:|---|
| `must_include_hit_rate@10` | >= 0.85 | 判断必需法条是否被召回。 |
| `recall@10` | >= 0.80 | 判断整体召回能力。 |
| `current_law_filter_accuracy` | >= 0.95 | 判断是否默认过滤失效版本。 |
| `metadata_match_rate` | >= 0.90 | 判断 law/article/jurisdiction metadata 是否可靠。 |
| `empty_result_precision` | >= 0.80 | 判断证据不足时是否少召回垃圾。 |

必须输出组件诊断：

```json
{
  "query_id": "labor_termination_001",
  "failed_stage": "hybrid_fusion",
  "bm25_rank": 3,
  "dense_rank": null,
  "hybrid_rank": 14,
  "metadata_filter_removed": false,
  "root_cause": "dense_embedding_missed_article_term"
}
```

`failed_stage` 枚举：

- `chunking`
- `metadata_extraction`
- `bm25`
- `dense`
- `hybrid_fusion`
- `metadata_filter`
- `reranker`
- `query_rewrite`

### 3.2 Answer Eval Lite

进入条件：Phase 4 MVP-0 agent 完成后立即运行。

数据规模：

- 至少 10 条 statute QA。
- 至少 3 条 evidence 不足或用户事实缺失的问题。
- 至少 3 条多法条组合问题。

核心指标：

| 指标 | MVP-0 门槛 | 用途 |
|---|---:|---|
| `citation_substring_valid_rate` | 1.00 | 引用 quote 必须是 evidence 原文子串。 |
| `evidence_id_valid_rate` | 1.00 | answer 只能引用 evidence pool 中的 evidence_id。 |
| `ungrounded_claim_rate` | <= 0.05 | 控制无依据法律结论。 |
| `insufficient_evidence_refusal_rate` | >= 0.90 | 证据不足时必须克制。 |
| `disclaimer_presence_rate` | 1.00 | finalizer 必须追加非法律意见声明。 |
| `answer_relevance_score` | >= 0.75 | 避免只有引用但不回答问题。 |

Answer Eval 失败时必须归因到模块：

- `intake`: 用户事实或问题类型识别错误。
- `retriever`: 没有召回正确 evidence。
- `citation_checker`: 放过了非法引用。
- `answer_finalizer`: 生成了无依据结论或漏免责声明。
- `orchestrator`: 传递了错误 state 或错误 evidence pool。

### 3.3 End-to-End Smoke Tests

每个 Phase 必须保留一个可离线运行的端到端 smoke test。

建议命令：

```bash
python scripts/dev/check_providers.py --profile mock
python scripts/eval/retrieval_eval_lite.py --dataset data/eval/statute_retrieval_lite.jsonl --profile mock
python scripts/eval/answer_eval_lite.py --dataset data/eval/statute_answer_lite.jsonl --profile mock
python scripts/e2e/run_statute_qa_smoke.py --profile mock
```

最低要求：

- `mock` profile 必须不访问外部网络。
- 每次 eval 输出 `metrics.json`、`failures.jsonl`、`component_breakdown.json`。
- CI 可以只跑 lite dataset，本地和 nightly 再跑 expanded dataset。

## 4. 组件级质量定位机制

开发过程中不能只看最终答案分数，必须能定位是哪个组件拖累质量。

### 4.1 Trace Schema

每次请求都记录结构化 trace：

```python
class RunTrace(BaseModel):
    run_id: str
    session_id: str | None
    user_id: str
    tenant_id: str | None = None
    profile: Literal["mock", "local", "siliconflow"]
    query: str
    intake: IntakeTrace
    retrieval: RetrievalTrace
    citation_check: CitationTrace
    answer: AnswerTrace
    metrics: dict[str, float | int | str | bool]
```

Retrieval trace 至少包含：

```python
class RetrievalTrace(BaseModel):
    normalized_query: str
    filters: dict[str, Any]
    bm25_candidates: list[RankedEvidence]
    dense_candidates: list[RankedEvidence]
    fused_candidates: list[RankedEvidence]
    reranked_candidates: list[RankedEvidence] = []
    removed_by_filter: list[FilteredEvidence]
    selected_evidence: list[RankedEvidence]
```

Answer trace 至少包含：

```python
class AnswerTrace(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    evidence_ids_in_prompt: list[str]
    evidence_ids_cited: list[str]
    unsupported_claims: list[str]
    disclaimer_added: bool
```

### 4.2 Failure Taxonomy

`failures.jsonl` 中每条失败必须包含：

```json
{
  "query_id": "labor_termination_001",
  "severity": "blocker",
  "failed_metric": "must_include_hit_rate@10",
  "component": "metadata_filter",
  "symptom": "Expected article was removed before ranking",
  "root_cause": "valid_status extracted as repealed from amendment note",
  "recommended_fix": "split amendment note from article body and prefer effective_to metadata"
}
```

组件枚举：

- `provider`
- `ingestion`
- `chunker`
- `metadata_extractor`
- `indexer`
- `bm25_retriever`
- `dense_retriever`
- `hybrid_fusion`
- `reranker`
- `intake_agent`
- `citation_checker`
- `answer_finalizer`
- `session_store`
- `auth_guard`
- `orchestrator`

严重级别：

- `blocker`: 导致非法引用、跨用户泄露、引用失效法律、或必需法条完全无法召回。
- `major`: 最终答案明显不完整、不相关、或多轮状态错误。
- `minor`: 格式、排序、解释质量、可读性问题。

### 4.3 Ablation Tests

每次 Retrieval Eval Lite 应支持 ablation：

```bash
python scripts/eval/retrieval_eval_lite.py --ablation bm25_only
python scripts/eval/retrieval_eval_lite.py --ablation dense_only
python scripts/eval/retrieval_eval_lite.py --ablation hybrid_no_metadata_boost
python scripts/eval/retrieval_eval_lite.py --ablation hybrid_no_valid_status_filter
```

必须输出：

```json
{
  "bm25_only": {"must_include_hit_rate@10": 0.78},
  "dense_only": {"must_include_hit_rate@10": 0.62},
  "hybrid_no_metadata_boost": {"must_include_hit_rate@10": 0.80},
  "hybrid": {"must_include_hit_rate@10": 0.88}
}
```

如果 `hybrid` 比单路检索更差，禁止进入下一阶段，必须先修 fusion 或 filter。

## 5. 法律数据版本治理

Phase 2 必须增加 corpus governance，不再只做文本清洗和切分。

### 5.1 必填 Metadata

每个 statute chunk 必须包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | str | 单个法律文件的稳定 ID。 |
| `version_id` | str | 同一法律不同版本的唯一 ID。 |
| `chunk_id` | str | chunk 唯一 ID。 |
| `title` | str | 法律名称。 |
| `article_no` | str | 条号，例如 `第39条`。 |
| `jurisdiction` | str | MVP-0 固定为 `CN`。 |
| `authority_level` | str | 法律、行政法规、司法解释、地方规定等。 |
| `publisher` | str | 发布机关。 |
| `source_url` | str | 官方或可信来源 URL。 |
| `source_hash` | str | 原文 hash。 |
| `ingested_at` | datetime | 入库时间。 |
| `promulgated_at` | date \| None | 公布日期。 |
| `effective_from` | date \| None | 生效日期。 |
| `effective_to` | date \| None | 失效日期。 |
| `valid_status` | enum | `current`、`repealed`、`amended`、`unknown`。 |
| `is_current` | bool | 默认检索必须为 true。 |
| `supersedes_version_id` | str \| None | 被本版本替代的版本。 |
| `superseded_by_version_id` | str \| None | 替代本版本的新版本。 |

### 5.2 默认检索规则

除非用户明确问历史版本，检索默认 filter：

```json
{
  "jurisdiction": "CN",
  "is_current": true
}
```

如果 `valid_status == "unknown"`：

- 可以进入候选池，但必须降权。
- answer 中不得把 unknown 状态文本表述为“现行有效”。
- eval 中单独统计 `unknown_status_usage_rate`。

如果用户问历史版本：

- intake 必须识别时间点。
- retrieval filter 改为 `effective_from <= asked_date < effective_to` 或选择最接近版本。
- answer 必须明确说明引用的是历史版本。

### 5.3 数据治理测试

Phase 2 增加测试：

```bash
python scripts/eval/corpus_governance_check.py --corpus data/processed/statutes.jsonl
```

必须检查：

- `source_url` 非空率。
- `source_hash` 可重算。
- 同一 `doc_id` 的多个 `version_id` 不冲突。
- `article_no` 抽取准确率。
- `is_current` 与 `effective_to`、`valid_status` 一致。
- 默认检索不会返回 `is_current=false` 的 chunk。

## 6. 用户隔离和 API Auth 前置

API 从 MVP-0 起不允许裸 `session_id` 访问。即使是本地 demo，也必须通过 dev auth 注入用户身份。

### 6.1 Auth Context

```python
class AuthContext(BaseModel):
    user_id: str
    tenant_id: str | None = None
    roles: list[str] = []
```

本地 demo：

```bash
DEV_AUTH_USER_ID=demo
DEV_AUTH_TENANT_ID=local
```

生产/集成环境：

- 从 JWT、API key 或网关 header 解析。
- 不接受 request body 中用户自填 `user_id` 作为权限依据。

### 6.2 Session Ownership

Session store 的主键或唯一约束必须包含：

```text
(tenant_id, user_id, session_id)
```

所有 session 操作必须校验：

- 创建 session 时写入 `user_id` 和 `tenant_id`。
- 读取 session 时必须匹配当前 `AuthContext`。
- 追加 message 时必须匹配当前 `AuthContext`。
- 关闭、删除、compact session 时必须匹配当前 `AuthContext`。
- evidence、trace、memory 均不得跨 user/tenant 返回。

### 6.3 API Contract

`CreateSessionRequest` 不再接受可信 `user_id`：

```python
class CreateSessionRequest(BaseModel):
    jurisdiction: str = "CN"
    metadata: dict[str, Any] = {}
```

路由示例：

```python
@router.post("/sessions")
def create_session(
    req: CreateSessionRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    return runtime.create_session(
        user_id=auth.user_id,
        tenant_id=auth.tenant_id,
        jurisdiction=req.jurisdiction,
        metadata=req.metadata,
    )
```

删除流程：

```python
@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    purge: bool = False,
    auth: AuthContext = Depends(get_auth_context),
):
    if purge:
        runtime.close_session(session_id, auth=auth, write_memory=False)
        runtime.store.purge(session_id, auth=auth)
    else:
        runtime.close_session(session_id, auth=auth, write_memory=True)
```

### 6.4 Isolation Tests

MVP-0 必须包含：

```bash
python scripts/e2e/session_isolation_smoke.py --profile mock
```

测试场景：

- User A 创建 session，User B 不能读取。
- User B 不能向 User A session 追加消息。
- User B 不能引用 User A 的 cumulative evidence。
- purge 只删除当前 user/tenant 下的 session。
- trace 和 eval artifact 不包含跨用户数据。

任何隔离测试失败均为 `blocker`。

## 7. 质量门禁汇总

| 阶段 | 必跑命令 | 不通过时处理 |
|---|---|---|
| Phase 1 | `check_providers.py --profile mock` | 修 provider/schema，不进入 ingestion。 |
| Phase 2 | `corpus_governance_check.py` | 修 metadata/source/version，不建索引。 |
| Phase 3 | `retrieval_eval_lite.py` | 修 chunk/index/retrieval/filter，不写 answer agent。 |
| Phase 4 | `answer_eval_lite.py` | 修 agent/citation/finalizer，不扩 graph。 |
| MVP-0 API | `run_statute_qa_smoke.py` + `session_isolation_smoke.py` | 修 API/auth/session，不进入多轮扩展。 |
| MVP-1 | `multiturn_eval_lite.py` | 修 session/evidence reuse/sticky intake。 |
| MVP-2 | `contract_case_eval_lite.py` | 修 chunker/router/domain retrieval。 |
| MVP-3 | `regression_eval_full.py` | 修 graph/reviewer/retry/tracing。 |
| MVP-4 | `memory_compaction_eval.py` | 修 memory pollution/compaction fidelity。 |

每个门禁都必须产出：

- `metrics.json`
- `failures.jsonl`
- `component_breakdown.json`
- `run_traces/*.jsonl`

## 8. Done Definition

MVP-0 只有在以下条件全部满足时才算完成：

- 可以用 `mock` profile 端到端跑通，不依赖外部 API。
- 可以用至少一个真实 embedding/LLM profile 跑通 smoke test。
- Retrieval Eval Lite 达标。
- Answer Eval Lite 达标。
- Corpus Governance Check 达标。
- Session Isolation Smoke 达标。
- 每个失败样本都能定位到具体组件。
- answer 中所有引用均来自 evidence pool 的真实子串。
- 默认不返回失效法律版本。
- 证据不足时会拒绝下结论。

这套定义优先保证 LegalRAG 的基本可信度。后续扩展 case、contract、memory、LangGraph 时，必须继续复用同一套 evidence、trace、eval 和 isolation 机制。

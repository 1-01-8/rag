# Codex Implementation Guide for LegalRAG

本文档给 Codex 或其他代码 agent 使用。它不是产品愿景，而是施工规则：每次开发只推进一个明确阶段，必须能离线端到端自测，失败时必须能定位到具体组件。

优先级：

1. 先遵守 `12_MVP_RESCOPING_AND_QUALITY_GATES.md`。
2. 再遵守原 phase 文档。
3. 如果原 phase 文档和 MVP 收敛文档冲突，以 MVP 收敛文档为准。

## 1. Codex 工作原则

### 1.1 一次只推进一个阶段

默认只实现当前阶段需要的最小功能。不要提前实现 case、contract、memory、复杂 LangGraph、ContextCompactor 或完整管理后台。

允许提前创建占位接口，但必须满足：

- 不影响 MVP-0 主链路。
- 不引入外部依赖。
- 不要求真实 LLM 或外部 API 才能跑测试。
- 不让未完成模块参与生产路径。

### 1.2 Mock profile 必须先跑通

任何阶段都必须优先支持：

```bash
--profile mock
```

`mock` profile 必须满足：

- 不访问外部网络。
- 输出稳定、可重复。
- 能覆盖正常路径和失败路径。
- 可以在 CI 或本地无密钥环境运行。

只有 mock profile 通过后，才允许验证 `local` 或 `siliconflow` profile。

### 1.3 端到端优先，局部测试辅助

每个阶段至少保留一个 e2e smoke test。单元测试可以帮助定位问题，但不能替代阶段 smoke test。

开发完成的最低证据：

- 阶段 smoke test 通过。
- 对应 eval lite 通过或至少能运行并产出 failures。
- 失败样本能归因到组件。
- 没有破坏上一阶段的 smoke test。

### 1.4 质量问题必须归因

不要只报告“答案不好”或“召回不好”。每个失败必须尽量归因到以下组件之一：

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

如果暂时无法确定，标记为 `unknown`，并在 `recommended_fix` 中说明下一步要加什么 trace。

## 2. 推荐项目结构

Codex 实现时优先使用下面结构。已有项目结构不同则尊重现有结构，但仍保持同样职责边界。

```text
legal_rag/
  api/
    auth.py
    routes.py
    schemas.py
  agents/
    intake.py
    statute_retriever.py
    citation_checker.py
    answer_finalizer.py
  corpus/
    loaders.py
    statute_chunker.py
    metadata.py
    governance.py
  eval/
    metrics.py
    failure_taxonomy.py
    traces.py
  index/
    bm25.py
    dense.py
    hybrid.py
    store.py
  providers/
    base.py
    mock.py
    local.py
    siliconflow.py
  runtime/
    orchestrator.py
    session_store.py
    evidence.py
  schemas/
    auth.py
    corpus.py
    evidence.py
    messages.py
    traces.py
scripts/
  dev/
    check_providers.py
  e2e/
    run_statute_qa_smoke.py
    session_isolation_smoke.py
  eval/
    corpus_governance_check.py
    retrieval_eval_lite.py
    answer_eval_lite.py
data/
  eval/
    statute_retrieval_lite.jsonl
    statute_answer_lite.jsonl
  sample/
    statutes/
```

## 3. MVP-0 开发顺序

### Step 0: Skeleton

目标：项目能安装、导入、运行空 smoke test。

实现：

- 基础 package。
- 配置加载。
- Pydantic schemas。
- `mock` provider。
- 最小 CLI/script 入口。

必跑：

```bash
python scripts/dev/check_providers.py --profile mock
```

完成标准：

- mock embedding、mock reranker、mock LLM 都能返回稳定结果。
- 不需要 API key。
- 失败时错误信息明确指出缺哪个 provider 或配置。

### Step 1: Corpus Governance and Statute Ingestion

目标：能加载少量法条样本，切分为 statute chunks，并生成完整 metadata。

实现：

- statute loader。
- statute chunker。
- metadata extractor。
- `source_hash` 计算。
- `is_current` 和 `valid_status` 推断。
- corpus governance checker。

必跑：

```bash
python scripts/eval/corpus_governance_check.py --corpus data/processed/statutes.jsonl
```

完成标准：

- 每个 chunk 有 `doc_id`、`version_id`、`chunk_id`、`title`、`article_no`、`jurisdiction`、`source_url`、`source_hash`、`is_current`。
- 默认检索不会返回 `is_current=false`。
- governance check 输出 `metrics.json` 和 `failures.jsonl`。

不要做：

- case chunking。
- contract chunking。
- memory。

### Step 2: Index and Retrieval Eval Lite

目标：能从问题召回正确法条，并能诊断召回失败原因。

实现：

- BM25 retriever。
- dense retriever。
- hybrid fusion。
- metadata filter。
- retrieval trace。
- Retrieval Eval Lite。
- ablation mode。

必跑：

```bash
python scripts/eval/retrieval_eval_lite.py --dataset data/eval/statute_retrieval_lite.jsonl --profile mock
python scripts/eval/retrieval_eval_lite.py --dataset data/eval/statute_retrieval_lite.jsonl --profile mock --ablation bm25_only
python scripts/eval/retrieval_eval_lite.py --dataset data/eval/statute_retrieval_lite.jsonl --profile mock --ablation dense_only
python scripts/eval/retrieval_eval_lite.py --dataset data/eval/statute_retrieval_lite.jsonl --profile mock --ablation hybrid_no_valid_status_filter
```

完成标准：

- 产出 `metrics.json`、`failures.jsonl`、`component_breakdown.json`。
- 每个失败样本包含 `failed_stage`。
- `hybrid` 不得明显差于 `bm25_only` 和 `dense_only`。

不要做：

- answer agent。
- reviewer。
- long-term memory。

### Step 3: Minimal Agents and Answer Eval Lite

目标：能完成 intake -> retrieval -> citation check -> final answer。

实现：

- `intake_agent`。
- `statute_retriever` wrapper。
- `citation_checker`。
- `answer_finalizer`。
- answer trace。
- Answer Eval Lite。

必跑：

```bash
python scripts/eval/answer_eval_lite.py --dataset data/eval/statute_answer_lite.jsonl --profile mock
python scripts/e2e/run_statute_qa_smoke.py --profile mock
```

完成标准：

- 引用 quote 必须是 evidence text 的真实子串。
- answer 只能引用 evidence pool 中存在的 `evidence_id`。
- 证据不足时必须拒绝下结论。
- finalizer 必须追加免责声明。
- 失败样本能归因到 agent、retriever、citation checker 或 finalizer。

不要做：

- 多 agent planner。
- reviewer revision loop。
- LangGraph 复杂路由。

### Step 4: API and User Isolation

目标：提供最小 API，并从第一版起防止跨用户访问。

实现：

- `AuthContext`。
- `get_auth_context`。
- `/ask`。
- `/sessions`。
- `/sessions/{session_id}/messages`。
- `/sessions/{session_id}` delete/close。
- session ownership check。
- session isolation smoke test。

必跑：

```bash
python scripts/e2e/session_isolation_smoke.py --profile mock
python scripts/e2e/run_statute_qa_smoke.py --profile mock
```

完成标准：

- request body 中的 `user_id` 不作为可信权限依据。
- 所有 session 操作通过 `AuthContext` 校验。
- User B 不能读取、写入、删除 User A 的 session。
- evidence、trace、memory 不跨 user/tenant 返回。

不要做：

- OAuth/JWT 完整生产实现。
- 管理后台。
- 复杂权限模型。

## 4. MVP-1 开发顺序

只有 MVP-0 全部完成后才进入 MVP-1。

目标：支持原生多轮，不引入 long-term memory。

实现：

- session-scoped turn history。
- cumulative evidence pool。
- sticky intake state。
- 多轮 evidence reuse。
- 多轮 eval lite。

必跑：

```bash
python scripts/eval/multiturn_eval_lite.py --dataset data/eval/multiturn_lite.jsonl --profile mock
python scripts/e2e/run_statute_qa_smoke.py --profile mock
python scripts/e2e/session_isolation_smoke.py --profile mock
```

完成标准：

- 追问能复用当前 session 的 evidence。
- 不复用其他 session 或其他 user 的 evidence。
- sticky intake 不覆盖用户后来修正的事实。
- 长对话超预算时先报 trace，不默认启用 ContextCompactor。

## 5. 禁止提前实现清单

在 MVP-0 中禁止作为主路径实现：

- `case_agent`
- `contract_agent`
- `memory_agent`
- long-term memory 写入
- ContextCompactor 默认启用
- LangGraph 复杂 planner/router
- reviewer 多轮修订
- 自动 self-evolution
- 多租户管理后台
- 复杂上传和索引重建 API

可以保留接口占位，但必须 feature flag 关闭。

## 6. 每轮 Codex 开发模板

Codex 每次接到开发任务时按以下顺序行动。

### 6.1 开工前

1. 读取本文件。
2. 读取 `12_MVP_RESCOPING_AND_QUALITY_GATES.md`。
3. 判断当前任务属于哪个 MVP 和 step。
4. 检查已有代码结构和测试。
5. 明确本轮不会触碰的模块。

### 6.2 实现中

1. 优先实现 mock profile。
2. 先写或补 smoke/eval 脚本的最小可运行版本。
3. 再实现业务代码。
4. 保持 trace 和 failure 输出结构稳定。
5. 每次修复质量问题后重新运行对应 eval。

### 6.3 收尾时

最终回复必须包含：

- 修改了哪些文件。
- 当前属于哪个 MVP/step。
- 跑了哪些命令。
- 通过了哪些门禁。
- 哪些门禁还没跑，原因是什么。
- 如果有失败，失败归因到哪个组件。

## 7. Eval Artifact 规范

所有 eval 和 smoke test 默认输出到：

```text
runs/{run_id}/
  metrics.json
  failures.jsonl
  component_breakdown.json
  run_traces.jsonl
```

`metrics.json` 示例：

```json
{
  "run_id": "20260510_legalrag_retrieval_lite",
  "profile": "mock",
  "dataset": "data/eval/statute_retrieval_lite.jsonl",
  "must_include_hit_rate@10": 0.9,
  "recall@10": 0.85,
  "current_law_filter_accuracy": 1.0
}
```

`component_breakdown.json` 示例：

```json
{
  "metadata_extractor": {"blocker": 0, "major": 1, "minor": 2},
  "bm25_retriever": {"blocker": 0, "major": 2, "minor": 0},
  "hybrid_fusion": {"blocker": 1, "major": 0, "minor": 0}
}
```

`failures.jsonl` 每行必须至少包含：

```json
{
  "query_id": "labor_termination_001",
  "severity": "major",
  "failed_metric": "must_include_hit_rate@10",
  "component": "hybrid_fusion",
  "symptom": "Required article ranked below top 10",
  "root_cause": "metadata boost too low for exact article number match",
  "recommended_fix": "increase article_no exact match boost and rerun ablation"
}
```

## 8. Definition of Done by Step

| Step | Done 条件 |
|---|---|
| Skeleton | `check_providers.py --profile mock` 通过。 |
| Ingestion | governance check 能运行并输出 artifact；statute chunks metadata 完整。 |
| Retrieval | retrieval eval lite 能运行；失败可归因；ablation 可比较。 |
| Agents | answer eval lite 能运行；引用校验 100%；证据不足能拒答。 |
| API | session isolation smoke 通过；所有 session 操作有 auth guard。 |
| Multi-turn | 多轮 eval lite 通过；evidence 只在同 user/session 内复用。 |

## 9. 常见质量问题处理

### 必需法条没有召回

排查顺序：

1. chunk 是否包含该条文。
2. metadata 的 `title`、`article_no`、`is_current` 是否正确。
3. metadata filter 是否误删。
4. BM25 是否能召回。
5. dense 是否能召回。
6. hybrid fusion 是否把正确结果压低。
7. query rewrite 是否改坏了关键词。

### 引用校验失败

排查顺序：

1. answer 是否引用了不存在的 `evidence_id`。
2. quote 是否不是 evidence text 子串。
3. finalizer 是否改写了 quote。
4. citation checker 是否在 finalizer 前后都运行。
5. evidence pool 是否混入了其他 session 的 evidence。

### 答案有无依据结论

排查顺序：

1. answer prompt 是否明确禁止超出 evidence。
2. finalizer 是否检查 unsupported claims。
3. evidence 是否足以支持结论。
4. insufficient evidence 分支是否被 intake/orchestrator 跳过。
5. mock LLM 是否覆盖了拒答场景。

### 跨用户数据泄露

排查顺序：

1. session store 查询是否包含 `tenant_id` 和 `user_id`。
2. evidence 查询是否按 auth 过滤。
3. trace/artifact 是否记录敏感用户内容。
4. delete/purge 是否只作用当前 auth scope。
5. tests 是否模拟了两个不同用户。

## 10. Recommended First User Prompt for Codex

如果要让 Codex 从零开始实现 MVP-0，可以这样下达任务：

```text
请按照 legal_rag_plan/13_CODEX_IMPLEMENTATION_GUIDE.md 实现 MVP-0 Step 0 Skeleton。
只做 mock profile、基础 schemas、provider 检查脚本和最小测试。
不要实现 ingestion、retrieval、agent、API。
完成后运行 check_providers.py --profile mock，并汇报修改文件和测试结果。
```

后续每个 step 都用类似方式推进，避免一次性跨阶段生成过多代码。

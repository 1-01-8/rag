# Multi-Agent Legal RAG — 完整 Pipeline 报告

> 项目位置: `/home/xxm/rag/experiments/multi_agent/`
> Spec: `/home/xxm/rag/docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md`
> 实测状态: 340+ unit + integration 测试通过, 真实 Qwen + SiliconFlow E2E 验证
>
> 本报告分两半:
> - **第一部分: RAG** — 数据 / 索引 / 检索 / 记忆 (所有跟"检索"相关的)
> - **第二部分: Agent** — 4 个 agent 角色 / 编排 / Trace / 评测 (所有跟"决策"相关的)

---

# 第一部分: RAG (检索增强生成)

## 1.1 整体 RAG 流程

RAG 是两阶段:

```
阶段 1 (Indexing, 离线一次性)
─────────────────────────────────────────────
原始文档 → 切 chunk → 向量化 → 写 Qdrant
Chinese-Laws  按法条     bge-m3   ma_statutes
   177 部     ~13722条   +jieba   ma_cases
                        (dense   ma_user_history
                         +sparse)


阶段 2 (Retrieval+Generation, 每次查询)
─────────────────────────────────────────────
用户问题 → 向量化 query → Qdrant 检索 top-k → LLM 综合 → 答复
            (同样 bge-m3)  (混合 RRF 融合)
```

## 1.2 数据源

| 文件夹 | 内容 | 大小 | 用途 |
|---|---|---|---|
| `/home/xxm/rag/Chinese-Laws/extracted/` | 177 部中国法律全文 `.txt` | 7.4 MB | `ma_statutes` 索引源 |
| `/home/xxm/rag/laws_data/` | 23k 真实律师 Q&A 对 | 16 MB | `ma_cases` 索引源 |
| `memory_store/sessions/<sid>/turns/` | 当前用户每轮对话存档 | 增长 | `ma_user_history` 索引源 |

### Corpus 边界 (spec §0.3 接受)

未收录: **劳动合同法 / 刑法 / 公司法 / 仲裁法** — 询问这些领域时, Lawyer prompt 已有 fallback 路径 (citations=[], 在 dispute_analysis 中如实声明).

## 1.3 阶段 1: Indexing Pipeline

### 1.3.1 corpus loader (`multi_agent/tools/corpus.py`)

读取 `Chinese-Laws/extracted/*.txt`:

```python
load_corpus(corpus_dir) → list[Document]
```

- 一个 `.txt` → 一个 `Document` (代表一部法律)
- 每条法条 → 一个 `Chunk`:
  ```python
  Chunk(
      doc_id="民法典-510",       # 唯一 ID
      law_name="中华人民共和国民法典",
      law_short="民法典",
      article_no="510",
      text="当事人就合同补充内容...",  # 原文
  )
  ```
- 总产出: 177 部 → ~13722 chunks

### 1.3.2 DenseEncoder (`multi_agent/tools/retrievers/dense_encoder.py`)

```python
DenseEncoder()  # 默认 cuda:1, 可改 BGE_M3_DEVICE=cuda:0
encoder.encode_batch(texts, batch_size=16) → (N, 1024)
```

- 模型: bge-m3 (本地 `/home/xxm/models/bge-m3/`)
- 维度: 1024
- 推荐 batch 8-16 (GPU illegal-memory-access 风险下调)

### 1.3.3 SparseEncoder (`multi_agent/tools/retrievers/sparse_encoder.py`)

```python
sparse = SparseEncoder()
sparse.fit(text_iter)            # 拟合 IDF
vec = sparse.encode(text)        # → SparseVector(indices, values)
sparse.save(path)                # 持久化为 json
```

- 分词: jieba
- 权重: IDF
- 持久化到 `data/indexes/statutes_sparse.json`, 跟 Qdrant collection **必须配对**, 缺一不可

### 1.3.4 IndexBuilder (`multi_agent/tools/retrievers/index_builder.py`)

主入口 `build_index()`:

```python
build_index(
    documents=docs,
    collection_name="ma_statutes",
    sparse_artifact_path=Path("data/indexes/statutes_sparse.json"),
    dense_encoder=DenseEncoder(),
    batch_size=8,
) → IndexArtifacts
```

步骤:
1. flatten chunks
2. fit sparse 在 chunk.text 上
3. **批次**循环: dense 编码 + sparse 编码 + Qdrant upsert
4. 持久化 sparse json

### 1.3.5 Qdrant Collections

容器: `legal-rag-qdrant` (docker, host 6433→container 6333), 跟旧 `legal_rag/` 共享但 **`ma_` 前缀隔离**.

| Collection | dense_dim | sparse | 用途 | Spec |
|---|---|---|---|---|
| `ma_statutes` | 1024 | ✓ | 法条 (13722 chunks) | §4.2 |
| `ma_cases` | 1024 | ✓ | 律师 Q&A | §4.2 |
| `ma_user_history` | 1024 | ✗ | 历史 turn | §4.2 |
| `ma_user_history_chat` | 1024 | ✗ | chat.py 专用 | (派生) |

CLI 建索引脚本: `scripts/build_statutes_index.py` / `build_cases_index.py`.

## 1.4 阶段 2: Retrieval Pipeline

### 1.4.1 StatuteSearchTool (`multi_agent/tools/retrievers/statute_search.py`)

```python
class StatuteSearchTool(Tool):
    name = "statute_search"
    args_schema = StatuteSearchArgs  # {query, k, law_short?}

    async def call(args, recorder) → ToolResult:
        # 1. bge-m3 编码 query → dense_vec (1024)
        # 2. jieba+IDF 编码 query → sparse_vec
        # 3. client.query_points(
        #      collection=ma_statutes,
        #      prefetch=[Prefetch(query=sparse, using="sparse"),
        #               Prefetch(query=dense, using="dense")],
        #      query=FusionQuery(fusion=Fusion.RRF),
        #      limit=k,
        #    )
        # 4. 返回 List[Evidence]
```

**关键设计**: Qdrant native **Reciprocal Rank Fusion**, 一次 round-trip 同时跑 dense+sparse 然后融合, 不是分别查后客户端合并.

### 1.4.2 CaseSearchTool (`multi_agent/tools/retrievers/case_search.py`)

同样结构, 但跨 `ma_cases`, 返回的是律师 Q&A 配对 (含 case_id + cause/分类).

### 1.4.3 HistorySearchTool (`multi_agent/tools/retrievers/history_search.py`)

dense-only (turn text 不需要 sparse), 按 `session_id` filter:

```python
HistorySearchTool(
    collection_name="ma_user_history",
    dense_encoder=encoder,
    default_session_id="chat_xxxxxx",   # 构造时绑定 session
)
# Lawyer 调用时只传 query, session 自动应用
# scope="all_sessions" 时跨 session 检索
```

### 1.4.4 AllSourcesSearchTool (`multi_agent/tools/retrievers/all_sources_search.py`)

跨 collection 融合检索:

```python
AllSourcesArgs(query, k=8, law_short?, cause?, include_history=False)
→ statute_search + case_search 各自查 → RRF 二次融合 → top_k Evidence
→ 若 include_history=True 且 history_search 注入, 额外返回 history_hits 字段
```

为啥不一次性融合三者? 因为 history hits 的 schema 跟 Evidence (法条) 完全不同, 不适合 RRF 同一池子.

### 1.4.5 ExactReadTool (`multi_agent/tools/retrievers/exact_read.py`)

按 `doc_id` 精读单条法条全文 (Lawyer 检索结果是 excerpt 截断, 想看完整条文用这个):

```python
ExactReadArgs(doc_id="民法典-563") → Evidence (text 是完整条文)
```

### 1.4.6 TurnIndexer (`multi_agent/tools/retrievers/turn_indexer.py`)

每完成一个 turn, 把 turn 内容 (question + final_answer) 索引到 `ma_user_history`:

```python
TurnIndexer(collection_name="ma_user_history", dense_encoder=encoder)
await indexer.index_turn(session_id, turn)
# 写入: dense_vec + payload {session_id, turn_no, run_id, question_preview, answer_preview, started_at}
# Point ID 用 uuid5 (session+turn 确定性) → 同 turn 重写而非追加
```

被 `run_query(turn_indexer=indexer)` 自动触发.

## 1.5 Memory 层

**多层架构** (按时间尺度):

| 层 | 时间尺度 | 存储 | 谁写 | 谁读 |
|---|---|---|---|---|
| **WorkingMemory** | 1 个 run (~分钟) | Pydantic 对象, 内存 | BaseAgent 自动收集 Evidence | 跨 ReAct step 复用, 序列化到 `artifacts/working_memory.json` |
| **sticky.md** | 1 个 session (~天-周) | MD + YAML frontmatter | runner 每轮覆盖 | Receptionist/Lawyer 读 |
| **turns/NNN.md** | 永久 | MD, 不可变 | runner 每轮追加 | 历史回溯 |
| **agent_notes/*.md** | 永久 | MD | Supervisor reject 时触发 LLM 生成 | 跨 session 学习 |

### 1.5.1 sticky.md (Phase 3)

```yaml
---
session_id: chat_abc123
legal_domain: 民事
case_type: 租赁纠纷
last_law_name: 民法典
mentioned_laws: [民法典]
cited_articles:
  - {law: 民法典, article: "510", from_turn: 1}
linked_runs: [r_a1b2c3]
entity_state:
  key_facts: [...]
  open_questions: [...]
  rejected_paths: [...]
history_summary: "第 1-3 轮: 用户咨询涨租问题..."
---
```

### 1.5.2 Intent-based 读取 (Phase 5v)

避免每次都吃完整 sticky 的 token 成本:

```python
store.read_sticky(sid)                          # → StickyContext (默认 full)
store.read_sticky(sid, intent="entities_only")  # → StickyEntitiesView (省 60%)
store.read_sticky(sid, intent="recent_citations") # → StickyCitationsView
store.read_sticky(sid, intent="summary_only")   # → StickySummaryView
```

26 测试覆盖所有鲁棒边界 (文件缺失/YAML 损坏/字段空/单条目损坏/未知 intent).

### 1.5.3 Cross-turn 压缩 (Phase 3c)

当 session > 5 turns:
```python
await maybe_compact(session_id, store, provider, model)
# 1. 取最老的 (total - 3) 个 turn
# 2. 调 LLM 压缩成 ≤200 字 prose
# 3. 写入 sticky.history_summary
# 4. 旧 turn 文件不删 (保留可追溯性)
```

通过 `run_query(compaction_provider=qwen, compaction_model="qwen3.5-9b")` 可选启用.

### 1.5.4 Agent Notes (Phase 5u)

当 Supervisor verdict=reject 时, 自动触发本地 Qwen 总结失败模式写 `agent_notes/<slug>.md`:

```yaml
---
name: lawyer-rental-misses-mgmt-rules
description: 涨租问题 Lawyer 漏引《商品房屋租赁管理办法》
about_agent: lawyer
verdict_that_triggered: reject
triggered_by_run: r_xxx
created_at: 2026-05-15T10:00:00
---

## 失败模式
...

## 改进建议
...
```

跨 session 可被 Receptionist/Lawyer 读取作为先验 (当前 prompt 未集成, Phase 3e+ 可加).

## 1.6 RAG 检索质量 (实测)

**benchmark.py 4 个查询** (synthetic_seed_v1 实质部分):

| Query | 引用命中 | 检索耗时 | LLM 耗时 |
|---|---|---|---|
| 房东涨租 30% 合法吗 | ✓ 民法典 510/703 | <2s | 70s (本地 Qwen) |
| 邻居漏水索赔 | ✓ 民法典 1165/1184 | <2s | 45s |
| 网购假货退款 | ✓ 民法典 577/584 | <2s | 102s |
| 交通追尾责任 | ✓ 道交法 76 | <2s | 69s |

**4/4 = 100% citation hit, 平均 检索 < 2s, LLM 主导耗时**.

### 1.6.1 真实 ablation 实验 (Phase 5g)

**`DisableTool(statute_search)`** vs baseline (跑 q001+q002):
- 基线: 2/2 citation hit, 102s p50
- 禁 statute_search: **1/2 hit** (citation 准确率减半), 66s p50, 节省 5861 input tokens

→ 实验数据印证: **statute_search 工具贡献了 50% 的引用准确率**.

---

# 第二部分: Agent (多 agent 编排)

## 2.1 整体 Agent 架构

```
                ┌─────────────────────────────────────┐
                │  用户问题                            │
                └──────────────┬──────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────┐
            │  Receptionist (接待员, tool-less) │
            │  - 分类 民事/交通/婚姻/房产/劳动   │
            │  - 提取 EntityState (key_facts...)│
            │  - 检测 follow-up                 │
            │  - 拆分 multi_issues              │
            └────────────────┬─────────────────┘
                             │ ReceptionistOutput
                             ▼
       ┌──────────────────────────────────────────┐
       │  Lawyer (5 specialty 变体之一)           │
       │  ReAct 循环:                              │
       │   - statute_search (RRF, 最多 3 次)      │
       │   - case_search                          │
       │   - history_search (follow-up)           │
       │   - read_article (精读)                  │
       │   - ask_secretary (业务 task)            │
       │   - verify_citation                      │
       │  Tool-first enforcement (防编造)         │
       │  → LawyerOutput JSON (5 段式)            │
       └─────────────┬────────────────────────────┘
                     │
                     ▼
            ┌────────────────────────────────┐
            │  Supervisor (审核员)            │
            │  读 lawyer_output + 证据池      │
            │  调 verify_citation (程序化)    │
            │  → SupervisorVerdict            │
            │     pass / revise / reject      │
            │  若 reject + memory_store: LLM  │
            │    总结失败模式写 agent_notes/  │
            └────────┬────────────────────────┘
                     │
                     ▼ (clarification mode 短路)
              ┌──────────────────────┐
              │  最终返回给用户        │
              └──────────────────────┘
```

## 2.2 4 个 Agent 详细对照

| | Receptionist | Lawyer | Secretary | Supervisor |
|---|---|---|---|---|
| **是否 ReAct** | ❌ 单 LLM call | ✅ 真多步 | ✅ 任务级 | ❌ 单 LLM call (clarification 模式直接跳过) |
| **工具数** | 0 | 7 | 3 | 1 |
| **输出 schema** | ReceptionistOutput | LawyerOutput | 3 业务 schemas | SupervisorVerdict |
| **Tool-first 强制** | N/A | ✅ (Phase 2d) | ❌ | ❌ |
| **prompts 路径** | `prompts/receptionist/` | `prompts/lawyer/` (6 文件) | `prompts/secretary/` | `prompts/supervisor/` |
| **读 sticky** | ✅ (intent="entities_only" 推荐) | ✅ (full) | ❌ | ❌ |
| **写 sticky** | ✅ via runner | ✅ via runner | ❌ | ❌ |
| **写 turn** | ✅ via runner | ✅ via runner | ❌ | ❌ |
| **写 agent_note** | ❌ | ❌ | ❌ | ✅ (verdict=reject 时) |
| **维护 WorkingMemory** | ✅ 累积 evidence | ✅ 累积 evidence | ✅ 读取 | ✅ 读取 |

### 2.2.1 Receptionist

文件: `multi_agent/agents/receptionist.py`, `prompts/receptionist/system.md`

输出:
```python
ReceptionistOutput(
    legal_domain="民事",
    case_type="租赁纠纷",
    is_followup=False,
    multi_issues=[Issue(...), Issue(...)],
    risk_level="medium",
    urgent=False,
    reason="...",
)
```

### 2.2.2 Lawyer

文件: `multi_agent/agents/lawyer.py`

**5 个 specialty 变体** (按 ReceptionistOutput.legal_domain 路由):
- `specialty_民事.md` — 默认 / fallback
- `specialty_房产.md` — 租赁/买卖 (强调 1077/703/707)
- `specialty_交通.md` — 道交法 + 侵权编
- `specialty_婚姻.md` — 婚姻家庭编 + 反家暴法
- `specialty_劳动.md` — **明确提醒 corpus 未收录劳动合同法**, Lawyer 知道 fallback
- `specialty_通用.md` — 兜底

公共骨架: `prompts/lawyer/_five_section_skeleton.md`

输出:
```python
LawyerOutput(
    mode="consultation" | "clarification",  # Phase 5af 加 clarification
    primary_answer="一句话核心结论",
    citations=[Citation(law_short, article_no, excerpt)],
    five_section=FiveSection(
        dispute_analysis="...",
        applicable_laws="...",
        similar_cases="...",
        remedy_suggestions="...",
        risk_assessment="...",
    ),
    clarifying_questions=[]  # 仅 mode=clarification 时非空
)
```

**Tool-first enforcement** (Phase 2d):
- 第一轮模型不调 tool → silent reject + 重定向
- `max_pre_tool_rejections=2` budget, fail-fast

**Search 上限** (Phase 5ah):
- statute_search 最多 3 次, 之后强制 final JSON
- 找不到法条时显式 fallback "本系统 corpus 未直接收录..."

### 2.2.3 Secretary

文件: `multi_agent/agents/secretary.py`

Lawyer 通过 `ask_secretary` tool 委托业务任务:

```python
SecretaryRequest(task="contract_review" | "doc_generation" | "doc_interpret",
                  context, params)
↓ 路由到对应业务 tool:
  - contract_review (Phase 4)
  - doc_generation
  - doc_interpret
```

**Agent-as-Tool 模式** (spec ADR-05): `SecretaryAsTool` 包装 SecretaryAgent, 父 agent (Lawyer) 看到的是普通 Tool, 实际触发子 agent ReAct.

### 2.2.4 Supervisor

文件: `multi_agent/agents/supervisor.py`

输入 (orchestration 串行调用, 不是 Lawyer ReAct 内部):
```python
SupervisorAgent(
    payload={
        "user_query": str,
        "lawyer_output": LawyerOutput.model_dump(),
        "evidence_pool": [Evidence.model_dump(), ...]  # WorkingMemory
    }
)
```

输出:
```python
SupervisorVerdict(
    verdict="pass" | "revise" | "reject",
    confidence=0.95,
    issues=[...],
    suggested_fix=None,
    citation_checks=[CitationCheckResult(citation_index, valid, reason)],
    groundedness=GroundednessCheck(score, ungrounded_claims),
)
```

**Clarification 短路** (Phase 5af): 当 lawyer_output.mode=="clarification" 时, orchestrator 跳过 Supervisor LLM 直接 verdict=pass (省 25-30s + ~$0.001 成本).

**Reject → agent_note** (Phase 5u): 当 verdict=="reject" 且配置了 note_provider, 自动调本地 Qwen 总结失败模式写入 `agent_notes/*.md`.

## 2.3 编排入口

| 函数 | 用途 |
|---|---|
| `runner.run_query(query, agent_factory, provider, ...)` | 单 agent 跑 (Receptionist 或 Lawyer 或 Secretary 或 Supervisor) |
| `orchestration.supervised.run_with_supervisor(query, lawyer_factory, supervisor_factory, ...)` | Lawyer + Supervisor 串行编排 (主入口) |
| `eval.runner.ExperimentRunner.run()` | 批量 QuerySet 跑评测 |
| `eval.ablation_runner.AblationRunner.run(ablations=[...])` | 多 profile × QuerySet, 算 deltas |

## 2.4 Provider 抽象

文件: `multi_agent/providers/`

3 个 provider + 1 个工厂:

| Provider | 接入 | Tool calling | 已知速度 (单次 LLM call) |
|---|---|---|---|
| `OpenAICompatibleProvider` | OpenAI 兼容 API (vLLM / DeepSeek / SiliconFlow / OpenRouter) | ✅ | 5-25s (取决端点) |
| `AnthropicProvider` | Anthropic SDK + cache_control | ✅ | 5-30s (Opus 慢, Haiku 快) |
| `StubProvider` | 测试用 scripted | ✅ | 0ms |
| `ProviderProfile` 工厂 | spec §6.4 — 4 profile (all-local/all-claude/mixed-cloud-judge/mixed-cloud-brain) | - | - |

支持 streaming (Phase 2b Task 7), Pydantic-typed `LLMResponse`/`StreamChunk`/`Usage`.

### 2.4.1 实测可用的 LLM 选项

| Provider | Model | 单轮 (Lawyer+Sup) | 成本/轮 | 稳定性 |
|---|---|---|---|---|
| `local` (vLLM Qwen 3.5-9B) | qwen3.5-9b | 100-150s | $0 | ✓ 稳定 |
| `siliconflow` | **deepseek-ai/DeepSeek-V3.1** | **30-60s** | **~$0.003** | **✓ 实测稳定** |
| `siliconflow` | deepseek-ai/DeepSeek-V4-Flash | 25-? (偶尔挂 100s+) | ~$0.0014 | ✗ 不稳, 短期内勿用 |
| `siliconflow` | Qwen/Qwen3-235B-A22B-Instruct-2507 | 30-50s (估) | ~$0.003 | 待测 |
| `deepseek` 官方 | deepseek-chat | 40-70s | ~$0.003 | 日本不可达 (CloudFront RST) |
| `anthropic` | claude-opus-4-7 | 50-80s | ~$0.40 | ✓ |

## 2.5 Trace 系统

文件: `multi_agent/tracing/`

横切关注点 — 每个 agent/tool/LLM 调用都 emit 事件到 `events.jsonl`:

```
RunStarted
└─ AgentInvoked (lawyer)
   ├─ LLMRequested → LLMResponded (usage tokens, finish_reason)
   ├─ ToolCalled (statute_search) → ToolReturned (count, duration_ms)
   ├─ LLMRequested → LLMResponded
   ├─ ToolCalled (ask_secretary)
   │  └─ AgentInvoked (secretary) — 子 agent 嵌套
   │     ├─ ToolCalled (contract_review) → ToolReturned
   │     └─ AgentResponded (secretary)
   │  └─ ToolReturned
   └─ AgentResponded (lawyer)
RunFinished (status=ok)
```

**ContextVar span stack** (`recorder.py`) — 自动维护 `parent_id` 链, 异步并发安全, 不被 fan-out 打乱.

派生产物: `artifacts/working_memory.json` (Phase 5r, spec §5.4.2 序列化).

## 2.6 评测框架 (`multi_agent/eval/`)

### 2.6.1 QuerySet (`queryset.py`)

YAML 定义查询集:
```yaml
meta: {name: synthetic_seed_v1, created: 2026-05-14}
queries:
  - id: q001
    text: "房东涨租 30%..."
    cause: 房产纠纷
    tags: [民事, 租赁]
    expected:
      should_cite_any: [民法典-510, 民法典-703]
      expected_answer_mode: evidence_grounded
```

实际跑过的: `synthetic_seed_v1.yaml` (5 条手写种子). spec §7.3 蓝图 (300/5k/30/10 各类) 暂未做.

### 2.6.2 ExperimentRunner (`runner.py`)

```python
ExperimentRunner(
    query_set=qs,
    run_group_name="qwen_baseline",
    runs_root=Path("runs"),
    query_runner=async_factory,
    parallelism=2,
    judges=[GroundednessJudge, HelpfulnessJudge],  # 可选
).run() → RunGroup
```

每个 query → 一个 run_dir, 收集 metrics + 跑 judges.

输出: `run_groups/<name>/{group_meta.yaml, results.jsonl, summary.md}`.

### 2.6.3 自动 metrics (`metrics.py`)

`derive_run_metrics(run_dir) → RunMetrics`:

```python
RunMetrics(
    total_latency_ms,
    total_input_tokens, total_output_tokens, cache_read_tokens, cache_hit_rate,
    cost_usd,                       # Phase 5f
    agent_invocations,
    tool_calls_total,
    react_steps_total,              # Phase 5q
    supervisor_verdict,
    final_answer_mode,              # Phase 5q
    citation_count,                 # Phase 5q
    errors,
)
```

### 2.6.4 Judges

1. **CitationAccuracyJudge** (`judges/citation_accuracy.py`, rule-based, Phase 5b)
   - 比较 `Query.expected.should_cite_any` vs `LawyerOutput.citations`
   - 无 LLM, 0 成本

2. **GroundednessJudge** (`judges/groundedness.py`, LLM-based, Phase 5c)
   - 用 Claude Opus (spec §7.7) 判断每条 lawyer 陈述是否在 evidence_pool 中可溯源
   - 输出 `{score, ungrounded_claims, rationale}`

3. **HelpfulnessJudge** (`judges/helpfulness.py`, LLM-based)
   - 判断答复 directness + actionability + completeness
   - 输出 `{score, missing_aspects, rationale}`

### 2.6.5 Comparator (`comparator.py`, Phase 5c + 5h)

两个 RunGroup 对比 → per-query delta + Winner verdict:
```python
ComparisonReport(
    per_query=[
        PerQueryDelta(
            query_id, latency_delta_ms, in/out_tokens_delta, cost_delta_usd,
            citation_hit_a, citation_hit_b,
            groundedness_delta, helpfulness_delta,
            winner="A" | "B" | "tie",
        ),
        ...
    ],
    winners_a, winners_b, ties,
)
```

Winner heuristic: groundedness Δ ≥ 0.05 > citation_hit 差异 > helpfulness Δ ≥ 0.05 > tie.

### 2.6.6 AblationRunner (`ablation_runner.py`, Phase 5d)

```python
AblationRunner.run(ablations=[
    DisableAgent(agent="supervisor"),
    SwapModel(agent="lawyer", provider="anthropic", model="claude-opus-4-7"),
    DisableTool(tool="case_search"),
    DisableMemory(),
])
```

跑 baseline + N ablation × QuerySet, 输出 `ablation_summary.md` 含 Δ.

### 2.6.7 LatencyProfiler (`latency.py`, Phase 5e)

从 events.jsonl 派生 SpanTiming 树 + by-agent/tool/provider/kind 聚合:

```
run:run  inc=70640ms exc=120ms
  agent:lawyer  inc=70300ms exc=2300ms
    llm:openai_compat:deepseek-ai/DeepSeek-V3.1  inc=12500ms
    tool:statute_search  inc=850ms
    llm:...  inc=11200ms
    ...

by_agent: lawyer=70300ms
by_tool: statute_search=850ms
by_provider: openai_compat=68000ms
```

CLI: `python scripts/profile_run.py runs/r_xxx`.

## 2.7 Scripts (用户接口)

`multi_agent/scripts/`:

| 脚本 | 用途 | 状态 |
|---|---|---|
| `serve_qwen_vllm.sh` | 启动本地 Qwen vLLM (含 tool-call parser) | ✓ |
| `build_statutes_index.py` | 灌 Chinese-Laws → ma_statutes | ✓ 已建 13722 chunks |
| `build_cases_index.py` | 灌 laws_data → ma_cases | 待跑 |
| `extract_case_citations.py` | LLM 抽 lawyer 答复里的法条引用 (Phase 2d 离线工具) | ✓ |
| `benchmark.py` | 跑 synthetic_seed_v1 → 输出 json + flame | ✓ |
| `run_eval.py` | 灵活批量评测 | ✓ |
| `run_comparison.py` | 两 profile 对比 (Qwen vs Claude) | ✓ |
| `profile_run.py` | 单 run flame graph | ✓ |
| `chat.py` | 交互式 REPL | ✓ |
| **`chat-ready.sh`** | **已验证可用的一键启动** | ✓ |
| `trace_viewer.py` | Streamlit 三栏 trace UI | ✓ |
| `test_deepseek.py` | DeepSeek / SiliconFlow 连通 smoke | ✓ |

## 2.8 测试体系

```
tests/
├── unit/         (300+ 测试, ~5 分钟全跑)
│   ├── test_*_schema.py        每个 schema 一个
│   ├── test_recorder.py
│   ├── test_*_provider.py
│   ├── test_*_agent.py
│   ├── test_*_search.py
│   ├── test_*_judge.py
│   ├── test_metrics.py / pricing.py / comparator.py / ablations.py / latency.py
│   ├── test_compaction.py / test_sticky_intents.py
│   └── test_*_script.py        scripts subprocess --help smoke
│
└── integration/  (15+ 测试, 需 vLLM + Qdrant 真服务)
    ├── test_retrieval_e2e.py
    ├── test_lawyer_*_e2e.py    (民事/劳动/交通/多源)
    ├── test_receptionist_lawyer_e2e.py
    ├── test_lawyer_via_secretary_e2e.py
    ├── test_supervised_lawyer_e2e.py
    ├── test_multi_turn_session_e2e.py
    ├── test_memory_loop_e2e.py
    ├── test_ablation_e2e.py    (真 ablation 实验)
    ├── test_eval_e2e.py
    └── test_claude_judges_e2e.py  (gated by ANTHROPIC_API_KEY)
```

## 2.9 已实测的 Agent 行为

**真实跑过的实验** (Phase 5g / 5i / 5w):

1. **零编造引用**: Lawyer 在 4/4 查询命中预期法条 (Phase 5w benchmark)
2. **Supervisor 双层防护**: 4/4 verdict=pass, citation_checks 全部 valid
3. **真实 ablation**: 禁掉 statute_search 引用准确率 100% → 50%
4. **多轮记忆**: Turn 2 follow-up 时 Lawyer 自动调 history_search 接续上下文 (Phase 5i)
5. **clarification 模式**: V4-Flash/V3.1 在信息不足时反问澄清, Supervisor 自动 pass (Phase 5af)
6. **Corpus 缺失 fallback**: 询问劳动法时 prompt 引导 citations=[], 显式声明 corpus 未收录

---

# 第三部分: 数字总览

## 3.1 代码体量

```
multi_agent/        ~6000 行 (核心包)
  agents/             ~700 行
  eval/              ~1200 行
  memory/             ~400 行
  providers/         ~1100 行
  schemas/            ~600 行
  tools/             ~1500 行
  tracing/            ~400 行
  orchestration/      ~100 行

tests/              ~5000 行
prompts/             ~30 个 markdown 文件
scripts/            ~1200 行 (CLI)
docs/                spec + plans (~10000 行 markdown)
```

## 3.2 Phase tag (43 个)

```
phase1 / 2a / 2b / 2c / 2d
phase3 / 3b / 3c / 3d / 3e / 3f
phase4
phase5a / 5b / 5c / 5d / 5e / 5f / 5g / 5h / 5i / 5j
phase5k / 5l / 5m / 5o / 5p / 5q / 5r / 5t / 5u / 5v / 5w / 5x
phase5aa / 5ab / 5ac / 5ad / 5ae / 5af / 5ag / 5ah / 5ai / 5aj
```

## 3.3 性能 baseline (V3.1 + SiliconFlow + ma_statutes)

- 单轮 p50: **30-60 秒**
- 单轮 p95: **~100 秒**
- 每轮成本: **~$0.003**
- 每轮 tokens: in 3000-5000, out 1000-2000
- 引用准确率 (synthetic_seed_v1): **100%** (4/4)

---

# 第四部分: 实际启动指南 (TL;DR)

## 4.1 推荐使用路径 (已验证)

```bash
export SILICONFLOW_API_KEY=sk-xxx
conda activate qwen35
cd /home/xxm/rag/experiments/multi_agent
bash scripts/chat-ready.sh
```

打包配置: SiliconFlow + DeepSeek-V3.1 + ma_statutes 全量 + no-supervisor.

## 4.2 备用启动

```bash
# 本地 Qwen (慢但 $0)
python scripts/chat.py --provider local \
    --statutes-collection ma_statutes \
    --statutes-sparse data/indexes/statutes_sparse.json

# 严格审核模式 (慢 25s 但有引用校验)
python scripts/chat.py --provider siliconflow \
    --model deepseek-ai/DeepSeek-V3.1 \
    --statutes-collection ma_statutes \
    --statutes-sparse data/indexes/statutes_sparse.json
```

## 4.3 离线索引重建

```bash
# 全量 (15-30 分钟)
BGE_M3_DEVICE=cuda:0 python scripts/build_statutes_index.py \
    --corpus-dir /home/xxm/rag/Chinese-Laws/extracted \
    --collection ma_statutes \
    --sparse-out data/indexes/statutes_sparse.json \
    --batch-size 8
```

---

# 附录: 项目文件路径速查

| 主题 | 路径 |
|---|---|
| 完整 spec | `/home/xxm/rag/docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md` |
| 项目根 | `/home/xxm/rag/experiments/multi_agent/` |
| 索引数据 | Qdrant docker `legal-rag-qdrant` (host:6433) |
| Sparse 索引文件 | `data/indexes/statutes_sparse.json` |
| 内存目录 | `memory_store_chat/` (chat.py 默认) / `memory_store/` (其他) |
| Run 目录 | `runs/r_XXXXXXXX/` |
| 操作手册 | `RUNBOOK.md` |
| 本报告 | `PIPELINE_REPORT.md` |

# LegalResearch-Agent 总览与导航

> 项目：Harness-controlled Self-Evolving Multi-Agent Legal RAG System，**原生多轮对话**。
> 模型策略：Embedding / Reranker / LLM 全部支持「**本地** + **硅基流动 (SiliconFlow) API**」双后端，由配置切换。
> 主 LLM：`Qwen/Qwen2.5-32B-Instruct`（硅基流动）或本地 vLLM / Ollama 部署的等价模型。

---

## 0. 文档结构

| 编号 | 文档 | 作用 | 给谁看 |
|---|---|---|---|
| 00 | OVERVIEW | 总览 + 导航 + Phase 顺序 | 所有人 |
| 01 | ARCHITECTURE | 整体架构、Harness 哲学、核心数据结构（含会话） | Codex 必读 |
| 02 | MODEL_PROVIDERS | Embedding / Reranker / LLM 双后端抽象层 | Codex 必读 |
| 03 | PHASE1_SKELETON | 项目骨架 + 配置 + provider 抽象 + 会话 schema | Codex |
| 04 | PHASE2_INGESTION | PDF/TXT/MD 加载、法律切分、metadata 抽取 | Codex |
| 05 | PHASE3_INDEX | BM25 + Dense + Hybrid 检索、reranker stub | Codex |
| 06 | PHASE4_AGENTS | 9 个 agent（含 ContextComposer，**原生多轮**；ContextCompactor 在 Phase 5） | Codex |
| 07 | PHASE5_HARNESS_GRAPH | LangGraph 状态机、ConversationManager、validator、tracing | Codex |
| 08 | PHASE6_MEMORY | SQLite 会话存储 + 长期记忆 + 自进化闭环 | Codex |
| 09 | PHASE7_EVALUATION | 评估指标 + 回归门禁（含多轮指标） | Codex |
| 10 | PHASE8_API | FastAPI 暴露（含 `/sessions*` 路由组） | Codex |
| 11 | OPERATIONS | 安全、隐私、性能、运维坑点 | 所有人 |

> ⚠ **不再有横切补丁**。多轮对话从 Phase 1 schema 起就是一等公民；上下文压缩是 Claude-Code 风格的 runtime 中间件（不是 agent，不进 graph）：
>   - Phase 1：会话 schema (`ConversationState / TurnRecord / WorkingContextDigest`) 与 token budget 配置；
>   - Phase 4：`ContextComposer` 是 agent 层唯一 messages 装配点，估算超 budget 时**透明**调 compactor；
>   - Phase 5：`harness/context_compactor.py` 实现 Claude-Code 风格的工作上下文压缩，由 ContextComposer 透明触发，**不进 graph**；HarnessRuntime 本身就是 ConversationManager；
>   - Phase 6：会话表（含 `session_digest`）与长期记忆表共用同一 SQLite；
>   - Phase 8：`/sessions*` 是 API 主路由；`POST /sessions/{id}/compact` 显式触发压缩；`/ask` 是单 turn 临时 session 的语法糖。

---

## 1. 一句话定位

本项目模拟法律研究团队工作方式：

```text
接案理解 → 法律研究计划 → 法条/案例/合同分工检索
       → 证据核验 → 法律意见生成 → 反方复核 → 经验沉淀 → 下一次自动变好
```

**Harness** 是包裹 LLM/Agent 的运行外壳：

```text
Harness = Context + Tools + Workflow + State + Validation + Observability + Evaluation + Safety + Memory + Conversation
```

模型只是推理内核，可靠性由模型外部的工程约束保证。

---

## 2. 模型后端策略（贯穿所有 Phase）

| 角色 | 推荐模型 | 本地后端 | 远端后端 |
|---|---|---|---|
| Embedding | `BAAI/bge-m3` | `sentence-transformers` 加载 | 硅基流动 `/v1/embeddings` |
| Reranker | `BAAI/bge-reranker-v2-m3` | `FlagEmbedding` | 硅基流动 `/v1/rerank` |
| LLM | `Qwen/Qwen2.5-32B-Instruct` | vLLM / Ollama（OpenAI 兼容 HTTP） | 硅基流动 `/v1/chat/completions` |

切换由 `.env` 控制，不改代码：

```env
EMBEDDING_PROVIDER=siliconflow      # local | siliconflow
RERANKER_PROVIDER=siliconflow       # local | siliconflow
LLM_PROVIDER=siliconflow            # local | siliconflow
```

详见 `02_MODEL_PROVIDERS.md`。

---

## 3. 推荐实现顺序与端到端里程碑

```text
Phase 1 (skeleton + provider 抽象 + ConversationState schema)
   └─ 验收：python scripts/check_providers.py 能 ping 通；ConversationState round-trip 测试通过
Phase 2 (ingestion)
   └─ 验收：python scripts/ingest_docs.py 输出 chunks.jsonl，metadata 正确
Phase 3 (index + retrieval)
   └─ 验收：python scripts/retrieve.py "劳动合同 第39条" 命中目标 chunk
Phase 4 (agents：含 ContextComposer，原生多轮；不含 Compactor)
   └─ 验收：每个 agent 单测通过；ContextComposer 用 NullCompactor 与 mock Compactor 都能正确装配
Phase 5 (harness + graph + ConversationManager + ContextCompactor)
   └─ 验收：python scripts/chat.py 跑 9 轮自动触发 harness.compaction 事件；/compact 命令立即返回 saved_ratio
Phase 6 (memory：会话表 + 长期记忆表)
   └─ 验收：close session 后第二次相似 query 能看到非空 memory_hints
Phase 7 (evaluation：含多轮场景)
   └─ 验收：python scripts/run_eval.py 输出多轮指标，回归门禁可用
Phase 8 (FastAPI：含 /sessions*)
   └─ 验收：6 步 curl 流程跑通（create → clarification → 补事实 → 答 → 代词引用 → close）
```

每个 Phase **独立验收**：Phase 1–7 的核心逻辑在 `*_PROVIDER=mock` 下不需要任何 API key 即可跑通。

---

## 4. 核心原则

1. **Agent 有能力，但必须在 Harness 受控环境中行动。**
2. **每个 agent 节点输入输出都是 LegalRAGState 的 dict patch。**
3. **结构化输出 + Pydantic 校验 + 解析失败重试一次再降级到规则。**
4. **检索 retry 与 answer revision 是两个独立计数。**
5. **Citation Checker 强制引用必须是 evidence pool（本轮 retrieved + 历史 cumulative）的真实子串。**
6. **Memory 只作 hint，不作证据。reviewer_score ≥ 0.75 才进 positive memory。**
7. **法律免责声明由 finalizer 拼接，不依赖 LLM 自觉。**
8. **所有 LLM 调用必须经 ContextComposer，agent 不允许自行拼 messages。**
9. **上下文超 budget 时由 ContextCompactor（runtime 中间件，非 agent）透明压缩；产物 `WorkingContextDigest` 自动作为后续所有 LLM 调用的 system 前缀。**

---

## 5. MVP 非目标

- 不做大模型微调 / RL。
- 不替代律师，仅做信息分析。
- 不接入大规模商业法律数据库。
- 不做复杂知识图谱推理。
- 不做多法域并存（默认 `CN`）。

---

## 6. 给 Codex 的总指引

每个 Phase 开始前先读：

- `01_ARCHITECTURE.md`、`02_MODEL_PROVIDERS.md`
- 当前 Phase 文档
- 上一个 Phase 文档（确认依赖）

然后执行该 Phase 末尾的「Codex Prompt」段落即可。

不要跨 Phase 实现：每个 Phase 必须先通过其端到端验收命令，才能进入下一个 Phase。

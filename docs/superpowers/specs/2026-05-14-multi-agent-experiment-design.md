# Multi-Agent Legal RAG 实验项目 —— 设计规范

- **状态**：草案 v1
- **日期**：2026-05-14
- **作者**：xxm（设计协作：Claude Opus 4.7）
- **位置**：`/home/xxm/rag/experiments/multi_agent/`

---

## 0. 概述

### 0.1 项目背景

`/home/xxm/rag/legal_rag/` 是基于 LangGraph 的 hybrid (规则+LLM) agentic RAG 实现。本项目（`experiments/multi_agent/`）是一个**完全独立的、纯 Python 的、全 LLM agent 化的**实验性平行项目，目标是从零探索 multi-agent 编排在法律咨询场景下的能力边界、失败模式与延迟/成本特征。

### 0.2 目标

1. 用 4 个角色（Receptionist / Lawyer / Secretary / Supervisor）的 multi-agent 架构跑通法律咨询任务
2. 建立 **trace-first** 的可观察性基础——所有 agent 交互、LLM 调用、tool 调用全程结构化记录、可回放、可对比
3. 支持 **本地 Qwen 9B vs 云端 Claude** 的全维度对比实验
4. 通过 Ablation 框架观察每个 agent / 工具 / memory 机制的真实贡献

### 0.3 Non-Goals

- 不上线、不追求生产级延迟/成本（实验项目）
- 不复用 `legal_rag/` 任何代码（仅共享 `data/` 与 `Chinese-Laws/` 语料文件）
- 不补充法律 corpus（接受 177 部现有法律的边界，主要做 4 类民事咨询）
- 不做 ReAct 之外的 agent 自主性范式（如 LLM-as-OS）
- 不做 vector memory（用 Markdown + tag 索引代替）

### 0.4 顶层架构原则

1. **trace 是横切关注点（cross-cutting concern）**，所有层通过统一 `Recorder.emit(event)` 写事件
2. **严格分层**：tracing → schemas → providers/memory/tools → agents → eval
3. **没有全局 state**，所有状态通过 Pydantic `RunState` 对象传递
4. **Agent-as-Tool 统一抽象**：子 agent（如 Secretary）被包装成 Tool 给父 agent（Lawyer）调用
5. **失败永不静默**：所有错误 emit 事件，不跨 provider fallback

### 0.5 时间表预期（粗估）

| 阶段 | 周次 | 交付 |
|---|---|---|
| Phase 1 | W1 | trace 系统 + schema + stub agent + 单元测试 |
| Phase 2 | W2-3 | Qdrant 索引 + BM25/dense/hybrid retriever + 1 个真 Lawyer agent |
| Phase 3 | W4 | Receptionist + memory store + multi-issue 支持 |
| Phase 4 | W5 | Secretary 拆出（agent-as-tool） |
| Phase 5 | W6 | Supervisor + eval framework + 第一轮 ablation |

实际推进根据观察到的失败模式调整，不强求时间表。

---

## 1. 整体架构

### 1.1 层级依赖（自底向上）

```
┌─────────────────────────────────────────────────────────┐
│  eval/                   QuerySet · Runner · Comparator  │
│                          Judges · Ablation               │
├─────────────────────────────────────────────────────────┤
│  agents/                 Receptionist · Lawyer           │
│                          Secretary · Supervisor          │
│                          (继承 BaseAgent)                │
├─────────────────────────────────────────────────────────┤
│  tools/                  retrievers · query_rewrite      │
│                          citation · corpus               │
│  memory/                 MarkdownMemoryStore             │
│  providers/              AnthropicProvider               │
│                          OpenAICompatibleProvider        │
├─────────────────────────────────────────────────────────┤
│  schemas/                所有 Pydantic 类型               │
│  tracing/                events · recorder · profiler    │
│                          viewer · replay                 │
└─────────────────────────────────────────────────────────┘
              ↑ tracing 横切,任何层都可 emit 事件
```

### 1.2 关键架构规则

- **`tracing/` 不依赖任何业务层**——只知道事件 schema
- **`agents/base.py` 自动注入 trace hook**——子类只写 prompt + output_schema
- **每次 `run(query)` 产生一个 `runs/<run_id>/` 目录**，包含完整 trace
- **`RunState` 通过函数参数传递**，避免 LangGraph 风格的 80+ 字段 TypedDict

### 1.3 目录结构

```
experiments/multi_agent/
├── pyproject.toml
├── README.md
├── docker-compose.yml              # Qdrant 服务
├── tracing/
│   ├── events.py                   # Pydantic Event types
│   ├── recorder.py                 # JSONL + SQLite 双写
│   ├── profiler.py                 # 延迟分析
│   ├── viewer.py                   # Streamlit timeline
│   └── replay.py                   # 单步重跑
├── schemas/
│   ├── messages.py                 # AgentMessage / ToolCall / ToolResult
│   ├── state.py                    # RunState
│   ├── evidence.py
│   ├── verdicts.py
│   └── memory.py                   # StickyContext / Turn / AgentNote
├── providers/
│   ├── base.py
│   ├── anthropic.py
│   └── openai_compatible.py
├── tools/
│   ├── base.py                     # Tool ABC
│   ├── retrievers/
│   │   ├── qdrant_client.py
│   │   ├── dense_encoder.py
│   │   ├── sparse_encoder.py
│   │   ├── index_builder.py
│   │   ├── dense_search.py
│   │   ├── sparse_search.py
│   │   ├── hybrid_search.py
│   │   └── exact_read.py
│   ├── query_rewrite.py
│   ├── citation_check.py
│   └── corpus.py
├── memory/
│   ├── store.py                    # MarkdownMemoryStore
│   └── schema.py
├── agents/
│   ├── base.py                     # BaseAgent + ReAct loop
│   ├── receptionist.py
│   ├── lawyer.py
│   ├── secretary.py
│   └── supervisor.py
├── prompts/
│   ├── receptionist/
│   │   ├── system.md
│   │   └── few_shot.yaml
│   ├── lawyer/specialty_*.md       # 6 个专业律师 prompt
│   ├── secretary/system.md
│   └── supervisor/system.md
├── eval/
│   ├── query_sets/
│   │   ├── synthetic_v1.yaml
│   │   ├── golden_qa_v1.yaml       # laws_data 抽取后的
│   │   └── multi_issue_v1.yaml
│   ├── runner.py
│   ├── judges.py
│   ├── comparator.py
│   ├── ablation.py
│   └── corpus_audit.py
├── memory_store/                   # MD-based memory(runtime data)
├── runs/                           # 每个 query 一个 trace 目录
├── run_groups/                     # ExperimentRunner 产物
├── qdrant_storage/                 # Qdrant 持久化(gitignore)
└── tests/
    ├── conftest.py
    ├── unit/
    ├── integration/
    └── smoke/
```

---

## 2. Trace 数据模型

### 2.1 事件类型（Pydantic discriminated union）

```python
class BaseEvent(BaseModel):
    event_id: str           # ULID
    run_id: str
    timestamp: datetime
    parent_id: str | None   # 形成调用树
    event_type: str         # discriminator

# 顶层
class RunStarted(BaseEvent):    query: str; config: dict
class RunFinished(BaseEvent):
    status: Literal["ok","error","timeout"]
    final_answer: str | None
    error: str | None

# Agent
class AgentInvoked(BaseEvent):  agent_name: str; role: str; input: dict
class AgentResponded(BaseEvent):output: dict; duration_ms: int

# LLM
class LLMRequested(BaseEvent):
    provider: str; model: str
    messages: list; params: dict
class LLMResponded(BaseEvent):
    raw_response: str; usage: Usage; duration_ms: int
    finish_reason: Literal["end_turn","tool_use","max_tokens","refusal"]

# Tool
class ToolCalled(BaseEvent):    tool_name: str; args: dict; agent_name: str
class ToolReturned(BaseEvent):
    result: dict | None; error: str | None; duration_ms: int

# Memory
class MemoryRead(BaseEvent):
    target: Literal["sticky","turn","agent_notes"]
    query: dict; hits: list
    agent_name: str
class MemoryWritten(BaseEvent):
    target: str; payload: dict; path: str
    agent_name: str

# Supervisor
class SupervisorVerdict(BaseEvent):
    verdict: Literal["pass","revise","reject"]
    issues: list[str]
```

### 2.2 调用树通过 `parent_id` 串起

```
RunStarted (parent=None)
└── AgentInvoked (lawyer)
    ├── LLMRequested
    ├── LLMResponded
    ├── ToolCalled (search_hybrid)
    │   └── ToolReturned
    └── AgentResponded
RunFinished
```

### 2.3 存储：双写 JSONL + SQLite

```
runs/2026-05-15_租房涨租_a1b2c3/
├── meta.json                # query / config / started / finished / status
├── events.jsonl             # 时间顺序、append-only、grep 友好
├── events.db                # SQLite 索引版,跨 run 查询
├── final.md                 # 人类可读概览
└── artifacts/               # 大件单独存
    ├── composed_prompts/
    │   └── lawyer_round_1.txt
    └── retrieval_results/
        └── round_1.json
```

**为什么双写**：JSONL 写快、git diff 友好；SQLite 用于跨 run 查询。

**大对象走 artifacts/**：events.jsonl 里只存路径引用，避免单行膨胀至 MB 级。

### 2.4 Recorder API

```python
class Recorder:
    def __init__(self, run_id: str, run_dir: Path): ...
    def emit(self, event: BaseEvent) -> None: ...
    @contextmanager
    def span(self, kind: str, **attrs) -> Iterator[Span]:
        """自动 emit start + end 事件,记录耗时,生成 parent_id."""
```

### 2.5 Replay 语义（V1 最简）

- `replay.py <run_dir>` 读 events.jsonl
- 给定一个 `event_id`，可"重跑这一次 LLM 调用"，可换模型/换 prompt
- **不做** 整个 run 的确定性 replay（temperature>0 不可重现）

### 2.6 关键 invariants

- 任何 run 退出，`events.jsonl` 末尾必有 `RunFinished`
- 任何 `LLMRequested` 必有对应的 `LLMResponded` 或 error
- 任何 `ToolCalled` 必有对应的 `ToolReturned`
- timestamp 精度到毫秒

---

## 3. Agent 抽象层

### 3.1 BaseAgent

```python
class BaseAgent(BaseModel, ABC):
    name: str
    role: str
    provider: LLMProvider
    recorder: Recorder
    max_steps: int = 10
    max_total_tokens: int = 20_000
    max_tool_calls: int = 8
    timeout_seconds: int = 60
    tools: list[Tool] = []

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def output_schema(self) -> type[BaseModel]: ...

    def run(self, input: AgentInput) -> AgentOutput:
        """模板方法,子类不重写."""
        with self.recorder.span("agent_invoke", agent_name=self.name, role=self.role):
            return self._react_loop(input)
```

### 3.2 ReAct 循环（基类实现）

```
messages = [system_prompt, user_input]
for step in range(max_steps):
    with recorder.span("llm_call"):
        response = provider.complete(messages, tools=self.tools)

    if response.tool_calls:
        for tc in response.tool_calls:
            with recorder.span("tool_call", tool=tc.name):
                result = self._dispatch_tool(tc)
            messages.append(tool_result_message(tc, result))
        continue

    if response.has_final_answer:
        validated = self.output_schema().model_validate(response.parsed)
        return AgentOutput(payload=validated, steps_used=step+1)

raise BudgetExceeded(f"{self.name} hit max_steps")
```

### 3.3 Tool 抽象

```python
class Tool(BaseModel):
    name: str
    description: str           # 给 LLM 看
    args_schema: type[BaseModel]

    @abstractmethod
    def call(self, args: BaseModel, recorder: Recorder) -> ToolResult: ...
```

### 3.4 Agent-as-Tool 模式

子 agent（Secretary）被包装成 Tool 给父 agent（Lawyer）调用：

```python
class SecretaryAsTool(Tool):
    name = "ask_secretary"
    description = "委托秘书完成检索/取证/起草任务"
    args_schema = SecretaryRequest

    def __init__(self, secretary_agent: SecretaryAgent):
        self._agent = secretary_agent

    def call(self, args, recorder):
        result = self._agent.run(AgentInput.from_request(args))
        return ToolResult(payload=result.payload.model_dump())
```

事件树自动嵌套（Lawyer 的 ToolCalled → Secretary 的 AgentInvoked → 等）。

### 3.5 各 Agent 角色

| Agent | 主要工具 | 输出 schema |
|---|---|---|
| **Receptionist** | （内置）classify_intent / detect_safety / decompose_case | `ReceptionistOutput`：specialty, sub_cases, urgency |
| **Lawyer** | ask_secretary / ask_user_clarify / finalize | `LawyerOutput`：draft, citations, reasoning |
| **Secretary** | search_hybrid / search_bm25 / search_dense / read_article / verify_citation / rewrite_query | `SecretaryResponse`：evidences, notes, confidence |
| **Supervisor** | check_groundedness / check_compliance / verify_citation | `SupervisorVerdict`：verdict, issues, suggested_fix |

### 3.6 Multi-Issue 支持（基于 laws_data 数据洞察）

实际用户咨询常含多议题，Receptionist 输出加 `sub_cases`：

```python
class SubCase(BaseModel):
    issue: str
    specialty: str
    priority: int
    requires_separate_retrieval: bool

class ReceptionistOutput(BaseModel):
    primary_specialty: str
    sub_cases: list[SubCase]
    is_multi_issue: bool
    case_type: str
    urgency: Literal["低","中","高"]
    risk_flag: str | None
    need_clarification: bool
    clarification_q: str | None
    initial_facts: list[str]
    normalized_query: str
```

**V1**：Lawyer **顺序处理** sub_cases，挨个调 Secretary，最后串成综合答复
**V2**：并发多 specialty Lawyer 并融合（不在 V1 范围）

### 3.7 预算控制

| 控制 | 默认值 |
|---|---|
| `max_steps` | Lawyer 10 / Receptionist 3 / Secretary 5 / Supervisor 5 |
| `max_total_tokens` | 20k |
| `max_tool_calls` | 8 |
| `timeout` | 60s per agent |

预算用尽前**先 emit 事件，再抛 `BudgetExceeded`**——trace 永远完整。

---

## 4. Tool / Retriever 层

### 4.1 Tool 清单（V1）

| 工具 | 输入 | 输出 |
|---|---|---|
| `search_bm25` | query, k, filters | list[Evidence] |
| `search_dense` | query, k, filters | list[Evidence] |
| `search_hybrid` | query, k, filters | list[Evidence] |
| `read_article` | law_name, article_no | Evidence |
| `verify_citation` | citation, evidence | bool + reason |
| `rewrite_query` | raw_query, history, sticky | RewrittenQuery |
| `read_memory` | session_id, target | list[Memory] |
| `write_memory` | session_id, payload | bool |

### 4.2 向量数据库：Qdrant（路线 B —— 一站式 sparse+dense+RRF）

- 本地 Docker：`qdrant/qdrant:v1.12.0`
- Collection：`statutes`（含 sparse + dense 双 vector）
- Hybrid 搜索原生支持（`query_points + prefetch + Fusion.RRF`）

```python
client.query_points(
    collection_name="statutes",
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=20),
        Prefetch(query=sparse_vec, using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10,
    query_filter=filter,
)
```

### 4.3 Embedding 与编码

- **Dense**：`BAAI/bge-m3`（sentence-transformers）
- **Sparse**：`jieba` 分词 + IDF 权重
- **首次索引构建**：CPU ~30-60min，GPU 5-10min（可接受）

### 4.4 Chunking 策略

**核心决定**：`chunk = 1 article`，不拆不并。

**理由**：法律语义最小单元就是"条"；用户问"依据哪条"必须以条为粒度命中。

#### Chunk schema

```python
chunk = {
    "doc_id": "民法典-510",
    "law_name": "中华人民共和国民法典",
    "law_short": "民法典",
    "article_no": "510",
    "text": "当事人就合同补充内容没有约定...",     # 纯条文
    "book": "合同编",                            # 章节信息(方案 B 先跳过)
    "chapter": "合同的订立",
    "cross_refs": ["第511条"],
    "preceding_text": "...",
    "following_text": "...",
    "concepts": ["合同补充", "履行规则", "合同漏洞填补"],  # LLM 预生成
}
```

#### Embedding 拼接（dense）

```
《民法典》合同编·合同的订立·第510条: 当事人就合同补充内容没有约定...
```

把 `law_name + book + chapter + article_no` 都拼进去——dense search 时关键词信号更强。

#### Sparse 编码

只用条文正文（不拼前缀，避免拉高常见词权重）。

#### Qdrant payload

```python
payload = {
    "doc_id", "law_name", "law_short", "article_no",
    "book", "chapter", "text", "concepts",
    "cross_refs", "preceding_text", "following_text",
}
```

`law_short`、`book`、`article_no` 等是 filterable attributes。

#### 章节信息：先方案 B（跳过）

V0 不补 book/chapter，召回会损失 5-10%。V1 看失败 case 决定是否手维护 `law_structure.yaml`（约 1 天工作量）。

#### Concepts 字段（实验性）

用本地 Qwen 9B 一次性给每条 chunk 生成 3-5 个"关键概念标签"：
- 一晚跑完（23k chunk × ~2s）
- 用途：sparse search 加权
- V0 跑

### 4.5 Query Rewriting

#### 两层 rewrite

**1）Receptionist 做"轻 rewrite"**——代词消解 + prior_facts 拼接

```python
class ReceptionistOutput:
    normalized_query: str        # 已消解代词
    initial_facts: list[str]     # 抽出的事实陈述
    ...
```

**2）Secretary 提供 `rewrite_query` tool**——术语映射 + 概念扩展（Lawyer 显式调用）

```python
class RewrittenQuery(BaseModel):
    canonical: str               # 主查询(法律语言)
    expanded_terms: list[str]    # 多路扩展词
    hypothetical: str | None     # HyDE(V2)
    concepts: list[str]          # 关键概念
    intent: Literal["statute_lookup","case_advice","procedure","amount"]
```

#### 各技术 V0/V1 范围

| 技术 | V0 | V1 | 说明 |
|---|---|---|---|
| 代词消解 | ✅ Receptionist | | 必做 |
| Prior facts 拼接 | ✅ Receptionist | | 必做 |
| 术语映射（口语→法律） | ❌ | ✅ Secretary | LLM 一次调用 |
| 多查询扩展 | ❌ | ✅ Secretary | 3 个角度查询 |
| HyDE | ❌ | ⚠️ V2 | 慢且烧 token |
| Concept-weighted sparse | ❌ | ✅ | 利用预生成 concepts |
| Stepback prompting | ❌ | ❌ | 不适合法律 |

### 4.6 Evidence schema

```python
class Evidence(BaseModel):
    doc_id: str
    law_name: str
    article_no: str
    text: str
    score: float
    retriever: Literal["bm25","dense","hybrid","exact","memory"]
    metadata: dict
```

---

## 5. Memory 层（MD-based）

### 5.1 设计原则

- 全部用 Markdown + YAML frontmatter，**不用 SQLite**
- 可读、可 git diff、LLM 可直接读写
- 文件即真相之源

### 5.2 三种文件分工

| 文件 | 性质 | 写入时机 |
|---|---|---|
| `sticky.md` | **可变**：当前 session 状态 | 每轮覆盖 |
| `turns/NNN.md` | **不可变**：历史 turn 记录 | 每轮追加 |
| `agent_notes/*.md` | **稀疏写**：跨 session 学习 | Supervisor reject 时由 LLM 产出 |

### 5.3 目录结构

```
memory_store/
├── MEMORY.md                          # auto-generated 人类速读索引
├── _index.json                        # auto-generated 机器查询索引
├── sessions/
│   └── s_abc123_2026-05-14/
│       ├── sticky.md
│       └── turns/
│           ├── 001-涨租问题.md
│           └── 002-依据哪条.md
└── agent_notes/                       # (原 lessons,改名)
    ├── lawyer-misses-rental-mgmt-rules.md
    └── supervisor-too-strict-on-hedging.md
```

### 5.4 sticky.md 格式

```markdown
---
session_id: s_abc123_2026-05-14
created_at: 2026-05-14T14:00:00
updated_at: 2026-05-14T15:30:00
legal_domain: 民事
case_type: 租赁纠纷
last_law_name: 民法典
mentioned_laws: [民法典, 商品房屋租赁管理办法]
cited_articles:
  - {law: 民法典, article: "510", from_turn: 1}
user_facts:
  - 用户去年签了一年的租房合同
  - 已住三个月
  - 房东要求涨租 30%
linked_runs: [r_a1b2c3, r_d4e5f6]      # ← trace 双向链接
---

# Session s_abc123
最近主题:租赁纠纷 / 房东单方涨租
```

**注意**：sticky 不存详细引用文本（在 turns 里），只存 `{law, article, from_turn}` 短指针——避免 single source of truth 违例。

### 5.5 turns/NNN-slug.md 格式

```markdown
---
turn: 1
run_id: r_a1b2c3                       # ← trace 双向链接
started_at: 2026-05-14T14:05:23
finished_at: 2026-05-14T14:05:48
answer_mode: evidence_grounded
supervisor_verdict: pass
agents_invoked: [receptionist, lawyer, secretary, supervisor]
duration_ms: 25180
total_tokens: 8420
---

## Q
房东要涨我 30% 房租...

## A
房东在合同期内单方调整租金通常需要双方协商一致...

## 引用
- 《民法典》第 510 条:...

## 关键决策点
- Receptionist:民事/租赁纠纷/低风险
- Lawyer 调 Secretary 2 次
- Supervisor: pass
```

### 5.6 agent_notes/*.md 格式

```markdown
---
name: lawyer-misses-rental-mgmt-rules
description: 涨租问题 Lawyer 漏引《商品房屋租赁管理办法》
produced_by: supervisor                # 哪个 agent 写
about_agent: lawyer                    # 关于哪个 agent
verdict_that_triggered: reject
tags: [涨租, 民法典-510, 租赁]
triggered_by_run: r_a1b2c3             # ← trace 链接
used_in_runs: [r_d4e5f6, r_g7h8i9]    # ← trace 链接
created_at: 2026-05-14
usage_count: 3
---

## 教训
当用户问"房东能否单方涨租"时,Lawyer 默认引《民法典》510 条,
但漏掉了更具体的《商品房屋租赁管理办法》第 7 条。

## 应该这样
1. 识别"租赁 + 单方变更"模式
2. 先查租赁管理办法,再回到民法典通则
```

### 5.7 索引（自动生成）

#### _index.json

```json
{
  "version": 1,
  "regenerated_at": "2026-05-14T15:30:00",
  "sessions": {
    "s_abc123": {
      "path": "sessions/s_abc123_2026-05-14/sticky.md",
      "turn_count": 3,
      "tags": ["租赁纠纷"],
      "linked_runs": ["r_a1b2c3", "r_d4e5f6"]
    }
  },
  "notes_by_tag": {"涨租": ["lawyer-misses-rental-mgmt-rules"]},
  "notes_by_about_agent": {"lawyer": ["lawyer-misses-rental-mgmt-rules"]},
  "notes_by_name": {
    "lawyer-misses-rental-mgmt-rules": {
      "path": "agent_notes/lawyer-misses-rental-mgmt-rules.md",
      "produced_by": "supervisor",
      "usage_count": 3
    }
  }
}
```

#### MEMORY.md

```markdown
<!-- AUTO-GENERATED. DO NOT EDIT. -->
<!-- Last regenerated: 2026-05-14T15:30:00 -->

## Sessions (12)
- [s_abc123 (租赁纠纷, 3 turns)](sessions/s_abc123_2026-05-14/sticky.md)
...

## Agent Notes by Tag
- **涨租**: [lawyer-misses-rental-mgmt-rules](agent_notes/...)
...

## Agent Notes by Subject
- **about lawyer (2)**: ...
```

### 5.8 索引维护：Eager 重建

每次 write 后**立刻**重建 `_index.json` 和 `MEMORY.md`（scan 全目录约 100ms，可接受）。**永远不要手动改两个索引文件**。

### 5.9 Store API

```python
class MarkdownMemoryStore:
    def __init__(self, root: Path): ...

    # Sessions
    def read_sticky(self, session_id: str) -> StickyContext: ...
    def write_sticky(self, session_id: str, ctx: StickyContext) -> None: ...

    # Turns
    def append_turn(self, session_id: str, turn: Turn) -> Path: ...
    def recent_turns(self, session_id: str, n: int = 5) -> list[Turn]: ...

    # Agent Notes
    def write_note(self, note: AgentNote) -> Path: ...
    def find_notes(self, tags: list[str] | None = None,
                   produced_by: str | None = None,
                   about_agent: str | None = None,
                   limit: int = 3) -> list[AgentNote]: ...
    def bump_usage(self, note_name: str, run_id: str) -> None: ...

    # Index
    def regenerate_index(self) -> None: ...
```

### 5.10 Atomic write

写 `<file>.tmp` 然后 `os.replace()`（POSIX 原子）。单进程实验场景无并发问题。

### 5.11 Trace ↔ Memory 双向链接

- turn.md frontmatter 有 `run_id` → 反向找 trace
- agent_note frontmatter 有 `triggered_by_run` 和 `used_in_runs[]`
- sticky.md frontmatter 有 `linked_runs[]`
- trace `MemoryRead` 事件返回 hit paths

任何一个 run 都能反向找出用过的 notes；任何 note 都能反向找出影响过的 run。**这是 ablation 的必备能力**。

### 5.12 跨 session 查询

- 结构化：读 `_index.json`
- 探索性：`grep -rl "涨租" memory_store/sessions/*/turns/`

不上 vector search、不上 SQLite。

---

## 6. LLM Provider 层

### 6.1 双 Provider 共存

```
LLMProvider (ABC)
├── AnthropicProvider          # Claude (cloud)
└── OpenAICompatibleProvider   # 本地 Qwen + OpenAI + 任何 OAI-兼容
```

agent 代码完全 provider-agnostic。

### 6.2 统一接口

```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: type[BaseModel] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        cache_breakpoints: list[int] | None = None,
        recorder: Recorder,
        agent_name: str,
    ) -> LLMResponse: ...

class LLMResponse(BaseModel):
    text: str
    parsed: BaseModel | None
    tool_calls: list[ToolCallRequest]
    usage: Usage
    raw: dict
    duration_ms: int
    finish_reason: Literal["end_turn","tool_use","max_tokens","refusal"]
```

### 6.3 关键差异点处理

| 维度 | Anthropic | OpenAI 兼容/vLLM |
|---|---|---|
| **Tool use 格式** | `tool_use` content block | `tool_calls` field |
| **Prompt caching** | 显式 `cache_control: {type: "ephemeral"}` | vLLM 自动 KV prefix cache（要求字节级一致前缀） |
| **结构化输出** | schema-in-prompt + Pydantic validate | 同上 |
| **上下文** | 200k | Qwen 9B = 32k（必须做 ContextComposer） |
| **Token usage 字段** | cache_creation_input + cache_read_input | vLLM 0.6+ 暴露 prefix_cache_hit_tokens |

### 6.4 配置：四个 profile

```python
PROVIDER_PROFILES = {
    "all-claude": {
        "receptionist": ("anthropic", "claude-haiku-4-5-20251001"),
        "lawyer":       ("anthropic", "claude-sonnet-4-6"),
        "secretary":    ("anthropic", "claude-sonnet-4-6"),
        "supervisor":   ("anthropic", "claude-haiku-4-5-20251001"),
    },
    "all-local": {
        "receptionist": ("openai_compat", "qwen3.5-9b"),
        "lawyer":       ("openai_compat", "qwen3.5-9b"),
        "secretary":    ("openai_compat", "qwen3.5-9b"),
        "supervisor":   ("openai_compat", "qwen3.5-9b"),
    },
    "mixed-cloud-judge": {  # 本地干活,云端审判
        "receptionist": ("openai_compat", "qwen3.5-9b"),
        "lawyer":       ("openai_compat", "qwen3.5-9b"),
        "secretary":    ("openai_compat", "qwen3.5-9b"),
        "supervisor":   ("anthropic", "claude-haiku-4-5-20251001"),
    },
    "mixed-cloud-brain": {  # 云端推理,本地路由
        "receptionist": ("openai_compat", "qwen3.5-9b"),
        "lawyer":       ("anthropic", "claude-sonnet-4-6"),
        "secretary":    ("openai_compat", "qwen3.5-9b"),
        "supervisor":   ("anthropic", "claude-haiku-4-5-20251001"),
    },
}
```

### 6.5 本地 Qwen 部署

服务地址：`http://localhost:8000/v1`（OpenAI 兼容）
启动命令：`cd /home/xxm/models/qwen3.5-9b && bash serve_vllm.sh`

详见 `/home/xxm/models/qwen3.5-9b/USAGE.md`。

### 6.6 Fail loud，不 fallback

vLLM 不可达 / Anthropic 429 → **直接 throw，不偷偷切 provider**。否则实验结果污染。

### 6.7 Pre-flight 检查

启动 ExperimentRunner 前检查活跃 profile 涉及的所有 provider 可达。

### 6.8 失败处理（与 §8 一致）

| 失败 | 行为 |
|---|---|
| 429 / 5xx | 指数退避重试 `max_retries=3` |
| JSON 不合 schema | 带错误塞回 prompt 重试 1 次 |
| Tool args 不合 schema | 同上,最多 2 次 |
| Refusal | 不重试,emit 事件,agent 转 fallback |
| Timeout 60s | 抛 `LLMTimeoutError` |

---

## 7. Eval 框架

### 7.1 三个核心抽象

```
QuerySet → ExperimentRunner → RunGroup → Comparator
```

### 7.2 QuerySet

YAML 定义：

```yaml
meta:
  name: golden_qa_v1
  description: laws_data 加工后的真实用户 Q&A,4 类(去除劳动)
  
queries:
  - id: q001
    text: 房东要涨我 30% 房租,合法吗?
    jurisdiction: CN
    cause: 房产纠纷
    source: laws_data
    source_id: train_001234
    tags: [民事, 租赁, 涨租]
    expected:
      should_cite_any: [民法典-510, 民法典-563]
      expected_answer_mode: evidence_grounded
      confidence: high
    audit:
      corpus_coverage: complete
      last_verified: 2026-05-14
```

### 7.3 QuerySet 来源（基于数据集体检调整后）

| Set | 来源 | 量级 | 用途 |
|---|---|---|---|
| `synthetic_v1.yaml` | 反向生成（Chinese-Laws + Claude Opus） | 300-500 | 基线，100% corpus 覆盖 |
| `golden_qa_v1.yaml` | laws_data LLM 抽取（过滤劳动） | 5-7k | 真实用户语言 |
| `multi_issue_v1.yaml` | 案件拆分数据集挑选 | 30-50 | multi-issue 压测 |
| `safety_v1.yaml` | 手写 | 10-20 | safety 拒答 |

**注意**：`laws_data` 加工后**直接过滤 cause=劳动纠纷**（corpus 无劳动合同法）。

### 7.4 ExperimentRunner

```python
class ExperimentRunner:
    def __init__(self, profile: str, query_set: QuerySet,
                 run_group_name: str, parallelism: int = 1):
        self.profile = profile
        self.query_set = query_set
        self.group_dir = Path(f"run_groups/{run_group_name}")

    def run(self) -> RunGroup: ...
```

并行度：Anthropic=1（避 rate limit），本地 Qwen=4-8（vLLM 12 路并发）。

### 7.5 RunGroup 产物

```
run_groups/
└── 2026-05-15_baseline_all-claude/
    ├── group_meta.yaml         # profile / query_set / git_sha / 时间
    ├── results.jsonl           # 每 query 一行:run_id / status / metrics
    ├── runs/                   # symlink 到 runs/<run_id>/
    └── summary.md
```

### 7.6 自动指标（从 trace 派生）

| 指标 | 来源 |
|---|---|
| total_latency_ms | RunStarted→RunFinished |
| total_input_tokens | sum LLMResponded.usage.input |
| total_output_tokens | sum LLMResponded.usage.output |
| cache_read_tokens | sum LLMResponded.usage.cache_read |
| cache_hit_rate | cache_read / total_input |
| agent_invocations | count AgentInvoked |
| tool_calls_total | count ToolCalled |
| react_steps_total | sum agent.steps_used |
| supervisor_verdict | 最后一次 SupervisorVerdict |
| final_answer_mode | RunFinished.answer_mode |
| citation_count | len(final.citations) |
| errors | count events with error |

### 7.7 LLM Judges（质量评估）

```python
class CitationAccuracyJudge:    # 纯规则
class GroundednessJudge:        # 用 Claude Opus
class HelpfulnessJudge:         # 用 Claude Opus
```

**关键决定**：**Judge 永远用 Claude Opus**，不管被审 run 是 Claude 还是 Qwen。原因：
- 避免自我评分偏差
- 给本地实验公平天花板对照

成本：每 query ~$0.01-0.03，30 个 query ~$0.3-1.0。

### 7.8 Comparator

```python
class Comparator:
    def compare(self, group_a: RunGroup, group_b: RunGroup) -> ComparisonReport: ...
```

输出 `comparison_reports/<a>_vs_<b>_<date>.md`：含总览表、逐 query diff、失败模式聚类、wins/regressions。

### 7.9 AblationRunner

```python
@dataclass
class Ablation:
    name: str

class DisableAgent(Ablation):     agent: str
class SwapModel(Ablation):         agent: str; new_provider: str; new_model: str
class DisableMemory(Ablation):     ...
class DisableTool(Ablation):       tool: str

class AblationRunner:
    def run(self, base_profile: str, ablations: list[Ablation],
            query_set: QuerySet) -> AblationReport: ...
```

### 7.10 Latency Profiler

```python
class SpanTiming(BaseModel):
    span_id: str
    kind: str
    label: str
    inclusive_ms: int
    exclusive_ms: int
    children: list[SpanTiming]
    metadata: dict

class LatencyProfile(BaseModel):
    run_id: str
    total_ms: int
    spans: SpanTiming
    by_agent: dict[str, AgentTiming]
    by_tool: dict[str, ToolTiming]
    by_provider: dict[str, ProviderTiming]
    by_kind: dict[str, int]
```

Profiler 是**纯派生层**，不改 trace schema。输出：
1. Per-run flame graph (CLI 缩进树)
2. Aggregate report (跨 query bottleneck 分析)
3. Streamlit viewer Latency tab

### 7.11 Trace Viewer

Streamlit 50-100 行，三栏：
- 左：时间线（缩进表示 parent_id 层级）
- 中：点击事件 → 完整 input/output
- 右：相关 memory 链接（双向引用）

### 7.12 数据集体检结论

详见附录。简要：
- Chinese-Laws (177 部，7.4MB)：民事/行政/治安/交通覆盖完整；劳动/刑事/商事缺核心法律；司法解释行政法规层完全空白
- laws_data (23k Q&A，16MB)：5 类 cause；4 类可用、劳动类不可用；需 LLM 抽取法条引用
- 整体充分性：在去除劳动类后，足够支持整个 multi-agent 实验

---

## 8. 错误处理

### 8.1 失败模式汇总

| 失败类型 | 检测点 | 行为 |
|---|---|---|
| Provider 不可达 | Provider 层 | 退避重试 3 次,抛 `ProviderUnavailable` |
| LLM JSON 不合 schema | Provider 层 | 带错误塞回 prompt 重试 1 次,仍失败抛 `ResponseValidationError` |
| Tool args 不合 schema | Tool dispatch | 返回 error 给 LLM 让它修正,最多 2 次 |
| Agent 超 `max_steps` | 基类循环 | 抛 `BudgetExceeded`,强制 finalize |
| Agent 超 `max_total_tokens` | 基类累加 | 同上 |
| Agent 超时 60s | wall clock | 抛 `AgentTimeout` |
| Tool 内部异常 | Tool.call | 包装成 `ToolResult(error=...)` 返给 LLM |
| LLM refusal | Provider 解析 | emit 事件,agent 转 fallback |
| Supervisor reject | Supervisor 节点 | 转 fallback 模板,不重生成 |
| memory 读失败 | MarkdownMemoryStore | emit `MemoryReadError`,返回空 context |
| memory 写失败 | atomic write | 抛 `MemoryWriteError`,rollback .tmp |
| Qdrant 不可达 | retriever tool | 返回 error 给 LLM |
| 未捕获异常 | 顶层 try/except | emit `RunFinished(status="error")`,不重试 |

### 8.2 设计原则

1. **All errors are events**——任何失败 emit trace 事件，永不静默
2. **No silent fallback across providers**——vLLM 挂了不偷偷换 Claude
3. **Budgets always hold**——硬上限永远先 emit 再抛
4. **Tool errors are LLM-visible**——让 LLM 自己决定怎么办
5. **Memory failures don't crash run**——读失败给空 context，写失败才整体失败

### 8.3 关键 invariant

```python
def run_query(query: str, ...):
    run_id = fresh_id()
    recorder = Recorder(run_id, ...)
    try:
        recorder.emit(RunStarted(...))
        result = orchestrate(query, recorder, ...)
        recorder.emit(RunFinished(status="ok", ...))
        return result
    except Exception as e:
        recorder.emit(RunFinished(status="error", error=str(e)))
        raise
    finally:
        recorder.close()
```

任何路径退出都 emit `RunFinished`——`runs/` 目录里永远不会有"未完成"的 trace。

---

## 9. 测试策略

### 9.1 三层测试

```
unit/        快、纯函数、不依赖外部
integration/ 中、真实 Qdrant + 真实 memory,stub LLM
smoke/       慢、真实 LLM,1-2 个端到端 query
```

### 9.2 Unit Tests（30-50 个）

| 模块 | 测试什么 |
|---|---|
| schemas/events.py | 序列化/反序列化、discriminated union |
| tracing/recorder.py | JSONL+SQLite 一致、span 嵌套、atomic close |
| tracing/profiler.py | inclusive/exclusive 计算 |
| memory/store.py | frontmatter 解析、atomic write、索引重建 |
| tools/retrievers/qdrant_*.py | mock client、参数构造 |
| tools/retrievers/sparse_encoder.py | jieba 分词 + IDF |
| providers/anthropic.py & openai_compatible.py | mock HTTP、请求构造、tool_call 翻译 |
| agents/base.py | ReAct 循环、budget 触发 |
| eval/comparator.py | diff 计算 |

工具：`pytest` + `pytest-asyncio` + optional `hypothesis`

### 9.3 Integration Tests（10 个）

走真实 Qdrant + memory，stub LLM：

| 测试 | 验证 |
|---|---|
| 单 agent 端到端 | trace 文件完整 |
| Lawyer 调 Secretary | parent_id 链正确嵌套 |
| Memory 跨 turn 持久化 | sticky 在 2 轮间正确更新 |
| Lawyer 超 max_steps | BudgetExceeded 抛出 |
| Qdrant 不可达 | tool 返 error,LLM 看到 |
| 4 agent profile 跑通 | 4 个 AgentInvoked 事件 |
| Schema 失败重试 | 第二次成功 |
| Replay 单 LLM 调用 | 输入输出可重现 |
| Profile 切换 | provider 字段正确 |
| Comparator 对比 | diff 报告正确 |

### 9.4 Smoke Tests（3-5 个，手动）

```bash
ANTHROPIC_API_KEY=$1 pytest tests/smoke -v --slow
```

每次成本 $0.20-0.50（Claude）或免费（本地）。

| 测试 | 内容 |
|---|---|
| test_smoke_all_claude_simple | 简单 query 走 Claude profile |
| test_smoke_all_local_simple | 同上,走本地 Qwen |
| test_smoke_multi_issue | 验证 sub_cases 拆分 |
| test_smoke_safety_refusal | 验证 safety_gate |
| test_smoke_eval_runner | 跑 3 个 query 的 mini set |

### 9.5 不做

- 单元覆盖率门槛
- mutation / fuzz testing
- perf benchmark 单测（用 profiler）
- CI 集成

### 9.6 测试目录

```
tests/
├── conftest.py
├── unit/
├── integration/
├── smoke/
└── fixtures/
    ├── stub_responses/
    └── sample_corpus/
```

### 9.7 关键 invariants 测试

不论 V1 如何演进都必须守住：

1. 任何 run 退出，`events.jsonl` 末尾有 `RunFinished`
2. 任何 `LLMRequested` 必有对应 `LLMResponded` 或 error
3. 任何 `ToolCalled` 必有对应 `ToolReturned`
4. memory 写入 atomic
5. `_index.json` 与目录内容一致

---

## 10. 设计决策记录（ADR 摘要）

| ID | 决策 | 备选 | 理由 |
|---|---|---|---|
| ADR-01 | 完全重写，不 import legal_rag | 复用部分代码 | 实验目标是学全栈，平行项目可独立演进 |
| ADR-02 | 纯 Python + Pydantic，不用 LangGraph/AutoGen/CrewAI | 用现成框架 | 学习目标 + 最大控制 + trace 设计自由 |
| ADR-03 | Trace-First Walking Skeleton | Schema-first / 单文件 prototype | 锁定接口、每周可见成果 |
| ADR-04 | Trace 双写 JSONL + SQLite | 纯 JSONL / 纯 DB | 写快 + 查方便 |
| ADR-05 | Agent-as-Tool 统一抽象 | Message-passing 平级 | 失控风险低、契约清晰 |
| ADR-06 | Memory 用 MD + frontmatter | SQLite | 可读、git diff、LLM 可直读 |
| ADR-07 | Memory 不用 vector search | 加 vector memory | 实验规模 grep + tag 索引够用 |
| ADR-08 | Qdrant 路线 B（sparse+dense+RRF 全在 Qdrant） | bm25s 单独 + Qdrant 仅 dense | 单存储、原生 RRF |
| ADR-09 | chunk = 1 article | sub-article 切分 / 合并多条 | 法律语义最小单元 |
| ADR-10 | Concepts 字段用本地 Qwen 一次性生成 | 不生成 / 用 Claude | 免费 + 提升 sparse 召回 |
| ADR-11 | Query rewrite 两层 | 单层 / 总在 Receptionist | Lawyer 自主决定何时 rewrite 是观察点 |
| ADR-12 | Provider profile 切换 | 单一 provider / 自由配置 | 实验对照清晰 |
| ADR-13 | Fail loud，不跨 provider fallback | 自动降级 | 避免实验污染 |
| ADR-14 | LLM Judge 永远用 Claude Opus | 同 provider 自审 | 避免自评偏差 |
| ADR-15 | 不补充 corpus | 补 5 部法律 | 接受劳动类失效，专注 agent 设计 |
| ADR-16 | 案件拆分数据不入向量库 | 单独 collection | 不是法条，作 Receptionist few-shot |
| ADR-17 | Receptionist 输出加 sub_cases，V1 顺序处理 | 单 specialty 路由 | 真实查询 30-50% multi-issue |
| ADR-18 | Trace ↔ Memory 双向链接 | 单向 | Ablation 必备 |

---

## 11. 开放问题（先记下，不阻塞 V1）

1. Qwen 9B 的 tool use 可靠性如何？预计失败率 10-20%，需要在 trace 里量化
2. ContextComposer 何时介入？V1 简单截断够吗？
3. 6 个 specialty Lawyer 的 prompt 怎么写？Cold-start 没有 few-shot 数据
4. agent_notes 的生命周期管理：何时归档/删除？
5. multi-issue 顺序处理的 token 成本：累积上下文是否爆炸？
6. Qdrant 索引版本管理：corpus 更新后 incremental update？

---

## 12. 附录

### 12.1 数据集详情（Chinese-Laws）

177 部法律全文（7.4MB）。

**关键覆盖**：民法典（1259 条，完整）、消费者权益保护法、行政诉讼/复议/处罚/许可/强制、治安管理处罚法、道路交通安全法、社会保险法、反家庭暴力法、未成年人保护法、刑事/民事诉讼法、商业银行法、证券法、票据法、合伙企业法、著作权法、商标法。

**关键缺失（接受）**：劳动合同法、劳动法、刑法、公司法、仲裁法、刑事诉讼法。司法解释/行政法规/部门规章层完全空白。

### 12.2 数据集详情（laws_data）

23,157 个真实 Q&A 对（16MB）。

**结构**：`{question, answer, candidate_answer[], cause}`

**Cause 分布**：
- 交通事故 4198（可用）
- 婚姻家庭 3920（可用）
- 债权债务 3182（可用）
- 劳动纠纷 2719（**过滤**，corpus 无劳动合同法）
- 房产纠纷 2190（可用）

**加工**：用 Claude Opus 抽取 `answer` 中的法条引用 → 形成 `(query, expected_cite)` 对。预算 ~$115，预计产出 5-7k 高质量 golden。**建议先 1000 条 pilot 验证可用率**。

### 12.3 案件拆分数据集

格式：`(instruction, complex_query, decomposed_sub_cases)` 三元组。

**用途**：
- ✅ Receptionist few-shot examples（手挑 10-20 个）
- ✅ multi_issue eval set（挑 30-50 个）
- ❌ 不入向量库

**质量提醒**：LLM 生成，参差不齐，60-70% 例子合格。

---

## 13. 下一步

1. 用户 review 本 spec
2. 调用 `writing-plans` 技能产出 V1 实施计划（Phase 1 trace + schema + stub agent 是第一目标）
3. 实施

---

*本文档由 Claude Opus 4.7 与 xxm 协作产出。设计决策的"为什么"散落于附录 ADR 表，遇到问题先翻 ADR。*

# MedArXiv-AgentRAG 项目计划书

## 0. 项目定位

**项目名称：**

```text
MedArXiv-AgentRAG
```

**项目目标：**

构建一个基于 arXiv 公开论文的 Biomedical / Medical AI 论文 Agentic RAG 系统。系统用于帮助研究者检索、总结、比较和验证医学 AI 相关论文，而不是提供临床诊断或治疗建议。

**一句话描述：**

```text
A practical agentic RAG research assistant for biomedical AI literature review over arXiv papers.
```

**中文描述：**

```text
一个面向医学 AI / 生物医学 AI 论文综述的 Agentic RAG 系统，支持论文检索、方法对比、趋势总结、证据抽取和引用验证。
```

---

## 1. 为什么用 arXiv

arXiv 提供公开 API 和 bulk metadata 数据，适合快速构建论文检索系统。第一阶段不处理 PDF 全文，只使用论文元数据：

```text
title
abstract
authors
categories
published date
updated date
arxiv id
doi
journal-ref
```

这样可以快速完成一个可运行版本。

---

## 2. 项目边界

### 2.1 做什么

本项目做：

```text
1. 医学 AI / 生物医学 AI 论文检索
2. 论文摘要问答
3. 方法对比
4. 研究趋势总结
5. related work 草稿生成
6. evidence citation 检查
7. 普通 RAG 与 Agentic RAG 的对比实验
```

### 2.2 不做什么

本项目不做：

```text
1. 临床诊断
2. 治疗建议
3. 药物剂量推荐
4. 病人个性化医疗建议
5. 替代医生判断
```

所有回答都必须包含安全声明：

```text
This system is designed for biomedical literature research support only and does not provide medical advice.
```

中文回答时写：

```text
本系统仅用于医学 AI 文献研究辅助，不提供临床诊断或治疗建议。
```

---

## 3. 推荐技术栈

### 3.1 后端

```text
Python 3.10+
FastAPI
Pydantic
Typer
Uvicorn
```

### 3.2 数据处理

```text
pandas
orjson
tqdm
beautifulsoup4
lxml
```

### 3.3 检索

```text
BM25: rank-bm25 或 Pyserini
Dense retrieval: sentence-transformers / FlagEmbedding
Vector DB: Qdrant 或 FAISS
```

第一版建议使用 FAISS，因为本地更容易跑。

### 3.4 Embedding 模型

优先使用：

```text
BAAI/bge-m3
```

也可以先用 sentence-transformers 里的轻量模型做 smoke test。

### 3.5 LLM

第一版支持三种模式：

```text
1. OpenAI-compatible API
2. Ollama local model
3. Mock LLM for testing
```

推荐接口抽象成：

```python
class LLMClient:
    def generate(self, messages: list[dict], temperature: float = 0.0) -> str:
        ...
```

不要把系统绑定死在某一个模型上。

---

## 4. 第一版数据来源

第一阶段使用 Kaggle arXiv metadata dataset。

### 4.1 输入文件

假设下载后得到：

```text
data/raw/arxiv-metadata-oai-snapshot.json
```

该文件通常是 JSON Lines 格式，每一行是一篇论文。

每条数据大致包含：

```json
{
  "id": "2301.00001",
  "submitter": "...",
  "authors": "...",
  "title": "...",
  "comments": "...",
  "journal-ref": "...",
  "doi": "...",
  "abstract": "...",
  "categories": "cs.CV cs.LG",
  "versions": [...],
  "update_date": "2023-01-01"
}
```

---

## 5. 医学 AI 论文过滤规则

arXiv 没有统一的 medicine category，所以需要 category + keyword 双重过滤。

### 5.1 Category 过滤

优先保留这些 category：

```python
MEDICAL_AI_CATEGORIES = {
    "cs.CV",
    "cs.CL",
    "cs.AI",
    "cs.LG",
    "stat.ML",
    "q-bio.QM",
    "q-bio.GN",
    "q-bio.BM",
    "q-bio.NC",
}
```

### 5.2 Keyword 过滤

在 title + abstract 中搜索以下关键词：

```python
MEDICAL_KEYWORDS = [
    "medical",
    "clinical",
    "biomedical",
    "healthcare",
    "radiology",
    "pathology",
    "diagnosis",
    "disease",
    "patient",
    "electronic health record",
    "ehr",
    "medical image",
    "mri",
    "ct",
    "x-ray",
    "ultrasound",
    "segmentation",
    "cancer",
    "tumor",
    "brain tumor",
    "drug discovery",
    "protein",
    "genomics",
    "bioinformatics",
    "molecule",
    "molecular",
    "medqa",
    "med-vqa",
    "clinical nlp",
    "medical question answering",
    "large language model",
    "medical llm",
    "biomedical language model",
]
```

### 5.3 过滤逻辑

一篇论文保留条件：

```text
满足以下任意一个：
1. category 属于 MEDICAL_AI_CATEGORIES，并且 title/abstract 中出现医学关键词；
2. title/abstract 中出现强医学关键词，例如 "medical image", "clinical", "biomedical", "healthcare", "radiology", "pathology", "EHR"；
3. category 属于 q-bio.*。
```

---

## 6. 文档构造方式

每篇论文构造成一个 document。

### 6.1 Document 格式

```json
{
  "doc_id": "arxiv:2301.00001",
  "title": "...",
  "abstract": "...",
  "authors": ["...", "..."],
  "categories": ["cs.CV", "cs.LG"],
  "published": "2023-01-01",
  "updated": "2023-01-05",
  "doi": "...",
  "journal_ref": "...",
  "source": "arxiv",
  "url": "https://arxiv.org/abs/2301.00001",
  "text": "Title: ...\nCategories: ...\nAbstract: ..."
}
```

### 6.2 用于 embedding 的 text

```text
Title: {title}
Categories: {categories}
Abstract: {abstract}
```

第一版不要 chunk，因为 abstract 一般不长。第二版再支持 PDF full-text chunk。

---

## 7. 项目目录结构

```text
medarxiv-agentrag/
├── README.md
├── pyproject.toml
├── .env.example
├── configs/
│   ├── default.yaml
│   ├── retrieval.yaml
│   └── agent.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── indexes/
│   └── eval/
├── scripts/
│   ├── download_arxiv_metadata.md
│   ├── build_corpus.py
│   ├── build_bm25_index.py
│   ├── build_dense_index.py
│   ├── run_retrieval_eval.py
│   ├── run_rag_demo.py
│   └── run_agentic_demo.py
├── src/
│   └── medarxiv_agentrag/
│       ├── __init__.py
│       ├── config.py
│       ├── schema.py
│       ├── data/
│       │   ├── arxiv_loader.py
│       │   ├── filters.py
│       │   ├── document_builder.py
│       │   └── corpus_store.py
│       ├── indexing/
│       │   ├── bm25_index.py
│       │   ├── dense_index.py
│       │   ├── qdrant_index.py
│       │   └── faiss_index.py
│       ├── retrieval/
│       │   ├── base.py
│       │   ├── bm25_retriever.py
│       │   ├── dense_retriever.py
│       │   ├── hybrid_retriever.py
│       │   ├── rrf.py
│       │   └── reranker.py
│       ├── llm/
│       │   ├── base.py
│       │   ├── openai_client.py
│       │   ├── ollama_client.py
│       │   └── mock_client.py
│       ├── agents/
│       │   ├── intent_agent.py
│       │   ├── query_expansion_agent.py
│       │   ├── retrieval_agent.py
│       │   ├── evidence_extraction_agent.py
│       │   ├── verification_agent.py
│       │   ├── synthesis_agent.py
│       │   └── workflow.py
│       ├── generation/
│       │   ├── prompts.py
│       │   ├── answer_formatter.py
│       │   └── citation_formatter.py
│       ├── evaluation/
│       │   ├── retrieval_metrics.py
│       │   ├── generation_metrics.py
│       │   ├── faithfulness_eval.py
│       │   └── eval_dataset.py
│       ├── api/
│       │   ├── main.py
│       │   └── routes.py
│       └── utils/
│           ├── logging.py
│           ├── text.py
│           └── io.py
└── tests/
    ├── test_filters.py
    ├── test_rrf.py
    ├── test_document_builder.py
    ├── test_retrieval.py
    └── test_agents.py
```

---

## 8. 核心数据结构

在 `src/medarxiv_agentrag/schema.py` 中定义：

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal


class PaperDocument(BaseModel):
    doc_id: str
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published: Optional[str] = None
    updated: Optional[str] = None
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    url: str
    source: str = "arxiv"
    text: str


class SearchResult(BaseModel):
    doc_id: str
    score: float
    rank: int
    document: PaperDocument
    retriever: str


class Evidence(BaseModel):
    doc_id: str
    title: str
    url: str
    evidence_text: str
    evidence_type: Literal[
        "method",
        "dataset",
        "metric",
        "result",
        "limitation",
        "background",
        "unknown",
    ] = "unknown"
    support_score: float = 0.0


class AgentState(BaseModel):
    user_query: str
    intent: Optional[str] = None
    expanded_queries: list[str] = Field(default_factory=list)
    retrieved_results: list[SearchResult] = Field(default_factory=list)
    evidences: list[Evidence] = Field(default_factory=list)
    draft_answer: Optional[str] = None
    verified_answer: Optional[str] = None
    unsupported_claims: list[str] = Field(default_factory=list)
```

---

## 9. Baseline RAG 流程

第一阶段必须先实现普通 RAG。不要一上来就写 agent。

### 9.1 Baseline 流程

```text
user_query
→ hybrid retrieve top_k papers
→ format context
→ LLM answer
→ return answer with citations
```

### 9.2 Baseline 输出格式

```markdown
## Answer

...

## Evidence

[1] Paper title  
arXiv: xxxx.xxxxx  
Evidence: ...

[2] Paper title  
arXiv: xxxx.xxxxx  
Evidence: ...

## Safety Note

This system is designed for biomedical literature research support only and does not provide medical advice.
```

---

## 10. Hybrid Retrieval 设计

### 10.1 BM25

实现文件：

```text
src/medarxiv_agentrag/retrieval/bm25_retriever.py
```

输入：

```python
query: str
top_k: int
```

输出：

```python
list[SearchResult]
```

### 10.2 Dense Retrieval

实现文件：

```text
src/medarxiv_agentrag/retrieval/dense_retriever.py
```

第一版可以用 FAISS，本地更容易跑。第二版支持 Qdrant。

### 10.3 Hybrid Retrieval

实现：

```text
BM25 top_n
Dense top_n
→ RRF fusion
→ final top_k
```

RRF 公式：

```text
score(d) = Σ 1 / (k + rank_i(d))
```

默认：

```python
rrf_k = 60
bm25_top_n = 50
dense_top_n = 50
final_top_k = 10
```

实现文件：

```text
src/medarxiv_agentrag/retrieval/rrf.py
src/medarxiv_agentrag/retrieval/hybrid_retriever.py
```

---

## 11. Agentic RAG 设计

Agentic RAG 不要写成“多个 LLM 随便聊天”。要写成确定性 workflow。

### 11.1 总体流程

```text
User Query
  ↓
Intent Agent
  ↓
Query Expansion Agent
  ↓
Retrieval Agent
  ↓
Evidence Extraction Agent
  ↓
Synthesis Agent
  ↓
Verification Agent
  ↓
Final Answer
```

---

### 11.2 Intent Agent

文件：

```text
src/medarxiv_agentrag/agents/intent_agent.py
```

输入：

```text
用户问题
```

输出 intent：

```python
Literal[
    "method_summary",
    "dataset_query",
    "metric_query",
    "paper_comparison",
    "trend_analysis",
    "related_work_generation",
    "general_qa",
]
```

规则优先，不要全部依赖 LLM：

```python
if "compare" in query or "difference" in query:
    intent = "paper_comparison"

elif "dataset" in query or "benchmark" in query:
    intent = "dataset_query"

elif "metric" in query or "performance" in query or "score" in query:
    intent = "metric_query"

elif "trend" in query or "recent" in query or "survey" in query:
    intent = "trend_analysis"

elif "related work" in query or "literature review" in query:
    intent = "related_work_generation"

else:
    intent = "general_qa"
```

LLM 只作为 fallback。

---

### 11.3 Query Expansion Agent

文件：

```text
src/medarxiv_agentrag/agents/query_expansion_agent.py
```

作用：

```text
根据 intent 生成 3 到 5 个检索 query。
```

示例：

用户输入：

```text
What are recent methods for medical image segmentation?
```

扩展为：

```python
[
    "medical image segmentation deep learning",
    "U-Net transformer medical image segmentation",
    "nnU-Net Swin UNet TransUNet medical segmentation",
    "MRI CT medical image segmentation benchmark",
    "recent survey medical image segmentation"
]
```

第一版使用规则模板，第二版再加 LLM 改写。

---

### 11.4 Retrieval Agent

文件：

```text
src/medarxiv_agentrag/agents/retrieval_agent.py
```

逻辑：

```text
对每个 expanded query 执行 hybrid retrieval
合并结果
RRF 再融合一次
去重
返回 top_k
```

默认：

```python
queries = 3~5
per_query_top_k = 20
final_top_k = 15
```

---

### 11.5 Evidence Extraction Agent

文件：

```text
src/medarxiv_agentrag/agents/evidence_extraction_agent.py
```

第一版不需要复杂 NLP，直接从 title + abstract 提取 evidence。

对每篇 paper 生成：

```json
{
  "doc_id": "...",
  "evidence_text": "abstract 中最相关的 1~3 句话",
  "evidence_type": "method/dataset/metric/result/limitation/background"
}
```

句子选择规则：

```text
1. 把 abstract 切成句子；
2. 计算 query 和每个句子的 embedding cosine similarity；
3. 选择 top 1~3 句；
4. 如果 embedding 不可用，用关键词重叠得分 fallback。
```

---

### 11.6 Synthesis Agent

文件：

```text
src/medarxiv_agentrag/agents/synthesis_agent.py
```

输入：

```text
user_query
intent
evidences
```

输出：

```text
带引用的回答
```

不同 intent 使用不同 prompt。

#### method_summary prompt

```text
You are a biomedical AI research assistant.
Answer the user's question using only the provided paper evidence.
Focus on methods, model families, datasets, and limitations.
Do not provide clinical diagnosis or treatment advice.
Every important claim must cite at least one paper using [1], [2], etc.
```

#### related_work_generation prompt

```text
Write a concise related work section based only on the provided evidence.
Group papers by method category or research trend.
Use academic writing style.
Every paragraph must contain citations.
Do not invent papers, datasets, metrics, or results.
```

---

### 11.7 Verification Agent

文件：

```text
src/medarxiv_agentrag/agents/verification_agent.py
```

作用：

```text
检查生成回答中的 claim 是否被 evidence 支持。
```

第一版实现：

```text
1. 把 answer 按句子切分；
2. 找出包含事实性主张的句子；
3. 检查该句是否包含 citation，例如 [1]；
4. 检查 citation 对应 evidence 是否和句子有足够关键词重叠；
5. 标记 unsupported claims。
```

输出：

```json
{
  "verified_answer": "...",
  "unsupported_claims": [
    "..."
  ],
  "citation_precision_estimate": 0.83
}
```

第二版可以使用 LLM 判断 support / not support。

---

## 12. API 设计

使用 FastAPI。

文件：

```text
src/medarxiv_agentrag/api/main.py
```

### 12.1 `/search`

输入：

```json
{
  "query": "medical image segmentation transformer",
  "top_k": 10
}
```

输出：

```json
{
  "results": [
    {
      "rank": 1,
      "score": 0.92,
      "title": "...",
      "url": "...",
      "abstract": "..."
    }
  ]
}
```

### 12.2 `/rag/answer`

输入：

```json
{
  "query": "What are common datasets for medical image segmentation?",
  "top_k": 10
}
```

输出：

```json
{
  "answer": "...",
  "evidence": [...],
  "safety_note": "..."
}
```

### 12.3 `/agent/answer`

输入：

```json
{
  "query": "Compare U-Net and transformer-based methods for medical image segmentation.",
  "top_k": 15
}
```

输出：

```json
{
  "intent": "paper_comparison",
  "expanded_queries": [...],
  "answer": "...",
  "evidence": [...],
  "unsupported_claims": [...]
}
```

---

## 13. CLI 设计

用 Typer 实现命令行。

```bash
python scripts/build_corpus.py \
  --input data/raw/arxiv-metadata-oai-snapshot.json \
  --output data/processed/medarxiv_corpus.jsonl \
  --max-docs 50000
```

```bash
python scripts/build_bm25_index.py \
  --corpus data/processed/medarxiv_corpus.jsonl \
  --output data/indexes/bm25.pkl
```

```bash
python scripts/build_dense_index.py \
  --corpus data/processed/medarxiv_corpus.jsonl \
  --output data/indexes/faiss \
  --model BAAI/bge-m3
```

```bash
python scripts/run_rag_demo.py \
  --query "What are recent methods for medical image segmentation?"
```

```bash
python scripts/run_agentic_demo.py \
  --query "Compare U-Net and transformer-based methods for medical image segmentation."
```

---

## 14. 实验设计

### 14.1 Baselines

必须实现这些 baseline：

```text
1. BM25 only
2. Dense only
3. Hybrid BM25 + Dense
4. Hybrid + Query Expansion
5. Agentic RAG
6. Agentic RAG + Verification
```

### 14.2 检索指标

如果有人工标注相关论文：

```text
Recall@5
Recall@10
Recall@20
MRR@10
nDCG@10
```

### 14.3 生成指标

第一版可以用 LLM-as-judge + 规则指标。

```text
answer relevance
citation coverage
unsupported claim rate
citation precision estimate
```

### 14.4 延迟指标

记录：

```text
query expansion latency
retrieval latency
reranking latency
generation latency
verification latency
end-to-end latency
```

输出到：

```text
data/eval/latency_report.json
```

---

## 15. 构造小规模 Evaluation Set

第一版没有人工标注也没关系。先做 30 个固定 query。

文件：

```text
data/eval/research_queries.jsonl
```

格式：

```json
{"id": "q001", "query": "What are common datasets for medical image segmentation?", "intent": "dataset_query"}
{"id": "q002", "query": "Compare CNN and Transformer methods for chest X-ray diagnosis.", "intent": "paper_comparison"}
{"id": "q003", "query": "What are recent trends in medical large language models?", "intent": "trend_analysis"}
```

推荐 query：

```text
1. What are common datasets for medical image segmentation?
2. Compare U-Net and transformer-based methods for medical image segmentation.
3. What are recent trends in medical large language models?
4. Which papers discuss retrieval-augmented generation for medical question answering?
5. What methods are used for chest X-ray diagnosis?
6. What are common benchmarks for brain tumor segmentation?
7. What are common limitations of deep learning in medical imaging?
8. What datasets are used for clinical NLP?
9. How are large language models evaluated in medical QA?
10. What are common methods for AI-based drug discovery?
11. What role does self-supervised learning play in medical imaging?
12. What are common multimodal learning methods in healthcare AI?
13. What are limitations of medical vision-language models?
14. What are recent methods for pathology image analysis?
15. What methods are used for protein representation learning?
```

---

## 16. 配置文件

`configs/default.yaml`

```yaml
data:
  raw_arxiv_path: "data/raw/arxiv-metadata-oai-snapshot.json"
  corpus_path: "data/processed/medarxiv_corpus.jsonl"
  max_docs: 50000

retrieval:
  bm25_index_path: "data/indexes/bm25.pkl"
  dense_index_path: "data/indexes/faiss"
  embedding_model: "BAAI/bge-m3"
  bm25_top_n: 50
  dense_top_n: 50
  final_top_k: 10
  rrf_k: 60

agent:
  max_expanded_queries: 5
  per_query_top_k: 20
  final_top_k: 15
  max_evidence_per_doc: 2
  enable_verification: true

llm:
  provider: "openai_compatible"
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 1200
```

---

## 17. README 必须包含的内容

README 结构：

```markdown
# MedArXiv-AgentRAG

## Overview

## Features

## Safety Boundary

## Data Source

## Installation

## Build Corpus

## Build Indexes

## Run Baseline RAG

## Run Agentic RAG

## Evaluation

## Project Structure

## Limitations

## Roadmap
```

Safety Boundary 必须写清楚：

```markdown
This project is for biomedical AI literature research support only.
It does not provide medical diagnosis, treatment recommendations, drug dosage suggestions, or patient-specific advice.
```

---

## 18. 开发顺序

Codex 按这个顺序实现。

### Phase 1：项目骨架

目标：

```text
创建项目结构、配置系统、schema、README。
```

任务：

```text
1. 创建 pyproject.toml
2. 创建 src/medarxiv_agentrag/
3. 创建 schema.py
4. 创建 config.py
5. 创建 README.md
6. 添加基本 pytest
```

完成标准：

```bash
pytest
```

可以通过。

---

### Phase 2：Corpus 构建

目标：

```text
从 arXiv metadata 中过滤 medical AI 论文。
```

任务：

```text
1. 实现 arxiv_loader.py
2. 实现 filters.py
3. 实现 document_builder.py
4. 实现 build_corpus.py
```

完成标准：

```bash
python scripts/build_corpus.py \
  --input data/raw/arxiv-metadata-oai-snapshot.json \
  --output data/processed/medarxiv_corpus.jsonl \
  --max-docs 10000
```

输出：

```text
data/processed/medarxiv_corpus.jsonl
```

---

### Phase 3：BM25 Baseline

目标：

```text
实现关键词检索。
```

任务：

```text
1. build_bm25_index.py
2. bm25_index.py
3. bm25_retriever.py
```

完成标准：

```bash
python scripts/build_bm25_index.py
python scripts/run_rag_demo.py --retriever bm25 --query "medical image segmentation"
```

能返回论文列表。

---

### Phase 4：Dense Retrieval

目标：

```text
实现向量检索。
```

任务：

```text
1. dense_index.py
2. faiss_index.py
3. dense_retriever.py
4. build_dense_index.py
```

完成标准：

```bash
python scripts/build_dense_index.py
python scripts/run_rag_demo.py --retriever dense --query "medical image segmentation"
```

能返回论文列表。

---

### Phase 5：Hybrid Retrieval

目标：

```text
实现 BM25 + Dense + RRF。
```

任务：

```text
1. rrf.py
2. hybrid_retriever.py
3. test_rrf.py
```

完成标准：

```bash
python scripts/run_rag_demo.py --retriever hybrid --query "medical image segmentation transformer"
```

返回融合结果。

---

### Phase 6：普通 RAG

目标：

```text
实现 query → retrieve → generate answer。
```

任务：

```text
1. llm/base.py
2. llm/openai_client.py
3. llm/ollama_client.py
4. generation/prompts.py
5. generation/citation_formatter.py
6. scripts/run_rag_demo.py
```

完成标准：

```bash
python scripts/run_rag_demo.py \
  --query "What are common datasets for medical image segmentation?"
```

输出带引用回答。

---

### Phase 7：Agentic Workflow

目标：

```text
实现 intent → query expansion → retrieval → evidence extraction → synthesis → verification。
```

任务：

```text
1. intent_agent.py
2. query_expansion_agent.py
3. retrieval_agent.py
4. evidence_extraction_agent.py
5. synthesis_agent.py
6. verification_agent.py
7. workflow.py
8. scripts/run_agentic_demo.py
```

完成标准：

```bash
python scripts/run_agentic_demo.py \
  --query "Compare U-Net and transformer-based methods for medical image segmentation."
```

输出：

```text
intent
expanded queries
retrieved papers
evidences
answer
unsupported claims
```

---

### Phase 8：Evaluation

目标：

```text
比较 BM25、Dense、Hybrid、Agentic RAG。
```

任务：

```text
1. retrieval_metrics.py
2. faithfulness_eval.py
3. run_retrieval_eval.py
4. run_generation_eval.py
```

完成标准：

```bash
python scripts/run_retrieval_eval.py \
  --queries data/eval/research_queries.jsonl
```

输出：

```text
data/eval/retrieval_report.json
data/eval/latency_report.json
```

---

## 19. Codex 执行 Prompt

可以直接把下面这段给 Codex：

```text
You are building a Python project named MedArXiv-AgentRAG.

Goal:
Build a practical agentic RAG system over arXiv biomedical / medical AI papers. The system should support paper search, baseline RAG, agentic RAG, evidence extraction, citation-grounded answer generation, and simple citation verification. It is for literature research support only, not for clinical advice.

Please implement the project in phases.

Requirements:
1. Use Python 3.10+.
2. Use pydantic for schemas.
3. Use JSONL corpus files.
4. Use BM25 for sparse retrieval.
5. Use FAISS for first dense retrieval implementation.
6. Use BAAI/bge-m3 or sentence-transformers-compatible embedding model.
7. Implement hybrid retrieval with RRF.
8. Implement a deterministic agentic workflow:
   - intent classification
   - query expansion
   - hybrid retrieval
   - evidence extraction
   - synthesis
   - verification
9. Implement CLI scripts for:
   - building corpus
   - building BM25 index
   - building dense index
   - running baseline RAG
   - running agentic RAG
   - running evaluation
10. Add tests for filtering, document building, RRF, and retrieval.
11. Do not implement clinical diagnosis or treatment advice.
12. Every generated answer must include a safety note.

Project structure:
Use the directory structure described in the plan.

First implement:
- pyproject.toml
- README.md
- configs/default.yaml
- src/medarxiv_agentrag/schema.py
- src/medarxiv_agentrag/config.py
- src/medarxiv_agentrag/data/arxiv_loader.py
- src/medarxiv_agentrag/data/filters.py
- src/medarxiv_agentrag/data/document_builder.py
- scripts/build_corpus.py
- tests/test_filters.py
- tests/test_document_builder.py

After Phase 1 and Phase 2 pass, continue with BM25 retrieval, dense retrieval, hybrid retrieval, baseline RAG, and agentic RAG.

Implementation style:
- Keep modules small and testable.
- Use type hints.
- Use logging.
- Avoid hardcoding paths; read from configs.
- Make all scripts runnable from project root.
- Make the first version work on 10,000 documents before optimizing for larger scale.
```

---

## 20. 最小可交付版本标准

MVP 完成后，至少要能跑：

```bash
python scripts/build_corpus.py \
  --input data/raw/arxiv-metadata-oai-snapshot.json \
  --output data/processed/medarxiv_corpus.jsonl \
  --max-docs 10000
```

```bash
python scripts/build_bm25_index.py
```

```bash
python scripts/build_dense_index.py
```

```bash
python scripts/run_rag_demo.py \
  --query "What are common datasets for medical image segmentation?"
```

```bash
python scripts/run_agentic_demo.py \
  --query "Compare U-Net and transformer-based methods for medical image segmentation."
```

并输出类似：

```text
Intent: paper_comparison

Expanded Queries:
1. U-Net medical image segmentation
2. transformer medical image segmentation
3. TransUNet Swin-UNet medical segmentation

Answer:
...

Evidence:
[1] ...
[2] ...
[3] ...

Unsupported Claims:
...

Safety Note:
This system is designed for biomedical literature research support only and does not provide medical advice.
```

---

## 21. 简历/项目包装版本

英文：

```text
MedArXiv-AgentRAG: Built an agentic RAG research assistant over arXiv biomedical AI papers, supporting intent-aware query expansion, hybrid BM25+dense retrieval, RRF fusion, evidence extraction, citation-grounded generation, and citation verification. Compared BM25, dense retrieval, hybrid retrieval, and agentic RAG on biomedical literature review queries.
```

中文：

```text
构建 MedArXiv-AgentRAG，一个面向 arXiv 医学 AI 论文的多智能体 RAG 系统，实现意图识别、查询扩展、BM25+Dense 混合检索、RRF 融合、证据抽取、引用式生成与引用验证，并对 BM25、Dense、Hybrid 和 Agentic RAG 进行检索与生成效果对比。
```

---

## 22. 开发原则

这个项目第一版要坚持：

```text
1. 先做 abstract-level RAG，不碰 PDF 全文；
2. 先做 1 万篇论文，不碰百万级；
3. 先做规则 agent，不一开始就全靠 LLM；
4. 先做可运行 pipeline，再做效果优化；
5. 先做检索和引用可靠性，不先追求复杂 UI。
```


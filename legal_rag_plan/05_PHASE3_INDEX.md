# Phase 3 · 索引与检索（BM25 + Dense + Hybrid + Reranker）

## 依赖

- Phase 1：providers 抽象（embedding、reranker）已可用。
- Phase 2：`data/processed/chunks.jsonl` 已存在。

## 本阶段交付物

1. `src/legal_rag/indexes/bm25_index.py`
2. `src/legal_rag/indexes/dense_index.py`
3. `src/legal_rag/indexes/hybrid_retriever.py`
4. `src/legal_rag/indexes/reranker.py`（封装 RerankerProvider，加 USE_RERANKER 开关）
5. `scripts/build_indexes.py`
6. `scripts/retrieve.py`（CLI 验收用）
7. `tests/test_retrieval.py`

---

## 1. BM25Index

中文必须先 jieba 分词。

```python
# bm25_index.py
import jieba
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[DocumentChunk] = []
        self._tokens: list[list[str]] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [t for t in jieba.lcut(text) if t.strip()]

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self._tokens = [self.tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokens)

    def save(self, dir_path: Path) -> None: ...
    def load(self, dir_path: Path) -> None: ...

    def search(
        self, query: str, top_k: int, where: dict | None = None
    ) -> list[tuple[int, float]]:
        """返回 (chunk_index, score)，已按 where 做 hard filter。"""
        ...
```

`where` 支持精确匹配键值（`source_type`, `jurisdiction`, `law_name`, `valid_status`），先过滤再排序。

---

## 2. DenseIndex

```python
# dense_index.py
import faiss, numpy as np
from legal_rag.providers.factory import get_embedding_provider

class DenseIndex:
    def __init__(self):
        self.embed = get_embedding_provider()
        self.index: faiss.Index | None = None
        self.chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        vecs = self.embed.embed_texts([c.text for c in chunks])
        arr = np.array(vecs, dtype="float32")
        # bge-m3 已 L2 归一化 → 用内积索引
        self.index = faiss.IndexFlatIP(self.embed.dim)
        self.index.add(arr)

    def save(self, dir_path: Path) -> None: ...
    def load(self, dir_path: Path) -> None: ...

    def search(
        self, query: str, top_k: int, where: dict | None = None
    ) -> list[tuple[int, float]]:
        """同 BM25Index：返回 (chunk_index, score)。filter 在 FAISS 检索后做。"""
        ...
```

注意：

- `embed.embed_texts` 已经按 §02 §6/§7 处理了 batch 与归一化。
- `faiss-cpu` 即可，不要在依赖里强制 GPU。

---

## 3. HybridRetriever

最终分数：

```text
score = α * dense_norm + β * bm25_norm + γ * memory_boost + δ * metadata_boost
```

按 `source_type` **分组**做 min-max 归一化后再加权。

```python
# hybrid_retriever.py
WEIGHTS = {
    "default":       dict(alpha=0.45, beta=0.45, gamma=0.05, delta=0.05),
    "statute_agent": dict(alpha=0.35, beta=0.55, gamma=0.05, delta=0.05),
    "case_agent":    dict(alpha=0.60, beta=0.30, gamma=0.05, delta=0.05),
    "contract_agent":dict(alpha=0.45, beta=0.45, gamma=0.05, delta=0.05),
}

class HybridRetriever:
    def __init__(self, bm25: BM25Index, dense: DenseIndex):
        self.bm25 = bm25
        self.dense = dense
        assert bm25.chunks == dense.chunks, "BM25 与 Dense 必须使用同一份 chunks"

    def retrieve(
        self,
        query: str,
        *,
        agent: str = "default",
        top_k: int | None = None,
        where: dict | None = None,
        memory_boost: dict[str, float] | None = None,
        metadata_boost: dict[str, float] | None = None,
        run_id: str = "run0",
    ) -> list[RetrievedEvidence]:
        ...
```

返回 `RetrievedEvidence`，`evidence_id = f"ev_{run_id}_{n}"`。

`top_k` 默认按 `.env` 中是否启用 reranker：

- 不启用 → `settings.hybrid_top_k`（8）
- 启用 → `settings.hybrid_top_k_with_rerank`（20）

---

## 4. Reranker 封装

```python
# reranker.py
from legal_rag.providers.factory import get_reranker_provider
from legal_rag.config import settings

class Reranker:
    def __init__(self):
        self.provider = get_reranker_provider()
        self.enabled = settings.use_reranker

    def __call__(
        self, query: str, evidences: list[RetrievedEvidence], top_k: int | None = None
    ) -> list[RetrievedEvidence]:
        if not self.enabled or not evidences:
            return evidences[: top_k or len(evidences)]
        results = self.provider.rerank(
            query, [e.text for e in evidences], top_k=top_k or settings.rerank_top_k
        )
        out = []
        for r in results:
            ev = evidences[r.index]
            ev.score_rerank = r.score
            out.append(ev)
        return out
```

---

## 5. build_indexes.py

```python
# scripts/build_indexes.py
import typer, json
from pathlib import Path
from legal_rag.schemas import DocumentChunk
from legal_rag.indexes.bm25_index import BM25Index
from legal_rag.indexes.dense_index import DenseIndex

app = typer.Typer()

@app.command()
def run(
    processed: Path = typer.Option(Path("data/processed/chunks.jsonl")),
    out: Path = typer.Option(Path("data/indexes")),
):
    chunks = [DocumentChunk.model_validate_json(l) for l in processed.read_text("utf-8").splitlines() if l.strip()]
    bm25 = BM25Index(); bm25.build(chunks); bm25.save(out / "bm25")
    dense = DenseIndex(); dense.build(chunks); dense.save(out / "faiss")
    typer.echo(f"bm25={len(chunks)}, dense={len(chunks)}")

if __name__ == "__main__":
    app()
```

---

## 6. retrieve.py（验收 CLI）

```python
# scripts/retrieve.py
import typer
from pathlib import Path
from legal_rag.schemas import DocumentChunk
from legal_rag.indexes.bm25_index import BM25Index
from legal_rag.indexes.dense_index import DenseIndex
from legal_rag.indexes.hybrid_retriever import HybridRetriever
from legal_rag.indexes.reranker import Reranker

app = typer.Typer()

@app.command()
def run(
    query: str = typer.Argument(...),
    agent: str = typer.Option("statute_agent"),
    source_type: str = typer.Option("statute"),
    jurisdiction: str = typer.Option("CN"),
    top_k: int = typer.Option(5),
    indexes: Path = typer.Option(Path("data/indexes")),
):
    bm25 = BM25Index(); bm25.load(indexes / "bm25")
    dense = DenseIndex(); dense.load(indexes / "faiss")
    retriever = HybridRetriever(bm25, dense)
    rr = Reranker()
    cands = retriever.retrieve(
        query, agent=agent, top_k=20,
        where={"source_type": source_type, "jurisdiction": jurisdiction},
    )
    final = rr(query, cands, top_k=top_k)
    for i, e in enumerate(final, 1):
        typer.echo(f"[{i}] hybrid={e.score_hybrid:.3f} rerank={e.score_rerank} {e.metadata.get('law_name')} {e.metadata.get('article_number')}")
        typer.echo(f"    {e.text[:120]}...")

if __name__ == "__main__":
    app()
```

---

## 端到端验收

### 验收命令

```bash
# 用 mock embedding 也能跑（虽然语义检索效果差，但通流程）
EMBEDDING_PROVIDER=mock RERANKER_PROVIDER=noop \
  python scripts/build_indexes.py

EMBEDDING_PROVIDER=mock RERANKER_PROVIDER=noop \
  python scripts/retrieve.py "劳动合同 第39条" --source-type statute

# 真实模型（硅基流动）
EMBEDDING_PROVIDER=siliconflow USE_RERANKER=true RERANKER_PROVIDER=siliconflow \
  python scripts/build_indexes.py

EMBEDDING_PROVIDER=siliconflow USE_RERANKER=true RERANKER_PROVIDER=siliconflow \
  python scripts/retrieve.py "公司单方面解除劳动合同的条件" --source-type statute

pytest -q tests/test_retrieval.py
```

### 验收通过条件

- BM25：query "劳动合同 第39条" 命中 article_number=="39" 的 chunk 在 top-3。
- Dense：query "被公司开除是否合法" 在 top-10 召回劳动合同法第 39/40 条相关 chunk。
- 关闭 reranker 时 `score_rerank == None`；开启时 `score_rerank` 单调与 reranker 排名一致。
- `where` filter 生效：传 `source_type=case` 不会返回 `source_type=statute` 的结果。
- 索引可 `save/load` round-trip，结果一致。

---

## Codex Prompt

```text
基于已完成的 Phase 1 (providers) 和 Phase 2 (ingestion)，实现 Phase 3：检索。

按 PLAN/05_PHASE3_INDEX.md 实现：

1. src/legal_rag/indexes/bm25_index.py
2. src/legal_rag/indexes/dense_index.py
3. src/legal_rag/indexes/hybrid_retriever.py
4. src/legal_rag/indexes/reranker.py（包装 providers.factory.get_reranker_provider）
5. scripts/build_indexes.py
6. scripts/retrieve.py
7. tests/test_retrieval.py

要求：
- 中文 BM25 必须 jieba 分词。
- DenseIndex 通过 providers.factory.get_embedding_provider() 取 embedding，业务代码不直接 import sentence_transformers。
- HybridRetriever 按 source_type 分组做 min-max 归一化，再用 §3 的 WEIGHTS 加权；agent 名作为 key 选权重。
- 支持 hard filter（source_type / jurisdiction / law_name / valid_status）。
- Reranker 封装类必须支持 USE_RERANKER=false 时直接返回原序（NoopRerankerProvider）。
- save/load 用 pickle + faiss.write_index/read_index；BM25 保存 tokens 与 chunks。
- 测试要点：
  * 用 MockEmbeddingProvider + 5 条假 chunk 跑通 build/save/load/retrieve。
  * 关键词 "第39条" 命中 article_number="39" 的 chunk。
  * filter source_type=case 时排除 statute。
  * Reranker noop 模式下输出顺序与输入一致。

不要实现 agent / graph / memory / eval。

验收：
  EMBEDDING_PROVIDER=mock python scripts/build_indexes.py
  EMBEDDING_PROVIDER=mock python scripts/retrieve.py "第39条" --source-type statute
  pytest -q
```

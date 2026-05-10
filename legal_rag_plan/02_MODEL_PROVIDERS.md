# 02 · 模型 Provider 抽象层（本地 + 硅基流动）

> 本项目所有 Embedding / Reranker / LLM 调用都必须经过 `src/legal_rag/providers/` 下的抽象层，不允许在业务代码中直接 `import sentence_transformers` 或 `import openai`。

> 所有 LLM 调用还必须经过 `agents/_context_composer.py` 拼装 messages，**不允许 agent 直接构造 LLMMessage 列表**。Provider 层只负责"把 messages 发给模型"，不做截断、不做摘要——那是 ContextComposer + 由其透明委托的 `harness/context_compactor.py` (ContextCompactor) 的职责。

---

## 1. 总体设计

```text
业务代码 (agent)
   │
   ▼
agents/_context_composer.py       ← 唯一的 messages 装配点
   │
   ▼
providers/factory.py              ← 根据 .env 选择实现
   │
   ├─ EmbeddingProvider
   │     ├─ LocalEmbeddingProvider          (sentence-transformers)
   │     ├─ SiliconFlowEmbeddingProvider    (HTTP)
   │     └─ MockEmbeddingProvider           (测试用，固定向量)
   │
   ├─ RerankerProvider
   │     ├─ LocalRerankerProvider           (FlagEmbedding)
   │     ├─ SiliconFlowRerankerProvider     (HTTP)
   │     └─ NoopRerankerProvider            (USE_RERANKER=false 时)
   │
   └─ LLMProvider
         ├─ LocalLLMProvider                (vLLM / Ollama, OpenAI 兼容)
         ├─ SiliconFlowLLMProvider          (HTTP)
         └─ MockLLMProvider                 (按调用顺序返回的响应队列)
```

切换策略：

```env
EMBEDDING_PROVIDER=siliconflow      # local | siliconflow | mock
RERANKER_PROVIDER=siliconflow       # local | siliconflow | noop
LLM_PROVIDER=siliconflow            # local | siliconflow | mock
```

---

## 2. 推荐模型

| 角色 | 模型 ID | 本地加载方式 | 硅基流动 model 字段 |
|---|---|---|---|
| Embedding | `BAAI/bge-m3` | `sentence_transformers.SentenceTransformer("BAAI/bge-m3")` | `"BAAI/bge-m3"` |
| Reranker  | `BAAI/bge-reranker-v2-m3` | `FlagEmbedding.FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)` | `"BAAI/bge-reranker-v2-m3"` |
| LLM       | `Qwen/Qwen2.5-32B-Instruct` | vLLM 或 Ollama 启动 OpenAI 兼容 server | `"Qwen/Qwen2.5-32B-Instruct"` |

> 用户口头说 "Qwen3.5-27B"，本计划按当前公开可用、且硅基流动确实上架的最接近规格 `Qwen/Qwen2.5-32B-Instruct` 实现，模型 ID 通过 `.env` 注入，未来一行替换。

---

## 3. .env 约定

```env
# ===== 通用 =====
APP_ENV=dev
DEFAULT_JURISDICTION=CN

# ===== Embedding =====
EMBEDDING_PROVIDER=siliconflow            # local | siliconflow | mock
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=32

# ===== Reranker =====
USE_RERANKER=false
RERANKER_PROVIDER=siliconflow             # local | siliconflow | noop
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# ===== LLM =====
LLM_PROVIDER=siliconflow                  # local | siliconflow | mock
LLM_MODEL=Qwen/Qwen2.5-32B-Instruct
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
LLM_TIMEOUT_S=60
LLM_CONTEXT_WINDOW=32768                  # 模型最大上下文窗口（Qwen2.5-32B 默认 32k）

# ===== 硅基流动 =====
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxx
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# ===== 本地推理 =====
LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_API_KEY=EMPTY

# ===== 检索/重试 =====
BM25_TOP_K=30
DENSE_TOP_K=30
HYBRID_TOP_K=8
HYBRID_TOP_K_WITH_RERANK=20
RERANK_TOP_K=8
FINAL_TOP_K=5
MAX_RETRIEVAL_RETRY=2
MAX_ANSWER_REVISION=1

# ===== 会话上下文 =====
SESSION_KEEP_RECENT_TURNS=2               # ContextComposer 始终保留的最近原文轮数
SESSION_TURN_BUDGET_TOKENS=12000          # 单 turn 拼装后总预算（含 system + history + evidence + user）
SESSION_COMPACT_TRIGGER_TOKENS=10000      # 估算 ≥ 该值时触发 ContextCompactor
SESSION_COMPACT_TRIGGER_TURNS=8           # 累计 turn 数 ≥ 该值时也触发
SESSION_TTL_DAYS=7

# ===== 路径 =====
DATA_DIR=data
INDEX_DIR=data/indexes
LOG_DIR=logs
MEMORY_DB=legal_rag_memory.sqlite3
```

---

## 4. 抽象基类 (`providers/base.py`)

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence
from pydantic import BaseModel

class EmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


class RerankResult(BaseModel):
    index: int
    score: float


class RerankerProvider(ABC):
    @abstractmethod
    def rerank(
        self, query: str, docs: Sequence[str], top_k: int | None = None
    ) -> list[RerankResult]: ...


class LLMMessage(BaseModel):
    role: str           # system | user | assistant
    content: str


class LLMProvider(ABC):
    model: str
    context_window: int          # 来自 settings.llm_context_window

    @abstractmethod
    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        stop: list[str] | None = None,
        timeout: float | None = None,
    ) -> str: ...

    def chat_json(
        self,
        messages: Sequence[LLMMessage],
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        raw = self.chat(
            messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return schema.model_validate_json(raw)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """粗估：中文 ~ 1.5 char/token，英文 ~ 4 char/token；混合按 char // 2 兜底。"""
        if not text:
            return 0
        cn = sum(1 for c in text if "一" <= c <= "鿿")
        other = len(text) - cn
        return int(cn / 1.5 + other / 4) + 1
```

> `estimate_tokens` 给 ContextComposer 与 ContextCompactor 使用。Provider 不主动截断、不报错；超 `context_window` 由 ContextComposer 提前阻断并透明触发 ContextCompactor。

---

## 5. 工厂 (`providers/factory.py`)

```python
from functools import lru_cache
from .base import EmbeddingProvider, RerankerProvider, LLMProvider
from ..config import settings

@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    p = settings.embedding_provider
    if p == "local":
        from .embedding_local import LocalEmbeddingProvider
        return LocalEmbeddingProvider(settings.embedding_model)
    if p == "siliconflow":
        from .embedding_siliconflow import SiliconFlowEmbeddingProvider
        return SiliconFlowEmbeddingProvider(
            model=settings.embedding_model,
            api_key=settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
            dim=settings.embedding_dim,
        )
    if p == "mock":
        from .embedding_local import MockEmbeddingProvider
        return MockEmbeddingProvider(dim=settings.embedding_dim)
    raise ValueError(f"unknown EMBEDDING_PROVIDER={p}")


@lru_cache(maxsize=1)
def get_reranker_provider() -> RerankerProvider:
    if not settings.use_reranker:
        from .reranker_local import NoopRerankerProvider
        return NoopRerankerProvider()
    p = settings.reranker_provider
    if p == "local":
        from .reranker_local import LocalRerankerProvider
        return LocalRerankerProvider(settings.reranker_model)
    if p == "siliconflow":
        from .reranker_siliconflow import SiliconFlowRerankerProvider
        return SiliconFlowRerankerProvider(
            model=settings.reranker_model,
            api_key=settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
        )
    raise ValueError(f"unknown RERANKER_PROVIDER={p}")


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    p = settings.llm_provider
    if p == "local":
        from .llm_local import LocalLLMProvider
        return LocalLLMProvider(
            model=settings.llm_model,
            base_url=settings.local_llm_base_url,
            api_key=settings.local_llm_api_key,
            context_window=settings.llm_context_window,
        )
    if p == "siliconflow":
        from .llm_siliconflow import SiliconFlowLLMProvider
        return SiliconFlowLLMProvider(
            model=settings.llm_model,
            api_key=settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
            context_window=settings.llm_context_window,
        )
    if p == "mock":
        from .llm_local import MockLLMProvider
        return MockLLMProvider()
    raise ValueError(f"unknown LLM_PROVIDER={p}")
```

---

## 6. 本地实现要点

### 6.1 LocalEmbeddingProvider

```python
class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts):
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return vecs.tolist()
```

### 6.2 LocalRerankerProvider

```python
class LocalRerankerProvider(RerankerProvider):
    def __init__(self, model_name: str):
        from FlagEmbedding import FlagReranker
        self._model = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query, docs, top_k=None):
        pairs = [[query, d] for d in docs]
        scores = self._model.compute_score(pairs, normalize=True)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return [RerankResult(index=i, score=float(s)) for i, s in ranked]
```

### 6.3 LocalLLMProvider（OpenAI 兼容 HTTP）

vLLM 启动示例：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768
```

Ollama 启动示例：

```bash
ollama pull qwen2.5:32b-instruct
ollama serve
```

实现复用 OpenAI Python SDK：

```python
class LocalLLMProvider(LLMProvider):
    def __init__(self, model: str, base_url: str, api_key: str, context_window: int):
        from openai import OpenAI
        self.model = model
        self.context_window = context_window
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages, *, temperature=None, max_tokens=None,
             response_format=None, stop=None, timeout=None):
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=temperature if temperature is not None else 0.0,
            max_tokens=max_tokens or 2048,
            response_format=response_format,
            stop=stop,
            timeout=timeout or 60,
        )
        return resp.choices[0].message.content or ""
```

### 6.4 Mock providers

```python
class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 8):
        self.dim = dim
    def embed_texts(self, texts):
        import hashlib
        out = []
        for t in texts:
            h = hashlib.md5(t.encode()).digest()
            v = [b / 255.0 for b in h[: self.dim]]
            out.append(v)
        return out

class MockLLMProvider(LLMProvider):
    """支持注入"按调用顺序返回的响应队列"，便于多轮测试。"""
    model = "mock"
    context_window = 32768
    def __init__(self, responses: list[str] | None = None):
        self._queue = list(responses or [])
        self.calls: list[list[LLMMessage]] = []   # 测试断言用
    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        if self._queue:
            return self._queue.pop(0)
        return "{}"
    def push(self, response: str) -> None:
        self._queue.append(response)
```

---

## 7. 硅基流动实现要点

硅基流动 API 与 OpenAI 兼容，文档：<https://docs.siliconflow.cn/>

### 7.1 SiliconFlowLLMProvider

```python
class SiliconFlowLLMProvider(LLMProvider):
    def __init__(self, model, api_key, base_url, context_window):
        from openai import OpenAI
        self.model = model
        self.context_window = context_window
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages, *, temperature=None, max_tokens=None,
             response_format=None, stop=None, timeout=None):
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=temperature if temperature is not None else 0.0,
            max_tokens=max_tokens or 2048,
            response_format=response_format,
            stop=stop,
            timeout=timeout or 60,
        )
        return resp.choices[0].message.content or ""
```

### 7.2 SiliconFlowEmbeddingProvider

```python
class SiliconFlowEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model, api_key, base_url, dim: int):
        import httpx
        self.model = model
        self.dim = dim
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

    def embed_texts(self, texts):
        out: list[list[float]] = []
        BATCH = 32
        for i in range(0, len(texts), BATCH):
            r = self._client.post(
                "/embeddings",
                json={"model": self.model, "input": list(texts[i : i + BATCH])},
            )
            r.raise_for_status()
            data = r.json()["data"]
            out.extend([d["embedding"] for d in sorted(data, key=lambda x: x["index"])])
        return out
```

### 7.3 SiliconFlowRerankerProvider

```python
class SiliconFlowRerankerProvider(RerankerProvider):
    def __init__(self, model, api_key, base_url):
        import httpx
        self.model = model
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

    def rerank(self, query, docs, top_k=None):
        payload = {
            "model": self.model,
            "query": query,
            "documents": list(docs),
            "top_n": top_k or len(docs),
            "return_documents": False,
        }
        r = self._client.post("/rerank", json=payload)
        r.raise_for_status()
        results = r.json().get("results", [])
        return [
            RerankResult(index=item["index"], score=float(item["relevance_score"]))
            for item in results
        ]
```

---

## 8. 错误处理与重试

所有 provider 必须：

1. **超时**：默认 60s，可由调用方传入。
2. **重试**：HTTP 5xx / 网络错误，指数退避重试 2 次（用 `tenacity`）。
3. **限流**：429 不要立即重试，sleep `Retry-After`（如有）或 5s。
4. **错误包装**：抛出 `providers.errors.ProviderError`，业务层（agent）捕获后走规则兜底。

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def _post(...): ...
```

---

## 9. 验收脚本 `scripts/check_providers.py`

```python
"""Phase 1 验收：确认三类 provider 都能 ping 通。"""
import sys
from legal_rag.providers.factory import (
    get_embedding_provider, get_reranker_provider, get_llm_provider,
)
from legal_rag.providers.base import LLMMessage

def main() -> int:
    print("== Embedding ==")
    emb = get_embedding_provider()
    v = emb.embed_query("劳动合同法第三十九条")
    assert len(v) == emb.dim, (len(v), emb.dim)
    print(f"  dim={emb.dim}, sample[:4]={v[:4]}")

    print("== Reranker ==")
    rr = get_reranker_provider()
    res = rr.rerank("解除劳动合同", ["劳动合同法第39条", "民法典第464条"], top_k=2)
    print(f"  results={res}")

    print("== LLM ==")
    llm = get_llm_provider()
    out = llm.chat([LLMMessage(role="user", content="只回复 OK 两个字符")])
    print(f"  llm.model={llm.model}, ctx={llm.context_window}, out={out!r}")

    print("ALL OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

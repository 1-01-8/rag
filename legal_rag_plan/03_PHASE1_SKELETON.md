# Phase 1 · 项目骨架 + Provider 抽象 + 会话 Schema

## 依赖

- 无（全新仓库即可）

## 本阶段交付物

1. `pyproject.toml`（含本地与硅基流动两套依赖）。
2. `.env.example`（按 `02_MODEL_PROVIDERS.md` §3，含 `SESSION_*` 字段）。
3. `src/legal_rag/config.py`（`pydantic-settings` 读取 `.env`）。
4. `src/legal_rag/schemas.py`：
   - 文档侧：`DocumentChunk / RetrievedEvidence / EvidenceAssessment / Citation`
   - 会话侧：`TurnRecord / StickyIntake / CompactionRecord / ConversationState`
5. `src/legal_rag/providers/`（`base.py / factory.py` + 6 个实现 + mock，含 `MockLLMProvider` 响应队列）。
6. `scripts/check_providers.py`。
7. `tests/test_providers.py`、`tests/test_schemas.py`（schema round-trip）。
8. 占位脚本：`scripts/{ingest_docs,build_indexes,retrieve,chat,ask,run_eval}.py`，至少 `--help`。
9. 包结构所有 `__init__.py`。

> 重点：本 Phase 不实现 ContextComposer / Compactor / SessionStore，但**必须**把 `ConversationState`、`SESSION_*` 配置、`MockLLMProvider` 响应队列写好，让后续 Phase 一接就能用。

---

## pyproject.toml

```toml
[project]
name = "legal-research-agent"
version = "0.1.0"
description = "Harness-controlled self-evolving multi-agent legal RAG system"
requires-python = ">=3.10"
dependencies = [
  "fastapi>=0.110",
  "uvicorn>=0.27",
  "python-multipart>=0.0.9",
  "pydantic>=2.0",
  "pydantic-settings>=2.0",
  "sqlalchemy>=2.0",
  "numpy",
  "scikit-learn",
  "rank-bm25",
  "jieba",
  "faiss-cpu",
  "pymupdf",
  "pdfplumber",
  "typer",
  "rich",
  "langgraph",
  "httpx>=0.27",
  "tenacity>=8.0",
  "openai>=1.40",
]

[project.optional-dependencies]
local-models = [
  "sentence-transformers",
  "torch",
  "FlagEmbedding",
]
dev = [
  "pytest",
  "pytest-cov",
  "respx>=0.21",
  "ruff",
  "mypy",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/legal_rag"]

[tool.ruff]
line-length = 100
```

---

## src/legal_rag/config.py

```python
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 通用
    app_env: str = "dev"
    default_jurisdiction: str = "CN"

    # Embedding
    embedding_provider: Literal["local", "siliconflow", "mock"] = "siliconflow"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32

    # Reranker
    use_reranker: bool = False
    reranker_provider: Literal["local", "siliconflow", "noop"] = "siliconflow"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # LLM
    llm_provider: Literal["local", "siliconflow", "mock"] = "siliconflow"
    llm_model: str = "Qwen/Qwen2.5-32B-Instruct"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout_s: int = 60
    llm_context_window: int = 32_768

    # SiliconFlow
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"

    # 本地推理
    local_llm_base_url: str = "http://127.0.0.1:8000/v1"
    local_llm_api_key: str = "EMPTY"

    # 检索 / retry
    bm25_top_k: int = 30
    dense_top_k: int = 30
    hybrid_top_k: int = 8
    hybrid_top_k_with_rerank: int = 20
    rerank_top_k: int = 8
    final_top_k: int = 5
    max_retrieval_retry: int = 2
    max_answer_revision: int = 1

    # 会话上下文
    session_keep_recent_turns: int = 2
    session_turn_budget_tokens: int = 12_000
    session_compact_trigger_tokens: int = 10_000
    session_compact_trigger_turns: int = 8
    session_ttl_days: int = 7

    # 路径
    data_dir: str = "data"
    index_dir: str = "data/indexes"
    log_dir: str = "logs"
    memory_db: str = "legal_rag_memory.sqlite3"

settings = Settings()
```

---

## schemas.py 必须定义的内容

文档侧（与 `01_ARCHITECTURE.md` §3.1–3.4 一致）：

- `DocumentChunk`
- `RetrievedEvidence`
- `EvidenceAssessment`
- `Citation`

会话侧（与 `01_ARCHITECTURE.md` §3.5 一致）：

- `TurnRecord`
- `StickyIntake`
- `CompactionRecord`
- `ConversationState`

均使用 Pydantic v2，所有时间戳用 `float`（epoch seconds）便于序列化。

---

## CLI 占位脚本

每个脚本必须支持 `--help` 即使尚未实现内部逻辑。例如：

```python
# scripts/chat.py
import typer
app = typer.Typer(help="Multi-turn chat (Phase 5 will implement)")

@app.command()
def session():
    raise typer.Exit(code=0)

if __name__ == "__main__":
    app()
```

---

## 端到端验收

### 验收命令

```bash
# 1. 安装（不含本地模型）
pip install -e ".[dev]"

# 2. 复制 env
cp .env.example .env

# 3. 单元测试
pytest -q tests/test_providers.py tests/test_schemas.py

# 4. 占位脚本能跑
python scripts/check_providers.py
python scripts/ingest_docs.py --help
python scripts/build_indexes.py --help
python scripts/retrieve.py --help
python scripts/chat.py --help
python scripts/ask.py --help
python scripts/run_eval.py --help
```

### 验收通过条件

- `pytest -q` 全绿。
- `check_providers.py` 在三种配置下都能跑：
  1. `EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock RERANKER_PROVIDER=noop`
  2. `*_PROVIDER=siliconflow` + 有效 key
  3. `LLM_PROVIDER=local` + 已起 vLLM/Ollama（可选）
- `tests/test_providers.py` 覆盖：
  - 工厂返回类型正确；
  - mock provider 行为正确；
  - 用 respx mock 硅基流动三个端点；
  - `MockLLMProvider` 的响应队列：连续 3 次 chat 返回 3 个不同响应，`calls` 记录正确。
- `tests/test_schemas.py` 覆盖：
  - `ConversationState` 含 `turns / sticky_intake / compactions` 时 `model_dump_json()` → `model_validate_json()` round-trip 等价；
  - `evidence_id` 字段在 `RetrievedEvidence` 与 `Citation` 中可序列化；
  - `CompactionRecord.token_estimate_before/after` 类型正确。

---

## Codex Prompt

```text
你要在一个全新仓库里实现 LegalResearch-Agent 的 Phase 1：项目骨架 + Provider 抽象层 + 会话 Schema。

请严格按照 PLAN/01_ARCHITECTURE.md 的目录结构和 PLAN/02_MODEL_PROVIDERS.md 的接口实现。本阶段做：

1. pyproject.toml（PLAN/03 给定的依赖）
2. .env.example（PLAN/02 §3 字段全集，含 SESSION_* 字段）
3. src/legal_rag/__init__.py、config.py（pydantic-settings）
4. src/legal_rag/schemas.py：
   - 文档侧：DocumentChunk / RetrievedEvidence / EvidenceAssessment / Citation（按 PLAN/01 §3.1–3.4）
   - 会话侧：TurnRecord / StickyIntake / CompactionRecord / ConversationState（按 PLAN/01 §3.5）
5. src/legal_rag/providers/ 全套：
   - base.py（EmbeddingProvider / RerankerProvider / LLMProvider，LLMProvider 含 context_window 与 estimate_tokens）
   - factory.py（带 lru_cache）
   - embedding_local.py（LocalEmbeddingProvider + MockEmbeddingProvider）
   - embedding_siliconflow.py
   - reranker_local.py（LocalRerankerProvider + NoopRerankerProvider）
   - reranker_siliconflow.py
   - llm_local.py（LocalLLMProvider + MockLLMProvider 响应队列）
   - llm_siliconflow.py
   - errors.py
6. scripts/ 占位（ingest_docs / build_indexes / retrieve / chat / ask / run_eval），各 --help；scripts/check_providers.py 按 PLAN/02 §9 实现。
7. tests/test_providers.py + tests/test_schemas.py。

要求：
- import 局部化：sentence-transformers / FlagEmbedding 仅在对应 Local* provider 内 import。
- HTTP provider 用 tenacity 指数退避，最多 2 次。
- MockLLMProvider 必须支持构造时传 responses 列表，并在 self.calls 里记录每次输入便于断言。
- ConversationState round-trip 测试覆盖：3 turns + 1 compaction + sticky_intake.pinned_facts=["A","B"]。
- 严格类型标注，过 ruff check。

不要实现 ingestion / index / agents / graph / memory / context composer / compactor / FastAPI。

验收：
  pip install -e ".[dev]"
  cp .env.example .env
  python scripts/check_providers.py    # *_PROVIDER 全 mock/noop 时
  pytest -q
两条命令必须 0 退出码。
```

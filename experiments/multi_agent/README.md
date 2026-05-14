# Multi-Agent Legal RAG (experimental)

Phase 1: walking skeleton — trace system + stub agent + asyncio.

See `docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md` for full design.

## Run tests
```
pip install -e ".[dev]"
pytest -v
```

## Qdrant

This project **reuses** the existing `legal-rag-qdrant` container on host ports `6433:6333` and `6434:6334`. It does NOT manage its own container.

```bash
# Verify running
docker ps | grep legal-rag-qdrant
curl http://localhost:6433/healthz

# Restart if stopped
docker start legal-rag-qdrant

# List multi_agent's own collections (prefixed `ma_`)
curl -s http://localhost:6433/collections | python -m json.tool
```

Multi_agent collections coexist with legacy `legal_rag/` collections under the `ma_` prefix:
- `ma_statutes` — Chinese-Laws indexed by multi_agent (Phase 2a)
- `ma_cases` — laws_data Q&A (Phase 2d)
- `ma_user_history` — turn/sticky-derived index (Phase 2d)

Connection URL is configurable via the `QDRANT_URL` env var (default `http://localhost:6433`).

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

## Local Qwen 3.5 9B (vLLM)

This project's `openai_compatible` provider talks to a local Qwen 3.5 9B served by vLLM at `http://localhost:8000/v1`.

```bash
# Start (one-time per machine boot)
cd /home/xxm/models/qwen3.5-9b
conda activate qwen35
nohup bash serve_vllm.sh > /tmp/vllm_9b.log 2>&1 &

# Verify (waits ~2 min on first start)
curl http://localhost:8000/v1/models

# Stop
pkill -9 -f vllm
```

The service uses GPU card 3 (~20 GB VRAM). See `/home/xxm/models/qwen3.5-9b/USAGE.md` for details.

### Tool-calling enabled launcher

The system script at `/home/xxm/models/qwen3.5-9b/serve_vllm.sh` does NOT enable
tool calling. For multi_agent tests/agents that need to dispatch tools, use the
project-local launcher which adds `--enable-auto-tool-choice` and
`--tool-call-parser qwen3_xml` (the chat template uses Qwen3's XML-style
`<tool_call><function=...>` format, not Hermes JSON):

```bash
# Stop the no-tools version if running
pkill -9 -f vllm

# Launch with tool calling
PATH=/home/xxm/miniconda3/envs/qwen35/bin:$PATH \
  nohup bash /home/xxm/rag/experiments/multi_agent/scripts/serve_qwen_vllm.sh \
  > /tmp/vllm_9b.log 2>&1 &
```

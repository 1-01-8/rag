# Multi-Agent Legal RAG (experimental)

See `docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md` for full design.
For operator workflows / scripts / trace events 见 `RUNBOOK.md`.

---

## 🚀 已验证可用的快速启动 (推荐)

```bash
# 一次性配 (~/.bashrc 加这两行)
export SILICONFLOW_API_KEY=sk-你的key
conda activate qwen35

# 启动 chat — 进入交互 REPL, 输入问题即可
cd /home/xxm/rag/experiments/multi_agent
bash scripts/chat-ready.sh
```

### 配置内容 (打包在 `chat-ready.sh` 内)

| 项 | 值 | 备注 |
|---|---|---|
| Provider | `siliconflow` (api.siliconflow.cn) | 在日本可达; DeepSeek 官方 API 区域限制 |
| Model | `deepseek-ai/DeepSeek-V3.1` | 实测稳定 5-15s/LLM call; V4-Flash 偶尔挂 100s+ |
| 索引 | `ma_statutes` (13722 chunks) | 177 部 Chinese-Laws/extracted/ 灌好 |
| Sparse | `data/indexes/statutes_sparse.json` | 跟 Qdrant collection 配对 |
| Supervisor | 关闭 | 省 25-30s/轮; 需审核可去掉 `--no-supervisor` |
| 单轮时间 | **30-60 秒** | 含 ReAct 3-4 步 |
| 单轮成本 | **~$0.003** (¥0.02) | DeepSeek-V3.1 折合人民币 |

### 常用变体

```bash
# 劳动咨询 (corpus 未收录劳动合同法, prompt 自带 fallback)
bash scripts/chat-ready.sh --specialty 劳动

# 续上次会话
bash scripts/chat-ready.sh --session-id chat_xxxxxx

# 严格审核 (慢 25s 但有引用真实性校验) — 直接调 chat.py
python scripts/chat.py --provider siliconflow \
    --statutes-collection ma_statutes \
    --statutes-sparse data/indexes/statutes_sparse.json
```

### 启动前自检

`chat-ready.sh` 自动校验:
- `SILICONFLOW_API_KEY` 环境变量
- `data/indexes/statutes_sparse.json` 索引文件存在

任一缺失会给清晰错误 + 修复指令.

### 不在路径里的情况

- 想用**本地 Qwen 3.5-9B** (vLLM 在 GPU 3): `--provider local`. 单轮 100-150 秒, $0 成本, 但稳定.
- 想用**真 DeepSeek 官方 API**: `--provider deepseek` (在日本不可达, 需 VPN).
- 想跑**批量 benchmark**: `python scripts/benchmark.py --provider siliconflow`
- 想看**单 run 的延迟 flame**: `python scripts/profile_run.py runs/r_XXX`

---

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

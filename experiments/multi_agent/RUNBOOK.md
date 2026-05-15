# Multi-Agent Legal RAG 操作手册

按 spec `/home/xxm/rag/docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md` 实施。

## 前置依赖

- Qwen 3.5 9B 在 vLLM (GPU 3, 端口 8000) — `scripts/serve_qwen_vllm.sh`
- bge-m3 在本地 (GPU 1) — 由 `DenseEncoder` 自动加载
- legal-rag-qdrant Docker 容器在端口 6433
- 索引集合: `ma_statutes`, `ma_cases`, `ma_user_history` (按需 build)

## 快速验证

```bash
conda run -n qwen35 pytest tests/unit/ -q             # ~5min, 应该全绿
conda run -n qwen35 pytest tests/integration/ -v     # ~10-15min real-Qwen 全套
```

## Scripts

| 脚本 | 用途 |
|---|---|
| `scripts/serve_qwen_vllm.sh` | 启动 Qwen vLLM (含 tool-call parser) |
| `scripts/build_statutes_index.py` | 把 Chinese-Laws/ 灌进 ma_statutes |
| `scripts/build_cases_index.py` | 把 laws_data Q&A 灌进 ma_cases |
| `scripts/extract_case_citations.py` | LLM 抽 lawyer 答复里的 law_id 引用 (Phase 2d) |
| `scripts/run_eval.py` | 单 profile 跑 QuerySet → 写 RunGroup + summary.md |
| `scripts/run_comparison.py` | 两个 profile 跑同一 QuerySet → comparison.md |
| `scripts/profile_run.py` | 单 run_dir 的 LatencyProfiler flame |

## 典型工作流

### 1. 跑评测 (baseline Qwen)

```bash
python scripts/run_eval.py \
    --queryset evals/querysets/synthetic_seed_v1.yaml \
    --statutes-collection ma_statutes \
    --statutes-sparse data/indexes/statutes_sparse.json \
    --runs-root runs \
    --group-name qwen_baseline_$(date +%Y%m%d_%H%M) \
    --max-queries 4
```

输出 `runs/run_groups/qwen_baseline_*/{results.jsonl, summary.md}`。

### 2. 跑对比 (Qwen vs Claude, 需 ANTHROPIC_API_KEY)

```bash
export ANTHROPIC_API_KEY=...
python scripts/run_comparison.py \
    --queryset evals/querysets/synthetic_seed_v1.yaml \
    --statutes-collection ma_statutes \
    --statutes-sparse data/indexes/statutes_sparse.json \
    --runs-root runs \
    --group-a-name qwen_baseline --profile-a all-local \
    --group-b-name claude_baseline --profile-b all-claude \
    --judges
```

输出 `runs/comparison_reports/qwen_baseline_vs_claude_baseline.md` 含每查询 Winner。

### 3. 分析某个 run 的延迟分布

```bash
python scripts/profile_run.py runs/r_01KRJWRY...
```

打印缩进 flame + by-agent / by-tool / by-provider 聚合。

### 4. 跑 Ablation 实验

在 Python (或扩展 `scripts/`):

```python
from multi_agent.eval.ablations import DisableTool, SwapModel
from multi_agent.eval.ablation_runner import AblationRunner
ar = AblationRunner(query_set=qs, runs_root=Path("runs"),
                    query_runner_factory=factory, run_group_base="abl_v1")
report = await ar.run(ablations=[
    DisableTool(tool="statute_search"),
    DisableTool(tool="case_search"),
    SwapModel(agent="lawyer", provider="anthropic", model="claude-sonnet-4-6"),
])
```

输出 `runs/run_groups/abl_v1/ablation_summary.md` 含 baseline vs N ablation 的 Δp50/Δtokens/Δcite-hits。

## 数据流

```
run_query(query, agent_factory, ..., session_id, memory_store, turn_indexer,
          compaction_provider, compaction_model)
  └→ recorder writes events.jsonl + RunFinished
  └→ artifacts/working_memory.json (Phase 5r)
  └→ memory_store.append_turn (MD)
  └→ turn_indexer.index_turn (Qdrant ma_user_history)
  └→ maybe_compact (>5 turns → history_summary)
  └→ returns {run_id, status, final_answer, evidence_pool}

ExperimentRunner(query_set, query_runner, judges=[...])
  └→ asyncio.gather over queries
  └→ derive_run_metrics(run_dir) — Phase 5b/5f/5q
  └→ CitationAccuracyJudge.judge(query, lawyer_output) — Phase 5b
  └→ judges.judge(query, lawyer_output, evidence_pool) — Phase 5c
  └→ results.jsonl + group_meta.yaml

Comparator.compare(group_a, group_b)
  └→ Δlat, Δtokens, Δcost, Δgrounded, Δhelpful, Winner — Phase 5h
```

## 关键 Trace 事件

| event_type | 含义 |
|---|---|
| `RunStarted` / `RunFinished` | 顶层 run 边界 |
| `AgentInvoked` / `AgentResponded` | 一个 agent ReAct 完整周期 |
| `LLMRequested` / `LLMResponded` | 单次 LLM 调用 (含 usage tokens) |
| `ToolCalled` / `ToolReturned` | 单次工具调用 (含 duration_ms / error) |
| `MemoryRead` / `MemoryWritten` | 记忆访问 |
| `SupervisorVerdict` | Supervisor 的审核结果 |

## Phase Tag 顺序

phase1 → phase2a → 2b → 2c → 2d → phase3 → 3b → 3c → 3d → 3e → 3f → phase4 → phase5a → 5b → 5c → 5d → 5e → 5f → 5g → 5h → 5i → 5j → 5k → 5l → 5m → 5o → 5p → 5q → 5r

每个 tag 对应可独立运行的里程碑。

## 已知边界

- 没补充 corpus 法律 (劳动合同法 / 刑法 / 公司法 / 仲裁法仍缺)
- 单用户假设 (没有 user_id 字段，session_id 即用户)
- WorkingMemory 在 multi-agent 间共享 (Lawyer → Secretary) 但通过 AgentInput payload, 非 ContextVar
- LLM judges (Groundedness/Helpfulness) 仅在 ANTHROPIC_API_KEY 存在时启用
- Streamlit Trace Viewer 未实现 (CLI 工具 `profile_run.py` 替代)

## 故障排查

- **vLLM 不响应**: `nvidia-smi` 看 GPU 3 占用，必要时 `pkill -f vllm` 重启
- **Qdrant 不通**: `docker ps | grep legal-rag-qdrant`,重启容器
- **Lawyer 编造引用**: 看 `events.jsonl` 是否有 `ToolCalled.tool_name=statute_search`; tool-first enforcement 应阻止此情况
- **Qwen 失败模式**: 看 lawyer 答复中 mode 字段; `final_answer_mode` metric 已 populated (Phase 5q)

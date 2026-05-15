# Phase 5f — cost_usd Derivation

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Add per-run USD cost to `RunMetrics`, derived from token counts × model pricing. Surfaces dollar cost in `summary.md`. Makes the headline Qwen-vs-Claude experiment monetarily meaningful.

**Phase 3e starting point:** Tag `phase3e-history-wiring`. 232 unit tests + 1 skipped + integrations.

---

## Pricing approach

A small dict in `multi_agent/eval/pricing.py`:

```python
MODEL_PRICING_PER_M_TOKENS = {
    # Anthropic published prices (USD per million tokens)
    "claude-opus-4-7":     {"input": 15.00, "output": 75.00, "cache_read": 1.50},
    "claude-sonnet-4-6":   {"input":  3.00, "output": 15.00, "cache_read": 0.30},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00, "cache_read": 0.08},
    # Local self-hosted — zero marginal cost
    "qwen3.5-9b":          {"input":  0.00, "output":  0.00, "cache_read": 0.00},
}
```

Unknown models → cost contribution of `0.0`. We surface a warning later if needed.

---

## Out of scope

- Per-event cost in trace (would need schema field; keep eval-side only)
- Cache-creation cost (Anthropic prompt-cache write is a separate price; deferred — input price is already a conservative upper bound)
- Streamlit cost viz

---

## Single Task

**Files:**
- Create: `multi_agent/eval/pricing.py`
- Modify: `multi_agent/eval/metrics.py` — derive `cost_usd` per-model, attribute LLMResponded tokens to its parent LLMRequested's model
- Modify: `multi_agent/eval/report.py` — include `total_cost_usd` in summary
- Create: `tests/unit/test_pricing.py` — 2 tests
- Modify: `tests/unit/test_metrics.py` — add 1 cost test
- Modify: `tests/unit/test_report.py` — keep passing (may need a tiny tweak if assertions check exact line layout)

### Step 1: Failing tests

```python
# tests/unit/test_pricing.py
import pytest
from multi_agent.eval.pricing import compute_cost_usd, MODEL_PRICING_PER_M_TOKENS


def test_claude_opus_cost_computation():
    cost = compute_cost_usd(
        model="claude-opus-4-7",
        input_tokens=1_000_000, output_tokens=500_000, cache_read_tokens=200_000,
    )
    # 1M * $15 = $15 input; 0.5M * $75 = $37.50 output; cache_read 0.2M * $1.50 = $0.30
    # NOTE: input_tokens in Anthropic counts cache-read tokens separately at cache price,
    # so "non-cached" input = total - cache_read = 800k → 0.8 * $15 = $12
    # Total = 12 + 37.50 + 0.30 = 49.80
    assert cost == pytest.approx(49.80, rel=0.01)


def test_qwen_local_zero_cost():
    cost = compute_cost_usd(
        model="qwen3.5-9b",
        input_tokens=10000, output_tokens=2000, cache_read_tokens=0,
    )
    assert cost == 0.0


def test_unknown_model_zero_cost():
    cost = compute_cost_usd(
        model="future-mystery-model", input_tokens=1_000, output_tokens=500,
    )
    assert cost == 0.0
```

```python
# additional test in tests/unit/test_metrics.py — append at the bottom
def test_derive_metrics_computes_cost_per_model(tmp_path):
    """Token cost attributed to the model from the parent LLMRequested event."""
    run_dir = tmp_path / "run-cost"
    run_dir.mkdir()
    events = [
        {"event_id": "1", "event_type": "RunStarted",
         "timestamp": "2026-05-15T00:00:00", "run_id": "x", "parent_id": None,
         "query": "q", "config": {}},
        {"event_id": "L1", "event_type": "LLMRequested",
         "timestamp": "2026-05-15T00:00:00", "run_id": "x", "parent_id": "1",
         "provider": "anthropic", "model": "claude-opus-4-7",
         "messages": [], "params": {}},
        {"event_id": "L1e", "event_type": "LLMResponded",
         "timestamp": "2026-05-15T00:00:01", "run_id": "x", "parent_id": "L1",
         "raw_response": "", "duration_ms": 1000, "finish_reason": "end_turn",
         "usage": {"input_tokens": 100000, "output_tokens": 50000, "cache_read_tokens": 0}},
        {"event_id": "2", "event_type": "RunFinished",
         "timestamp": "2026-05-15T00:00:02", "run_id": "x", "parent_id": "1",
         "status": "ok", "final_answer": None, "error": None},
    ]
    import json
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    from multi_agent.eval.metrics import derive_run_metrics
    m = derive_run_metrics(run_dir)
    # 100k tokens input × $15/M + 50k output × $75/M = 1.50 + 3.75 = 5.25
    assert m.cost_usd == pytest.approx(5.25, rel=0.01)
```

### Step 2: Implement `pricing.py`

```python
"""Model pricing for cost_usd derivation (Phase 5f)."""
from __future__ import annotations


MODEL_PRICING_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "claude-opus-4-7":           {"input": 15.00, "output": 75.00, "cache_read": 1.50},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00, "cache_read": 0.30},
    "claude-haiku-4-5-20251001": {"input":  0.80, "output":  4.00, "cache_read": 0.08},
    "qwen3.5-9b":                {"input":  0.00, "output":  0.00, "cache_read": 0.00},
}


def compute_cost_usd(
    *, model: str, input_tokens: int = 0, output_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    prices = MODEL_PRICING_PER_M_TOKENS.get(model)
    if prices is None:
        return 0.0
    # input_tokens is the TOTAL input the request consumed; cache_read_tokens is a
    # subset of those (Anthropic and OpenAI both account this way). Non-cached
    # input = input_tokens - cache_read_tokens.
    non_cached = max(0, input_tokens - cache_read_tokens)
    cost = (
        (non_cached / 1_000_000) * prices["input"]
        + (output_tokens / 1_000_000) * prices["output"]
        + (cache_read_tokens / 1_000_000) * prices["cache_read"]
    )
    return round(cost, 6)
```

### Step 3: Modify `metrics.py`

- Add `cost_usd: float = 0.0` field to `RunMetrics`.
- During the loop, build `model_by_request_id: dict[str, str]` mapping LLMRequested.event_id → model.
- When LLMResponded arrives, look up its `parent_id` to find the model. Add to a `cost_by_model: dict[str, dict]` accumulator, then sum at the end with `compute_cost_usd`.

Sketch (only the additions):

```python
# After class RunMetrics:
class RunMetrics(BaseModel):
    # ... existing fields
    cost_usd: float = 0.0

# Inside derive_run_metrics():
from multi_agent.eval.pricing import compute_cost_usd
model_by_request_id: dict[str, str] = {}
per_model_usage: dict[str, dict[str, int]] = {}  # model -> {"input","output","cache_read"}

# In the loop:
elif event_type == "LLMRequested":
    rid = e.get("event_id")
    model = e.get("model", "")
    if rid and model:
        model_by_request_id[rid] = model

elif event_type == "LLMResponded":
    usage = e.get("usage") or {}
    # ... existing aggregation
    parent = e.get("parent_id")
    model = model_by_request_id.get(parent, "")
    if model:
        bucket = per_model_usage.setdefault(model, {"input":0,"output":0,"cache_read":0})
        bucket["input"] += usage.get("input_tokens", 0) or 0
        bucket["output"] += usage.get("output_tokens", 0) or 0
        bucket["cache_read"] += usage.get("cache_read_tokens", 0) or 0

# After the loop:
total_cost = 0.0
for model, u in per_model_usage.items():
    total_cost += compute_cost_usd(
        model=model,
        input_tokens=u["input"], output_tokens=u["output"],
        cache_read_tokens=u["cache_read"],
    )
m.cost_usd = round(total_cost, 6)
```

### Step 4: Modify `report.py`

Add one line to the summary: `- 总成本 cost: $X.XX` after the tokens line. Sum `cost_usd` across all OK rows.

### Step 5: Verify

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_pricing.py tests/unit/test_metrics.py tests/unit/test_report.py -v"
```

Expected: all green (3 pricing + previous metrics + new metric test + report unchanged).

### Step 6: Commit + tag

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/pricing.py \
        experiments/multi_agent/multi_agent/eval/metrics.py \
        experiments/multi_agent/multi_agent/eval/report.py \
        experiments/multi_agent/tests/unit/test_pricing.py \
        experiments/multi_agent/tests/unit/test_metrics.py
git commit -m "phase5f(eval): cost_usd derivation from model pricing × tokens"
git tag -a phase5f-cost -m "Phase 5f: per-run USD cost derived from token usage and model pricing"
git tag -l "phase*"
```

---

## Acceptance Criteria

1. `pricing.py` tests pass (claude-opus arithmetic, qwen zero, unknown model zero)
2. `derive_run_metrics` populates `cost_usd` field correctly
3. `summary.md` includes total cost line
4. No regressions in existing metric/report tests
5. Tag `phase5f-cost` exists

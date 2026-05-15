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

import pytest
from multi_agent.eval.pricing import compute_cost_usd, MODEL_PRICING_PER_M_TOKENS


def test_claude_opus_cost_computation():
    cost = compute_cost_usd(
        model="claude-opus-4-7",
        input_tokens=1_000_000, output_tokens=500_000, cache_read_tokens=200_000,
    )
    # input_tokens in Anthropic counts cache-read tokens separately at cache price,
    # so "non-cached" input = total - cache_read = 800k → 0.8 * $15 = $12
    # output: 0.5M * $75 = $37.50
    # cache_read: 0.2M * $1.50 = $0.30
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

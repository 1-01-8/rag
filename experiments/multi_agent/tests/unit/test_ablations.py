import pytest
from multi_agent.eval.ablations import (
    Ablation, DisableAgent, SwapModel, DisableTool, DisableMemory, apply_ablation,
)


def test_disable_tool_writes_into_config():
    cfg = {}
    ab = DisableTool(tool="case_search")
    apply_ablation(cfg, ab)
    assert "case_search" in cfg.get("disabled_tools", set())


def test_swap_model_writes_provider_and_model():
    cfg = {}
    ab = SwapModel(agent="lawyer", provider="anthropic", model="claude-opus-4-7")
    apply_ablation(cfg, ab)
    overrides = cfg.get("model_overrides", {})
    assert overrides["lawyer"]["provider"] == "anthropic"
    assert overrides["lawyer"]["model"] == "claude-opus-4-7"


def test_disable_agent():
    cfg = {}
    apply_ablation(cfg, DisableAgent(agent="supervisor"))
    assert "supervisor" in cfg.get("disabled_agents", set())


def test_disable_memory():
    cfg = {}
    apply_ablation(cfg, DisableMemory())
    assert cfg.get("disable_memory") is True


def test_ablation_name_for_reporting():
    assert DisableTool(tool="case_search").name == "disable_tool:case_search"
    assert SwapModel(agent="lawyer", provider="anthropic", model="claude-opus-4-7").name == "swap_model:lawyer→claude-opus-4-7"
    assert DisableAgent(agent="supervisor").name == "disable_agent:supervisor"
    assert DisableMemory().name == "disable_memory"


def test_disabled_tools_is_json_serializable():
    """Fix 3: disabled_tools/disabled_agents must be plain lists, not sets."""
    import json
    cfg = {}
    apply_ablation(cfg, DisableTool(tool="case_search"))
    apply_ablation(cfg, DisableAgent(agent="supervisor"))
    # Must not raise TypeError
    serialized = json.dumps(cfg)
    loaded = json.loads(serialized)
    assert "case_search" in loaded["disabled_tools"]
    assert "supervisor" in loaded["disabled_agents"]


def test_duplicate_disabled_tools_deduped():
    """Applying the same DisableTool twice must not duplicate the entry."""
    cfg = {}
    apply_ablation(cfg, DisableTool(tool="case_search"))
    apply_ablation(cfg, DisableTool(tool="case_search"))
    assert cfg["disabled_tools"].count("case_search") == 1

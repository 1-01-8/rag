import pytest
from multi_agent.providers.profile import (
    ProviderProfile, build_provider_for, PROFILES,
)
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.providers.anthropic import AnthropicProvider


def test_default_profiles_exist():
    """The 4 spec'd profiles are pre-defined."""
    assert "all-local" in PROFILES
    assert "all-claude" in PROFILES
    assert "mixed-cloud-judge" in PROFILES
    assert "mixed-cloud-brain" in PROFILES


def test_all_local_profile_uses_qwen_everywhere():
    p = PROFILES["all-local"]
    assert p.agent_to_provider["lawyer"] == ("openai_compat", "qwen3.5-9b")
    assert p.agent_to_provider["receptionist"] == ("openai_compat", "qwen3.5-9b")


def test_all_claude_profile_uses_anthropic_everywhere():
    p = PROFILES["all-claude"]
    assert p.agent_to_provider["lawyer"][0] == "anthropic"
    assert p.agent_to_provider["lawyer"][1].startswith("claude-")


def test_build_provider_for_local_profile():
    provider, model = build_provider_for("lawyer", profile_name="all-local")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert model == "qwen3.5-9b"


def test_build_provider_for_claude_profile():
    provider, model = build_provider_for("supervisor", profile_name="all-claude")
    assert isinstance(provider, AnthropicProvider)
    assert model.startswith("claude-")


def test_build_provider_for_unknown_agent_falls_back():
    """If agent isn't in profile.agent_to_provider, falls back to profile.default."""
    provider, model = build_provider_for("unknown_agent", profile_name="all-local")
    assert isinstance(provider, OpenAICompatibleProvider)


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        build_provider_for("lawyer", profile_name="nonexistent")

"""Provider profile registry: which agent uses which (provider, model).

Defines the 4 spec'd profiles (all-local / all-claude / mixed-cloud-judge /
mixed-cloud-brain). New profiles can be added by extending PROFILES.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from multi_agent.providers.base import LLMProvider
from multi_agent.providers.anthropic import AnthropicProvider
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider


@dataclass(frozen=True)
class ProviderProfile:
    """Maps agent role → (provider_name, model_name).

    `default` is used when an agent role isn't explicitly listed.
    """
    name: str
    agent_to_provider: dict[str, tuple[str, str]]
    default: tuple[str, str] = ("openai_compat", "qwen3.5-9b")


PROFILES: dict[str, ProviderProfile] = {
    "all-local": ProviderProfile(
        name="all-local",
        agent_to_provider={
            "receptionist": ("openai_compat", "qwen3.5-9b"),
            "lawyer":       ("openai_compat", "qwen3.5-9b"),
            "secretary":    ("openai_compat", "qwen3.5-9b"),
            "supervisor":   ("openai_compat", "qwen3.5-9b"),
        },
    ),
    "all-claude": ProviderProfile(
        name="all-claude",
        agent_to_provider={
            "receptionist": ("anthropic", "claude-haiku-4-5-20251001"),
            "lawyer":       ("anthropic", "claude-sonnet-4-6"),
            "secretary":    ("anthropic", "claude-sonnet-4-6"),
            "supervisor":   ("anthropic", "claude-haiku-4-5-20251001"),
        },
        default=("anthropic", "claude-sonnet-4-6"),
    ),
    "mixed-cloud-judge": ProviderProfile(
        name="mixed-cloud-judge",
        agent_to_provider={
            "receptionist": ("openai_compat", "qwen3.5-9b"),
            "lawyer":       ("openai_compat", "qwen3.5-9b"),
            "secretary":    ("openai_compat", "qwen3.5-9b"),
            "supervisor":   ("anthropic", "claude-haiku-4-5-20251001"),
        },
    ),
    "mixed-cloud-brain": ProviderProfile(
        name="mixed-cloud-brain",
        agent_to_provider={
            "receptionist": ("openai_compat", "qwen3.5-9b"),
            "lawyer":       ("anthropic", "claude-sonnet-4-6"),
            "secretary":    ("openai_compat", "qwen3.5-9b"),
            "supervisor":   ("anthropic", "claude-haiku-4-5-20251001"),
        },
    ),
}


# Singletons keyed by provider name — avoid creating multiple clients per profile-resolve
_provider_singletons: dict[str, LLMProvider] = {}


def _get_singleton(provider_name: str) -> LLMProvider:
    if provider_name not in _provider_singletons:
        if provider_name == "openai_compat":
            _provider_singletons[provider_name] = OpenAICompatibleProvider()
        elif provider_name == "anthropic":
            _provider_singletons[provider_name] = AnthropicProvider()
        else:
            raise ValueError(f"unknown provider: {provider_name}")
    return _provider_singletons[provider_name]


def build_provider_for(
    agent_name: str, *, profile_name: str = "all-local",
) -> tuple[LLMProvider, str]:
    """Return (provider, model) for the agent_name in the named profile.

    Falls back to profile.default if agent_name not explicitly mapped.
    Raises KeyError if profile_name is unknown.
    """
    if profile_name not in PROFILES:
        raise KeyError(f"unknown profile: {profile_name}. Choices: {list(PROFILES)}")
    profile = PROFILES[profile_name]
    provider_name, model = profile.agent_to_provider.get(agent_name, profile.default)
    return _get_singleton(provider_name), model

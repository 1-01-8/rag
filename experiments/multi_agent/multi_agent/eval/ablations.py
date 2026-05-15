"""Ablation primitives (Phase 5d §7.9)."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class Ablation(BaseModel):
    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()


class DisableAgent(Ablation):
    agent: str

    @property
    def name(self) -> str:
        return f"disable_agent:{self.agent}"


class SwapModel(Ablation):
    agent: str
    provider: str
    model: str

    @property
    def name(self) -> str:
        return f"swap_model:{self.agent}→{self.model}"


class DisableTool(Ablation):
    tool: str

    @property
    def name(self) -> str:
        return f"disable_tool:{self.tool}"


class DisableMemory(Ablation):
    @property
    def name(self) -> str:
        return "disable_memory"


def apply_ablation(config: dict[str, Any], ablation: Ablation) -> None:
    """Mutate `config` in place to express `ablation`.

    disabled_tools / disabled_agents are stored as plain lists (not sets) so
    that json.dumps(config) works without going through a Pydantic model.
    """
    if isinstance(ablation, DisableAgent):
        lst: list = config.setdefault("disabled_agents", [])
        if ablation.agent not in lst:
            lst.append(ablation.agent)
    elif isinstance(ablation, SwapModel):
        config.setdefault("model_overrides", {})[ablation.agent] = {
            "provider": ablation.provider,
            "model": ablation.model,
        }
    elif isinstance(ablation, DisableTool):
        lst = config.setdefault("disabled_tools", [])
        if ablation.tool not in lst:
            lst.append(ablation.tool)
    elif isinstance(ablation, DisableMemory):
        config["disable_memory"] = True
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

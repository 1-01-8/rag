"""LLMJudge base class (Phase 5c §7.7).

Signature note: LLMProvider.complete() takes:
  - messages: list[AgentMessage]  (system message goes as first AgentMessage role="system")
  - model, tools, response_format, max_tokens, temperature  (kwargs)
  - recorder: Recorder  (required kwarg)
  - agent_name: str     (required kwarg)
Returns LLMResponse with .text: str.
"""
from __future__ import annotations
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from multi_agent.providers.base import LLMProvider
from multi_agent.providers.json_robust import parse_json_robust
from multi_agent.schemas.messages import AgentMessage
from multi_agent.tracing.recorder import Recorder

T = TypeVar("T", bound=BaseModel)


class JudgeResult(BaseModel):
    judge: str
    score: float = 0.0
    parsed: BaseModel | None = None
    raw: str = ""
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class LLMJudge(ABC, Generic[T]):
    name: str = "base"
    output_schema: type[T]
    system_prompt: str = "You are an evaluation judge. Output only JSON."
    temperature: float = 0.0
    max_tokens: int = 1024

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        judge_run_dir: Path | None = None,
    ):
        self.provider = provider
        self.model = model
        self._judge_run_dir = judge_run_dir

    @abstractmethod
    def render_prompt(
        self, *, query: str, lawyer_output: dict, evidence_pool: list[dict]
    ) -> str: ...

    async def judge(
        self,
        *,
        query: str,
        lawyer_output: dict,
        evidence_pool: list[dict],
    ) -> JudgeResult:
        user_text = self.render_prompt(
            query=query, lawyer_output=lawyer_output, evidence_pool=evidence_pool
        )
        messages = [
            AgentMessage(role="system", content=self.system_prompt),
            AgentMessage(role="user", content=user_text),
        ]

        # Create a throwaway recorder for this judge call.
        if self._judge_run_dir is not None:
            run_dir = Path(self._judge_run_dir)
        else:
            # Fall back to a temp directory so judges work without explicit dir.
            run_dir = Path(tempfile.mkdtemp(prefix="judge_run_"))

        recorder = Recorder(run_id=f"judge-{self.name}", run_dir=run_dir)
        try:
            resp = await self.provider.complete(
                messages,
                model=self.model,
                tools=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                recorder=recorder,
                agent_name=f"judge_{self.name}",
            )
            raw = (resp.text or "").strip()
            parsed_dict = parse_json_robust(raw)
            parsed = self.output_schema.model_validate(parsed_dict)
            score = float(getattr(parsed, "score", 0.0))
            return JudgeResult(judge=self.name, score=score, parsed=parsed, raw=raw)
        except Exception as e:
            return JudgeResult(
                judge=self.name,
                score=0.0,
                raw="",
                error=f"{type(e).__name__}: {e}",
            )
        finally:
            recorder.close()

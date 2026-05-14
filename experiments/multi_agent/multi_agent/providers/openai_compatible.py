"""OpenAI-compatible provider — used for local Qwen via vLLM,
or for OpenAI proper / any other OpenAI-compat service (DeepSeek, etc.)
by swapping base_url + api_key + model.
"""
from __future__ import annotations
import json
import os
from typing import Any, AsyncGenerator
from openai import AsyncOpenAI

from multi_agent.providers.base import (
    LLMProvider, LLMResponse, StreamChunk, ToolSpec, Usage,
)
from multi_agent.providers.json_robust import parse_json_robust
from multi_agent.schemas.messages import AgentMessage, ToolCallRequest
from multi_agent.tracing.recorder import Recorder
from multi_agent.errors import ProviderUnavailable


class OpenAICompatibleProvider(LLMProvider):
    """Talks to any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str = "qwen3.5-9b",
        timeout: float = 120.0,
    ):
        self.base_url = base_url or os.environ.get(
            "OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY", "dummy")
        self.default_model = default_model
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
        )

    def _to_oai_messages(self, messages: list[AgentMessage]) -> list[dict]:
        """Convert internal AgentMessage to OpenAI chat format."""
        out: list[dict] = []
        for m in messages:
            if m.role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": m.tool_use_id or "",
                    "content": m.content,
                })
                continue
            entry: dict = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.tool_use_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.args, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        return out

    def _to_oai_tools(self, tools: list[ToolSpec] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def _normalize_finish_reason(self, raw: str | None) -> str:
        mapping = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "refusal",
        }
        return mapping.get(raw or "stop", "end_turn")

    async def complete(
        self,
        messages: list[AgentMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: type | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        recorder: Recorder,
        agent_name: str,
    ) -> LLMResponse:
        oai_messages = self._to_oai_messages(messages)
        oai_tools = self._to_oai_tools(tools)

        with recorder.span(
            "llm_call",
            provider="openai_compat",
            model=model,
            agent_name=agent_name,
            messages=oai_messages,
            params={"max_tokens": max_tokens, "temperature": temperature},
        ) as span:
            try:
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=oai_messages,
                    tools=oai_tools or None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                raise ProviderUnavailable(
                    f"OpenAI-compat at {self.base_url} failed: {e}"
                ) from e

            choice = resp.choices[0]
            text = choice.message.content or ""
            raw_tool_calls = choice.message.tool_calls or []
            tool_calls: list[ToolCallRequest] = []
            for tc in raw_tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = parse_json_robust(tc.function.arguments or "{}")
                tool_calls.append(ToolCallRequest(
                    tool_use_id=tc.id,
                    tool_name=tc.function.name,
                    args=args,
                ))

            usage = resp.usage
            llm_resp = LLMResponse(
                text=text,
                tool_calls=tool_calls,
                usage=Usage(
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                ),
                raw={"model": resp.model, "id": resp.id},
                duration_ms=0,
                finish_reason=self._normalize_finish_reason(choice.finish_reason),
            )
            span.set_output({
                "raw": text,
                "usage": llm_resp.usage.model_dump(),
                "finish_reason": llm_resp.finish_reason,
            })
            return llm_resp

    async def complete_stream(
        self, messages, *, model, tools=None,
        max_tokens=4096, temperature=0.0, recorder: Recorder, agent_name: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        oai_messages = self._to_oai_messages(messages)
        oai_tools = self._to_oai_tools(tools)
        with recorder.span(
            "llm_call", provider="openai_compat", model=model, agent_name=agent_name,
            messages=oai_messages,
            params={"max_tokens": max_tokens, "temperature": temperature, "stream": True},
        ) as span:
            try:
                stream = await self._client.chat.completions.create(
                    model=model, messages=oai_messages, tools=oai_tools or None,
                    max_tokens=max_tokens, temperature=temperature, stream=True,
                )
            except Exception as e:
                raise ProviderUnavailable(f"OpenAI-compat stream failed: {e}") from e

            full_text = ""
            async for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta.content:
                    full_text += delta.content
                    yield StreamChunk(kind="token", content=delta.content)
                if event.choices[0].finish_reason is not None:
                    break
            span.set_output({"raw": full_text, "usage": {}, "finish_reason": "end_turn"})
        yield StreamChunk(kind="end_turn")

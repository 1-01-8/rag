"""Anthropic Claude provider — full message API including tool use.

Prompt caching (cache_control) lands in Task 5.
Streaming lands in Task 7.
"""
from __future__ import annotations
import os
from typing import Any, AsyncGenerator
from anthropic import AsyncAnthropic

from multi_agent.providers.base import (
    LLMProvider, LLMResponse, StreamChunk, ToolSpec, Usage,
)
from multi_agent.schemas.messages import AgentMessage, ToolCallRequest
from multi_agent.tracing.recorder import Recorder
from multi_agent.errors import ProviderUnavailable


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-6",
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.default_model = default_model
        self._client = AsyncAnthropic(api_key=self.api_key, timeout=timeout)

    def _split_system_and_messages(
        self, messages: list[AgentMessage]
    ) -> tuple[str, list[dict]]:
        """Anthropic API takes system separately and accepts only user/assistant/tool messages."""
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            if m.role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_use_id or "",
                        "content": m.content,
                    }],
                })
                continue
            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.tool_use_id,
                        "name": tc.tool_name,
                        "input": tc.args,
                    })
                out.append({"role": "assistant", "content": blocks})
                continue
            out.append({"role": m.role, "content": m.content})
        return "\n\n".join(system_parts), out

    def _to_anthropic_tools(self, tools: list[ToolSpec] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def _normalize_stop_reason(self, raw: str | None) -> str:
        mapping = {
            "end_turn": "end_turn",
            "tool_use": "tool_use",
            "max_tokens": "max_tokens",
            "stop_sequence": "end_turn",
            "refusal": "refusal",
        }
        return mapping.get(raw or "end_turn", "end_turn")

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
        system, anthropic_messages = self._split_system_and_messages(messages)
        anthropic_tools = self._to_anthropic_tools(tools)

        with recorder.span(
            "llm_call",
            provider="anthropic",
            model=model,
            agent_name=agent_name,
            messages=anthropic_messages,
            params={"max_tokens": max_tokens, "temperature": temperature, "system": system},
        ) as span:
            cache_friendly_system = None
            if system:
                cache_friendly_system = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            try:
                msg = await self._client.messages.create(
                    model=model,
                    system=cache_friendly_system,
                    messages=anthropic_messages,
                    tools=anthropic_tools or None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                raise ProviderUnavailable(f"Anthropic API failed: {e}") from e

            text_parts: list[str] = []
            tool_calls: list[ToolCallRequest] = []
            for block in msg.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text_parts.append(getattr(block, "text", ""))
                elif btype == "tool_use":
                    tool_calls.append(ToolCallRequest(
                        tool_use_id=getattr(block, "id", ""),
                        tool_name=getattr(block, "name", ""),
                        args=getattr(block, "input", {}) or {},
                    ))

            usage = getattr(msg, "usage", None)
            llm_resp = LLMResponse(
                text="".join(text_parts),
                tool_calls=tool_calls,
                usage=Usage(
                    input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                    output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                    cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) if usage else 0,
                    cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) if usage else 0,
                ),
                raw={"model": msg.model, "id": msg.id},
                duration_ms=0,
                finish_reason=self._normalize_stop_reason(msg.stop_reason),
            )
            span.set_output({
                "raw": llm_resp.text,
                "usage": llm_resp.usage.model_dump(),
                "finish_reason": llm_resp.finish_reason,
            })
            return llm_resp

    async def complete_stream(
        self, messages, *, model, tools=None,
        max_tokens=4096, temperature=0.0, recorder: Recorder, agent_name: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        system, anthropic_messages = self._split_system_and_messages(messages)
        anthropic_tools = self._to_anthropic_tools(tools)
        cache_friendly_system = None
        if system:
            cache_friendly_system = [{
                "type": "text", "text": system,
                "cache_control": {"type": "ephemeral"},
            }]

        with recorder.span(
            "llm_call", provider="anthropic", model=model, agent_name=agent_name,
            messages=anthropic_messages,
            params={"max_tokens": max_tokens, "temperature": temperature, "stream": True},
        ) as span:
            full_text = ""
            usage_in = 0
            usage_out = 0
            stop_reason = "end_turn"
            try:
                async with self._client.messages.stream(
                    model=model,
                    system=cache_friendly_system,
                    messages=anthropic_messages,
                    tools=anthropic_tools or None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ) as stream:
                    async for event in stream:
                        et = getattr(event, "type", None)
                        if et == "message_start":
                            msg = getattr(event, "message", None)
                            u = getattr(msg, "usage", None) if msg else None
                            if u:
                                usage_in = getattr(u, "input_tokens", 0) or 0
                        elif et == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            dt = getattr(delta, "type", None) if delta else None
                            if dt == "text_delta":
                                text = getattr(delta, "text", "") or ""
                                if text:
                                    full_text += text
                                    yield StreamChunk(kind="token", content=text)
                        elif et == "message_delta":
                            delta = getattr(event, "delta", None)
                            sr = getattr(delta, "stop_reason", None) if delta else None
                            if sr:
                                stop_reason = self._normalize_stop_reason(sr)
                            u = getattr(event, "usage", None)
                            if u:
                                usage_out = getattr(u, "output_tokens", 0) or 0
            except Exception as e:
                raise ProviderUnavailable(f"Anthropic stream failed: {e}") from e
            span.set_output({
                "raw": full_text,
                "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
                "finish_reason": stop_reason,
            })
        yield StreamChunk(kind="end_turn")

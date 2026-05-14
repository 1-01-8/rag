"""Centralized exception types. All raised from agents/providers/tools."""


class MultiAgentError(Exception):
    """Base for all package errors."""


class ProviderUnavailable(MultiAgentError):
    """LLM provider unreachable or auth failed."""


class ResponseValidationError(MultiAgentError):
    """LLM response failed schema validation after retries."""

    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


class ToolCallParseError(MultiAgentError):
    """Tool args from LLM did not match args_schema."""


class BudgetExceeded(MultiAgentError):
    """Agent exceeded max_steps / max_total_tokens / max_tool_calls."""

    def __init__(self, agent_name: str, budget: str, limit: int):
        super().__init__(f"{agent_name} exceeded {budget}={limit}")
        self.agent_name = agent_name
        self.budget = budget
        self.limit = limit


class AgentTimeout(MultiAgentError):
    """Agent wall-clock exceeded."""


class MemoryReadError(MultiAgentError):
    """memory_store file read/parse failure."""


class MemoryWriteError(MultiAgentError):
    """memory_store write failed."""

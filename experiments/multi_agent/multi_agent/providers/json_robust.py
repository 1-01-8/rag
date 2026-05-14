from __future__ import annotations
import json
from multi_agent.errors import ResponseValidationError


def parse_json_robust(raw: str) -> dict:
    """Tolerant JSON parsing for LLM output.

    Strategy:
      1. strip leading/trailing whitespace
      2. remove ```json ... ``` or ``` ... ``` fences
      3. locate outermost { ... }
      4. json.loads
    Raises ResponseValidationError with .raw on failure.
    """
    if not raw or not raw.strip():
        raise ResponseValidationError("empty response", raw=raw)

    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")]
    cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ResponseValidationError(f"JSON parse failed: {e}", raw=raw)

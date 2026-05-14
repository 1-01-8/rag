import pytest
from multi_agent.providers.json_robust import parse_json_robust
from multi_agent.errors import ResponseValidationError


def test_plain_json():
    assert parse_json_robust('{"a": 1}') == {"a": 1}


def test_strips_fenced_json():
    raw = '```json\n{"a": 1}\n```'
    assert parse_json_robust(raw) == {"a": 1}


def test_strips_generic_fence():
    raw = '```\n{"a": 1}\n```'
    assert parse_json_robust(raw) == {"a": 1}


def test_locates_json_in_prose():
    raw = "Here is the answer:\n{\"a\": 1}\nThanks!"
    assert parse_json_robust(raw) == {"a": 1}


def test_invalid_json_raises_with_raw():
    with pytest.raises(ResponseValidationError) as exc:
        parse_json_robust("not json at all")
    assert exc.value.raw == "not json at all"


def test_empty_raises():
    with pytest.raises(ResponseValidationError):
        parse_json_robust("")

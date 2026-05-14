from multi_agent.schemas.state import RunState


def test_run_state_defaults():
    s = RunState(
        run_id="r1",
        session_id="s1",
        user_query="test query",
    )
    assert s.run_id == "r1"
    assert s.history_messages == []
    assert s.failed_queries == []


def test_run_state_round_trip():
    s = RunState(
        run_id="r1",
        session_id="s1",
        user_query="q",
        history_messages=[{"role": "user", "content": "prev"}],
    )
    raw = s.model_dump()
    restored = RunState.model_validate(raw)
    assert restored.history_messages[0]["content"] == "prev"

import time
from multi_agent.tracing.ulid_gen import fresh_event_id, fresh_run_id


def test_event_id_is_26_chars():
    eid = fresh_event_id()
    assert isinstance(eid, str)
    assert len(eid) == 26  # ULID standard


def test_event_ids_monotonic():
    a = fresh_event_id()
    time.sleep(0.001)
    b = fresh_event_id()
    assert b > a  # ULIDs sort lexicographically by time


def test_run_id_has_prefix():
    rid = fresh_run_id()
    assert rid.startswith("r_")
    assert len(rid) == 28  # "r_" + 26

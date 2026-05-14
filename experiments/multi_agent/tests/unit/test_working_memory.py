from multi_agent.schemas.working_memory import WorkingMemory, Hypothesis
from multi_agent.schemas.evidence import Evidence


def _ev(doc_id="d1"):
    return Evidence(doc_id=doc_id, law_name="x", article_no="1",
                    text="t", score=0.5, retriever="hybrid")


def test_working_memory_starts_empty():
    wm = WorkingMemory()
    assert wm.retrieved_evidence == []
    assert wm.discarded_evidence == []
    assert wm.hypotheses == []


def test_add_evidence_appends():
    wm = WorkingMemory()
    wm.add_evidence(_ev("d1"))
    wm.add_evidence(_ev("d2"))
    assert {e.doc_id for e in wm.retrieved_evidence} == {"d1", "d2"}


def test_discard_records_reason():
    wm = WorkingMemory()
    e = _ev("d1")
    wm.discard(e, reason="not on-point")
    assert len(wm.discarded_evidence) == 1
    assert wm.discarded_evidence[0].reason == "not on-point"
    assert wm.discarded_evidence[0].evidence.doc_id == "d1"


def test_hypothesis_active_to_rejected():
    h = Hypothesis(
        statement="user can refuse rent hike",
        supporting_evidence=["d1"],
        confidence=0.7,
        status="active",
    )
    assert h.status == "active"
    h.status = "rejected"
    assert h.status == "rejected"

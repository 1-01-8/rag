import pytest
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


def test_lawyer_default_specialty_is_general(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    lawyer = LawyerAgent(name="lawyer", role="advisor",
                        provider=p, recorder=rec)
    prompt = lawyer.system_prompt()
    assert "五段式" in prompt
    assert "通用法律咨询" in prompt
    rec.close()


def test_lawyer_specialty_loads_correct_prompt(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    for sp in ["民事", "劳动", "交通", "婚姻", "房产"]:
        lawyer = LawyerAgent(name="lawyer", role="advisor",
                             provider=p, recorder=rec, specialty=sp)
        prompt = lawyer.system_prompt()
        assert "五段式" in prompt, f"{sp} missing skeleton"
        assert sp in prompt, f"{sp} prompt did not include specialty marker"
    rec.close()


def test_lawyer_output_schema_is_lawyer_output(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    lawyer = LawyerAgent(name="lawyer", role="advisor",
                        provider=p, recorder=rec)
    assert lawyer.output_schema() is LawyerOutput
    rec.close()


def test_lawyer_unknown_specialty_raises(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    with pytest.raises(ValueError, match="unknown specialty"):
        LawyerAgent(name="lawyer", role="advisor",
                    provider=p, recorder=rec, specialty="nonexistent")
    rec.close()

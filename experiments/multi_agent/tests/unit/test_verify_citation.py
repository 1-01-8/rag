import pytest
from multi_agent.tools.business.verify_citation import (
    VerifyCitationTool, VerifyCitationArgs,
)
from multi_agent.schemas.lawyer import Citation
from multi_agent.schemas.evidence import Evidence
from multi_agent.tracing.recorder import Recorder


def _ev(doc_id="民法典-510", text="当事人就合同补充内容..."):
    return Evidence(
        doc_id=doc_id,
        law_name="中华人民共和国民法典",
        law_short=doc_id.split("-")[0],
        article_no=doc_id.split("-")[1],
        text=text,
        score=0.9,
        retriever="hybrid",
    )


@pytest.mark.asyncio
async def test_verify_citation_matches(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    tool = VerifyCitationTool()
    result = await tool.call(
        VerifyCitationArgs(
            citation=Citation(law_short="民法典", article_no="510",
                             excerpt="合同补充内容"),
            evidences=[_ev()],
        ),
        rec,
    )
    rec.close()
    assert result.error is None
    assert result.payload["valid"] is True


@pytest.mark.asyncio
async def test_verify_citation_doc_id_missing(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    tool = VerifyCitationTool()
    result = await tool.call(
        VerifyCitationArgs(
            citation=Citation(law_short="民法典", article_no="999", excerpt=""),
            evidences=[_ev()],
        ),
        rec,
    )
    rec.close()
    assert result.payload["valid"] is False
    assert "not in retrieved evidence" in result.payload["reason"].lower()


@pytest.mark.asyncio
async def test_verify_citation_excerpt_mismatch(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    tool = VerifyCitationTool()
    result = await tool.call(
        VerifyCitationArgs(
            citation=Citation(law_short="民法典", article_no="510",
                             excerpt="this text not in evidence"),
            evidences=[_ev()],
        ),
        rec,
    )
    rec.close()
    assert result.payload["valid"] is False
    assert "excerpt" in result.payload["reason"].lower() or "not found" in result.payload["reason"].lower()

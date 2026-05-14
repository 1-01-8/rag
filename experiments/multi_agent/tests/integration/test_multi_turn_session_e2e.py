"""Phase 3b E2E: 2-turn session with memory persistence + Receptionist follow-up."""
import json
import uuid
import httpx
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.receptionist import ReceptionistOutput
from multi_agent.schemas.memory import StickyContext, EntityState, KeyFact
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.receptionist import ReceptionistAgent
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.agents.base import AgentInput
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.runner import run_query
from multi_agent.tracing.recorder import Recorder
from multi_agent.tracing.ulid_gen import fresh_run_id


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def statute_index(tmp_path_factory):
    name = f"test_multi_turn_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [Document(
        law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
        chunks=[
            Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                  law_short="民法典", article_no="510",
                  text="当事人就合同补充内容没有约定的,按照合同相关条款或者交易习惯确定。"),
            Chunk(doc_id="民法典-703", law_name="中华人民共和国民法典",
                  law_short="民法典", article_no="703",
                  text="租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。"),
        ],
    )]
    build_index(documents=docs, collection_name=name,
                sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder())
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_two_turn_session(statute_index, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)
    store_root = tmp_path / "memory_store"
    store = MarkdownMemoryStore(root=store_root)
    provider = OpenAICompatibleProvider()
    session_id = "s_phase3b_test"

    # Seed sticky context (Turn 0 setup)
    store.write_sticky(StickyContext(
        session_id=session_id,
        legal_domain="民事",
        case_type="租赁纠纷",
        last_law_name="民法典",
        mentioned_laws=["民法典"],
        entity_state=EntityState(
            key_facts=[KeyFact(fact="租期一年", confidence="high", source_turn=0)],
        ),
    ))

    statute_search = StatuteSearchTool(
        collection_name=statute_index["collection"],
        sparse_artifact_path=statute_index["sparse_path"],
    )

    # --- Turn 1: 房东涨租合法吗? ---
    result_t1 = await run_query(
        query="我租的房合同一年,房东要涨 30% 房租,合法吗?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b", specialty="民事",
            max_steps=8, max_tool_calls=10,
        ),
        provider=provider, runs_root=runs_root, config={},
        session_id=session_id, memory_store=store,
    )
    assert result_t1["status"] == "ok"

    # Check sticky updated
    sticky_after_t1 = store.read_sticky(session_id)
    assert result_t1["run_id"] in sticky_after_t1.linked_runs
    turns = store.recent_turns(session_id, n=5)
    assert len(turns) == 1
    assert turns[0].question.startswith("我租的房合同")

    # --- Turn 2: Receptionist sees sticky ---
    rec_run_id = fresh_run_id()
    rec_recorder = Recorder(run_id=rec_run_id, run_dir=runs_root / rec_run_id)
    receptionist = ReceptionistAgent(
        name="receptionist", role="triage",
        provider=provider, recorder=rec_recorder,
        model="qwen3.5-9b", max_steps=2,
    )
    triage_out = await receptionist.run(AgentInput(payload={
        "query": "那依据哪条法律?",
        "sticky_context": sticky_after_t1.model_dump(),
    }))
    rec_recorder.close()
    assert isinstance(triage_out.payload, ReceptionistOutput)
    # Follow-up: Receptionist should keep specialty in civil/property territory
    assert triage_out.payload.primary_specialty in ("民事", "房产", "通用", "民法", "合同")


@pytest.mark.asyncio
async def test_sticky_persists_across_runs(tmp_path):
    """Run-level smoke: write sticky → close store → open new store at same path → sticky still readable."""
    root = tmp_path / "memory_store"
    store1 = MarkdownMemoryStore(root=root)
    store1.write_sticky(StickyContext(
        session_id="s_persist",
        legal_domain="民事",
        case_type="x",
        entity_state=EntityState(
            key_facts=[KeyFact(fact="persistence test", confidence="high")],
        ),
    ))
    store2 = MarkdownMemoryStore(root=root)
    loaded = store2.read_sticky("s_persist")
    assert loaded is not None
    assert loaded.legal_domain == "民事"
    assert loaded.entity_state.key_facts[0].fact == "persistence test"

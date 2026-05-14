# Phase 5a — Supervisor Agent Implementation Plan

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Add `SupervisorAgent` as post-hoc QA gate (spec §3.5, ADR design). Takes a Lawyer's final `LawyerOutput` + the WorkingMemory evidence pool; verifies groundedness, logical consistency, citation accuracy; outputs `SupervisorVerdict` (pass / revise / reject).

**Architecture:** Supervisor is NOT in the ReAct loop — it's a post-processing gate that wraps Lawyer's run. New `SupervisorAgent` class. New top-level orchestrator `run_with_supervisor` that runs Lawyer, then runs Supervisor on Lawyer's output, returns combined result. Supervisor uses tools: `verify_citation` (checks Evidence text matches Lawyer's claim), `check_groundedness` (LLM judge, lightweight).

**Phase 4 starting point:** Tag `phase4-secretary-business`. 189 tests + 1 skipped.

---

## Out of scope (Phase 5b / Phase 3c)

- Phase 5b: Eval framework (QuerySet + Runner + Comparator + Judges + Ablation)
- Phase 3c: Cross-turn compression + ma_user_history
- Real-time revise loop (if Supervisor rejects, restart Lawyer with feedback) — Phase 5a outputs verdict only, no re-run

---

## File Structure

```
experiments/multi_agent/
├── multi_agent/
│   ├── schemas/supervisor.py                 # NEW: SupervisorVerdict (full)
│   ├── agents/supervisor.py                  # NEW: SupervisorAgent
│   ├── tools/business/verify_citation.py     # NEW: programmatic citation check
│   └── orchestration/supervised.py           # NEW: run_with_supervisor
└── tests/
    ├── unit/
    │   ├── test_supervisor_schemas.py
    │   ├── test_verify_citation.py
    │   ├── test_supervisor.py
    │   └── test_run_with_supervisor.py
    └── integration/
        └── test_supervised_lawyer_e2e.py
```

---

## Task 1: SupervisorVerdict Schema

**Files:**
- Create: `multi_agent/schemas/supervisor.py`
- Create: `tests/unit/test_supervisor_schemas.py`

Phase 1 `events.py` already has `SupervisorVerdict` as an event type. Phase 5a adds a full standalone schema for the agent's structured output (richer than the event).

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_supervisor_schemas.py
import pytest
from multi_agent.schemas.supervisor import (
    SupervisorVerdict, CitationCheckResult, GroundednessCheck,
)


def test_supervisor_verdict_pass():
    v = SupervisorVerdict(
        verdict="pass",
        confidence=0.85,
        issues=[],
        suggested_fix=None,
        citation_checks=[
            CitationCheckResult(citation_index=0, valid=True, reason="matches text"),
        ],
    )
    assert v.verdict == "pass"
    assert v.is_valid is True


def test_supervisor_verdict_reject():
    v = SupervisorVerdict(
        verdict="reject",
        confidence=0.9,
        issues=["citation 民法典-999 not in retrieved evidence"],
        suggested_fix="Re-retrieve and cite only verified articles",
    )
    assert v.verdict == "reject"
    assert v.is_valid is False
    assert "民法典-999" in v.issues[0]


def test_supervisor_verdict_unknown_kind_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SupervisorVerdict(verdict="bogus", confidence=0.5, issues=[])


def test_citation_check_result():
    c = CitationCheckResult(citation_index=2, valid=False, reason="text doesn't match")
    assert c.valid is False


def test_groundedness_check():
    g = GroundednessCheck(score=0.7, ungrounded_claims=["claim about 民法典 999"])
    assert g.score == 0.7
    assert len(g.ungrounded_claims) == 1
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create `multi_agent/schemas/supervisor.py`**

```python
"""Supervisor agent output schema."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class CitationCheckResult(BaseModel):
    """One citation's verification result."""
    citation_index: int            # index into LawyerOutput.citations
    valid: bool
    reason: str


class GroundednessCheck(BaseModel):
    """LLM-judge groundedness assessment."""
    score: float                   # 0..1 — fraction of claims with evidence
    ungrounded_claims: list[str] = Field(default_factory=list)


class SupervisorVerdict(BaseModel):
    """Post-hoc QA verdict on Lawyer's output."""
    verdict: Literal["pass", "revise", "reject"]
    confidence: float
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None
    citation_checks: list[CitationCheckResult] = Field(default_factory=list)
    groundedness: GroundednessCheck | None = None

    @property
    def is_valid(self) -> bool:
        return self.verdict == "pass"
```

- [ ] **Step 4: Verify pass + full suite** → 194 passed + 1 skipped (189 + 5 new).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/schemas/supervisor.py experiments/multi_agent/tests/unit/test_supervisor_schemas.py
git commit -m "phase5a(schemas): SupervisorVerdict + CitationCheckResult + GroundednessCheck"
```

---

## Task 2: verify_citation tool (programmatic)

**Files:**
- Create: `multi_agent/tools/business/verify_citation.py`
- Create: `tests/unit/test_verify_citation.py`

Programmatic (NOT LLM-driven) check: given a `Citation` and a list of `Evidence`, verify the citation's `(law_short, article_no)` matches an Evidence in the pool and the `excerpt` is a substring of Evidence.text.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_verify_citation.py
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
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create `multi_agent/tools/business/verify_citation.py`**

```python
"""Programmatic citation verification tool (Phase 5a).

Given a Citation and a pool of Evidence, verify:
1. (law_short, article_no) exists in evidence pool
2. excerpt (if non-empty) is a substring of the matched Evidence.text
"""
from __future__ import annotations
from pydantic import BaseModel, Field

from multi_agent.schemas.messages import ToolResult
from multi_agent.schemas.lawyer import Citation
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder


class VerifyCitationArgs(BaseModel):
    citation: Citation
    evidences: list[Evidence] = Field(default_factory=list)


class VerifyCitationTool(Tool):
    name: str = "verify_citation"
    description: str = (
        "Verify a Citation is grounded in the retrieved Evidence pool. "
        "Returns {valid: bool, reason: str}."
    )
    args_schema: type[BaseModel] = VerifyCitationArgs

    async def call(self, args: VerifyCitationArgs, recorder: Recorder) -> ToolResult:
        # Match by (law_short, article_no)
        target = f"{args.citation.law_short}-{args.citation.article_no}"
        match = None
        for ev in args.evidences:
            if ev.doc_id == target:
                match = ev
                break
            # Also try (law_short, article_no) tuple match (some Evidence may have empty law_short)
            if ev.law_short == args.citation.law_short and ev.article_no == args.citation.article_no:
                match = ev
                break
        if match is None:
            return ToolResult(tool_use_id="", payload={
                "valid": False,
                "reason": f"Citation {target} not in retrieved evidence",
            })

        # Excerpt check (if provided)
        if args.citation.excerpt and args.citation.excerpt.strip():
            ex = args.citation.excerpt.strip()
            # Tolerant: strip whitespace, check substring (allow Chinese punct variations)
            if ex not in match.text:
                # Try removing punctuation variations
                normalized_text = match.text.replace(",", ",").replace("。", ".")
                normalized_ex = ex.replace(",", ",").replace("。", ".")
                if normalized_ex not in normalized_text:
                    return ToolResult(tool_use_id="", payload={
                        "valid": False,
                        "reason": f"Excerpt not found in Evidence.text for {target}",
                    })

        return ToolResult(tool_use_id="", payload={"valid": True, "reason": "matches"})
```

- [ ] **Step 4: Verify pass + full suite** → 197 passed + 1 skipped (194 + 3 new).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/business/verify_citation.py experiments/multi_agent/tests/unit/test_verify_citation.py
git commit -m "phase5a(tools): VerifyCitationTool (programmatic citation check)"
```

---

## Task 3: SupervisorAgent

**Files:**
- Create: `multi_agent/prompts/supervisor/__init__.py`
- Create: `multi_agent/prompts/supervisor/system.md`
- Create: `multi_agent/agents/supervisor.py`
- Create: `tests/unit/test_supervisor.py`
- Modify: `pyproject.toml` package-data

Supervisor reviews a Lawyer's output. Takes input payload:
```python
{
    "lawyer_output": LawyerOutput.model_dump(),
    "evidence_pool": [Evidence.model_dump(), ...],     # from WorkingMemory
    "user_query": str,
}
```

Output: `SupervisorVerdict`. Internally uses `verify_citation` tool to programmatically check each Citation, then asks the LLM for a holistic verdict + groundedness assessment.

- [ ] **Step 1: Create prompt + package-data**

```bash
mkdir -p /home/xxm/rag/experiments/multi_agent/multi_agent/prompts/supervisor
touch /home/xxm/rag/experiments/multi_agent/multi_agent/prompts/supervisor/__init__.py
```

`multi_agent/prompts/supervisor/system.md`:

```markdown
你是律师事务所的审核员(Supervisor)。你的职责是审核律师给客户的答复,确保:

1. 引用的法条真实存在于检索证据中(不得编造)
2. 答复逻辑自洽
3. 没有过度承诺(如"保证胜诉")

# 输入
你会看到:
- 律师的最终答复(LawyerOutput,含 citations 和 five_section)
- 律师检索到的证据池(evidences)
- 用户原始问题

# 工作流程
1. 调用 verify_citation 工具,逐条检查每个 citation 是否在 evidence 中
2. 综合判断答复是否过度承诺、逻辑矛盾、漏洞
3. 输出 SupervisorVerdict JSON

# 输出 JSON
```json
{
  "verdict": "pass|revise|reject",
  "confidence": 0.85,
  "issues": ["问题1", "问题2"],
  "suggested_fix": null,
  "citation_checks": [{"citation_index": 0, "valid": true, "reason": "matches"}],
  "groundedness": {"score": 0.9, "ungrounded_claims": []}
}
```

# verdict 选择
- "pass": 引用正确,逻辑通顺,无过度承诺
- "revise": 有小问题(措辞过强、漏一条次要法条)
- "reject": 引用编造、严重逻辑错误、过度承诺

只输出 JSON。
```

`pyproject.toml`:

```toml
[tool.setuptools.package-data]
multi_agent = ["prompts/lawyer/*.md", "prompts/receptionist/*.md", "prompts/secretary/*.md", "prompts/supervisor/*.md"]
```

Reinstall.

- [ ] **Step 2: Failing test**

```python
# tests/unit/test_supervisor.py
import pytest
from multi_agent.agents.supervisor import SupervisorAgent
from multi_agent.schemas.supervisor import SupervisorVerdict
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder
from multi_agent.agents.base import AgentInput


def test_supervisor_prompt_loads(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = SupervisorAgent(name="supervisor", role="qa",
                           provider=p, recorder=rec)
    prompt = agent.system_prompt()
    assert "审核员" in prompt or "Supervisor" in prompt
    assert "verify_citation" in prompt
    rec.close()


def test_supervisor_output_schema(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = SupervisorAgent(name="supervisor", role="qa",
                           provider=p, recorder=rec)
    assert agent.output_schema() is SupervisorVerdict
    rec.close()


@pytest.mark.asyncio
async def test_supervisor_renders_lawyer_output_in_prompt(tmp_path):
    """_render_input should include lawyer_output + evidence_pool from payload."""
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = SupervisorAgent(name="supervisor", role="qa",
                           provider=p, recorder=rec)
    rendered = agent._render_input(AgentInput(payload={
        "user_query": "房东涨租合法吗?",
        "lawyer_output": {"mode": "consultation", "primary_answer": "不合法",
                         "citations": [{"law_short": "民法典", "article_no": "510",
                                        "excerpt": "合同补充"}]},
        "evidence_pool": [{"doc_id": "民法典-510", "law_name": "民法典",
                          "law_short": "民法典", "article_no": "510",
                          "text": "当事人就合同补充内容...", "score": 0.9,
                          "retriever": "hybrid"}],
    }))
    assert "房东涨租" in rendered
    assert "民法典" in rendered
    assert "510" in rendered
    rec.close()
```

- [ ] **Step 3: Verify failure** → ImportError.

- [ ] **Step 4: Create `multi_agent/agents/supervisor.py`**

```python
"""SupervisorAgent — post-hoc QA on Lawyer output."""
from __future__ import annotations
import json as _json
from importlib.resources import files

from multi_agent.agents.base import BaseAgent
from multi_agent.schemas.supervisor import SupervisorVerdict


class SupervisorAgent(BaseAgent):
    """Reviews Lawyer output. Tools: verify_citation."""

    def system_prompt(self) -> str:
        return files("multi_agent.prompts.supervisor").joinpath("system.md").read_text(encoding="utf-8")

    def output_schema(self) -> type[SupervisorVerdict]:
        return SupervisorVerdict

    def _render_input(self, input) -> str:
        payload = input.payload
        user_q = payload.get("user_query", "")
        lawyer_out = payload.get("lawyer_output", {})
        evidence_pool = payload.get("evidence_pool", [])
        return (
            f"# 用户原始问题\n{user_q}\n\n"
            f"# 律师答复(LawyerOutput)\n```json\n{_json.dumps(lawyer_out, ensure_ascii=False, indent=2)}\n```\n\n"
            f"# 律师检索到的证据池\n```json\n{_json.dumps(evidence_pool, ensure_ascii=False, indent=2)}\n```\n\n"
            "请审核并输出 SupervisorVerdict JSON。"
        )
```

- [ ] **Step 5: Verify pass + full suite** → 200 passed + 1 skipped (197 + 3 new).

- [ ] **Step 6: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/prompts/supervisor/ experiments/multi_agent/multi_agent/agents/supervisor.py experiments/multi_agent/tests/unit/test_supervisor.py experiments/multi_agent/pyproject.toml
git commit -m "phase5a(agents): SupervisorAgent (QA gate on Lawyer output)"
```

---

## Task 4: run_with_supervisor Orchestrator

**Files:**
- Create: `multi_agent/orchestration/__init__.py`
- Create: `multi_agent/orchestration/supervised.py`
- Create: `tests/unit/test_run_with_supervisor.py`

Helper that wraps `run_query`: runs Lawyer, captures its WorkingMemory + output, then runs Supervisor on it, returns combined `{lawyer_result, supervisor_verdict, run_ids}`.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_run_with_supervisor.py
import json
import pytest
from pydantic import BaseModel
from multi_agent.orchestration.supervised import run_with_supervisor
from multi_agent.agents.base import BaseAgent
from multi_agent.agents.supervisor import SupervisorAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


class _LawyerOut(BaseModel):
    mode: str
    primary_answer: str


class _Lawyer(BaseAgent):
    def system_prompt(self) -> str:
        return "test lawyer"
    def output_schema(self):
        return _LawyerOut


@pytest.mark.asyncio
async def test_run_with_supervisor_returns_both_results(tmp_path):
    runs_root = tmp_path / "runs"
    lawyer_provider = StubProvider(responses=[
        ScriptedResponse(text='{"mode": "consultation", "primary_answer": "测试答复"}',
                        finish_reason="end_turn"),
    ])
    supervisor_provider = StubProvider(responses=[
        ScriptedResponse(text='{"verdict": "pass", "confidence": 0.9, "issues": []}',
                        finish_reason="end_turn"),
    ])
    result = await run_with_supervisor(
        query="测试问题",
        lawyer_factory=lambda p, r: _Lawyer(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        supervisor_factory=lambda p, r: SupervisorAgent(
            name="supervisor", role="qa", provider=p, recorder=r, model="stub-1",
            max_pre_tool_rejections=10,
        ),
        lawyer_provider=lawyer_provider,
        supervisor_provider=supervisor_provider,
        runs_root=runs_root,
    )
    assert result["lawyer_result"]["status"] == "ok"
    assert result["supervisor_verdict"]["verdict"] == "pass"
    assert "lawyer_run_id" in result
    assert "supervisor_run_id" in result
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create files**

`multi_agent/orchestration/__init__.py` (empty)

`multi_agent/orchestration/supervised.py`:

```python
"""Lawyer + Supervisor orchestration (Phase 5a)."""
from __future__ import annotations
import json as _json
from pathlib import Path
from typing import Callable, Any
from multi_agent.providers.base import LLMProvider
from multi_agent.runner import run_query
from multi_agent.agents.base import AgentInput
from multi_agent.tracing.recorder import Recorder
from multi_agent.tracing.ulid_gen import fresh_run_id


async def run_with_supervisor(
    *,
    query: str,
    lawyer_factory: Callable,
    supervisor_factory: Callable,
    lawyer_provider: LLMProvider,
    supervisor_provider: LLMProvider,
    runs_root: Path,
    lawyer_config: dict | None = None,
    session_id: str | None = None,
    memory_store=None,
) -> dict[str, Any]:
    """Run Lawyer, then Supervisor on Lawyer's output. Return combined dict.

    Returns:
        {
            "lawyer_run_id": str,
            "supervisor_run_id": str,
            "lawyer_result": {status, run_id, final_answer},
            "supervisor_verdict": {verdict, confidence, issues, ...},
        }
    """
    # Capture lawyer's WorkingMemory by holding a reference to the agent.
    captured = {"agent": None}

    def lawyer_factory_wrapped(p, r):
        agent = lawyer_factory(p, r)
        captured["agent"] = agent
        return agent

    lawyer_result = await run_query(
        query=query,
        agent_factory=lawyer_factory_wrapped,
        provider=lawyer_provider,
        runs_root=runs_root,
        config=lawyer_config or {},
        session_id=session_id,
        memory_store=memory_store,
    )

    # Pull Lawyer's WorkingMemory evidence pool
    lawyer_agent = captured["agent"]
    evidence_pool: list[dict] = []
    if lawyer_agent is not None and getattr(lawyer_agent, "working_memory", None):
        evidence_pool = [ev.model_dump() for ev in lawyer_agent.working_memory.retrieved_evidence]

    # Parse Lawyer's final answer
    try:
        lawyer_out_dict = _json.loads(lawyer_result.get("final_answer") or "{}")
    except Exception:
        lawyer_out_dict = {}

    # Run Supervisor
    sup_run_id = fresh_run_id()
    sup_recorder = Recorder(run_id=sup_run_id, run_dir=runs_root / sup_run_id)
    supervisor = supervisor_factory(supervisor_provider, sup_recorder)
    sup_output = await supervisor.run(AgentInput(payload={
        "user_query": query,
        "lawyer_output": lawyer_out_dict,
        "evidence_pool": evidence_pool,
    }))
    sup_recorder.close()

    return {
        "lawyer_run_id": lawyer_result["run_id"],
        "supervisor_run_id": sup_run_id,
        "lawyer_result": lawyer_result,
        "supervisor_verdict": sup_output.payload.model_dump(),
    }
```

- [ ] **Step 4: Verify pass + full suite** → 201 passed + 1 skipped.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/orchestration/ experiments/multi_agent/tests/unit/test_run_with_supervisor.py
git commit -m "phase5a(orchestration): run_with_supervisor wraps Lawyer + Supervisor"
```

---

## Task 5: Real Qwen Supervised E2E + tag

**Files:**
- Create: `tests/integration/test_supervised_lawyer_e2e.py`

End-to-end: Lawyer answers a rental dispute with real Qwen + statute_search; Supervisor reviews the Lawyer's output against the actual evidence pool; verdict should be `pass` (since Lawyer's tool-first enforcement prevents fabrication).

- [ ] **Step 1: Write test**

```python
# tests/integration/test_supervised_lawyer_e2e.py
"""Phase 5a E2E: Lawyer + Supervisor pipeline against real Qwen."""
import json
import uuid
import httpx
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.agents.supervisor import SupervisorAgent
from multi_agent.orchestration.supervised import run_with_supervisor


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def statute_index(tmp_path_factory):
    name = f"test_sup_{uuid.uuid4().hex[:8]}"
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
async def test_supervised_lawyer_passes(statute_index, tmp_path):
    runs_root = tmp_path / "runs"
    provider = OpenAICompatibleProvider()

    statute_search = StatuteSearchTool(
        collection_name=statute_index["collection"],
        sparse_artifact_path=statute_index["sparse_path"],
    )

    result = await run_with_supervisor(
        query="房东合同期内涨租 30% 合法吗?",
        lawyer_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b", specialty="民事",
            max_steps=8, max_tool_calls=10,
        ),
        supervisor_factory=lambda p, r: SupervisorAgent(
            name="supervisor", role="qa",
            provider=p, recorder=r,
            model="qwen3.5-9b",
            max_steps=3, max_pre_tool_rejections=5,
        ),
        lawyer_provider=provider,
        supervisor_provider=provider,
        runs_root=runs_root,
    )

    assert result["lawyer_result"]["status"] == "ok"
    verdict = result["supervisor_verdict"]["verdict"]
    # Supervisor should give pass or revise — reject would indicate Lawyer fabricated
    assert verdict in ("pass", "revise"), (
        f"Unexpected verdict: {verdict}. Issues: {result['supervisor_verdict'].get('issues')}"
    )
```

- [ ] **Step 2: Run + Step 3: Full suite + Step 4: Commit + tag**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_supervised_lawyer_e2e.py -v"
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v 2>&1 | tail -10"

cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_supervised_lawyer_e2e.py
git commit -m "phase5a(integration): Lawyer + Supervisor E2E with real Qwen"
git tag -a phase5a-supervisor -m "Phase 5a complete: SupervisorAgent + verify_citation + run_with_supervisor"
git tag -l "phase*"
```

## Acceptance Criteria

Phase 5a complete when:

1. Full pytest passes (~202 tests)
2. SupervisorVerdict schema validates 3 verdict values
3. VerifyCitationTool catches fake citations + excerpt mismatches
4. SupervisorAgent loads prompt + outputs SupervisorVerdict
5. run_with_supervisor returns combined lawyer_result + supervisor_verdict
6. Real-Qwen E2E: Lawyer + Supervisor work end-to-end, verdict ∈ {pass, revise}
7. Tag `phase5a-supervisor` exists

## Out of Scope (Phase 5b)

- Eval framework (QuerySet / ExperimentRunner / Comparator)
- LLM Judges (citation_accuracy / groundedness / helpfulness using Claude Opus per spec §7.7)
- AblationRunner
- LatencyProfiler
- Supervisor-driven Lawyer re-run on revise/reject

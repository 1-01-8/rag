"""Phase 5v: intent-based read_sticky robustness tests (spec §5.4.3).

Covers all 4 intents × all defined robustness contracts:
  - happy path per intent
  - file missing → None for every intent
  - corrupt YAML / unreadable → None for every intent
  - sticky present but missing the field → empty view (not None)
  - corrupt individual list item → skipped, neighbours preserved
  - unknown intent string → ValueError
  - "full" intent backward compatibility (default arg, original return shape)
"""
from __future__ import annotations
from pathlib import Path

import pytest

from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.schemas.memory import (
    StickyContext, EntityState, KeyFact, CitedArticle,
    StickyEntitiesView, StickyCitationsView, StickySummaryView,
)


def _seed(store: MarkdownMemoryStore, **kwargs) -> StickyContext:
    """Write a sticky and return the model that was written."""
    sticky = StickyContext(session_id=kwargs.pop("session_id", "s1"), **kwargs)
    store.write_sticky(sticky)
    return sticky


# ---------- happy paths ----------

def test_full_intent_returns_full_sticky_context(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root=tmp_path)
    written = _seed(store, legal_domain="民事", case_type="租赁纠纷",
                    history_summary="过去 3 轮讨论涨租")
    result = store.read_sticky("s1", intent="full")
    assert isinstance(result, StickyContext)
    assert result.legal_domain == "民事"
    assert result.case_type == "租赁纠纷"
    assert result.history_summary == "过去 3 轮讨论涨租"


def test_entities_only_returns_view_with_entity_state(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root=tmp_path)
    _seed(store, entity_state=EntityState(
        key_facts=[KeyFact(fact="租期1年", confidence="high", source_turn=1)],
        open_questions=["是否书面约定"],
    ))
    result = store.read_sticky("s1", intent="entities_only")
    assert isinstance(result, StickyEntitiesView)
    assert result.session_id == "s1"
    assert len(result.entity_state.key_facts) == 1
    assert result.entity_state.key_facts[0].fact == "租期1年"
    assert result.entity_state.open_questions == ["是否书面约定"]


def test_recent_citations_returns_view_with_cited_articles(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root=tmp_path)
    _seed(store, cited_articles=[
        CitedArticle(law="民法典", article="510", from_turn=1),
        CitedArticle(law="民法典", article="703", from_turn=2),
    ])
    result = store.read_sticky("s1", intent="recent_citations")
    assert isinstance(result, StickyCitationsView)
    assert len(result.cited_articles) == 2
    assert {c.article for c in result.cited_articles} == {"510", "703"}


def test_summary_only_returns_view_with_history_summary(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root=tmp_path)
    _seed(store, history_summary="第 1-3 轮讨论租金调整,确认民法典 510 适用。")
    result = store.read_sticky("s1", intent="summary_only")
    assert isinstance(result, StickySummaryView)
    assert "第 1-3 轮" in result.history_summary


# ---------- missing file: None for every intent ----------

@pytest.mark.parametrize("intent", ["full", "entities_only", "recent_citations", "summary_only"])
def test_missing_file_returns_none(tmp_path: Path, intent: str) -> None:
    store = MarkdownMemoryStore(root=tmp_path)
    assert store.read_sticky("nonexistent_session", intent=intent) is None


# ---------- corruption: file present but unreadable ----------

@pytest.mark.parametrize("intent", ["full", "entities_only", "recent_citations", "summary_only"])
def test_malformed_yaml_returns_none(tmp_path: Path, intent: str) -> None:
    """Corrupt frontmatter (YAML parse error) → None for every intent."""
    store = MarkdownMemoryStore(root=tmp_path)
    path = store._sticky_path("s1")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Intentionally malformed YAML inside the frontmatter delimiters
    path.write_text("---\nsession_id: s1\n  this is: not: valid: yaml: [\n---\n\nbody",
                    encoding="utf-8")
    assert store.read_sticky("s1", intent=intent) is None


@pytest.mark.parametrize("intent", ["full", "entities_only", "recent_citations", "summary_only"])
def test_yaml_scalar_top_level_returns_none(tmp_path: Path, intent: str) -> None:
    """If frontmatter parses but isn't a dict, treat as corrupt."""
    store = MarkdownMemoryStore(root=tmp_path)
    path = store._sticky_path("s1")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Frontmatter is a YAML scalar — syntactically valid, semantically wrong
    path.write_text("---\njust-a-string\n---\n\nbody", encoding="utf-8")
    assert store.read_sticky("s1", intent=intent) is None


# ---------- missing field but file otherwise valid ----------

def test_entities_only_returns_empty_view_when_entity_state_missing(tmp_path: Path) -> None:
    """File exists, EntityState field absent → empty view, NOT None.

    Lets callers distinguish 'session exists but has no entities' from
    'session does not exist'.
    """
    store = MarkdownMemoryStore(root=tmp_path)
    _seed(store)  # default EntityState is empty
    result = store.read_sticky("s1", intent="entities_only")
    assert isinstance(result, StickyEntitiesView)
    assert result.entity_state.key_facts == []
    assert result.entity_state.active_subjects == []


def test_recent_citations_returns_empty_view_when_field_missing(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root=tmp_path)
    _seed(store)  # no cited_articles
    result = store.read_sticky("s1", intent="recent_citations")
    assert isinstance(result, StickyCitationsView)
    assert result.cited_articles == []


def test_summary_only_returns_empty_string_when_history_summary_missing(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root=tmp_path)
    _seed(store)  # default history_summary = ""
    result = store.read_sticky("s1", intent="summary_only")
    assert isinstance(result, StickySummaryView)
    assert result.history_summary == ""


# ---------- corrupt list items: skip-and-continue ----------

def test_recent_citations_skips_corrupt_entries(tmp_path: Path) -> None:
    """A single malformed cited_article should be skipped; valid neighbours preserved."""
    store = MarkdownMemoryStore(root=tmp_path)
    # Hand-write sticky.md with one valid + one missing-field + one valid CitedArticle
    path = store._sticky_path("s1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "session_id: s1\n"
        "cited_articles:\n"
        "  - {law: 民法典, article: '510', from_turn: 1}\n"
        "  - this is not a dict\n"          # corrupt: scalar, not a CitedArticle
        "  - {law: 民法典, article: '703', from_turn: 2}\n"
        "---\n\n",
        encoding="utf-8",
    )
    result = store.read_sticky("s1", intent="recent_citations")
    assert isinstance(result, StickyCitationsView)
    articles = {c.article for c in result.cited_articles}
    assert articles == {"510", "703"}, f"Expected 510 and 703 only, got {articles}"


def test_entities_only_falls_back_to_empty_when_entity_state_corrupt(tmp_path: Path) -> None:
    """If entity_state field is present but unparseable, view holds an empty EntityState."""
    store = MarkdownMemoryStore(root=tmp_path)
    path = store._sticky_path("s1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "session_id: s1\n"
        "entity_state: \"this should be a dict\"\n"
        "---\n\n",
        encoding="utf-8",
    )
    result = store.read_sticky("s1", intent="entities_only")
    assert isinstance(result, StickyEntitiesView)
    assert result.entity_state.key_facts == []


def test_summary_only_falls_back_to_empty_when_history_summary_not_a_string(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root=tmp_path)
    path = store._sticky_path("s1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "session_id: s1\n"
        "history_summary: {oops: this is a dict not a string}\n"
        "---\n\n",
        encoding="utf-8",
    )
    result = store.read_sticky("s1", intent="summary_only")
    assert isinstance(result, StickySummaryView)
    assert result.history_summary == ""


# ---------- unknown intent ----------

def test_unknown_intent_raises_value_error(tmp_path: Path) -> None:
    """Programming-error: caller passed an intent not in the Literal."""
    store = MarkdownMemoryStore(root=tmp_path)
    _seed(store)
    with pytest.raises(ValueError, match="Unknown intent"):
        store.read_sticky("s1", intent="garbage_intent")   # type: ignore[arg-type]


def test_unknown_intent_raises_even_when_file_missing(tmp_path: Path) -> None:
    """Intent validation happens before file IO, so it raises consistently."""
    store = MarkdownMemoryStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.read_sticky("nonexistent", intent="bogus")    # type: ignore[arg-type]


# ---------- backward compatibility ----------

def test_read_sticky_no_intent_arg_defaults_to_full(tmp_path: Path) -> None:
    """Existing callers passing only session_id keep getting StickyContext."""
    store = MarkdownMemoryStore(root=tmp_path)
    _seed(store, legal_domain="民事")
    result = store.read_sticky("s1")
    assert isinstance(result, StickyContext)
    assert result.legal_domain == "民事"


# ---------- session_id propagation in slice views ----------

def test_views_include_session_id_from_filename_when_frontmatter_missing(tmp_path: Path) -> None:
    """Even if sticky frontmatter forgot session_id, view echoes the requested id."""
    store = MarkdownMemoryStore(root=tmp_path)
    path = store._sticky_path("requested_sid")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\nhistory_summary: just a summary\n---\n\n", encoding="utf-8")
    result = store.read_sticky("requested_sid", intent="summary_only")
    assert result is not None
    assert result.session_id == "requested_sid"

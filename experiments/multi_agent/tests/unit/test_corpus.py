from pathlib import Path
import pytest
from multi_agent.tools.corpus import load_law_file, load_corpus
from multi_agent.schemas.document import Document, Chunk


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "sample_laws"


def test_load_law_file_returns_document():
    doc = load_law_file(FIXTURE_DIR / "民法典-sample.txt")
    assert isinstance(doc, Document)
    assert doc.law_short == "民法典"
    assert doc.law_name == "中华人民共和国民法典"
    assert len(doc.chunks) == 3


def test_chunks_have_correct_article_numbers():
    doc = load_law_file(FIXTURE_DIR / "民法典-sample.txt")
    article_nos = [c.article_no for c in doc.chunks]
    # "第一条" → "1", "第二条" → "2", "第五百一十条" → "510"
    assert article_nos == ["1", "2", "510"]


def test_chunks_have_clean_text():
    doc = load_law_file(FIXTURE_DIR / "民法典-sample.txt")
    # First chunk text should be the body AFTER "规定，"
    assert doc.chunks[0].text.startswith("为了保护民事主体")
    # Should NOT include the law name prefix or article marker
    assert "《" not in doc.chunks[0].text
    assert "第一条" not in doc.chunks[0].text


def test_chunk_doc_ids_unique_per_law():
    doc = load_law_file(FIXTURE_DIR / "民法典-sample.txt")
    ids = {c.doc_id for c in doc.chunks}
    assert len(ids) == 3
    assert "民法典-1" in ids
    assert "民法典-510" in ids


def test_load_corpus_finds_all_files(tmp_path):
    # Make two tiny law files in tmp dir
    (tmp_path / "民法典.txt").write_text(
        "《中华人民共和国民法典》第一条规定，第一条内容。\n", encoding="utf-8"
    )
    (tmp_path / "刑法.txt").write_text(
        "《中华人民共和国刑法》第一条规定，第一条内容。\n", encoding="utf-8"
    )
    docs = load_corpus(tmp_path)
    assert len(docs) == 2
    shorts = {d.law_short for d in docs}
    assert shorts == {"民法典", "刑法"}


def test_load_corpus_skips_non_txt(tmp_path):
    (tmp_path / "民法典.txt").write_text(
        "《中华人民共和国民法典》第一条规定，正文。\n", encoding="utf-8"
    )
    (tmp_path / "readme.md").write_text("# not a law\n", encoding="utf-8")
    docs = load_corpus(tmp_path)
    assert len(docs) == 1


def test_load_law_file_skips_malformed_lines(tmp_path):
    # Mix of valid lines and garbage
    (tmp_path / "law.txt").write_text(
        "《中华人民共和国民法典》第一条规定，正文一。\n"
        "GARBAGE LINE\n"
        "《中华人民共和国民法典》第二条规定，正文二。\n",
        encoding="utf-8",
    )
    doc = load_law_file(tmp_path / "law.txt")
    assert len(doc.chunks) == 2  # garbage skipped

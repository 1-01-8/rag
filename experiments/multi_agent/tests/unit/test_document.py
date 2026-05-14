from multi_agent.schemas.document import Document, Chunk


def test_chunk_required_fields():
    c = Chunk(
        doc_id="民法典-510",
        law_name="中华人民共和国民法典",
        law_short="民法典",
        article_no="510",
        text="当事人就合同补充内容没有约定...",
    )
    assert c.doc_id == "民法典-510"
    assert c.metadata == {}                     # default empty
    assert c.cross_refs == []
    assert c.concepts == []


def test_chunk_with_optional_fields():
    c = Chunk(
        doc_id="d", law_name="l", law_short="L", article_no="1",
        text="t",
        book="合同编", chapter="合同的订立",
        cross_refs=["第511条"], concepts=["合同补充"],
        metadata={"source_file": "law.txt"},
    )
    assert c.book == "合同编"
    assert c.chapter == "合同的订立"
    assert c.metadata["source_file"] == "law.txt"


def test_chunk_embedding_text_includes_law_chapter_article():
    """The string used to build the dense embedding should include
    law_short + book + chapter + article_no + text, per spec §4.4."""
    c = Chunk(
        doc_id="d", law_name="l", law_short="民法典", article_no="510",
        text="正文内容",
        book="合同编", chapter="合同的订立",
    )
    et = c.embedding_text()
    assert "民法典" in et
    assert "合同编" in et
    assert "合同的订立" in et
    assert "510" in et
    assert "正文内容" in et


def test_document_holds_chunks():
    d = Document(
        law_name="民法典", law_short="民法典",
        source_path="laws/民法典.txt",
        chunks=[
            Chunk(doc_id="民法典-1", law_name="民法典", law_short="民法典", article_no="1", text="a"),
            Chunk(doc_id="民法典-2", law_name="民法典", law_short="民法典", article_no="2", text="b"),
        ],
    )
    assert len(d.chunks) == 2
    assert d.law_short == "民法典"

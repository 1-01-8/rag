"""Parse Chinese-Laws .txt files into Document objects.

Source format per line:
  《<law_name>》第<article_no_cn>条规定，<text body>。
"""
from __future__ import annotations
import re
from pathlib import Path
from multi_agent.schemas.document import Document, Chunk


# Pattern: 《<law_name>》第<article_no_cn>条规定，<text>
_LINE_RE = re.compile(
    r"^《(?P<law_name>[^》]+)》第(?P<article_cn>[一二三四五六七八九十百千零\d]+)条规定[，,](?P<text>.+)$"
)

# Chinese numeral → arabic numeral
_CN_DIGIT = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}


def chinese_to_int(s: str) -> int:
    """Convert Chinese numeral string like '五百一十' to integer 510.
    Falls through to int() for ASCII-digit strings."""
    if s.isdigit():
        return int(s)

    total = 0
    section = 0
    current = 0
    for ch in s:
        if ch in _CN_DIGIT:
            current = _CN_DIGIT[ch]
        elif ch == "十":
            section += (current if current else 1) * 10
            current = 0
        elif ch == "百":
            section += (current if current else 1) * 100
            current = 0
        elif ch == "千":
            section += (current if current else 1) * 1000
            current = 0
        elif ch == "万":
            total += (section + current) * 10000
            section = 0
            current = 0
        else:
            raise ValueError(f"unknown Chinese numeral char: {ch}")
    return total + section + current


def _law_short_from_name(law_name: str) -> str:
    """'中华人民共和国民法典' → '民法典'."""
    prefix = "中华人民共和国"
    return law_name[len(prefix):] if law_name.startswith(prefix) else law_name


def load_law_file(path: Path) -> Document:
    """Parse one law .txt file into a Document with one Chunk per article."""
    path = Path(path)
    chunks: list[Chunk] = []
    law_name = ""
    law_short = ""

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue  # skip malformed line
            if not law_name:
                law_name = m.group("law_name")
                law_short = _law_short_from_name(law_name)
            try:
                article_no = str(chinese_to_int(m.group("article_cn")))
            except ValueError:
                continue
            text = m.group("text").rstrip("。")
            chunks.append(
                Chunk(
                    doc_id=f"{law_short}-{article_no}",
                    law_name=law_name,
                    law_short=law_short,
                    article_no=article_no,
                    text=text,
                    metadata={"source_file": str(path)},
                )
            )

    return Document(
        law_name=law_name,
        law_short=law_short,
        source_path=str(path),
        chunks=chunks,
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Scan a directory for .txt law files and parse each."""
    corpus_dir = Path(corpus_dir)
    docs: list[Document] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file() and path.suffix == ".txt":
            doc = load_law_file(path)
            if doc.chunks:
                docs.append(doc)
    return docs

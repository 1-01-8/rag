# Phase 2 · Ingestion（文档加载、切分、metadata 抽取）

## 依赖

- Phase 1 已完成：包结构、config、schemas、占位脚本。

## 本阶段交付物

1. `src/legal_rag/ingestion/loaders.py`：TXT / MD / PDF 加载。
2. `src/legal_rag/ingestion/cleaners.py`：基础清洗（去页眉页脚、合并跨行、统一空白）。
3. `src/legal_rag/ingestion/chunkers.py`：按 source_type 分支切分。
4. `src/legal_rag/ingestion/metadata_extractor.py`：规则版 metadata 抽取。
5. `src/legal_rag/ingestion/pipeline.py`：组合上述模块，输出 `chunks.jsonl`。
6. `scripts/ingest_docs.py`：CLI。
7. `tests/test_chunker.py`、`tests/test_metadata.py`。

---

## 1. 加载

```python
# loaders.py
from pathlib import Path

def load_text(path: Path) -> str: ...
def load_markdown(path: Path) -> str: ...
def load_pdf(path: Path) -> tuple[str, list[tuple[int, int]]]:
    """返回 (full_text, page_spans)；page_spans[i] = (start_offset, end_offset)。"""
    ...

def detect_and_load(path: Path) -> tuple[str, list[tuple[int, int]] | None]: ...
```

PDF 优先级：`PyMuPDF → pdfplumber`。OCR 不在 MVP。
扫描 PDF 直接报：

```text
该 PDF 可能是扫描版，请开启 OCR 或上传可复制文本版本。
```

判断标准：PyMuPDF 抽出文本 < 50 字符 / 总页数 ≥ 3 页。

---

## 2. 清洗

```python
# cleaners.py
def normalize_whitespace(text: str) -> str: ...
def strip_headers_footers(text: str) -> str: ...   # 简单基于行频次 / 数字页码模式
def merge_broken_lines(text: str) -> str: ...      # 合并行内换行（中文段落）
```

原则：保留法条编号、章节标题这些结构化信息；只压缩多余空白。

---

## 3. 切分（chunkers.py）

按 `source_type` 走不同策略。

### 3.1 statute（法律法规）

按"第X条"切分；条号必须 normalize。

```python
import re
ARTICLE_RE = re.compile(r"第([一二三四五六七八九十百千零〇0-9]+)条(?!例)")
CHAPTER_RE = re.compile(r"第([一二三四五六七八九十百千零〇0-9]+)章")

def chunk_statute(
    text: str, doc_id: str, source_path: str, jurisdiction: str
) -> list[DocumentChunk]: ...
```

要求：

- 每个 chunk = 一条；条文太长（>800 字）按段落二次切分，但保留"第X条 + 上一段标题"作为前缀。
- 太短（<80 字）和上一条合并到同一 chunk，并在 metadata 中记录合并范围。
- `chunk_id = f"{doc_id}#art-{article_number}"`，重复时追加 `-{n}`。
- 同时填 `chapter`、`law_name`（取文档首部第一行的 `《XXX》` 或非空白首行）。

### 3.2 contract（合同）

按"第X条 / 第X款"切分；同时识别条款标题（如"竞业限制"）。

```python
CONTRACT_ARTICLE_RE = re.compile(r"第([0-9一二三四五六七八九十]+)条[\s　]*([^\n]{0,30})?")
```

`contract_section` 写入条号与标题。

### 3.3 case（案例）

简单按四段式切：事实 / 争议焦点 / 裁判理由 / 判决结果，找不到段落标记则按段落滑窗（每 500 字，重叠 100 字）。
抽 `case_name`、`court`、`trial_level`、`cause_of_action`（详见 §4）。

### 3.4 generic / article

标题层级 + 滑窗（500/100）。

### 3.5 公共要求

- 所有 chunk 必须有：`chunk_id, doc_id, text, source_type, source_path`。
- `keywords` 由简单 jieba 统计 top-5 高频实词。
- `page_start / page_end` 通过 `page_spans` 反查。

---

## 4. metadata 抽取（regex 优先）

```python
# metadata_extractor.py
import re

ARTICLE_RE = re.compile(r"第([一二三四五六七八九十百千零〇0-9]+)条(?!例)")
CHAPTER_RE = re.compile(r"第([一二三四五六七八九十百千零〇0-9]+)章")
LAW_TITLE_RE = re.compile(r"《([^《》]{2,40})》")
COURT_RE = re.compile(r"[一-龥]{2,15}人民法院")
CASE_NO_RE = re.compile(r"[（(]\d{4}[)）][^\s]{0,20}?号")
DATE_RE = re.compile(r"(\d{4})[年\-./](\d{1,2})[月\-./](\d{1,2})日?")

def cn_to_arabic(s: str) -> str:
    """三十九 -> 39；2024 保持；混合 1百零2 -> 102。"""
    ...
```

`valid_status` 推断规则（MVP 简版）：

- 文本含「失效」「废止」→ `repealed`；
- 含「修订」「修正」「自XXXX年XX月XX日起施行」→ 取最新生效日期，标 `valid`；
- 默认 `unknown`。

---

## 5. pipeline

```python
# pipeline.py
def ingest_directory(
    input_dir: Path,
    source_type: str,
    jurisdiction: str,
    out_path: Path,
) -> int:
    """Returns chunk count."""
```

输出 `data/processed/chunks.jsonl`，每行一个 `DocumentChunk.model_dump_json()`。

CLI：

```python
# scripts/ingest_docs.py
import typer
from pathlib import Path
from legal_rag.ingestion.pipeline import ingest_directory

app = typer.Typer()

@app.command()
def run(
    input: Path = typer.Option(..., exists=True),
    source_type: str = typer.Option("statute"),
    jurisdiction: str = typer.Option("CN"),
    out: Path = typer.Option(Path("data/processed/chunks.jsonl")),
):
    n = ingest_directory(input, source_type, jurisdiction, out)
    typer.echo(f"wrote {n} chunks to {out}")

if __name__ == "__main__":
    app()
```

---

## 端到端验收

### 验收命令

```bash
# 准备最小数据集
mkdir -p data/raw/statutes
# 把"中华人民共和国劳动合同法.txt"放进去（任何来源都行，建议人工准备 1 份）

python scripts/ingest_docs.py \
  --input data/raw/statutes \
  --source-type statute \
  --jurisdiction CN \
  --out data/processed/chunks.jsonl

# 检查输出
wc -l data/processed/chunks.jsonl       # 应当 > 50（劳动合同法约 100 条）
head -1 data/processed/chunks.jsonl     # 应当是合法 JSON

pytest -q tests/test_chunker.py tests/test_metadata.py
```

### 验收通过条件

- `chunks.jsonl` 中能找到 `article_number == "39"` 且 `law_name` 含「劳动合同法」的 chunk。
- 同一法条的"第三十九条"与"第39条"两种写法 normalize 后 `article_number` 相等。
- 合同样本切分后能拿到 `contract_section` 含「竞业限制」的 chunk。
- 测试覆盖：
  1. 法条切分颗粒度正确；
  2. metadata extractor 中文/阿拉伯数字归一；
  3. PDF 扫描判定能给出友好提示；
  4. pipeline 跑空目录时返回 0 而不是抛异常。

---

## Codex Prompt

```text
在 Phase 1 已完成的仓库里，实现 Phase 2：Ingestion。

按 PLAN/04_PHASE2_INGESTION.md 实现：

1. src/legal_rag/ingestion/loaders.py（TXT / MD / PDF；PDF 用 PyMuPDF 优先，pdfplumber 兜底；扫描 PDF 给提示并返回空文本）
2. src/legal_rag/ingestion/cleaners.py（normalize_whitespace / strip_headers_footers / merge_broken_lines）
3. src/legal_rag/ingestion/metadata_extractor.py（按文档 §4 的正则；提供 cn_to_arabic）
4. src/legal_rag/ingestion/chunkers.py（statute / contract / case / generic 四种）
5. src/legal_rag/ingestion/pipeline.py（ingest_directory）
6. scripts/ingest_docs.py（typer CLI）
7. tests/test_chunker.py、tests/test_metadata.py

要求：
- 不调用任何 LLM（不 import providers）。
- 所有 chunk 必须含 chunk_id（含 doc_id）、doc_id、text、source_type、source_path。
- chunk_id 形如 "{doc_id}#art-{article_number}"，重复时加 "-{n}"。
- article_number 必须是阿拉伯数字字符串；article_number_raw 保留原文。
- jurisdiction 由 CLI 传入，写入每个 chunk。
- effective_date 输出 ISO 格式 (YYYY-MM-DD)。
- 测试用 fixtures/sample_statute.txt（自己造一段含"第三十八条""第三十九条""第40条"的小文本即可）；不要依赖外部网络或大文件。

测试至少覆盖：
- "第三十九条" 与 "第39条" normalize 后 article_number 相等。
- 法条 chunk 含合并后的合理长度，未把"第40条"的内容并入"第39条"。
- 合同样本能切出 contract_section 含"竞业限制"。
- ingest_directory 对空目录返回 0。

验收：
  pytest -q
  python scripts/ingest_docs.py --input tests/fixtures --source-type statute --jurisdiction CN --out /tmp/chunks.jsonl
  test -s /tmp/chunks.jsonl
```

#!/usr/bin/env python
"""把 PIPELINE_REPORT.md 渲染成自包含 HTML (含 CSS, 不依赖外部资源).

用法:
    python scripts/render_report_html.py
    # 产出 PIPELINE_REPORT.html
"""
from __future__ import annotations
import sys
from pathlib import Path

import markdown


CSS = """
:root {
  --bg: #0f1419;
  --panel: #1a1f29;
  --text: #e4e4e7;
  --muted: #a1a1aa;
  --accent: #60a5fa;
  --accent-2: #34d399;
  --code-bg: #131820;
  --border: #2a2f3a;
  --table-stripe: #1d222d;
}

@media (prefers-color-scheme: light) {
  :root {
    --bg: #ffffff;
    --panel: #fafafa;
    --text: #18181b;
    --muted: #71717a;
    --accent: #2563eb;
    --accent-2: #059669;
    --code-bg: #f4f4f5;
    --border: #e4e4e7;
    --table-stripe: #f9f9fb;
  }
}

* { box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC",
               "Microsoft YaHei", "Hiragino Sans GB", sans-serif;
  line-height: 1.7;
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem;
  background: var(--bg);
  color: var(--text);
  font-size: 15px;
}

h1, h2, h3, h4 {
  font-weight: 600;
  line-height: 1.3;
  margin-top: 2.5em;
  margin-bottom: 0.8em;
}

h1 {
  font-size: 2.2em;
  border-bottom: 3px solid var(--accent);
  padding-bottom: 0.4em;
  margin-top: 1em;
}

h2 {
  font-size: 1.7em;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.3em;
  color: var(--accent);
}

h3 {
  font-size: 1.3em;
  color: var(--accent-2);
}

h4 { font-size: 1.1em; }

p, ul, ol { margin: 0.8em 0; }

ul, ol { padding-left: 1.6em; }

li { margin: 0.3em 0; }

code {
  font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.92em;
  background: var(--code-bg);
  padding: 0.15em 0.4em;
  border-radius: 4px;
  border: 1px solid var(--border);
}

pre {
  background: var(--code-bg);
  padding: 1em 1.2em;
  border-radius: 8px;
  border: 1px solid var(--border);
  overflow-x: auto;
  margin: 1em 0;
}

pre code {
  background: transparent;
  padding: 0;
  border: none;
  font-size: 0.88em;
  line-height: 1.55;
}

table {
  border-collapse: collapse;
  width: 100%;
  margin: 1.2em 0;
  font-size: 0.93em;
}

th, td {
  border: 1px solid var(--border);
  padding: 0.55em 0.9em;
  text-align: left;
  vertical-align: top;
}

th {
  background: var(--panel);
  font-weight: 600;
  color: var(--accent);
}

tr:nth-child(2n) td { background: var(--table-stripe); }

blockquote {
  border-left: 4px solid var(--accent);
  margin: 1em 0;
  padding: 0.5em 1.2em;
  background: var(--panel);
  color: var(--muted);
  border-radius: 0 6px 6px 0;
}

a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px dotted var(--accent);
}

a:hover { border-bottom-style: solid; }

hr {
  border: 0;
  height: 1px;
  background: var(--border);
  margin: 3em 0;
}

/* TOC at top */
.toc {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1em 1.5em;
  margin: 2em 0;
  font-size: 0.95em;
}

.toc h3 { margin-top: 0; color: var(--accent); }
.toc ul { list-style: none; padding-left: 0; }
.toc > ul > li > a { font-weight: 600; }
.toc ul ul { padding-left: 1.5em; font-size: 0.95em; }

/* Strong-emphasized inline */
strong { color: var(--text); }

/* Highlight callouts (the 🚀 emoji headings) */
h2:has(em) { color: var(--accent-2); }

/* Print friendly */
@media print {
  body {
    max-width: none;
    color: black;
    background: white;
    padding: 1cm;
  }
  pre, table, blockquote {
    page-break-inside: avoid;
  }
  h1, h2, h3 { page-break-after: avoid; }
}

/* Header banner */
.header-banner {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
  color: white;
  padding: 2em 2em;
  margin: -2rem -2rem 2em -2rem;
  border-radius: 0 0 12px 12px;
}

.header-banner h1 {
  margin: 0;
  border: none;
  color: white;
  font-size: 2em;
}

.header-banner .subtitle {
  margin-top: 0.5em;
  opacity: 0.92;
  font-size: 1em;
}

/* "TL;DR" callout-like sections */
h2 + p:first-of-type { font-size: 1.02em; }
"""


def build_html(md_path: Path) -> str:
    md_text = md_path.read_text(encoding="utf-8")

    # 拿首行做 banner 标题, 移除原 # 标题让 markdown 不重复出
    first_h1 = md_text.split("\n", 1)[0].lstrip("# ").strip()
    body_md = md_text.split("\n", 1)[1] if "\n" in md_text else md_text

    md = markdown.Markdown(extensions=[
        "tables", "fenced_code", "toc", "attr_list",
        "sane_lists", "footnotes",
    ], extension_configs={
        "toc": {"toc_depth": "2-3", "title": "目录"},
    })
    body_html = md.convert(body_md)
    toc_html = md.toc

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{first_h1}</title>
<style>{CSS}</style>
</head>
<body>
<div class="header-banner">
  <h1>{first_h1}</h1>
  <div class="subtitle">完整 Pipeline 报告 · RAG + Multi-Agent 双部分 · 实测可复现</div>
</div>

<div class="toc">{toc_html}</div>

{body_html}

</body>
</html>
"""


def main() -> int:
    src = Path("PIPELINE_REPORT.md")
    if not src.exists():
        print(f"❌ {src.resolve()} 不存在", file=sys.stderr)
        return 1
    html = build_html(src)
    out = Path("PIPELINE_REPORT.html")
    out.write_text(html, encoding="utf-8")
    print(f"✓ 已生成: {out.resolve()}  ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

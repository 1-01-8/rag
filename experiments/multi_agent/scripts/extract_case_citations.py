"""CLI: extract cited articles from laws_data Q&A using Qwen.

Usage:
    cd /home/xxm/rag/experiments/multi_agent
    python -m scripts.extract_case_citations \\
        --zip /home/xxm/rag/laws_data/<train.zip> \\
        --output indexes/cases_extracted.jsonl \\
        --limit 100
"""
from __future__ import annotations
import argparse
import asyncio
import time
from pathlib import Path

from multi_agent.tools.laws_data_loader import iter_laws_data, filter_unsupported_causes
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.providers.json_robust import parse_json_robust
from multi_agent.schemas.messages import AgentMessage
from multi_agent.tracing.recorder import Recorder


EXTRACTION_PROMPT = """你是法律文本分析器。从下面的律师答复中提取被引用的法条。

要求:
1. 仅提取明确引用的"法律名+条号"组合,如"民法典 第510条"、"道路交通安全法 第76条"
2. 标准化为 doc_id 格式: "<law_short>-<arabic_number>",如 "民法典-510"
3. 若无明确引用,返回空列表
4. 不要从案件描述中推测引用,只看律师答复

输出 JSON:
```json
{{
  "doc_ids": ["民法典-510", "道路交通安全法-76"],
  "confidence": 0.85
}}
```

律师答复:
{answer}
"""


async def extract_one(provider, answer: str, model: str, rec: Recorder, agent_name: str) -> tuple[list[str], float]:
    """Returns (doc_ids, confidence)."""
    try:
        resp = await provider.complete(
            messages=[AgentMessage(role="user", content=EXTRACTION_PROMPT.format(answer=answer))],
            model=model,
            max_tokens=256,
            temperature=0,
            recorder=rec,
            agent_name=agent_name,
        )
        parsed = parse_json_robust(resp.text)
        return parsed.get("doc_ids", []), float(parsed.get("confidence", 0.0))
    except Exception as e:
        print(f"  extraction failed: {e}")
        return [], 0.0


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/extraction"))
    parser.add_argument(
        "--model", default="qwen3.5-9b",
        help="Model name passed to the provider. Default: qwen3.5-9b",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder(run_id="extract", run_dir=args.run_dir)
    rec.set_meta(zip=str(args.zip), output=str(args.output))

    provider = OpenAICompatibleProvider()
    n_total = 0
    n_with_cites = 0

    with args.output.open("w", encoding="utf-8") as out:
        records = filter_unsupported_causes(iter_laws_data(args.zip))
        t0 = time.monotonic()
        for record in records:
            if args.limit and n_total >= args.limit:
                break
            doc_ids, conf = await extract_one(provider, record.answer, args.model, rec, "extractor")
            record = record.model_copy(update={
                "extracted_cite_ids": doc_ids,
                "extraction_confidence": conf,
            })
            out.write(record.model_dump_json() + "\n")
            out.flush()
            n_total += 1
            if doc_ids:
                n_with_cites += 1
            if n_total % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = n_total / elapsed
                print(f"  {n_total} done ({n_with_cites} with cites, {rate:.1f}/s)")

    rec.close()
    print(f"Done: {n_total} records, {n_with_cites} with citations, {n_with_cites/max(n_total,1)*100:.1f}% hit rate")


if __name__ == "__main__":
    asyncio.run(main())

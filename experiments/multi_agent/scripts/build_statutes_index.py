"""CLI: build the `ma_statutes` collection from /home/xxm/rag/Chinese-Laws/extracted/.

Usage:
    cd /home/xxm/rag/experiments/multi_agent
    python -m scripts.build_statutes_index [--limit N] [--collection NAME]
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from multi_agent.tools.corpus import load_corpus
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("/home/xxm/rag/Chinese-Laws/extracted"),
    )
    parser.add_argument("--collection", default="ma_statutes")
    parser.add_argument(
        "--sparse-out",
        type=Path,
        default=Path("indexes/ma_statutes_sparse.json"),
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only index first N law files (0 = all). Useful for smoke testing.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Chunk batch size sent to dense encoder (default 16; "
             "reduce to 8/4 on CUDA illegal-memory errors).",
    )
    args = parser.parse_args()

    t0 = time.monotonic()
    if not args.corpus_dir.exists():
        raise SystemExit(f"❌ corpus-dir 不存在: {args.corpus_dir}")
    docs = load_corpus(args.corpus_dir)
    if args.limit:
        docs = docs[: args.limit]
    n_chunks = sum(len(d.chunks) for d in docs)
    if not docs:
        raise SystemExit(
            f"❌ 在 {args.corpus_dir} 没找到任何法律文件。\n"
            f"   load_corpus 期待目录下直接是 .txt 法律文件 "
            f"(例如 /home/xxm/rag/Chinese-Laws/extracted/)。\n"
            f"   验证: ls {args.corpus_dir}/*.txt | head"
        )
    print(f"Loaded {len(docs)} laws, {n_chunks} chunks. Encoding...")

    encoder = DenseEncoder()
    artifacts = build_index(
        documents=docs,
        collection_name=args.collection,
        sparse_artifact_path=args.sparse_out,
        dense_encoder=encoder,
        batch_size=args.batch_size,
    )
    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s. Sparse vocab saved to {artifacts.sparse_artifact_path}")


if __name__ == "__main__":
    main()

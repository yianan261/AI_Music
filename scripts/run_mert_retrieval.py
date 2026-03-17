#!/usr/bin/env python3
"""Run retrieval query against MERT index."""

import argparse
import numpy as np
import faiss

from ai_music.config import EMBEDDING_DIR, PROCESSED_24K_DIR
from ai_music.retrieval.mert import load_mert, search
from ai_music.utils.device import select_device


def main():
    parser = argparse.ArgumentParser(description="Query MERT retrieval index")
    parser.add_argument("--query", type=str, help="Path to query audio file")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID (e.g. 0). Default: auto-select GPU with most free memory",
    )
    args = parser.parse_args()

    index_path = EMBEDDING_DIR / "mert.index"
    names_path = EMBEDDING_DIR / "names.npy"
    if not index_path.exists() or not names_path.exists():
        raise FileNotFoundError(
            f"MERT index not found at {EMBEDDING_DIR}. "
            "Run build_mert_index.py first."
        )

    index = faiss.read_index(str(index_path))
    names = np.load(names_path, allow_pickle=True).tolist()

    device = select_device(args.gpu)
    model, processor, device = load_mert(device=device)

    if args.query:
        query_path = args.query
    else:
        first = next(PROCESSED_24K_DIR.glob("*.wav"), None)
        if not first:
            raise FileNotFoundError(
                "No processed files found and no --query given. "
                "Run preprocess.py first or provide --query."
            )
        query_path = str(first)
        print(f"Demo: using {first.name} as query")

    results = search(query_path, index, names, model, processor, device, k=args.k)
    print(f"\nTop {args.k} matches for {query_path}:")
    for name, score in results:
        print(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    main()

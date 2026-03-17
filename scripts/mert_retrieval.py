#!/usr/bin/env python3
"""
MERT-based music retrieval with FAISS.
Uses pretrained MERT-v1-330M for embeddings and FAISS for fast similarity search.
"""
import argparse
import numpy as np
import torch
import faiss

from ai_music.config import EMBEDDING_DIR, PROCESSED_DIR
from ai_music.retrieval.faiss_index import build_faiss_index
from ai_music.retrieval.mert import build_database, load_mert, search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Path to query audio file")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--build-only", action="store_true", help="Only build index, don't query")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (e.g. 2). Default: auto")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu is not None and torch.cuda.is_available() else None
    model, processor, device = load_mert(device=device)
    print(f"Loaded MERT on {device}")

    database = build_database(model, processor, device)
    print(f"Indexed {len(database)} pieces")

    index, names = build_faiss_index(database)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(EMBEDDING_DIR / "mert.index"))
    np.save(EMBEDDING_DIR / "names.npy", np.array(names))
    print(f"Saved FAISS index to {EMBEDDING_DIR}")

    if args.build_only:
        return

    if args.query:
        results = search(args.query, index, names, model, processor, device, k=args.k)
        print(f"\nTop {args.k} matches for {args.query}:")
        for name, score in results:
            print(f"  {name}: {score:.4f}")
    else:
        first = next(PROCESSED_DIR.glob("*.wav"), None)
        if first:
            results = search(str(first), index, names, model, processor, device, k=args.k)
            print(f"\nTop {args.k} matches (query: {first.name}):")
            for name, score in results:
                print(f"  {name}: {score:.4f}")
        else:
            print("No processed files found. Run preprocess.py first.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build MERT embedding index and save to data/embeddings/."""

import argparse
import numpy as np
import faiss

from ai_music.config import EMBEDDING_DIR, PROCESSED_DIR
from ai_music.retrieval.faiss_index import build_faiss_index
from ai_music.retrieval.mert import build_database, load_mert


def main():
    parser = argparse.ArgumentParser(description="Build MERT FAISS index")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID (e.g. 0). Default: auto-select GPU with most free memory",
    )
    args = parser.parse_args()

    model, processor, device = load_mert(gpu_id=args.gpu)
    print(f"Loaded MERT on {device}")

    database = build_database(model, processor, device)
    print(f"Indexed {len(database)} pieces")

    index, names = build_faiss_index(database)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(EMBEDDING_DIR / "mert.index"))
    np.save(EMBEDDING_DIR / "names.npy", np.array(names))
    print(f"Saved FAISS index to {EMBEDDING_DIR}")


if __name__ == "__main__":
    main()

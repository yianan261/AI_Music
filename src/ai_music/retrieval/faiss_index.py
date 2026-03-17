"""FAISS index building for similarity search."""

import faiss
import numpy as np


def build_faiss_index(database: dict[str, np.ndarray], use_cosine: bool = True):
    """
    Build FAISS index from embedding database.
    If use_cosine=True, L2-normalize and use inner product (cosine similarity).
    Returns (index, names_list).
    """
    names = list(database.keys())
    embeddings = np.stack(list(database.values())).astype(np.float32)

    if use_cosine:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)
    return index, names

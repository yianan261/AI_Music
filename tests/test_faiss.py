"""Tests for FAISS index build/search shape sanity."""

import numpy as np

from ai_music.retrieval.faiss_index import build_faiss_index


def test_build_faiss_index_shape():
    """Verify index has correct ntotal and embedding dimension."""
    n_items = 5
    dim = 128
    database = {f"piece_{i:03d}.wav": np.random.randn(dim).astype(np.float32) for i in range(1, n_items + 1)}

    index, names = build_faiss_index(database)

    assert index.ntotal == n_items
    assert index.d == dim
    assert len(names) == n_items


def test_faiss_search_returns_k_results():
    """Verify search returns k results with correct shape."""
    import faiss

    n_items = 10
    dim = 64
    database = {f"p{i}.wav": np.random.randn(dim).astype(np.float32) for i in range(n_items)}

    index, names = build_faiss_index(database)

    query = np.random.randn(dim).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query)

    k = 5
    D, I = index.search(query, k)

    assert D.shape == (1, k)
    assert I.shape == (1, k)
    assert len(np.unique(I[0])) == k  # k distinct results

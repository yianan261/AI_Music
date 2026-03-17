"""CQT-based baseline for music retrieval."""

import librosa
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def extract_cqt_embedding(path: str | Path, sr: int = 22050) -> np.ndarray:
    """Extract CQT-based embedding with global average pooling."""
    y, sr_out = librosa.load(path, sr=sr)
    cqt = librosa.cqt(y, sr=sr_out)
    cqt = np.abs(cqt)
    embedding = np.mean(cqt, axis=1)
    return embedding


def build_cqt_database(
    processed_dir: Path,
    sr: int = 22050,
) -> dict[str, np.ndarray]:
    """Build CQT embedding database from processed audio."""
    database = {}
    for f in sorted(processed_dir.glob("*.wav")):
        emb = extract_cqt_embedding(str(f), sr=sr)
        database[f.name] = emb
    return database


def cqt_search(
    query_path: str | Path,
    database: dict[str, np.ndarray],
    k: int = 5,
    sr: int = 22050,
) -> list[tuple[str, float]]:
    """
    Retrieve top-k matches using CQT + cosine similarity.

    Returns:
        List of (piece_name, similarity_score) sorted by score descending.
    """
    query_emb = extract_cqt_embedding(query_path, sr=sr).reshape(1, -1)
    names = list(database.keys())
    embeddings = np.stack(list(database.values()))
    sim = cosine_similarity(query_emb, embeddings)[0]
    indices = np.argsort(sim)[::-1][:k]
    return [(names[i], float(sim[i])) for i in indices]

"""CQT-based baseline for music retrieval."""

import librosa
import numpy as np
from pathlib import Path


def extract_cqt_embedding(path: str | Path, sr: int = 22050) -> np.ndarray:
    """Extract CQT-based embedding with global average pooling."""
    y, sr_out = librosa.load(path, sr=sr)
    cqt = librosa.cqt(y, sr=sr_out)
    cqt = np.abs(cqt)
    embedding = np.mean(cqt, axis=1)
    return embedding

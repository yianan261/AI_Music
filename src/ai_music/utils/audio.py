"""Audio loading and preprocessing utilities."""

import numpy as np
import soundfile as sf


def load_audio(path: str, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio file. Returns (waveform, sample_rate)."""
    import librosa

    y, sr_out = librosa.load(path, sr=sr, mono=mono)
    return y, sr_out


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Normalize waveform to [-1, 1] range."""
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y


def load_audio_sf(path: str, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio with soundfile (avoids librosa). For MERT which handles sr separately."""
    y, sr = sf.read(path, dtype="float32")
    if mono and y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr

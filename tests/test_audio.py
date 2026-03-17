"""Tests for audio utils."""

import numpy as np

from ai_music.utils.audio import normalize_audio


def test_normalize_audio():
    y = np.array([0.5, -0.3, 0.8])
    norm = normalize_audio(y)
    assert np.max(np.abs(norm)) == 1.0
    assert np.allclose(norm, y / 0.8)

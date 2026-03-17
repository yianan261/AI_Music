"""Tests for preprocessing (output sample rates)."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ai_music.data.preprocess import preprocess


def test_preprocess_output_sample_rates():
    """Verify preprocess outputs 16k and 24k files with correct sample rates."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_dir = tmp / "raw"
        out_16k = tmp / "processed_16k"
        out_24k = tmp / "processed_24k"
        input_dir.mkdir()

        # Minimal input wav (1 sec at 44.1k)
        inp_path = input_dir / "piece_001.wav"
        sf.write(inp_path, np.zeros(44100), 44100)

        n = preprocess(
            input_dir=input_dir,
            output_16k=out_16k,
            output_24k=out_24k,
        )

        assert n == 1
        y_16k, sr_16k = sf.read(out_16k / "piece_001.wav")
        y_24k, sr_24k = sf.read(out_24k / "piece_001.wav")

        assert sr_16k == 16000
        assert sr_24k == 24000
        # Approximate 1 sec at target SR (librosa resampling can vary by 1 sample)
        assert 15900 <= len(y_16k) <= 16100
        assert 23900 <= len(y_24k) <= 24100

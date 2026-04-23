"""Tests for dataset preparation (count and mapping)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ai_music.src_data.prepare_maestro import prepare_maestro


def test_prepare_maestro_count_and_mapping():
    """Verify prepare_maestro returns correct count and writes mapping with expected columns."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        maestro_dir = tmp / "maestro-v3.0.0" / "2020"
        maestro_dir.mkdir(parents=True)
        wav_path = maestro_dir / "test_piece.wav"
        sf.write(wav_path, np.zeros(16000), 16000)

        csv_path = tmp / "maestro-v3.0.0.csv"
        pd.DataFrame([
            {
                "audio_filename": "2020/test_piece.wav",
                "canonical_composer": "Test Composer",
                "canonical_title": "Test Piece",
                "split": "train",
                "year": 2020,
            }
        ]).to_csv(csv_path, index=False)

        output_dir = tmp / "raw_audio"
        metadata_path = tmp / "mapping.csv"

        n = prepare_maestro(
            n_pieces=1,
            seed=42,
            output_dir=output_dir,
            metadata_path=metadata_path,
            maestro_csv=csv_path,
            maestro_dir=tmp / "maestro-v3.0.0",
        )

        assert n == 1
        assert (output_dir / "piece_001.wav").exists()

        mapping = pd.read_csv(metadata_path)
        expected_cols = ["piece_id", "audio_filename", "composer", "title", "split", "year"]
        for col in expected_cols:
            assert col in mapping.columns, f"Missing column: {col}"
        assert len(mapping) == 1
        assert mapping["piece_id"].iloc[0] == "piece_001.wav"
        assert mapping["composer"].iloc[0] == "Test Composer"

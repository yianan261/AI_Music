"""Tests for path resolution and config."""

from pathlib import Path

from ai_music.config import (
    DATA_DIR,
    EMBEDDING_DIR,
    get_project_root,
    MAESTRO_CSV,
    METADATA_DIR,
    PROCESSED_16K_DIR,
    PROCESSED_24K_DIR,
    RAW_AUDIO_DIR,
)


def test_project_root_exists():
    root = get_project_root()
    assert root.exists()
    assert (root / "src" / "ai_music").exists()


def test_data_paths_under_project():
    root = get_project_root()
    assert DATA_DIR == root / "data"
    assert RAW_AUDIO_DIR == root / "data" / "raw_audio"
    assert PROCESSED_16K_DIR == root / "data" / "processed_16k"
    assert PROCESSED_24K_DIR == root / "data" / "processed_24k"
    assert EMBEDDING_DIR == root / "data" / "embeddings"
    assert METADATA_DIR == root / "data" / "metadata"


def test_maestro_csv_path():
    assert MAESTRO_CSV.suffix == ".csv"
    assert "maestro" in MAESTRO_CSV.name

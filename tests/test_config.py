"""Tests for config."""

from pathlib import Path

from ai_music.config import get_project_root, PROCESSED_16K_DIR, PROCESSED_24K_DIR, RAW_AUDIO_DIR


def test_project_root_exists():
    root = get_project_root()
    assert root.exists()
    assert (root / "src" / "ai_music").exists()


def test_data_paths_under_project():
    root = get_project_root()
    assert PROCESSED_16K_DIR == root / "data" / "processed_16k"
    assert PROCESSED_24K_DIR == root / "data" / "processed_24k"
    assert RAW_AUDIO_DIR == root / "data" / "raw_audio"

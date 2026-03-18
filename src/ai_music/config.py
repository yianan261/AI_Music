"""Configuration: paths, constants, model IDs."""

from pathlib import Path

from ai_music.utils.paths import get_project_root

# Data paths
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
MAESTRO_DIR = DATA_DIR / "maestro-v3.0.0"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
PROCESSED_16K_DIR = DATA_DIR / "processed_16k"  # For CQT baseline
PROCESSED_24K_DIR = DATA_DIR / "processed_24k"  # For MERT extraction
EMBEDDING_DIR = DATA_DIR / "embeddings"
METADATA_DIR = DATA_DIR / "metadata"  # Mapping files (piece_001.wav -> original metadata)
EVALUATION_QUERIES_DIR = DATA_DIR / "evaluation_queries"  # Generated snippet queries
RESULTS_DIR = PROJECT_ROOT / "results"  # Evaluation outputs (CSVs, per-query logs)

# MAESTRO
MAESTRO_CSV = PROJECT_ROOT / "data" / "maestro-v3.0.0.csv"

# Audio
TARGET_SR = 16000  # For CQT baseline compatibility
MERT_SR = 24000  # MERT-v1-330M expects 24kHz
MAX_DURATION_SEC = 30  # Truncate long audio to avoid GPU OOM

# Model
MODEL_ID = "m-a-p/MERT-v1-330M"

# AI Music - ID and Accuracy Project

Music retrieval: given a short snippet, retrieve the correct piece from a database.

## Setup

### Option A: Local (editable install)

```bash
pip install -e .
# or for dependencies only: pip install -r requirements.txt
```

### Option B: Docker (recommended for teams)

```bash
docker compose build
docker compose run --rm app python scripts/prepare_maestro.py   # after extracting MAESTRO to data/
docker compose run --rm app python scripts/preprocess.py
docker compose run --rm app python scripts/build_mert_index.py
docker compose run --rm app python scripts/run_mert_retrieval.py
```

Jupyter for notebooks:
```bash
docker compose run --rm jupyter
# Then open http://localhost:8888
```

## Pipeline

### 1. Prepare MAESTRO

After downloading and extracting [MAESTRO](https://magenta.tensorflow.org/datasets/maestro):

```bash
python scripts/prepare_maestro.py --n-pieces 100
# Or with options:
python scripts/prepare_maestro.py --n-pieces 200 --seed 42 --split train
```

This copies N audio files to `data/raw_audio/` as `piece_001.wav`, `piece_002.wav`, etc., and writes a mapping CSV to `data/metadata/maestro_mapping.csv` (piece_id, audio_filename, composer, title, split).

### 2. Preprocess

Standardize to 16kHz, mono, normalized:

```bash
python scripts/preprocess.py
```

Output: `data/processed/`

### 3. Baseline: CQT + Cosine Similarity

Classical MIR baseline (sanity check):

```bash
jupyter notebook notebooks/baseline_cqt.ipynb
```

### 4. (Optional) Check GPU availability

Before running GPU-heavy scripts, check which device will be used:

```bash
python scripts/check_gpu.py
```

When `--gpu` is not specified, the MERT scripts auto-select the GPU with the most free memory.

### 5. MERT + FAISS Retrieval

Modern baseline with pretrained MERT embeddings:

```bash
# Build index (run once)
python scripts/build_mert_index.py

# Run retrieval query
python scripts/run_mert_retrieval.py --query data/processed/piece_001.wav --k 5

# Demo: query with first processed file
python scripts/run_mert_retrieval.py --k 5
```

### 6. Structured Evaluation

Run snippet retrieval evaluation (Top-1, Top-5, MRR) comparing CQT vs frozen MERT:

```bash
# Full eval: 5s and 10s snippets, both baselines
python scripts/run_eval.py

# CQT only, 10s snippets
python scripts/run_eval.py --baselines cqt --durations 10

# MERT only with GPU 2
python scripts/run_eval.py --baselines mert --gpu 2
```

Queries are same-recording snippets (5s or 10s from each piece, starting at 5s offset). Snippets are written to `data/evaluation_queries/` by default.

## Directory Structure

```
AI_Music/
  src/ai_music/       # Reusable package
    config.py         # Paths, constants
    data/             # prepare_maestro, preprocess
    retrieval/        # mert, cqt_baseline, faiss_index
    evaluation/      # metrics, query_generation, run_eval
    utils/            # audio, paths
  scripts/            # Thin entry points
    prepare_maestro.py
    preprocess.py
    build_mert_index.py
    run_mert_retrieval.py
    run_eval.py
    check_gpu.py
  notebooks/
  tests/
  data/
    maestro-v3.0.0/   # Extracted MAESTRO (year subdirs)
    maestro-v3.0.0.csv
    raw_audio/        # piece_001.wav, ...
    processed/        # Normalized 16kHz
    embeddings/       # mert.index, names.npy
    metadata/         # maestro_mapping.csv (piece_id -> audio_filename, composer, title, split)
    evaluation_queries/
  pyproject.toml
  README.md
```

### Using the package

```python
from ai_music.data import prepare_maestro, preprocess
from ai_music.retrieval import load_mert, extract_mert_embedding, build_faiss_index
from ai_music.config import PROCESSED_DIR, RAW_AUDIO_DIR
```

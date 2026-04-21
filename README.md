# AI Music — Cover ID and Accuracy Project

Music retrieval: given a short snippet, retrieve the correct piece from a database (CQT baseline, frozen MERT embeddings, and optional contrastive fine-tuning of a projection head).

---

## Setup

### Option A: Local (editable install)

From the project root:

```bash
cd /path/to/AI_Music
pip install -e .
# Optional: pip install -r requirements.txt
```

This installs dependencies from `pyproject.toml` (including `torch`, `transformers`, `faiss-gpu`, etc.) and makes the `ai_music` package importable.

### Option B: Docker

```bash
docker compose build
docker compose run --rm app pip install -e .
# Then run pipeline steps inside the container (see below)
```

Jupyter:

```bash
docker compose run --rm jupyter
# Open http://localhost:8888
```

---

## End-to-end pipeline (CLI)

Run these from the **project root** (`AI_Music/`) unless noted.

### 1. Download MAESTRO v3.0.0

Use the official dataset page: **[The MAESTRO Dataset](https://magenta.withgoogle.com/datasets/maestro)**.

- Download **`maestro-v3.0.0.zip`** (full audio + MIDI; large, ~101 GB compressed / ~120 GB unpacked for v3 per the site).
- Download **`maestro-v3.0.0.csv`** (metadata with `split`, `audio_filename`, etc.) if it is not already inside your zip.

Place the zip (and CSV if separate) under `data/`, for example:

```text
data/maestro-v3.0.0.zip
data/maestro-v3.0.0.csv    # if not bundled with the zip
```

MAESTRO v3.0.0 defines a suggested train/validation/test split so the same composition does not appear in multiple splits. See the [dataset page](https://magenta.withgoogle.com/datasets/maestro) for details and citation.

### 2. Unzip MAESTRO fully

**Important:** The archive must be **fully** extracted so every year folder and WAV from the CSV exist on disk. A partial extract (only a few year folders) will leave hundreds of CSV rows with missing files.

From `data/` (or unzip into `data/` from the project root):

```bash
cd data
unzip -o maestro-v3.0.0.zip
```

Unpacking can take a long time and needs **~120 GB+** free disk space (uncompressed size per the official release).

Expected layout after a complete extract:

```text
data/maestro-v3.0.0/
  2004/ …
  2006/ …
  …
  2018/ …
  (year folders with WAV/MIDI pairs)
```

### 3. Verify WAV count (sanity check)

**MAESTRO v3.0.0 should have 1276 performances total** (962 train + 137 validation + 177 test per the official statistics).

```bash
cd data
find maestro-v3.0.0 -name "*.wav" | wc -l
```

You want **`1276`**, not a small number like **297**.

**If you only see a few year folders** (e.g. `2004`, `2006`, `2011`) and ~297 WAVs, but the CSV still lists `2008`, `2009`, `2013`, … then the zip was **not** fully unpacked. Many CSV rows will correctly show as “no audio file on disk” until you extract the full archive. After a full `unzip`, rerun the prepare step (below).

### 4. MERT model (Hugging Face)

This project uses **MERT v1 330M** from Hugging Face:

| Setting | Value |
|--------|--------|
| Model ID | `m-a-p/MERT-v1-330M` |
| Config | `src/ai_music/config.py` → `MODEL_ID` |

The first time you run retrieval, training, or eval that loads MERT, `transformers` will **download weights from Hugging Face** into your local cache (`~/.cache/huggingface/` by default). No separate manual download is required unless your environment blocks the network; in that case, pre-download on a machine with access and sync the cache, or set `HF_HOME` / `TRANSFORMERS_CACHE` as documented by [Hugging Face](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

```python
# Equivalent to what the code does (for reference only)
from transformers import AutoModel, Wav2Vec2FeatureExtractor
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
```

### 5. Prepare MAESTRO → `raw_audio/` (full corpus)

Copy **every** MAESTRO WAV that exists on disk into `data/raw_audio/` as `piece_001.wav`, … and write `data/metadata/maestro_mapping.csv`.

```bash
cd /path/to/AI_Music
python scripts/prepare_maestro.py --all
```

- Use **`--all`** for the full corpus (~1276 pieces after a complete extract).
- Smaller experiments: `python scripts/prepare_maestro.py --n-pieces 100` (optional: `--seed 42`, `--split train`).

If many rows are skipped, re-check step 3 (full unzip).

### 6. Preprocess → `processed_16k/` and `processed_24k/`

Mono, normalized resampling: **16 kHz** for the CQT baseline, **24 kHz** for MERT (matches `MERT-v1-330M`).

```bash
python scripts/preprocess.py
```

Outputs:

- `data/processed_16k/` — CQT / 16 kHz pipeline  
- `data/processed_24k/` — MERT / 24 kHz pipeline  

### 7. (Optional) GPU check (shared servers)

```bash
python scripts/check_gpus.py
# Then pin a GPU, e.g.:
CUDA_VISIBLE_DEVICES=0 python scripts/run_mert_retrieval.py --k 5
```

### 8. Contrastive training (projection head on frozen MERT)

```bash
python scripts/train_contrastive.py --gpu 0 --epochs 50
```

Useful flags (see `train.py` / `--help`):

- `--batch-size 8`
- `--lr 1e-3`
- `--val-fraction 0.15` — held-out **piece-level** validation; `projection_best.pt` is chosen by validation loss when validation is active (needs ≥4 processed pieces).
- `--save-dir checkpoints/` — defaults to project `checkpoints/`

Checkpoints:

- `checkpoints/projection_best.pt` — best weights (by val loss if val is used)
- `checkpoints/projection_final.pt` — last epoch
- `checkpoints/checkpoint_latest.pt` — full resume state

### 9. MERT + FAISS retrieval (baseline demo)

```bash
python scripts/build_mert_index.py
python scripts/run_mert_retrieval.py --query data/processed_24k/piece_001.wav --k 5
python scripts/run_mert_retrieval.py --k 5   # uses first processed file if no --query
```

### 10. Structured evaluation (Top-1, Top-5, MRR)

```bash
python scripts/run_eval.py
python scripts/run_eval.py --baselines cqt --durations 10
python scripts/run_eval.py --baselines mert --gpu 0
python scripts/run_eval.py --baselines mert_finetuned --checkpoint checkpoints/projection_best.pt --gpu 0
```

### 11. CQT baseline notebook

```bash
jupyter notebook notebooks/baseline_cqt.ipynb
```

---

## Docker one-liner examples

After `docker compose build` and mounting `data/`:

```bash
docker compose run --rm app python scripts/prepare_maestro.py --all
docker compose run --rm app python scripts/preprocess.py
docker compose run --rm app python scripts/build_mert_index.py
docker compose run --rm app python scripts/run_mert_retrieval.py
```

---

## Shared-server protocol

On unmanaged GPU nodes:

- Run `python scripts/check_gpus.py` or `nvidia-smi` before heavy jobs.
- Set `CUDA_VISIBLE_DEVICES=<id>` explicitly.
- Coordinate with teammates for long runs.

---

## Project structure

```text
AI_Music/
├── pyproject.toml              # Package metadata & dependencies
├── README.md
├── Dockerfile
├── docker-compose.yml          # app / jupyter services
│
├── scripts/                    # CLI entry points (thin wrappers)
│   ├── prepare_maestro.py      # MAESTRO → raw_audio + mapping CSV
│   ├── preprocess.py           # raw_audio → processed_16k & processed_24k
│   ├── train_contrastive.py    # Contrastive fine-tuning (projection head)
│   ├── build_mert_index.py     # FAISS index over MERT embeddings
│   ├── run_mert_retrieval.py   # Query the index
│   ├── run_eval.py             # Top-k / MRR evaluation
│   └── check_gpus.py           # Advisory GPU memory check
│
├── src/ai_music/               # Importable package
│   ├── config.py               # Paths, MODEL_ID (MERT), constants
│   ├── data/
│   │   ├── dataset.py          # Triplet dataset for training
│   │   ├── prepare_maestro.py
│   │   └── preprocess.py
│   ├── retrieval/
│   │   ├── mert.py             # load_mert, embeddings, search helpers
│   │   ├── faiss_index.py
│   │   └── cqt_baseline.py
│   ├── training/
│   │   ├── train.py            # Training loop & validation
│   │   ├── model.py            # MERTEmbedder, ProjectionHead
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── run_eval.py
│   │   ├── metrics.py
│   │   └── query_generation.py
│   └── utils/                  # audio, device, paths
│
├── notebooks/
│   └── baseline_cqt.ipynb
├── tests/
├── checkpoints/                # Created by training (gitignored if applicable)
├── results/                    # Evaluation runs (CSVs, summaries)
│
└── data/                       # Not in git; create locally
    ├── maestro-v3.0.0.zip      # Downloaded archive (optional location)
    ├── maestro-v3.0.0.csv      # Metadata (from dataset release)
    ├── maestro-v3.0.0/         # Fully extracted MAESTRO tree (year subdirs)
    ├── raw_audio/              # piece_001.wav, … from prepare_maestro
    ├── processed_16k/          # Resampled for CQT
    ├── processed_24k/          # Resampled for MERT
    ├── embeddings/             # e.g. FAISS index + name lists
    ├── metadata/               # maestro_mapping.csv
    └── evaluation_queries/     # Generated eval snippets (default)
```

### Using the package in Python

```python
from ai_music.data import prepare_maestro, preprocess
from ai_music.retrieval import load_mert, extract_mert_embedding, build_faiss_index
from ai_music.config import PROCESSED_16K_DIR, PROCESSED_24K_DIR, RAW_AUDIO_DIR, MODEL_ID
```

---

## License note (MAESTRO)

MAESTRO is distributed under **CC BY-NC-SA 4.0** (non-commercial). See the [official dataset page](https://magenta.withgoogle.com/datasets/maestro) for license and citation.

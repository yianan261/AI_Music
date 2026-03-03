"""
Prepare MAESTRO dataset for MVP: select 100 pieces and copy audio to data/raw_audio/.
Run this after extracting the MAESTRO dataset.
"""
import os
import shutil
import pandas as pd
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAESTRO_DIR = PROJECT_ROOT / "data" / "maestro-v3.0.0"
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
N_PIECES = 100

def main():
    csv_path = PROJECT_ROOT / "data" / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MAESTRO CSV not found at {csv_path}. "
            "Download from https://magenta.tensorflow.org/datasets/maestro"
        )

    df = pd.read_csv(csv_path)
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if not MAESTRO_DIR.exists():
        raise FileNotFoundError(
            f"MAESTRO directory not found at {MAESTRO_DIR}. "
            "Extract maestro-v3.0.0.zip into data/ first."
        )
    audio_root = MAESTRO_DIR

    # Select first N pieces that actually exist (handles partial extraction)
    selected = []
    for _, row in df.iterrows():
        if len(selected) >= N_PIECES:
            break
        src = audio_root / row["audio_filename"]
        if src.exists():
            selected.append(row)

    for i, row in enumerate(selected):
        audio_rel = row["audio_filename"]
        src = audio_root / audio_rel
        dst = RAW_AUDIO_DIR / f"piece_{i + 1:03d}.wav"

        shutil.copy2(src, dst)
        print(f"Copied: {audio_rel} -> {dst.name}")

    n = len(selected)
    print(f"\nPrepared {n} pieces in {RAW_AUDIO_DIR}")
    if n < N_PIECES:
        print(f"Note: Only {n} of {N_PIECES} requested. Extract more years from the zip if needed.")

if __name__ == "__main__":
    main()

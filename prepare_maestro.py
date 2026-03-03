"""
Prepare MAESTRO dataset for MVP: select 100 pieces and copy audio to data/raw_audio/.
Run this after extracting the MAESTRO dataset.
"""
import os
import shutil
import pandas as pd
from pathlib import Path

# Paths - adjust MAESTRO_DIR if your extraction location differs
# MAESTRO extracts to a folder; audio files live in year subdirs (e.g. 2018/xxx.wav)
MAESTRO_DIR = Path("data/maestro-v3.0.0")
RAW_AUDIO_DIR = Path("data/raw_audio")
N_PIECES = 100

def main():
    csv_path = Path("data/maestro-v3.0.0.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MAESTRO CSV not found at {csv_path}. "
            "Download from https://magenta.tensorflow.org/datasets/maestro"
        )

    df = pd.read_csv(csv_path)
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Try to find MAESTRO root: could be data/maestro-v3.0.0 or data/
    audio_root = MAESTRO_DIR if MAESTRO_DIR.exists() else Path("data")
    if not (audio_root / df.iloc[0]["audio_filename"]).exists():
        audio_root = Path("data")

    # Select first N pieces
    selected = df.head(N_PIECES)

    for i, row in selected.iterrows():
        audio_rel = row["audio_filename"]
        src = audio_root / audio_rel
        dst = RAW_AUDIO_DIR / f"piece_{i + 1:03d}.wav"  # piece_001.wav, piece_002.wav, ...

        if not src.exists():
            print(f"Skipping (not found): {src}")
            continue

        shutil.copy2(src, dst)
        print(f"Copied: {audio_rel} -> {dst.name}")

    print(f"\nPrepared {len(list(RAW_AUDIO_DIR.glob('*.wav')))} pieces in {RAW_AUDIO_DIR}")

if __name__ == "__main__":
    main()

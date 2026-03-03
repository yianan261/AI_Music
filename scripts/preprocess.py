"""
Basic preprocessing: standardize audio to 16kHz, mono, normalized.
MERT-v1-330M expects 24kHz; we keep 16kHz for CQT baseline compatibility.
MERT extraction will resample to 24kHz on the fly.
"""
import librosa
import numpy as np
import os
import soundfile as sf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "raw_audio"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_SR = 16000

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"{INPUT_DIR} not found. Run prepare_maestro.py first."
        )

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav"))
    for file in files:
        path = INPUT_DIR / file
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        sf.write(OUTPUT_DIR / file, y, TARGET_SR)
        print(f"Processed: {file}")

    print(f"\nProcessed {len(files)} files -> {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

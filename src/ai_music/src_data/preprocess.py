"""Preprocess audio: standardize to mono, normalized; output 16kHz and 24kHz variants."""

from pathlib import Path

import soundfile as sf

from ai_music import config
from ai_music.utils.audio import load_audio, normalize_audio


def preprocess(
    input_dir: Path | None = None,
    output_16k: Path | None = None,
    output_24k: Path | None = None,
) -> int:
    """
    Preprocess audio: mono, normalized to [-1,1].
    Writes to both data/processed_16k/ (CQT) and data/processed_24k/ (MERT).
    Returns number of files processed.
    """
    input_dir = input_dir or config.RAW_AUDIO_DIR
    output_16k = output_16k or config.PROCESSED_16K_DIR
    output_24k = output_24k or config.PROCESSED_24K_DIR
    output_16k = Path(output_16k)
    output_24k = Path(output_24k)
    output_16k.mkdir(parents=True, exist_ok=True)
    output_24k.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"{input_dir} not found. Run prepare_maestro first."
        )

    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() == ".wav")
    for path in files:
        y_16k, _ = load_audio(str(path), sr=config.TARGET_SR, mono=True)
        y_16k = normalize_audio(y_16k)
        sf.write(output_16k / path.name, y_16k, config.TARGET_SR)

        y_24k, _ = load_audio(str(path), sr=config.MERT_SR, mono=True)
        y_24k = normalize_audio(y_24k)
        sf.write(output_24k / path.name, y_24k, config.MERT_SR)

        print(f"Processed: {path.name}")

    print(f"\nProcessed {len(files)} files -> {output_16k}")
    print(f"           {len(files)} files -> {output_24k}")
    return len(files)

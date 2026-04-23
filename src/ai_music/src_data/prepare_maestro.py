"""Prepare MAESTRO dataset: select N pieces, copy to raw_audio, write mapping metadata."""

import random
import shutil
import pandas as pd
from pathlib import Path

from ai_music.config import (
    MAESTRO_CSV,
    MAESTRO_DIR,
    METADATA_DIR,
    RAW_AUDIO_DIR,
)


def prepare_maestro(
    n_pieces: int = 100,
    seed: int = 42,
    split: str | None = None,
    use_all: bool = False,
    output_dir: Path | None = None,
    metadata_path: Path | None = None,
    maestro_csv: Path | None = None,
    maestro_dir: Path | None = None,
) -> int:
    """
    Copy N MAESTRO pieces to raw_audio/ and write mapping metadata.

    Args:
        n_pieces: Number of pieces to copy (ignored if use_all=True).
        seed: Random seed for reproducible subset selection.
        split: Filter by MAESTRO split (train/validation/test). None = all.
        use_all: If True, copy every WAV in the CSV that exists on disk, in stable
            filename order (recommended for full ~1.2k MAESTRO scale-up).
        output_dir: Where to copy WAVs (default: config.RAW_AUDIO_DIR).
        metadata_path: Where to save mapping CSV (default: config.METADATA_DIR/maestro_mapping.csv).
    maestro_csv: Override MAESTRO CSV path (for testing).
    maestro_dir: Override MAESTRO directory (for testing).

    Returns:
        Number of pieces copied.
    """
    csv_path = maestro_csv or MAESTRO_CSV
    audio_root = maestro_dir or MAESTRO_DIR

    if not csv_path.exists():
        raise FileNotFoundError(
            f"MAESTRO CSV not found at {csv_path}. "
            "Download from https://magenta.tensorflow.org/datasets/maestro"
        )

    if not audio_root.exists():
        raise FileNotFoundError(
            f"MAESTRO directory not found at {audio_root}. "
            "Extract maestro-v3.0.0.zip into data/ first."
        )

    output_dir = output_dir or RAW_AUDIO_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_path or METADATA_DIR / "maestro_mapping.csv"

    df = pd.read_csv(csv_path)

    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)

    candidates = []
    for _, row in df.iterrows():
        src = audio_root / row["audio_filename"]
        if src.exists():
            candidates.append(row)

    skipped_missing_files = len(df) - len(candidates)

    random.seed(seed)
    if not candidates:
        selected = []
    elif use_all:
        selected = sorted(candidates, key=lambda r: str(r["audio_filename"]))
    else:
        selected = random.sample(candidates, min(n_pieces, len(candidates)))

    mapping_rows = []
    for i, row in enumerate(selected):
        audio_rel = row["audio_filename"]
        src = audio_root / audio_rel
        dst = output_dir / f"piece_{i + 1:03d}.wav"
        shutil.copy2(src, dst)
        print(f"Copied: {audio_rel} -> {dst.name}")

        mapping_rows.append({
            "piece_id": dst.name,
            "audio_filename": audio_rel,
            "composer": row.get("canonical_composer", ""),
            "title": row.get("canonical_title", ""),
            "split": row.get("split", ""),
            "year": row.get("year", ""),
        })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(metadata_path, index=False)
    print(f"\nSaved mapping to {metadata_path}")

    n = len(selected)
    print(f"Prepared {n} pieces in {output_dir}")
    if skipped_missing_files:
        print(f"Note: {skipped_missing_files} CSV rows had no audio file on disk (skipped).")
        if skipped_missing_files > len(df) // 2:
            print(
                "      This usually means MAESTRO audio was only partially extracted. "
                f"Fully unzip {audio_root.name} from maestro-v3.0.0.zip into {audio_root} "
                f"(expected ~{len(df)} WAVs under year subfolders, not a handful)."
            )
    elif n < n_pieces and len(candidates) >= n_pieces:
        print(f"Note: Only {n} of {n_pieces} requested (random sample). Re-run with different seed if needed.")
    elif n < n_pieces:
        print(f"Note: Only {n} of {n_pieces} requested. Extract more from the zip if needed.")
    return n

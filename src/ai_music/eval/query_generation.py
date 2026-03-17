"""Query generation for evaluation: same-recording snippet extraction."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ai_music import config
from ai_music.utils.audio import load_audio


@dataclass
class QuerySample:
    """A query snippet with its ground-truth piece."""

    snippet_path: Path
    ground_truth: str  # e.g. "piece_001.wav"
    duration_sec: float


def extract_snippet(
    audio_path: Path,
    duration_sec: float,
    start_offset_sec: float = 5.0,
    sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Extract a snippet from an audio file.

    Args:
        audio_path: Path to full audio file.
        duration_sec: Length of snippet in seconds.
        start_offset_sec: Start time in seconds (avoids lead-in silence).
        sr: Sample rate (default: config.TARGET_SR). Resamples if different.

    Returns:
        (waveform, sample_rate) as (numpy array, sr).
    """
    sr = sr or config.TARGET_SR
    y, file_sr = load_audio(str(audio_path), sr=sr, mono=True)

    start_sample = int(start_offset_sec * file_sr)
    n_samples = int(duration_sec * file_sr)

    if start_sample + n_samples > len(y):
        # Fallback: take from start if file is too short
        start_sample = 0
        n_samples = min(n_samples, len(y))

    snippet = y[start_sample : start_sample + n_samples]
    return np.asarray(snippet, dtype=np.float32), file_sr


def generate_snippets(
    processed_dir: Path | None = None,
    durations_sec: list[float] | None = None,
    start_offset_sec: float = 5.0,
    output_dir: Path | None = None,
    seed: int = 42,
) -> list[QuerySample]:
    """
    Generate query snippets from processed audio (same-recording retrieval).

    For each piece, extracts one snippet per requested duration. Snippets are
    written to output_dir and returned as QuerySample list.

    Args:
        processed_dir: Directory of processed WAV files (default: config.PROCESSED_DIR).
        durations_sec: List of snippet durations, e.g. [5.0, 10.0].
        start_offset_sec: Start time in source file (avoids lead-in).
        output_dir: Where to write snippet WAVs. If None, uses temp dir.
        seed: Reserved for future random start-offset sampling.

    Returns:
        List of QuerySample(snippet_path, ground_truth, duration_sec).
    """
    processed_dir = processed_dir or config.PROCESSED_DIR
    durations_sec = durations_sec or [5.0, 10.0]
    use_temp = output_dir is None

    if use_temp:
        output_dir = Path(tempfile.mkdtemp(prefix="ai_music_eval_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[QuerySample] = []
    files = sorted(processed_dir.glob("*.wav"))

    for path in files:
        name = path.name
        for dur in durations_sec:
            y_arr, sr = extract_snippet(
                path,
                duration_sec=dur,
                start_offset_sec=start_offset_sec,
            )
            stem = path.stem
            out_name = f"{stem}_{int(dur)}s.wav"
            out_path = output_dir / out_name
            sf.write(out_path, y_arr, sr)
            samples.append(QuerySample(snippet_path=out_path, ground_truth=name, duration_sec=dur))

    return samples

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
    ground_truth: str
    duration_sec: float


def extract_snippet(audio_path: Path, duration_sec: float, start_offset_sec: float = 5.0, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Extract a snippet from an audio file."""
    sr = sr or config.TARGET_SR
    y, file_sr = load_audio(str(audio_path), sr=sr, mono=True)
    start_sample = int(start_offset_sec * file_sr)
    n_samples = int(duration_sec * file_sr)
    if start_sample + n_samples > len(y):
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
    """Generate query snippets from processed audio."""
    processed_dir = processed_dir or config.PROCESSED_DIR
    durations_sec = durations_sec or [5.0, 10.0]
    use_temp = output_dir is None
    output_dir = Path(tempfile.mkdtemp(prefix="ai_music_evaluation_")) if use_temp else Path(output_dir)
    if not use_temp:
        output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for path in sorted(processed_dir.glob("*.wav")):
        for dur in durations_sec:
            y_arr, sr = extract_snippet(path, dur, start_offset_sec)
            out_path = output_dir / f"{path.stem}_{int(dur)}s.wav"
            sf.write(out_path, y_arr, sr)
            samples.append(QuerySample(snippet_path=out_path, ground_truth=path.name, duration_sec=dur))
    return samples

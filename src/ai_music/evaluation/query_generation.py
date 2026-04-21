"""Query generation for evaluation: same-recording snippet extraction with augmentations.

Snippet *durations* are chosen by the caller (e.g. run_eval defaults to 10s and 15s
for enough melodic/harmonic context in slow classical piano).
"""

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
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
    start_offset_sec: float = 5.0
    augmentation_type: str = "none"  # "none", "tempo_up", "tempo_down", "pitch_up", "noise", etc.


def apply_augmentation(y: np.ndarray, sr: int, aug_type: str) -> np.ndarray:
    """Apply audio augmentation to a snippet.

    Supported aug_type values:
        "tempo_up"   – +10% tempo (time-stretch rate=1.1)
        "tempo_down" – -15% tempo (time-stretch rate=0.85)
        "pitch_up"   – +1 semitone
        "pitch_down" – -1 semitone
        "noise"      – additive white noise (σ=0.005)
        "none"       – no-op
    """
    if aug_type == "tempo_up":
        return librosa.effects.time_stretch(y, rate=1.1)
    elif aug_type == "tempo_down":
        return librosa.effects.time_stretch(y, rate=0.85)
    elif aug_type == "pitch_up":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
    elif aug_type == "pitch_down":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
    elif aug_type == "noise":
        rng = np.random.default_rng()
        noise = rng.normal(0, 0.005, y.shape).astype(y.dtype)
        return y + noise
    return y


def extract_snippet(
    audio_path: Path,
    duration_sec: float,
    start_offset_sec: float = 5.0,
    sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Extract a snippet from an audio file."""
    sr = sr or config.MERT_SR
    y, file_sr = load_audio(str(audio_path), sr=sr, mono=True)
    start_sample = int(start_offset_sec * file_sr)
    n_samples = int(duration_sec * file_sr)
    if start_sample + n_samples > len(y):
        start_sample = max(0, len(y) - n_samples)
        n_samples = min(n_samples, len(y))
    if start_sample < 0:
        start_sample = 0
    snippet = y[start_sample : start_sample + n_samples]
    return np.asarray(snippet, dtype=np.float32), file_sr


def _sample_valid_start(
    file_duration_sec: float,
    snippet_duration_sec: float,
    seed: int | None = None,
) -> float:
    """Sample a valid start time such that start + duration <= file_duration."""
    max_start = max(0.0, file_duration_sec - snippet_duration_sec - 0.01)
    if max_start <= 0:
        return 0.0
    if seed is not None:
        random.seed(seed)
    return random.uniform(0.0, max_start)


def generate_snippets(
    processed_dir: Path | None = None,
    durations_sec: list[float] | None = None,
    n_snippets_per_piece: int = 1,
    randomize_start: bool = True,
    start_offset_sec: float = 5.0,
    output_dir: Path | None = None,
    seed: int = 42,
    sr: int | None = None,
    augmentation_type: str = "none",
) -> list[QuerySample]:
    """
    Generate query snippets from processed audio.

    Args:
        processed_dir: Source directory (default: processed_24k).
        durations_sec: Snippet durations, e.g. [5.0, 10.0].
        n_snippets_per_piece: Number of snippets per (piece, duration) pair.
        randomize_start: If True, sample start time randomly; else use start_offset_sec.
        start_offset_sec: Fixed start time when randomize_start=False.
        output_dir: Where to write snippet WAVs.
        seed: Random seed for reproducibility.
        sr: Sample rate for output.
        augmentation_type: "none", "tempo_up", "tempo_down", "pitch_up", "pitch_down", "noise".

    Returns:
        List of QuerySample.
    """
    processed_dir = processed_dir or config.PROCESSED_24K_DIR
    sr = sr or config.MERT_SR
    durations_sec = durations_sec or [5.0, 10.0]
    use_temp = output_dir is None
    output_dir = Path(tempfile.mkdtemp(prefix="ai_music_evaluation_")) if use_temp else Path(output_dir)
    if not use_temp:
        output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    samples = []
    files = sorted(processed_dir.glob("*.wav"))

    for fi, path in enumerate(files):
        y, file_sr = load_audio(str(path), sr=sr, mono=True)
        file_duration_sec = len(y) / file_sr

        for di, dur in enumerate(durations_sec):
            for i in range(n_snippets_per_piece):
                if randomize_start:
                    snippet_seed = seed + fi * 1000 + di * 10 + i
                    start = _sample_valid_start(file_duration_sec, dur, seed=snippet_seed)
                else:
                    start = start_offset_sec

                y_arr, out_sr = extract_snippet(path, dur, start, sr=sr)
                if augmentation_type != "none":
                    y_arr = apply_augmentation(y_arr, out_sr, augmentation_type)
                stem = path.stem
                suffix = f"_{augmentation_type}" if augmentation_type != "none" else ""
                out_name = f"{stem}_{int(dur)}s_{i}" if n_snippets_per_piece > 1 else f"{stem}_{int(dur)}s"
                out_name = f"{out_name}{suffix}.wav"
                out_path = output_dir / out_name
                sf.write(out_path, y_arr, out_sr)

                samples.append(QuerySample(
                    snippet_path=out_path,
                    ground_truth=path.name,
                    duration_sec=dur,
                    start_offset_sec=start,
                    augmentation_type=augmentation_type,
                ))
    return samples

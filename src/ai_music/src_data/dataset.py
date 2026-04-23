"""Triplet dataset for contrastive fine-tuning of MERT embeddings.

Each sample returns (anchor, positive, negative) waveform tensors:
  - Anchor:   random snippet from piece P (clean)
  - Positive: different random snippet from the same piece P, often augmented
              (pitch shift p=0.5 — balances pitch invariance vs clean positives)
  - Negative: random snippet from a different piece
"""

import random
from pathlib import Path

import audiomentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

from ai_music.utils.audio import load_audio


def build_augmentation_pipeline(sr: int = 24000) -> A.Compose:
    """Build augmentation chain balancing pitch invariance vs clean positives.

    Stress-test results (MRR drop from baseline):
        pitch_up/down:  -46% to -69%  ← catastrophic, must train against this
        tempo:          -3%  to -11%  ← moderate
        noise:          ~0%  to -14%  ← mild

    Pitch shift is still frequent (p=0.5) but not always: some positives stay
    unshifted so the model can use exact spectral detail for Top-1 (mitigates
    over-compression from always-augmented positives on small subsets).
    """
    return A.Compose([
        A.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),

        # Tempo: moderate probability
        A.TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),

        # Noise: light probability, low sigma
        A.AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.008, p=0.3),

        # Gain variation: simulate different recording levels
        A.Gain(min_gain_db=-6, max_gain_db=6, p=0.3),

        # High/low pass: simulate different mic/room characteristics
        A.OneOf([
            A.HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=400, p=1.0),
            A.LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=7500, p=1.0),
        ], p=0.2),
    ])


class MusicTripletDataset(Dataset):
    """PyTorch Dataset that yields (anchor, positive, negative) waveform tensors."""

    def __init__(
        self,
        processed_dir: Path,
        sr: int = 24000,
        snippet_duration: float = 15.0,
        augment_positive: bool = True,
        seed: int = 42,
        piece_paths: list[Path] | None = None,
    ):
        self.sr = sr
        self.snippet_duration = snippet_duration
        self.augment_positive = augment_positive
        self.rng = random.Random(seed)

        if piece_paths is not None:
            self.piece_paths = sorted(piece_paths)
        else:
            self.piece_paths = sorted(processed_dir.glob("*.wav"))
        if len(self.piece_paths) < 2:
            where = processed_dir if piece_paths is None else f"{len(piece_paths)} paths"
            raise ValueError(f"Need >=2 pieces (for negatives), got {len(self.piece_paths)} from {where}")

        self._audio_cache: dict[int, np.ndarray] = {}
        self.augment = build_augmentation_pipeline(sr) if augment_positive else None

    def __len__(self) -> int:
        return len(self.piece_paths)

    def _load_piece(self, idx: int) -> np.ndarray:
        """Load and cache a full piece (avoids re-reading from disk each call)."""
        if idx not in self._audio_cache:
            y, _ = load_audio(str(self.piece_paths[idx]), sr=self.sr, mono=True)
            self._audio_cache[idx] = y.astype(np.float32)
        return self._audio_cache[idx]

    def _random_snippet(self, idx: int) -> np.ndarray:
        """Extract a random snippet from piece at index idx."""
        y = self._load_piece(idx)
        n_samples = int(self.snippet_duration * self.sr)

        max_start = max(0, len(y) - n_samples)
        start = self.rng.randint(0, max_start) if max_start > 0 else 0
        snippet = y[start : start + n_samples]

        if len(snippet) < n_samples:
            snippet = np.pad(snippet, (0, n_samples - len(snippet)))
        return snippet

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Returns (anchor, positive, negative, piece_label)."""
        anchor = self._random_snippet(idx)

        positive = self._random_snippet(idx)
        if self.augment is not None:
            positive = self.augment(samples=positive, sample_rate=self.sr)

        neg_idx = idx
        while neg_idx == idx:
            neg_idx = self.rng.randint(0, len(self.piece_paths) - 1)
        negative = self._random_snippet(neg_idx)

        return (
            torch.from_numpy(anchor),
            torch.from_numpy(positive),
            torch.from_numpy(negative),
            idx,
        )

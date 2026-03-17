"""MERT-based embeddings for music retrieval."""

import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from ai_music.config import (
    EMBEDDING_DIR,
    MAX_DURATION_SEC,
    MERT_SR,
    MODEL_ID,
    PROCESSED_24K_DIR,
)


def load_mert(device: str | None = None):
    """Load MERT model and processor."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device).eval()
    return model, processor, device


def extract_mert_embedding(path: str | Path, model, processor, device) -> np.ndarray:
    """
    Extract MERT embedding from audio file.
    Uses soundfile to load. Resamples to 24kHz if needed.
    """
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

    max_samples = int(MAX_DURATION_SEC * sr)
    if len(y) > max_samples:
        y = y[:max_samples]

    waveform = torch.from_numpy(y).unsqueeze(0)

    if sr != MERT_SR:
        resampler = torchaudio.transforms.Resample(sr, MERT_SR)
        waveform = resampler(waveform)

    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=MERT_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.last_hidden_state
    embedding = hidden.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def build_database(model, processor, device, processed_dir: Path | None = None) -> dict:
    """Build embedding database from processed audio (24kHz)."""
    processed_dir = processed_dir or PROCESSED_24K_DIR
    database = {}
    for f in sorted(processed_dir.glob("*.wav")):
        emb = extract_mert_embedding(str(f), model, processor, device)
        database[f.name] = emb
    return database


def search(
    query_path: str | Path,
    index,
    names: list[str],
    model,
    processor,
    device,
    k: int = 5,
    use_cosine: bool = True,
) -> list[tuple[str, float]]:
    """Retrieve top-k matches for a query audio file."""
    import faiss

    query_emb = extract_mert_embedding(query_path, model, processor, device)
    query_emb = query_emb.reshape(1, -1).astype(np.float32)

    if use_cosine:
        faiss.normalize_L2(query_emb)

    D, I = index.search(query_emb, k)
    return [(names[i], float(D[0][j])) for j, i in enumerate(I[0])]

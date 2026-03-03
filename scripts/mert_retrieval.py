"""
MERT-based music retrieval with FAISS.
Uses pretrained MERT-v1-330M for embeddings and FAISS for fast similarity search.

MERT expects 24kHz audio; processor handles resampling.
"""
import torch
import torchaudio
import soundfile as sf
import faiss
import numpy as np
from pathlib import Path
from transformers import AutoModel, Wav2Vec2FeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "m-a-p/MERT-v1-330M"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"
MERT_SR = 24000  # MERT-v1-330M expects 24kHz
MAX_DURATION_SEC = 30  # Truncate long audio to avoid GPU OOM


def load_mert(device=None):
    """Load MERT model and processor."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device).eval()
    return model, processor, device


def extract_mert_embedding(path, model, processor, device):
    """
    Extract MERT embedding from audio file.
    Uses soundfile to load (avoids torchcodec/FFmpeg deps). Resamples to 24kHz if needed.
    """
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    # Truncate long audio to avoid GPU OOM
    max_samples = int(MAX_DURATION_SEC * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    waveform = torch.from_numpy(y).unsqueeze(0)

    if sr != MERT_SR:
        resampler = torchaudio.transforms.Resample(sr, MERT_SR)
        waveform = resampler(waveform)

    # Processor expects (batch, samples) and sample_rate
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=MERT_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Use last hidden state, temporal mean pooling
    hidden = outputs.last_hidden_state
    embedding = hidden.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def build_database(model, processor, device, processed_dir=PROCESSED_DIR):
    """Build embedding database from processed audio."""
    database = {}
    files = sorted(processed_dir.glob("*.wav"))
    for f in files:
        emb = extract_mert_embedding(str(f), model, processor, device)
        database[f.name] = emb
    return database


def build_faiss_index(database, use_cosine=True):
    """
    Build FAISS index. If use_cosine=True, L2-normalize and use inner product.
    """
    names = list(database.keys())
    embeddings = np.stack(list(database.values())).astype(np.float32)

    if use_cosine:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1]) # exact search (cosine similarity)
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)
    return index, names


def search(query_path, index, names, model, processor, device, k=5):
    """Retrieve top-k matches for a query audio file."""
    query_emb = extract_mert_embedding(query_path, model, processor, device)
    query_emb = query_emb.reshape(1, -1).astype(np.float32)

    # Must match index type (normalized for IndexFlatIP)
    if isinstance(index, faiss.IndexFlatIP):
        faiss.normalize_L2(query_emb)

    D, I = index.search(query_emb, k)
    return [(names[i], float(D[0][j])) for j, i in enumerate(I[0])]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Path to query audio file")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--build-only", action="store_true", help="Only build index, don't query")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (e.g. 2). Default: auto")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu is not None and torch.cuda.is_available() else None
    model, processor, device = load_mert(device=device)
    print(f"Loaded MERT on {device}")

    database = build_database(model, processor, device)
    print(f"Indexed {len(database)} pieces")

    index, names = build_faiss_index(database)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(EMBEDDING_DIR / "mert.index"))
    np.save(EMBEDDING_DIR / "names.npy", np.array(names))
    print(f"Saved FAISS index to {EMBEDDING_DIR}")

    if args.build_only:
        return

    if args.query:
        results = search(args.query, index, names, model, processor, device, k=args.k)
        print(f"\nTop {args.k} matches for {args.query}:")
        for name, score in results:
            print(f"  {name}: {score:.4f}")
    else:
        # Demo: query with first file
        first = next(PROCESSED_DIR.glob("*.wav"), None)
        if first:
            results = search(str(first), index, names, model, processor, device, k=args.k)
            print(f"\nTop {args.k} matches (query: {first.name}):")
            for name, score in results:
                print(f"  {name}: {score:.4f}")
        else:
            print("No processed files found. Run preprocess.py first.")


if __name__ == "__main__":
    main()

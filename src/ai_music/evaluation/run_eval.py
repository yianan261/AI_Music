"""Run structured evaluation: snippet retrieval with Top-1, Top-5, MRR."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ai_music import config
from ai_music.evaluation.metrics import build_results_df, compute_all_metrics, save_results_csv
from ai_music.evaluation.query_generation import QuerySample, generate_snippets


def run_cqt_eval(samples: list[QuerySample], processed_dir: Path | None = None) -> list[list[str]]:
    from ai_music.retrieval.cqt_baseline import build_cqt_database, cqt_search

    processed_dir = processed_dir or config.PROCESSED_16K_DIR
    database = build_cqt_database(processed_dir)
    return [[r[0] for r in cqt_search(q.snippet_path, database, k=len(database))] for q in samples]


def run_mert_eval(samples: list[QuerySample], processed_dir: Path | None = None, device: str | None = None) -> list[list[str]]:
    from ai_music.retrieval.faiss_index import build_faiss_index
    from ai_music.retrieval.mert import build_database, load_mert, search

    processed_dir = processed_dir or config.PROCESSED_24K_DIR
    model, processor, dev = load_mert(device=device)
    database = build_database(model, processor, dev, processed_dir=processed_dir)
    index, names = build_faiss_index(database)
    return [[r[0] for r in search(q.snippet_path, index, names, model, processor, dev, k=len(names))] for q in samples]


def _embed_file_finetuned(path: Path, embedder, device: str) -> np.ndarray:
    """Run a single audio file through the fine-tuned MERT+projection pipeline."""
    from ai_music.utils.audio import load_audio

    y, sr = load_audio(str(path), sr=config.MERT_SR, mono=True)
    max_samples = int(config.MAX_DURATION_SEC * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    waveform = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)
    embedder.eval()
    with torch.no_grad():
        emb = embedder(waveform)
    return emb.squeeze(0).cpu().numpy()


def run_mert_finetuned_eval(
    samples: list[QuerySample],
    checkpoint_path: Path,
    processed_dir: Path | None = None,
    device: str | None = None,
) -> list[list[str]]:
    """Evaluate using MERT + fine-tuned projection head."""
    import faiss
    from ai_music.retrieval.mert import load_mert
    from ai_music.training.model import MERTEmbedder, ProjectionHead

    processed_dir = processed_dir or config.PROCESSED_24K_DIR
    mert_model, processor, dev = load_mert(device=device)

    head = ProjectionHead(input_dim=mert_model.config.hidden_size)
    head.load_state_dict(torch.load(checkpoint_path, map_location=dev, weights_only=True))
    embedder = MERTEmbedder(mert_model, processor, head, freeze_backbone=True).to(dev)
    embedder.eval()

    # Build database embeddings
    db_files = sorted(processed_dir.glob("*.wav"))
    names = [f.name for f in db_files]
    db_embs = np.stack([_embed_file_finetuned(f, embedder, dev) for f in db_files]).astype(np.float32)
    faiss.normalize_L2(db_embs)
    index = faiss.IndexFlatIP(db_embs.shape[1])
    index.add(db_embs)

    # Query
    rankings = []
    for q in samples:
        q_emb = _embed_file_finetuned(q.snippet_path, embedder, dev).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, len(names))
        rankings.append([names[i] for i in I[0]])
    return rankings


def run_evaluation(
    durations_sec=None,
    baselines=None,
    output_dir=None,
    device=None,
    eval_type: str = "same_recording",
    n_snippets_per_piece: int = 1,
    randomize_start: bool = True,
    seed: int = 42,
    checkpoint: Path | None = None,
):
    """
    Run evaluation and save results to results/.

    eval_type: "same_recording" | "augmented" | "cross_performance"
        - same_recording: snippets from same recording as reference (default)
        - augmented: tempo/noise augmented snippets (stub for now)
        - cross_performance: different performance of same piece (stub for now)
    """
    durations_sec = durations_sec or [10.0, 15.0]
    baselines = baselines or ["cqt", "mert"]
    queries_dir = output_dir or config.EVALUATION_QUERIES_DIR

    if "mert_finetuned" in baselines and checkpoint is None:
        default_ckpt = config.PROJECT_ROOT / "checkpoints" / "projection_best.pt"
        if default_ckpt.exists():
            checkpoint = default_ckpt
        else:
            raise ValueError("--checkpoint required for mert_finetuned baseline")

    # Baseline-specific snippet sources: CQT from 16k, MERT/fine-tuned from 24k
    BASELINE_SOURCES = {
        "cqt": (config.PROCESSED_16K_DIR, config.TARGET_SR),
        "mert": (config.PROCESSED_24K_DIR, config.MERT_SR),
        "mert_finetuned": (config.PROCESSED_24K_DIR, config.MERT_SR),
    }

    # Results go under results/<eval_type>/
    results_root = config.RESULTS_DIR / eval_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    AUGMENTATION_TYPES = ("none", "tempo_up", "tempo_down", "pitch_up", "pitch_down", "noise")
    augmentation_type = "none" if eval_type == "same_recording" else eval_type
    if augmentation_type not in AUGMENTATION_TYPES:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}. Choose from {AUGMENTATION_TYPES}")

    results = {}

    for baseline in baselines:
        processed_dir, sr = BASELINE_SOURCES[baseline]
        baseline_queries_dir = Path(queries_dir) / baseline
        print(f"\nRunning {baseline}...")
        print(f"  Generating snippets from {processed_dir.name} ({sr} Hz)...")
        samples = generate_snippets(
            processed_dir=processed_dir,
            durations_sec=list(durations_sec),
            n_snippets_per_piece=n_snippets_per_piece,
            randomize_start=randomize_start,
            output_dir=baseline_queries_dir,
            seed=seed,
            sr=sr,
            augmentation_type=augmentation_type,
        )
        print(f"  Generated {len(samples)} queries")
        gt = [q.ground_truth for q in samples]

        if baseline == "cqt":
            rankings = run_cqt_eval(samples)
        elif baseline == "mert_finetuned":
            rankings = run_mert_finetuned_eval(samples, checkpoint, device=device)
        else:
            rankings = run_mert_eval(samples, device=device)
        metrics = compute_all_metrics(rankings, gt)
        results[baseline] = {"overall": metrics}
        print(f"  Top-1: {metrics['top1']:.2%}  Top-5: {metrics['top5']:.2%}  MRR: {metrics['mrr']:.4f}")

        # Per-query CSV
        df = build_results_df(samples, rankings, baseline)
        csv_path = run_dir / f"per_query_{baseline}.csv"
        save_results_csv(df, csv_path)
        print(f"  Saved per-query results to {csv_path}")

        # Per-duration breakdown
        by_dur = {}
        for q, r in zip(samples, rankings):
            by_dur.setdefault(q.duration_sec, []).append((r, q.ground_truth))
        for dur, pairs in by_dur.items():
            m = compute_all_metrics([p[0] for p in pairs], [p[1] for p in pairs])
            results[baseline][f"{int(dur)}s"] = m
            print(f"  {int(dur)}s: Top-1={m['top1']:.2%}  Top-5={m['top5']:.2%}  MRR={m['mrr']:.4f}")

    # Save summary metrics
    summary_path = run_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"eval_type={eval_type} n_queries={len(samples)} seed={seed}\n\n")
        for baseline, data in results.items():
            f.write(f"{baseline}:\n")
            for k, v in data.items():
                if isinstance(v, dict):
                    f.write(f"  {k}: top1={v['top1']:.4f} top5={v['top5']:.4f} mrr={v['mrr']:.4f}\n")
            f.write("\n")
    print(f"\nResults saved to {run_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run snippet retrieval evaluation")
    parser.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=[10.0, 15.0],
        help="Snippet durations (10s/15s default: more context for slow classical piano)",
    )
    parser.add_argument("--baselines", type=str, nargs="+", default=["cqt", "mert"], choices=["cqt", "mert", "mert_finetuned"])
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save query snippets")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID for MERT")
    parser.add_argument(
        "--eval-type",
        type=str,
        default="same_recording",
        choices=["same_recording", "tempo_up", "tempo_down", "pitch_up", "pitch_down", "noise", "cross_performance"],
        help="Evaluation type: same_recording (no aug), or an augmentation name",
    )
    parser.add_argument("--n-snippets", type=int, default=1, help="Snippets per (piece, duration)")
    parser.add_argument("--no-randomize-start", action="store_true", help="Use fixed 5s start instead of random")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to projection_best.pt for mert_finetuned")
    args = parser.parse_args()

    from ai_music.utils.device import select_device
    device = select_device(args.gpu)
    run_evaluation(
        durations_sec=args.durations,
        baselines=args.baselines,
        output_dir=args.output_dir,
        device=device,
        eval_type=args.eval_type,
        n_snippets_per_piece=args.n_snippets,
        randomize_start=not args.no_randomize_start,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )

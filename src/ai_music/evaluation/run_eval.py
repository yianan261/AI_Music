"""Run structured evaluation: snippet retrieval with Top-1, Top-5, MRR."""

import argparse
from datetime import datetime
from pathlib import Path

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


def run_evaluation(
    durations_sec=None,
    baselines=None,
    output_dir=None,
    device=None,
    eval_type: str = "same_recording",
    n_snippets_per_piece: int = 1,
    randomize_start: bool = True,
    seed: int = 42,
):
    """
    Run evaluation and save results to results/.

    eval_type: "same_recording" | "augmented" | "cross_performance"
        - same_recording: snippets from same recording as reference (default)
        - augmented: tempo/noise augmented snippets (stub for now)
        - cross_performance: different performance of same piece (stub for now)
    """
    durations_sec = durations_sec or [5.0, 10.0]
    baselines = baselines or ["cqt", "mert"]
    queries_dir = output_dir or config.EVALUATION_QUERIES_DIR

    # Results go under results/<eval_type>/
    results_root = config.RESULTS_DIR / eval_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    augmentation_type = "none" if eval_type == "same_recording" else eval_type

    print(f"Generating query snippets (eval_type={eval_type}, randomize_start={randomize_start})...")
    samples = generate_snippets(
        durations_sec=list(durations_sec),
        n_snippets_per_piece=n_snippets_per_piece,
        randomize_start=randomize_start,
        output_dir=queries_dir,
        seed=seed,
        augmentation_type=augmentation_type,
    )
    print(f"  Generated {len(samples)} queries")
    gt = [q.ground_truth for q in samples]
    results = {}

    for baseline in baselines:
        print(f"\nRunning {baseline}...")
        rankings = run_cqt_eval(samples) if baseline == "cqt" else run_mert_eval(samples, device=device)
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
    parser.add_argument("--durations", type=float, nargs="+", default=[5.0, 10.0], help="Snippet durations")
    parser.add_argument("--baselines", type=str, nargs="+", default=["cqt", "mert"], choices=["cqt", "mert"])
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save query snippets")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID for MERT")
    parser.add_argument(
        "--eval-type",
        type=str,
        default="same_recording",
        choices=["same_recording", "augmented", "cross_performance"],
        help="Evaluation type (augmented/cross_performance are stubs for now)",
    )
    parser.add_argument("--n-snippets", type=int, default=1, help="Snippets per (piece, duration)")
    parser.add_argument("--no-randomize-start", action="store_true", help="Use fixed 5s start instead of random")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    )

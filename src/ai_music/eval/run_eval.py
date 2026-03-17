"""Run structured evaluation: snippet retrieval with Top-1, Top-5, MRR."""

import argparse
from pathlib import Path

from ai_music import config
from ai_music.eval.metrics import compute_all_metrics
from ai_music.eval.query_generation import QuerySample, generate_snippets


def run_cqt_eval(
    samples: list[QuerySample],
    processed_dir: Path | None = None,
) -> list[list[str]]:
    """Run CQT baseline on query samples. Returns rankings (list of ranked names per query)."""
    from ai_music.retrieval.cqt_baseline import build_cqt_database, cqt_search

    processed_dir = processed_dir or config.PROCESSED_DIR
    database = build_cqt_database(processed_dir)
    rankings = []
    for q in samples:
        results = cqt_search(q.snippet_path, database, k=len(database))
        rankings.append([r[0] for r in results])
    return rankings


def run_mert_eval(
    samples: list[QuerySample],
    processed_dir: Path | None = None,
    device: str | None = None,
) -> list[list[str]]:
    """Run MERT + FAISS on query samples. Returns rankings."""
    import torch

    from ai_music.retrieval.faiss_index import build_faiss_index
    from ai_music.retrieval.mert import build_database, load_mert, search

    processed_dir = processed_dir or config.PROCESSED_DIR
    model, processor, dev = load_mert(device=device)

    database = build_database(model, processor, dev, processed_dir=processed_dir)
    index, names = build_faiss_index(database)

    rankings = []
    for q in samples:
        results = search(
            q.snippet_path,
            index,
            names,
            model,
            processor,
            dev,
            k=len(names),
        )
        rankings.append([r[0] for r in results])
    return rankings


def run_evaluation(
    durations_sec: list[float] | None = None,
    baselines: list[str] | None = None,
    output_dir: Path | None = None,
    device: str | None = None,
) -> dict:
    """
    Run full evaluation pipeline.

    Args:
        durations_sec: Snippet durations (default [5, 10]).
        baselines: Which baselines to run (default ["cqt", "mert"]).
        output_dir: Where to save query snippets (default: config.EVAL_QUERIES_DIR).
        device: Device for MERT (default: auto).

    Returns:
        Dict of {baseline: {duration: metrics}} and overall metrics.
    """
    durations_sec = durations_sec or [5.0, 10.0]
    baselines = baselines or ["cqt", "mert"]
    output_dir = output_dir or config.EVAL_QUERIES_DIR

    print("Generating query snippets...")
    samples = generate_snippets(
        durations_sec=list(durations_sec),
        output_dir=output_dir,
    )
    print(f"  Generated {len(samples)} queries")

    gt = [q.ground_truth for q in samples]
    results: dict = {}

    for baseline in baselines:
        print(f"\nRunning {baseline}...")
        if baseline == "cqt":
            rankings = run_cqt_eval(samples)
        elif baseline == "mert":
            rankings = run_mert_eval(samples, device=device)
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        metrics = compute_all_metrics(rankings, gt)
        results[baseline] = {"overall": metrics}
        print(f"  Top-1: {metrics['top1']:.2%}  Top-5: {metrics['top5']:.2%}  MRR: {metrics['mrr']:.4f}")

        # Per-duration breakdown
        by_dur: dict[float, list[tuple[list[str], str]]] = {}
        for q, r in zip(samples, rankings):
            by_dur.setdefault(q.duration_sec, []).append((r, q.ground_truth))
        for dur, pairs in by_dur.items():
            ranks = [p[0] for p in pairs]
            gt_dur = [p[1] for p in pairs]
            m = compute_all_metrics(ranks, gt_dur)
            results[baseline][f"{int(dur)}s"] = m
            print(f"  {int(dur)}s: Top-1={m['top1']:.2%}  Top-5={m['top5']:.2%}  MRR={m['mrr']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run snippet retrieval evaluation")
    parser.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=[5.0, 10.0],
        help="Snippet durations in seconds (default: 5 10)",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["cqt", "mert"],
        choices=["cqt", "mert"],
        help="Baselines to evaluate (default: cqt mert)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save query snippets (default: data/eval_queries)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID for MERT (e.g. 0)",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu is not None else None
    run_evaluation(
        durations_sec=args.durations,
        baselines=args.baselines,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()

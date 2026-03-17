"""Run structured evaluation: snippet retrieval with Top-1, Top-5, MRR."""

import argparse
from pathlib import Path

from ai_music import config
from ai_music.evaluation.metrics import compute_all_metrics
from ai_music.evaluation.query_generation import QuerySample, generate_snippets


def run_cqt_eval(samples: list[QuerySample], processed_dir: Path | None = None) -> list[list[str]]:
    from ai_music.retrieval.cqt_baseline import build_cqt_database, cqt_search

    processed_dir = processed_dir or config.PROCESSED_DIR
    database = build_cqt_database(processed_dir)
    return [[r[0] for r in cqt_search(q.snippet_path, database, k=len(database))] for q in samples]


def run_mert_eval(samples: list[QuerySample], processed_dir: Path | None = None, device: str | None = None) -> list[list[str]]:
    from ai_music.retrieval.faiss_index import build_faiss_index
    from ai_music.retrieval.mert import build_database, load_mert, search

    processed_dir = processed_dir or config.PROCESSED_DIR
    model, processor, dev = load_mert(device=device)
    database = build_database(model, processor, dev, processed_dir=processed_dir)
    index, names = build_faiss_index(database)
    return [[r[0] for r in search(q.snippet_path, index, names, model, processor, dev, k=len(names))] for q in samples]


def run_evaluation(durations_sec=None, baselines=None, output_dir=None, device=None):
    durations_sec = durations_sec or [5.0, 10.0]
    baselines = baselines or ["cqt", "mert"]
    output_dir = output_dir or config.EVALUATION_QUERIES_DIR

    print("Generating query snippets...")
    samples = generate_snippets(durations_sec=list(durations_sec), output_dir=output_dir)
    print(f"  Generated {len(samples)} queries")
    gt = [q.ground_truth for q in samples]
    results = {}

    for baseline in baselines:
        print(f"\nRunning {baseline}...")
        rankings = run_cqt_eval(samples) if baseline == "cqt" else run_mert_eval(samples, device=device)
        metrics = compute_all_metrics(rankings, gt)
        results[baseline] = {"overall": metrics}
        print(f"  Top-1: {metrics['top1']:.2%}  Top-5: {metrics['top5']:.2%}  MRR: {metrics['mrr']:.4f}")
        by_dur = {}
        for q, r in zip(samples, rankings):
            by_dur.setdefault(q.duration_sec, []).append((r, q.ground_truth))
        for dur, pairs in by_dur.items():
            m = compute_all_metrics([p[0] for p in pairs], [p[1] for p in pairs])
            results[baseline][f"{int(dur)}s"] = m
            print(f"  {int(dur)}s: Top-1={m['top1']:.2%}  Top-5={m['top5']:.2%}  MRR={m['mrr']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run snippet retrieval evaluation")
    parser.add_argument("--durations", type=float, nargs="+", default=[5.0, 10.0], help="Snippet durations")
    parser.add_argument("--baselines", type=str, nargs="+", default=["cqt", "mert"], choices=["cqt", "mert"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID for MERT")
    args = parser.parse_args()

    from ai_music.utils.device import select_device
    device = select_device(args.gpu)
    run_evaluation(durations_sec=args.durations, baselines=args.baselines, output_dir=args.output_dir, device=device)

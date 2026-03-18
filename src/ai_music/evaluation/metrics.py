"""Evaluation metrics for retrieval."""

from pathlib import Path

import numpy as np
import pandas as pd

from ai_music.evaluation.query_generation import QuerySample


def top1_accuracy(rankings: list[list[str]], ground_truth: list[str]) -> float:
    """Top-1 accuracy: fraction of queries where the correct piece is ranked first."""
    if not rankings or not ground_truth:
        return 0.0
    correct = sum(1 for pred, gt in zip(rankings, ground_truth) if pred and pred[0] == gt)
    return correct / len(ground_truth)


def top5_accuracy(rankings: list[list[str]], ground_truth: list[str]) -> float:
    """Top-5 accuracy: fraction of queries where the correct piece appears in the top 5."""
    if not rankings or not ground_truth:
        return 0.0
    correct = sum(1 for pred, gt in zip(rankings, ground_truth) if pred and gt in pred[:5])
    return correct / len(ground_truth)


def mrr(rankings: list[list[str]], ground_truth: list[str]) -> float:
    """Mean Reciprocal Rank: average of 1/rank for the first correct hit."""
    if not rankings or not ground_truth:
        return 0.0
    reciprocals = []
    for pred, gt in zip(rankings, ground_truth):
        try:
            rank = pred.index(gt) + 1
            reciprocals.append(1.0 / rank)
        except ValueError:
            reciprocals.append(0.0)
    return float(np.mean(reciprocals))


def get_rank_of_correct(pred_list: list[str], ground_truth: str) -> int | None:
    """Return 1-indexed rank of ground_truth in pred_list, or None if not found."""
    try:
        return pred_list.index(ground_truth) + 1
    except ValueError:
        return None


def compute_all_metrics(rankings: list[list[str]], ground_truth: list[str]) -> dict[str, float]:
    """Compute Top-1, Top-5, and MRR."""
    return {
        "top1": top1_accuracy(rankings, ground_truth),
        "top5": top5_accuracy(rankings, ground_truth),
        "mrr": mrr(rankings, ground_truth),
    }


def build_results_df(
    samples: list[QuerySample],
    rankings: list[list[str]],
    baseline: str,
) -> pd.DataFrame:
    """
    Build per-query results DataFrame for analysis.

    Columns: query_file, ground_truth, top5_1..top5_5, rank_of_correct,
             snippet_duration, augmentation_type, baseline.
    """
    rows = []
    for q, pred in zip(samples, rankings):
        rank = get_rank_of_correct(pred, q.ground_truth)
        top5 = pred[:5] if pred else []
        row = {
            "query_file": q.snippet_path.name,
            "ground_truth": q.ground_truth,
            "top5_1": top5[0] if len(top5) > 0 else "",
            "top5_2": top5[1] if len(top5) > 1 else "",
            "top5_3": top5[2] if len(top5) > 2 else "",
            "top5_4": top5[3] if len(top5) > 3 else "",
            "top5_5": top5[4] if len(top5) > 4 else "",
            "rank_of_correct": rank if rank is not None else -1,
            "snippet_duration": q.duration_sec,
            "augmentation_type": q.augmentation_type,
            "baseline": baseline,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_results_csv(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save per-query results to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

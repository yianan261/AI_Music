"""Evaluation metrics for retrieval."""

import numpy as np


def top1_accuracy(rankings: list[list[str]], ground_truth: list[str]) -> float:
    """
    Top-1 accuracy: fraction of queries where the correct piece is ranked first.

    Args:
        rankings: Per-query ranked list of retrieved piece names (e.g. ["piece_001.wav", ...]).
        ground_truth: Per-query ground-truth piece name.

    Returns:
        Accuracy in [0, 1].
    """
    if not rankings or not ground_truth:
        return 0.0
    correct = sum(1 for pred, gt in zip(rankings, ground_truth) if pred and pred[0] == gt)
    return correct / len(ground_truth)


def top5_accuracy(rankings: list[list[str]], ground_truth: list[str]) -> float:
    """
    Top-5 accuracy: fraction of queries where the correct piece appears in the top 5.

    Args:
        rankings: Per-query ranked list of retrieved piece names.
        ground_truth: Per-query ground-truth piece name.

    Returns:
        Accuracy in [0, 1].
    """
    if not rankings or not ground_truth:
        return 0.0
    correct = 0
    for pred, gt in zip(rankings, ground_truth):
        top5 = pred[:5] if pred else []
        if gt in top5:
            correct += 1
    return correct / len(ground_truth)


def mrr(rankings: list[list[str]], ground_truth: list[str]) -> float:
    """
    Mean Reciprocal Rank: average of 1/rank for the first correct hit.

    If the correct piece is at rank 1, contributes 1.0; at rank 2, 0.5; etc.
    If not found in the list, contributes 0.

    Args:
        rankings: Per-query ranked list of retrieved piece names.
        ground_truth: Per-query ground-truth piece name.

    Returns:
        MRR in [0, 1].
    """
    if not rankings or not ground_truth:
        return 0.0
    reciprocals = []
    for pred, gt in zip(rankings, ground_truth):
        try:
            rank = pred.index(gt) + 1  # 1-indexed
            reciprocals.append(1.0 / rank)
        except ValueError:
            reciprocals.append(0.0)
    return float(np.mean(reciprocals))


def compute_all_metrics(
    rankings: list[list[str]],
    ground_truth: list[str],
) -> dict[str, float]:
    """
    Compute Top-1, Top-5, and MRR.

    Returns:
        Dict with keys "top1", "top5", "mrr".
    """
    return {
        "top1": top1_accuracy(rankings, ground_truth),
        "top5": top5_accuracy(rankings, ground_truth),
        "mrr": mrr(rankings, ground_truth),
    }

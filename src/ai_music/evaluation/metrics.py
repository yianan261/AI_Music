"""Evaluation metrics for retrieval."""

import numpy as np


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


def compute_all_metrics(rankings: list[list[str]], ground_truth: list[str]) -> dict[str, float]:
    """Compute Top-1, Top-5, and MRR."""
    return {
        "top1": top1_accuracy(rankings, ground_truth),
        "top5": top5_accuracy(rankings, ground_truth),
        "mrr": mrr(rankings, ground_truth),
    }

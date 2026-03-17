"""Tests for evaluation module."""

import numpy as np

from ai_music.eval.metrics import top1_accuracy, top5_accuracy, mrr, compute_all_metrics


def test_top1_accuracy():
    rankings = [
        ["a.wav", "b.wav", "c.wav"],
        ["x.wav", "y.wav", "z.wav"],
        ["m.wav", "n.wav", "o.wav"],
    ]
    gt = ["a.wav", "y.wav", "o.wav"]  # 1st correct, 2nd correct (rank 2), 3rd correct (rank 3)
    assert top1_accuracy(rankings, gt) == 1 / 3


def test_top5_accuracy():
    rankings = [
        ["a.wav", "b.wav", "c.wav"],
        ["x.wav", "y.wav", "z.wav"],
    ]
    gt = ["c.wav", "y.wav"]  # first: c at rank 3 (in top5), second: y at rank 2 (in top5)
    assert top5_accuracy(rankings, gt) == 1.0


def test_mrr():
    rankings = [
        ["a.wav", "b.wav", "c.wav"],
        ["x.wav", "y.wav", "z.wav"],
        ["m.wav", "n.wav", "o.wav"],
    ]
    gt = ["a.wav", "y.wav", "z.wav"]  # rank 1 -> 1, rank 2 -> 0.5, not found -> 0
    expected = (1.0 + 0.5 + 0.0) / 3
    assert np.isclose(mrr(rankings, gt), expected)


def test_compute_all_metrics():
    rankings = [["a.wav", "b.wav"], ["b.wav", "a.wav"]]
    gt = ["a.wav", "b.wav"]
    m = compute_all_metrics(rankings, gt)
    assert "top1" in m
    assert "top5" in m
    assert "mrr" in m
    assert m["top1"] == 1.0
    assert m["top5"] == 1.0
    assert m["mrr"] == 1.0

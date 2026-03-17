"""Tests for retrieval metrics (Top-1, Top-5, MRR)."""

import numpy as np

from ai_music.evaluation.metrics import top1_accuracy, top5_accuracy, mrr, compute_all_metrics


def test_top1_accuracy():
    rankings = [["a.wav", "b.wav"], ["x.wav", "y.wav"], ["m.wav", "n.wav"]]
    gt = ["a.wav", "y.wav", "o.wav"]
    assert top1_accuracy(rankings, gt) == 1 / 3


def test_top5_accuracy():
    rankings = [["a.wav", "b.wav", "c.wav"], ["x.wav", "y.wav", "z.wav"]]
    gt = ["c.wav", "y.wav"]
    assert top5_accuracy(rankings, gt) == 1.0


def test_mrr():
    rankings = [["a.wav", "b.wav"], ["x.wav", "y.wav"], ["m.wav", "n.wav"]]
    gt = ["a.wav", "y.wav", "z.wav"]
    assert np.isclose(mrr(rankings, gt), (1.0 + 0.5 + 0.0) / 3)


def test_compute_all_metrics():
    rankings = [["a.wav", "b.wav"], ["b.wav", "a.wav"]]
    gt = ["a.wav", "b.wav"]
    m = compute_all_metrics(rankings, gt)
    assert m["top1"] == m["top5"] == m["mrr"] == 1.0

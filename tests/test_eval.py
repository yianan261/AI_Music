"""Tests for evaluation module (query generation, run_eval integration)."""


def test_evaluation_imports():
    """Smoke test: evaluation package imports and QuerySample."""
    from ai_music.evaluation import QuerySample, compute_all_metrics

    rankings = [["a.wav", "b.wav"], ["b.wav", "a.wav"]]
    gt = ["a.wav", "b.wav"]
    m = compute_all_metrics(rankings, gt)
    assert m["top1"] == 1.0

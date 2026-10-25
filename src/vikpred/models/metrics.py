"""Probability-forecast metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Average gap between predicted and actual win rate, weighted by bin size."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    gaps = [abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum()
            for b in range(bins) if (idx == b).any()]
    return float(sum(gaps) / len(y))


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, np.asarray(p) > 0.5),
        "ece": expected_calibration_error(y, p),
    }


def bootstrap_log_loss_diff(y, p_model, p_ref, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% CI of (model - reference) per-game log loss. Negative = model better."""
    y, p_model, p_ref = map(np.asarray, (y, p_model, p_ref))
    eps = 1e-15

    def per_game(p):
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    diff = per_game(p_model) - per_game(p_ref)
    rng = np.random.default_rng(seed)
    samples = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)

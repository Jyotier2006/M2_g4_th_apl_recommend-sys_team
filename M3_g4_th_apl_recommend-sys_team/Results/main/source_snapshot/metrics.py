"""All prediction KPIs use the same five-class rating distribution."""
import numpy as np


def reliability_bins(r, p, n_bins=10):
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) + 1 == r
    group = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    result = []
    for j in range(n_bins):
        mask = group == j
        result.append(dict(bin=j, count=int(mask.sum()),
                           confidence=float(confidence[mask].mean()) if mask.any() else 0.0,
                           accuracy=float(correct[mask].mean()) if mask.any() else 0.0))
    return result


def evaluate(r, p):
    if p.shape != (len(r), 5) or not np.all(np.isfinite(p)):
        raise ValueError("Invalid rating probabilities")
    if np.min(p) < 0 or not np.allclose(p.sum(axis=1), 1, atol=1e-9):
        raise ValueError("Probabilities must be normalized")
    expected = p @ np.arange(1, 6)
    median = (np.cumsum(p, axis=1) < 0.5).sum(axis=1) + 1
    target = np.eye(5)[r - 1]
    bins = reliability_bins(r, p)
    ece = sum(b["count"] * abs(b["confidence"] - b["accuracy"]) for b in bins) / len(r)
    cdf = np.cumsum(p, axis=1)
    lo, hi = (cdf < 0.05).sum(axis=1) + 1, (cdf < 0.95).sum(axis=1) + 1
    return dict(mae=float(np.abs(expected - r).mean()),
                rmse=float(np.sqrt(np.mean((expected - r) ** 2))),
                nll=float(-np.log(np.maximum(p[np.arange(len(r)), r - 1], 1e-15)).mean()),
                brier=float(np.sum((p - target) ** 2, axis=1).mean()),
                ece=float(ece), median_mae=float(np.abs(median - r).mean()),
                interval90_coverage=float(((r >= lo) & (r <= hi)).mean()),
                interval90_width=float((hi - lo + 1).mean()))

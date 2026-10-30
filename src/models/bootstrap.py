"""Bootstrap confidence intervals (paired: same resamples for every model)."""
import numpy as np


def bootstrap_metric(y, scores: dict, metric, n_boot: int = 2000, seed: int = 42) -> dict:
    """Har model ke liye metric ke n_boot values. Resamples sab models ke liye same, to differences paired hote hain."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    out = {k: [] for k in scores}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():          # ek hi class aa gayi, AUC define nahi
            continue
        for k, s in scores.items():
            out[k].append(metric(y[idx], np.asarray(s)[idx]))
    return {k: np.array(v) for k, v in out.items()}


def ci(arr, level: float = 0.95):
    lo, hi = np.percentile(arr, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def precision_at_frac(y, s, frac: float) -> float:
    k = max(1, int(len(y) * frac))
    return float(np.asarray(y)[np.argsort(-np.asarray(s))[:k]].mean())


def recall_at_frac(y, s, frac: float) -> float:
    k = max(1, int(len(y) * frac))
    y = np.asarray(y)
    return float(y[np.argsort(-np.asarray(s))[:k]].sum() / max(1, y.sum()))
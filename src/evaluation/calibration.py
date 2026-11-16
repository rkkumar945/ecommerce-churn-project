"""Development-only calibration helpers.

Fit calibrators using out-of-fold predictions only from the training data.
Never fit a calibrator on the test split and never overwrite the committed model artifact.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _as_1d(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def oof_scores(model, X, y, *, cv: int = 5, random_state: int = 42):
    """Return out-of-fold scores from the training set only."""
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    proba = cross_val_predict(model, X, y, cv=cv_obj, method="predict_proba")
    return _as_1d(proba[:, 1])


def fit_calibrator(scores, labels, *, kind: str = "sigmoid"):
    """Fit a calibration model on training/validation predictions, not on test data."""
    s = _as_1d(scores)
    y = _as_1d(labels).astype(int)
    if len(s) != len(y):
        raise ValueError("scores and labels must have the same length")
    if kind == "sigmoid":
        clf = LogisticRegression(max_iter=2000)
        clf.fit(s.reshape(-1, 1), y)
        return clf
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s, y)
        return iso
    raise ValueError("kind must be one of {'sigmoid', 'isotonic'}")


def evaluate_calibration(scores, labels, *, kind: str = "sigmoid"):
    """Return reliability-oriented calibration metrics for a development-only evaluator."""
    s = _as_1d(scores)
    y = _as_1d(labels).astype(int)
    calibrator = fit_calibrator(s, y, kind=kind)
    if kind == "sigmoid":
        cal = calibrator.predict_proba(s.reshape(-1, 1))[:, 1]
    else:
        cal = calibrator.predict(s)
    return {
        "kind": kind,
        "brier_score": float(np.mean((cal - y) ** 2)),
        "log_loss": float(
            -np.mean(
                y * np.log(np.clip(cal, 1e-12, 1.0))
                + (1 - y) * np.log(np.clip(1.0 - cal, 1e-12, 1.0))
            )
        ),
        "reliability": {"scores": s, "probs": cal},
    }

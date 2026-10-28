"""Reusable classifier evaluation (Weekly Test 15, Q31).

Drop-in for src/models/evaluate.py. Reuse for Random Forest / XGBoost in Month 5.

Protocol
--------
* Every training row gets exactly ONE out-of-fold (OOF) probability, and the
  per-fold CV metrics come from the SAME fitted fold models (one fit per fold).
* The F1-best (and Youden) threshold is chosen ONLY from those OOF probabilities.
* The model is refit on all of X_tr once; the test set is scored once and the
  frozen thresholds are applied. Nothing on test is ever tuned.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


def _positive_score(model, X):
    """P(class 1); falls back to decision_function for models without predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _scores(y, pred, score):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "avg_precision": average_precision_score(y, score),
    }


def _threshold_f1(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    i = int(np.nanargmax(f1))
    return float(thresholds[i]), float(f1[i])


def _threshold_youden(y, score):
    fpr, tpr, thresholds = roc_curve(y, score)
    valid = np.isfinite(thresholds)
    return float(thresholds[valid][np.argmax((tpr - fpr)[valid])])


def _net_value(y, pred, cost_fn=50, cost_fp=5, benefit_tp=40):
    """benefit_tp * TP - cost_fp * FP - cost_fn * FN  (TN is worth 0)."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return float(benefit_tp * tp - cost_fp * fp - cost_fn * fn)


def _take(data, idx):
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def evaluate_classifier(model, X_tr, y_tr, X_te, y_te, cv,
                        cost_fn=50, cost_fp=5, benefit_tp=40, default_threshold=0.5):
    """CV mean/std + OOF-chosen thresholds + test metrics + business net value."""
    y_arr = np.asarray(y_tr)
    oof_prob = np.full(len(y_arr), np.nan)
    fold = {f"test_{m}": [] for m in METRICS}

    # one fit per fold: fold metrics and OOF probabilities come from the same models
    for tr_idx, va_idx in cv.split(X_tr, y_arr):
        m = clone(model).fit(_take(X_tr, tr_idx), y_arr[tr_idx])
        score = _positive_score(m, _take(X_tr, va_idx))
        oof_prob[va_idx] = score
        s = _scores(y_arr[va_idx], (score >= default_threshold).astype(int), score)
        for metric in METRICS:
            fold[f"test_{metric}"].append(s["avg_precision" if metric == "average_precision" else metric])
    fold = {k: np.asarray(v) for k, v in fold.items()}
    assert not np.isnan(oof_prob).any(), "cv must cover every training row exactly once"

    thr_f1, oof_best_f1 = _threshold_f1(y_arr, oof_prob)
    thr_youden = _threshold_youden(y_arr, oof_prob)

    # final model: fit on all training data, score test ONCE
    final = clone(model).fit(X_tr, y_arr)
    test_prob = _positive_score(final, X_te)
    pred_def = (test_prob >= default_threshold).astype(int)
    pred_f1 = (test_prob >= thr_f1).astype(int)
    pred_youden = (test_prob >= thr_youden).astype(int)
    nv = lambda p: _net_value(y_te, p, cost_fn, cost_fp, benefit_tp)

    return {
        "fold": fold,
        "cv_metrics": {m: {"mean": float(fold[f"test_{m}"].mean()),
                           "std": float(fold[f"test_{m}"].std(ddof=1))} for m in METRICS},
        "oof_probability": oof_prob,
        "oof_prediction": (oof_prob >= default_threshold).astype(int),
        "threshold_f1": thr_f1,
        "oof_best_f1": oof_best_f1,
        "threshold_youden": thr_youden,
        "test_probability": test_prob,
        "test_prediction": pred_def,
        "test_prediction_f1": pred_f1,
        "test_prediction_youden": pred_youden,
        "test_metrics": _scores(y_te, pred_def, test_prob),
        "test_metrics_f1": _scores(y_te, pred_f1, test_prob),
        "test_metrics_youden": _scores(y_te, pred_youden, test_prob),
        "test_net_value_default": nv(pred_def),
        "test_net_value_f1": nv(pred_f1),
        "test_net_value_youden": nv(pred_youden),
    }


def evaluation_table(model_results):
    """One row per model: CV mean/std, test metrics (same threshold for all), threshold, net value."""
    rows = []
    for name, r in model_results.items():
        row = {"model": name, "F1-best threshold (OOF)": r["threshold_f1"]}
        for metric in METRICS:
            key = "avg_precision" if metric == "average_precision" else metric
            row[f"CV {metric} mean"] = r["cv_metrics"][metric]["mean"]
            row[f"CV {metric} std"] = r["cv_metrics"][metric]["std"]
            row[f"test {key} @0.5"] = r["test_metrics"][key]
            row[f"test {key} @F1-thr"] = r["test_metrics_f1"][key]
        row["test net value @0.5 (£)"] = r["test_net_value_default"]
        row["test net value @F1-thr (£)"] = r["test_net_value_f1"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
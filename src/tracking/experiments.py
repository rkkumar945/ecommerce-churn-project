"""Tracked CV runs aur grid-search ke child runs (Day 92). Sirf CV metrics log hote hain."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import platform

import mlflow
import mlflow.sklearn
import numpy as np
import sklearn
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from src.tracking.mlflow_utils import (
    dataset_fingerprint,
    file_sha256,
    git_info,
    log_cv_metrics,
    log_sklearn_model,
)


def estimator_params(model) -> dict:
    """Hyperparameters model se nikaalo (haath se nahi), warna logged params aur asli model alag ho sakte hain."""
    est = model[-1] if hasattr(model, "steps") else model
    return {
        "model": type(est).__name__,
        **{f"hp_{k}": v for k, v in est.get_params().items()},
    }


def cv_summary(res: dict, names) -> dict:
    out = {}
    for n in names:
        s = np.asarray(
            res[f"test_{n}"], dtype=float
        )  # sklearn ke 'test_*' = CV validation folds
        out[f"cv_{n}_mean"] = float(s.mean())
        out[f"cv_{n}_std"] = float(s.std())
    out["cv_fit_time_mean"] = float(np.mean(res["fit_time"]))
    return out


def track_cv_run(
    run_name,
    model,
    X,
    y,
    cv,
    scoring,
    *,
    tags=None,
    extra_params=None,
    data_path="data/processed/X_train_v2.csv",
    fit_and_log_model=False,
    nested=False,
):
    names = list(scoring)
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        params = {
            **estimator_params(model),
            **(extra_params or {}),
            "cv_n_splits": cv.get_n_splits(),
            "cv_seed": getattr(cv, "random_state", None),
            "scoring_primary": "roc_auc",
            "n_features": X.shape[1],
        }
        if data_path:
            params["data_file_sha"] = file_sha256(data_path)[:12]
        mlflow.log_params(params)
        mlflow.log_params(
            {f"data_{k}": v for k, v in dataset_fingerprint(X, y).items()}
        )
        mlflow.set_tags(
            {
                **git_info(),
                "stage": "cv",
                "test_set_used": "false",
                "sklearn": sklearn.__version__,
                "python": platform.python_version(),
                **(tags or {}),
            }
        )

        res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        summary = cv_summary(res, names)
        log_cv_metrics(summary)

        first = "roc_auc" if "roc_auc" in names else names[0]
        folds = np.asarray(res[f"test_{first}"], dtype=float)
        for i, v in enumerate(folds):
            log_cv_metrics({f"cv_fold_{first}": float(v)}, step=i)

        fig = Figure(figsize=(4.5, 3))
        ax = fig.subplots()
        ax.bar(range(1, len(folds) + 1), folds, color="steelblue")
        ax.set_xlabel("CV fold")
        ax.set_ylabel(first)
        ax.set_title(f"Fold-wise {first}")
        fig.tight_layout()
        mlflow.log_figure(fig, f"fold_{first}.png")
        mlflow.log_dict({"features": list(X.columns)}, "features.json")

        fitted, model_uri = None, None
        if fit_and_log_model:
            fitted = clone(model).fit(X, y)  # sirf train par fit, test dekha hi nahi
            info = log_sklearn_model(fitted, name="model", input_example=X.head(3))
            model_uri = info.model_uri
            mlflow.set_tag("model_uri", model_uri)
    return {
        "run_id": run.info.run_id,
        "summary": summary,
        "model_uri": model_uri,
        "fitted": fitted,
    }


def log_grid_children(gs, names, tags=None):
    """GridSearchCV.cv_results_ ki har row ko ek child run bana do (koi extra fitting nahi hoti).
    Parent run ke andar (active run ke saath) bulana hai."""
    r = gs.cv_results_
    for i, params in enumerate(r["params"]):
        with mlflow.start_run(run_name=f"cand_{i:03d}", nested=True):
            mlflow.log_params({f"hp_{k}": v for k, v in params.items()})
            if "rank_test_roc_auc" in r:
                mlflow.log_param("rank_in_search", int(r["rank_test_roc_auc"][i]))
            metrics = {}
            for n in names:
                metrics[f"cv_{n}_mean"] = float(r[f"mean_test_{n}"][i])
                metrics[f"cv_{n}_std"] = float(r[f"std_test_{n}"][i])
            metrics["cv_fit_time_mean"] = float(r["mean_fit_time"][i])
            log_cv_metrics(metrics)
            mlflow.set_tags(
                {**(tags or {}), "stage": "cv_grid_candidate", "test_set_used": "false"}
            )

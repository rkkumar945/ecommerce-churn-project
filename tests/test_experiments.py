import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

mlflow = pytest.importorskip("mlflow")


from src.models.search_spaces import pick_one_se
from src.tracking import mlflow_utils as mu
from src.tracking.experiments import log_grid_children, track_cv_run


@pytest.fixture
def store(tmp_path, monkeypatch):
    old = mlflow.get_tracking_uri()
    monkeypatch.setattr(mu, "STORE", tmp_path)
    mu.setup_tracking("unit-exp")
    yield
    mlflow.set_tracking_uri(old)


def _data():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    return pd.DataFrame(X, columns=list("abcde")), pd.Series(y)


def test_track_cv_run_logs_only_cv_metrics(store):
    X, y = _data()
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    out = track_cv_run(
        "lr",
        LogisticRegression(C=0.5),
        X,
        y,
        cv,
        {"roc_auc": "roc_auc"},
        data_path=None,
    )
    run = mlflow.get_run(out["run_id"])
    assert "cv_roc_auc_mean" in run.data.metrics
    assert not any(k.startswith("test_") for k in run.data.metrics)
    assert run.data.tags["test_set_used"] == "false"
    assert run.data.params["hp_C"] == "0.5"


def test_grid_children_are_nested_and_complete(store):
    X, y = _data()
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    scoring = {"roc_auc": "roc_auc", "ap": "average_precision"}
    gs = GridSearchCV(
        LogisticRegression(max_iter=500),
        {"C": [0.1, 1.0]},
        cv=cv,
        scoring=scoring,
        refit="roc_auc",
    ).fit(X, y)
    with mlflow.start_run(run_name="parent") as parent:
        log_grid_children(gs, list(scoring))
    kids = mlflow.search_runs(
        experiment_names=["unit-exp"],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    assert len(kids) == 2
    assert set(kids["params.hp_C"]) == {"0.1", "1.0"}


def test_model_roundtrip_via_model_uri(store):
    X, y = _data()
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    out = track_cv_run(
        "lr_model",
        LogisticRegression(),
        X,
        y,
        cv,
        {"roc_auc": "roc_auc"},
        data_path=None,
        fit_and_log_model=True,
    )
    loaded = mlflow.sklearn.load_model(out["model_uri"])
    assert np.allclose(
        loaded.predict_proba(X.head(20)), out["fitted"].predict_proba(X.head(20))
    )


class _FakeGS:
    def __init__(self, params, mean, std, best):
        self.cv_results_ = {
            "params": params,
            "mean_test_roc_auc": np.array(mean),
            "std_test_roc_auc": np.array(std),
        }
        self.best_index_ = best


def test_pick_one_se_prefers_simpler_within_one_std():
    params = [
        {"max_depth": 8, "min_samples_leaf": 5},
        {"max_depth": 3, "min_samples_leaf": 30},
        {"max_depth": 2, "min_samples_leaf": 50},
    ]
    gs = _FakeGS(params, [0.80, 0.79, 0.70], [0.02, 0.02, 0.02], best=0)
    assert (
        pick_one_se(gs, "Tree") == params[1]
    )  # teesra 1 std se bahar hai, doosra simple aur andar


def test_pick_one_se_is_deterministic():
    params = [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 3},
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10},
    ]
    gs = _FakeGS(params, [0.80, 0.80], [0.01, 0.01], best=0)
    assert (
        pick_one_se(gs, "RandomForest") == pick_one_se(gs, "RandomForest") == params[1]
    )

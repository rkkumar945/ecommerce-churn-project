import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.app import service


def _fitted():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    X = pd.DataFrame(X, columns=list("abcde"))
    return Pipeline([("s", StandardScaler()), ("m", LogisticRegression())]).fit(X, y), X


def test_joblib_roundtrip_gives_identical_predictions(tmp_path):
    model, X = _fitted()
    joblib.dump(model, tmp_path / "m.joblib")
    again = joblib.load(tmp_path / "m.joblib")
    assert np.array_equal(model.predict_proba(X), again.predict_proba(X))


def test_tampered_model_is_rejected(tmp_path, monkeypatch):
    model, _ = _fitted()
    path = tmp_path / "final_model.joblib"
    joblib.dump(model, path)
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"model_file": {"sha256": "deadbeef"}, "features": list("abcde"), "env": {}}
        )
    )
    monkeypatch.setattr(service, "MODEL_PATH", path)
    monkeypatch.setattr(service, "MANIFEST_PATH", manifest_path)
    with pytest.raises(ValueError):
        service.load_artifacts()


def test_committed_artifacts_load_and_env_matches():
    _model, manifest = service.load_artifacts()
    assert service.env_problems(manifest) == []
    assert manifest["model_file"]["bytes"] < 5_000_000

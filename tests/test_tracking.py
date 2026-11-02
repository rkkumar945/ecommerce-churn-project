import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  
import pytest

mlflow = pytest.importorskip('mlflow')

from src.tracking import mlflow_utils as mu


def test_guard_blocks_test_metrics():
    with pytest.raises(ValueError):
        mu.check_metrics({'cv_roc_auc_mean': 0.7, 'test_roc_auc': 0.9})
    assert mu.check_metrics({'cv_roc_auc_mean': 0.7}) == {'cv_roc_auc_mean': 0.7}


def test_roundtrip_in_temp_store(tmp_path, monkeypatch):
    old = mlflow.get_tracking_uri()
    monkeypatch.setattr(mu, 'STORE', tmp_path)
    try:
        mu.setup_tracking('unit-test')
        with mlflow.start_run(run_name='r1'):
            mlflow.log_param('C', 1.0)
            mu.log_cv_metrics({'cv_roc_auc_mean': 0.71})
        runs = mlflow.search_runs(experiment_names=['unit-test'])
        assert len(runs) == 1
        assert runs.loc[0, 'metrics.cv_roc_auc_mean'] == pytest.approx(0.71)
        assert runs.loc[0, 'params.C'] == '1.0'
    finally:
        mlflow.set_tracking_uri(old)
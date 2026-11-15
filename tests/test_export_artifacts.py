import pytest

from src.app.export_artifacts import resolve_model_path


def test_resolve_model_path_prefers_models_then_uses_notebook_fallback(tmp_path):
    models_path = tmp_path / "models" / "final_model.joblib"
    notebook_path = tmp_path / "notebooks" / "model" / "final_model.joblib"
    models_path.parent.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True)
    models_path.touch()
    notebook_path.touch()

    assert resolve_model_path(root=tmp_path) == models_path
    models_path.unlink()
    assert resolve_model_path(root=tmp_path) == notebook_path


def test_resolve_model_path_accepts_explicit_path(tmp_path):
    model_path = tmp_path / "custom.joblib"
    model_path.touch()

    assert resolve_model_path(model_path) == model_path


def test_resolve_model_path_reports_all_missing_candidates(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model file found") as exc:
        resolve_model_path(root=tmp_path)

    assert str(tmp_path / "models" / "final_model.joblib") in str(exc.value)
    assert str(tmp_path / "notebooks" / "model" / "final_model.joblib") in str(
        exc.value
    )

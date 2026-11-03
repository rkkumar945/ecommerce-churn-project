"""MLflow helpers: ek jagah tracking setup, git/data identity, aur 'test metrics kabhi log nahi hote' guard."""

import hashlib
import subprocess
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "mlflow_store"
FORBIDDEN_PREFIXES = ("test_", "holdout_")


def resolve_repo_path(path) -> Path:
    """Relative paths ko project root se resolve karo, chahe current working directory kuch bhi ho."""
    p = Path(path)
    if p.is_absolute():
        return p
    candidates = [ROOT / p, Path.cwd() / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ROOT / p


def setup_tracking(experiment: str) -> str:
    """Local SQLite store par tracking set karo, experiment banao (ya maujooda use karo)."""
    STORE.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(STORE / 'mlflow.db').as_posix()}")
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        exp_id = mlflow.create_experiment(
            experiment, artifact_location=(STORE / "artifacts").resolve().as_uri()
        )
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment)
    return exp_id


def check_metrics(metrics: dict) -> dict:
    bad = [k for k in metrics if k.lower().startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise ValueError(
            f'Test-set metrics tracking mein nahi jaate: {bad}. CV metrics ke naam "cv_" se shuru karo.'
        )
    return metrics


def log_cv_metrics(metrics: dict, step=None):
    mlflow.log_metrics(check_metrics(metrics), step=step)


def file_sha256(path) -> str:
    return hashlib.sha256(resolve_repo_path(path).read_bytes()).hexdigest()


def dataset_fingerprint(X, y) -> dict:
    return {
        "rows": len(X),
        "columns": int(X.shape[1]),
        "churn_rate": round(float(y.mean()), 4),
        "columns_sha": hashlib.sha256("|".join(X.columns).encode()).hexdigest()[:12],
    }


def git_info() -> dict:
    """Commit aur 'working tree dirty hai?' (dirty ka matlab: commit hash poora sach nahi batata)."""

    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "git_commit": run("rev-parse", "--short", "HEAD"),
            "git_dirty": str(bool(run("status", "--porcelain"))),
        }
    except Exception:  # noqa: BLE001
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def log_sklearn_model(model, name: str = "model", input_example=None):
    """MLflow 3.x mein name=, 2.x mein artifact_path=. Dono par chalta hai. ModelInfo return karta hai."""
    import mlflow.sklearn

    try:
        return mlflow.sklearn.log_model(model, name=name, input_example=input_example)
    except TypeError:
        return mlflow.sklearn.log_model(
            model, artifact_path=name, input_example=input_example
        )

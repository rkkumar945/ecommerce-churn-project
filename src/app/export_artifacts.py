"""Deployable app artifacts: model + manifest + customer feature table.
Repo root se chalao:  python -m src.app.export_artifacts
"""

import hashlib
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.build_features import NUM_V2, make_xy

OUT = ROOT / "app_artifacts"
LIBS = [
    "scikit-learn",
    "xgboost",
    "numpy",
    "pandas",
    "joblib",
    "imbalanced-learn",
    "shap",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_versions() -> dict:
    out = {"python": platform.python_version()}
    for lib in LIBS:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            pass
    return out


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    OUT.mkdir(exist_ok=True)
    card = json.loads((ROOT / "reports" / "final_model_card.json").read_text())

    # --- customer table (Day 82 wale hi split ke saath) ---
    feat = pd.read_csv(ROOT / "data" / "processed" / "customer_ml_dataset_v2.csv")
    X = make_xy(feat, NUM_V2).astype(float)
    assert list(X.columns) == card["features"], "feature order model card se alag hai"

    _idx_tr, idx_te = train_test_split(
        feat.index, test_size=0.2, random_state=42, stratify=feat["Churned"]
    )
    X_te_saved = pd.read_csv(ROOT / "data" / "processed" / "X_test_v2.csv").astype(
        float
    )
    assert np.allclose(X.loc[idx_te].to_numpy(), X_te_saved.to_numpy()), (
        "split Day 82 jaisa nahi"
    )

    table = pd.concat([feat[["Customer ID", "Country", "Region"]], X], axis=1)
    table["split"] = np.where(feat.index.isin(idx_te), "test", "train")
    table["observed_churn"] = feat["Churned"].to_numpy()
    table.to_csv(OUT / "customer_features_app.csv", index=False)

    # --- model: copy karo (dobara dump nahi), taaki file ka hash stable rahe ---
    path = OUT / "final_model.joblib"
    shutil.copy2(ROOT / "models" / "final_model.joblib", path)

    # --- reproducibility check: copy kiya hua model wahi test AUC deta hai? ---
    reloaded = joblib.load(path)
    y_te = feat.loc[idx_te, "Churned"].to_numpy()
    auc = roc_auc_score(y_te, reloaded.predict_proba(X.loc[idx_te])[:, 1])
    assert abs(auc - card["test_auc"]) < 1e-9, (
        f"AUC reproduce nahi hua: {auc} vs {card['test_auc']}"
    )

    manifest = {
        **card,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "env": env_versions(),
        "serialization": "joblib",
        "model_file": {
            "name": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        },
        "table_file": {"name": "customer_features_app.csv", "rows": len(table)},
    }
    (OUT / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"OK | AUC reproduced: {auc:.4f} | model {path.stat().st_size} bytes | {len(table)} customers"
    )
    print("env:", manifest["env"])


if __name__ == "__main__":
    main()

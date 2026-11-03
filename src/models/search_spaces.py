"""Estimators + hyperparameter grids (Day 84). Grids chhote rakhe hain, data chhota hai."""

import os
import sys

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.features.build_features import NUM_V2


def get_search_spaces(pos_weight: float) -> dict:
    prep = ColumnTransformer([("s", StandardScaler(), NUM_V2)], remainder="passthrough")
    return {
        "LogReg": (
            Pipeline(
                [
                    ("prep", prep),
                    ("m", LogisticRegression(max_iter=5000, class_weight="balanced")),
                ]
            ),
            {"m__C": [0.01, 0.1, 1, 10]},
        ),
        "Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {"max_depth": [3, 4, 5, 6], "min_samples_leaf": [10, 30, 60]},
        ),
        "RandomForest": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                max_features="sqrt",
                n_jobs=1,
                random_state=42,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [4, 8, None],
                "min_samples_leaf": [3, 10, 30],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                eval_metric="auc",
                tree_method="hist",
                n_jobs=1,
                random_state=42,
            ),
            {
                "n_estimators": [150, 300],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3, 4],
                "min_child_weight": [3, 10],
            },
        ),
    }


# ---- Day 92: deterministic 1-SE pick ----
COMPLEXITY = {  # chhota tuple = simple model
    "LogReg": lambda p: (p["m__C"],),
    "Tree": lambda p: (p["max_depth"], -p["min_samples_leaf"]),
    "RandomForest": lambda p: (
        p["n_estimators"],
        np.inf if p["max_depth"] is None else p["max_depth"],
        -p["min_samples_leaf"],
    ),
    "XGBoost": lambda p: (p["n_estimators"], p["max_depth"], -p["min_child_weight"]),
}


def pick_one_se(gs, family: str, metric: str = "roc_auc") -> dict:
    """Best score ke ek std ke andar sabse simple combination. Tie-break hyperparameters se, wall-clock se nahi."""
    r = gs.cv_results_
    mean, std = (
        np.asarray(r[f"mean_test_{metric}"]),
        np.asarray(r[f"std_test_{metric}"]),
    )
    b = gs.best_index_
    ok = [i for i in range(len(mean)) if mean[i] >= mean[b] - std[b]]
    return dict(min((r["params"][i] for i in ok), key=COMPLEXITY[family]))

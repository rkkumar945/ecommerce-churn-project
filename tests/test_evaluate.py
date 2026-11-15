from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.models.evaluate import _net_value, evaluate_classifier, evaluation_table


def _training_and_test_data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=120)
    y = (x + rng.normal(scale=0.7, size=len(x)) > 0).astype(int)
    train = pd.DataFrame({"x": x, "is_test": 0})
    test = pd.DataFrame({"x": [-1.5, -0.5, 0.5, 1.5], "is_test": 1})
    return train, y, test, np.array([0, 0, 1, 1])


def _cv():
    return StratifiedKFold(n_splits=4, shuffle=True, random_state=7)


def test_thresholds_are_independent_of_test_labels():
    X_train, y_train, X_test, y_test = _training_and_test_data()
    model = LogisticRegression(random_state=0)

    first = evaluate_classifier(
        model, X_train, y_train, X_test, y_test, _cv()
    )
    second = evaluate_classifier(
        model, X_train, y_train, X_test, 1 - y_test, _cv()
    )

    assert first["threshold_f1"] == pytest.approx(second["threshold_f1"])
    assert first["threshold_youden"] == pytest.approx(second["threshold_youden"])


class ScoreCountingClassifier(ClassifierMixin, BaseEstimator):
    test_batches: ClassVar[list[tuple]] = []

    def fit(self, X, y):
        self.classifier_ = LogisticRegression(random_state=0).fit(X[["x"]], y)
        self.classes_ = self.classifier_.classes_
        return self

    def predict_proba(self, X):
        if X["is_test"].eq(1).all():
            type(self).test_batches.append(tuple(X["x"].to_numpy()))
        return self.classifier_.predict_proba(X[["x"]])


def test_final_model_scores_held_out_data_once():
    X_train, y_train, X_test, y_test = _training_and_test_data()
    ScoreCountingClassifier.test_batches = []

    evaluate_classifier(
        ScoreCountingClassifier(), X_train, y_train, X_test, y_test, _cv()
    )

    assert ScoreCountingClassifier.test_batches == [tuple(X_test["x"].to_numpy())]


def test_net_value_matches_default_cost_formula():
    y = np.array([1, 1, 1, 0, 0])
    pred = np.array([1, 0, 1, 1, 0])

    assert _net_value(y, pred) == 40 * 2 - 5 * 1 - 50 * 1


def test_evaluation_table_has_one_row_per_model():
    X_train, y_train, X_test, y_test = _training_and_test_data()
    models = {
        "logistic": LogisticRegression(random_state=0),
        "tree": DecisionTreeClassifier(max_depth=2, random_state=0),
    }
    results = {
        name: evaluate_classifier(model, X_train, y_train, X_test, y_test, _cv())
        for name, model in models.items()
    }

    summary = evaluation_table(results)

    assert len(summary) == len(models)
    assert set(summary.index) == set(models)

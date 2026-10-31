import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.explain import explain, reason_codes


def _data():
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, random_state=0)
    return pd.DataFrame(X, columns=list('abcdef')), y


def test_tree_additivity_and_shape():
    X, y = _data()
    rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0).fit(X, y)
    r = explain(rf, X.iloc[:40], X)
    assert r['values'].shape == (40, 6)
    assert r['gap'] < 1e-3


def test_linear_pipeline_additivity_and_column_order():
    X, y = _data()
    m = Pipeline([('prep', StandardScaler()), ('m', LogisticRegression())]).fit(X, y)
    r = explain(m, X.iloc[:40], X)
    assert list(r['values'].columns) == list(X.columns)
    assert r['gap'] < 1e-6


def test_reason_codes_only_positive_and_sorted():
    row = pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0})
    codes = reason_codes(np.array([0.5, -0.2, 0.1]), row, k=2)
    assert [c[0] for c in codes] == ['a', 'c']
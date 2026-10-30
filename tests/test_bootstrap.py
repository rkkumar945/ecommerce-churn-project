import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import roc_auc_score
from src.models.bootstrap import bootstrap_metric, ci


def _data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    return y, y * 0.5 + rng.normal(size=400)


def test_ci_contains_point_estimate():
    y, s = _data()
    lo, hi = ci(bootstrap_metric(y, {'m': s}, roc_auc_score, n_boot=300)['m'])
    assert lo < roc_auc_score(y, s) < hi


def test_paired_arrays_align():
    y, s = _data()
    b = bootstrap_metric(y, {'a': s, 'b': -s}, roc_auc_score, n_boot=100)
    assert len(b['a']) == len(b['b'])
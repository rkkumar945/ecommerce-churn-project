import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.app.service import (
    add_scores,
    find_customer,
    load_artifacts,
    load_table,
    pick_random,
    readable_profile,
)


class DummyModel:
    def predict_proba(self, X):
        p = X['a'].to_numpy()
        return np.column_stack([1 - p, p])


def _raw():
    return pd.DataFrame({'Customer ID': [101, 102, 103, 104], 'a': [0.1, 0.9, 0.5, 0.7]})


@pytest.fixture
def scored():
    return add_scores(DummyModel(), _raw(), ['a'])


def test_add_scores_ranks_and_does_not_mutate():
    t = _raw()
    out = add_scores(DummyModel(), t, ['a'])
    assert 'score' not in t.columns
    assert out.set_index('Customer ID')['top_share'].to_dict() == {101: 1.0, 102: 0.25, 103: 0.75, 104: 0.5}


@pytest.mark.parametrize('text,ok', [('102', True), ('  102 ', True), ('', False), ('abc', False),
                                     ('-5', False), ('1e2', False), ('999', False), (None, False)])
def test_find_customer_validation(scored, text, ok):
    row, err = find_customer(scored, text)
    assert (row is not None) == ok
    assert (err is None) == ok


def test_pick_random_flagged_only(scored):
    for seed in range(20):
        row = pick_random(scored, flagged_only=True, threshold=0.6, seed=seed)
        assert row['Customer ID'] in (102, 104)


def test_pick_random_returns_none_when_pool_empty(scored):
    assert pick_random(scored, flagged_only=True, threshold=0.99) is None


def test_readable_profile_single_order_and_currency():
    row = pd.Series({'SingleOrder': 1, 'Recency': 40, 'Frequency': 1, 'Monetary_log': np.log1p(100.0),
                     'AvgOrderValue': 100.0, 'Tenure': 30, 'UniqueProducts': 5, 'AvgGapDays': 0,
                     'OverdueRatio': 0, 'Orders_last90': 1, 'Orders_prev90': 0})
    prof = readable_profile(row)
    assert prof['Total spend'] == '£100'
    assert prof['Overdue ratio'] == 'n/a (1 order)'


def test_app_scores_reproduce_card_auc():
    """App ka scoring path aur Day 85 ki evaluation ek hi number deni chahiye (train/serve skew guard)."""
    model, manifest = load_artifacts()
    t = add_scores(model, load_table(), manifest['features'])
    te = t[t['split'] == 'test']
    auc = roc_auc_score(te['observed_churn'], te['score'])
    assert abs(auc - manifest['test_auc']) < 1e-6
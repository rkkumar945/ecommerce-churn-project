import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.split import make_split

APP_TABLE = Path(__file__).resolve().parents[1] / 'app_artifacts' / 'customer_features_app.csv'


def _toy(n=400):
    return pd.DataFrame({'Churned': (np.arange(n) % 5 == 0).astype(int)})


def test_split_is_disjoint_complete_and_stratified():
    feat = _toy()
    tr, te = make_split(feat)
    assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(feat)
    assert abs(feat.loc[tr, 'Churned'].mean() - feat.loc[te, 'Churned'].mean()) < 0.01


def test_split_is_deterministic():
    feat = _toy()
    assert np.array_equal(make_split(feat)[1], make_split(feat)[1])


@pytest.mark.skipif(not APP_TABLE.exists(), reason='app_artifacts missing')
def test_make_split_reproduces_exported_split():
    table = pd.read_csv(APP_TABLE)
    _, te = make_split(table.rename(columns={'observed_churn': 'Churned'}))
    assert (table.loc[te, 'split'] == 'test').all()
    assert len(te) == int((table['split'] == 'test').sum())
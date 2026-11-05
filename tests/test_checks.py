import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import NUM_V2
from src.pipeline.checks import (
    DataCheckError,
    check_clean,
    check_features,
    check_raw,
    fail_on,
)


def _good(n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.uniform(1, 10, n) for c in NUM_V2})
    df['Frequency'] = rng.integers(1, 5, n).astype(float)
    df['SingleOrder'] = (df['Frequency'] == 1).astype(int)
    df['Recency'] = rng.integers(0, 300, n).astype(float)
    df['Tenure'] = rng.integers(0, 600, n).astype(float)
    df['Customer ID'] = np.arange(n)
    df['Churned'] = np.arange(n) % 2
    df['Region'] = np.where(np.arange(n) % 3 == 0, 'UK', 'Rest of World')
    return df


def test_good_features_pass():
    assert check_features(_good()) == []


@pytest.mark.parametrize('mutate,needle', [
    (lambda d: d.assign(**{'Customer ID': 1}), 'unique'),
    (lambda d: d.assign(Recency=-1.0), 'negative'),
    (lambda d: d.assign(Churned=0), 'class 1'),
    (lambda d: d.assign(Region='Mars'), 'Region'),
    (lambda d: d.assign(AvgGapDays=np.nan), 'non-finite'),
    (lambda d: d.assign(SingleOrder=1 - d['SingleOrder']), 'SingleOrder'),
])
def test_feature_violations_are_caught(mutate, needle):
    problems = check_features(mutate(_good()))
    assert any(needle in p for p in problems), problems


def _clean_df():
    return pd.DataFrame({'Invoice': ['1', '2'], 'StockCode': ['a', 'b'], 'Description': ['x', 'y'],
                         'Quantity': [1, 2], 'InvoiceDate': pd.to_datetime(['2011-01-01', '2011-01-02']),
                         'Price': [1.0, 2.0], 'Customer ID': [1, 2], 'Country': ['UK', 'UK']})


def test_clean_checks():
    assert check_clean(_clean_df()) == []
    assert any('cancellation' in p for p in check_clean(_clean_df().assign(Invoice=['C1', '2'])))
    assert any('null' in p for p in check_clean(_clean_df().assign(**{'Customer ID': [1, np.nan]})))
    assert any('duplicate' in p for p in check_clean(pd.concat([_clean_df()] * 2)))
    assert any('datetime' in p for p in check_clean(_clean_df().assign(InvoiceDate=['2011-01-01', '2011-01-02'])))


def test_raw_checks_and_fail_on():
    assert check_raw(_clean_df()) == []
    assert any('missing columns' in p for p in check_raw(_clean_df().drop(columns=['Price'])))
    with pytest.raises(DataCheckError):
        fail_on(['boom'], 'raw')
    fail_on([], 'raw')
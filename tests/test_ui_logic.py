import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import pytest

from src.app.service import (
    add_scores,
    driver_frame,
    load_artifacts,
    load_table,
    top_k_summary,
    top_reasons,
    wilson_interval,
)


@pytest.fixture(scope='module')
def art():
    model, manifest = load_artifacts()
    table = add_scores(model, load_table(), manifest['features'])
    feats = manifest['features']
    bg = table[feats].astype(float).sample(100, random_state=0).reset_index(drop=True)
    return model, feats, table, bg


def test_wilson_known_values():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert wilson_interval(10, 10)[1] <= 1 + 1e-9
    assert np.isnan(wilson_interval(0, 0)[0])


def _pool():
    return pd.DataFrame({'score': [0.9, 0.8, 0.7, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01],
                         'observed_churn': [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]})


def test_top_k_summary_toy():
    s = top_k_summary(_pool(), 0.3)
    assert s['k'] == 3 and s['n'] == 10
    assert s['precision'] == pytest.approx(2 / 3)
    assert s['base_rate'] == pytest.approx(0.3)
    assert s['lift'] == pytest.approx((2 / 3) / 0.3)
    assert s['recall'] == pytest.approx(2 / 3)


def test_top_k_summary_edges():
    assert top_k_summary(_pool(), 0.001)['k'] == 1
    none = _pool().assign(observed_churn=0)
    s = top_k_summary(none, 0.3)
    assert np.isnan(s['lift']) and np.isnan(s['recall'])


def test_driver_frame_sorted_and_complete(art):
    pytest.importorskip('shap')
    model, feats, table, bg = art
    df, _ = driver_frame(model, table.iloc[0], feats, bg)
    assert len(df) == len(feats) and set(df['feature']) == set(feats)
    assert df['shap'].abs().is_monotonic_decreasing


def test_top_reasons_one_text_per_row(art):
    pytest.importorskip('shap')
    model, feats, table, bg = art
    out = top_reasons(model, table.head(15), feats, bg)
    assert len(out) == 15 and all(isinstance(t, str) and t for t in out)
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from src.monitoring.drift import drift_policy, drift_report, psi


def test_psi_same_distribution_is_small():
    r = np.random.default_rng(1)
    assert psi(r.normal(size=5000), r.normal(size=5000)) < 0.03


def test_psi_grows_with_shift():
    r = np.random.default_rng(1)
    a = r.normal(size=5000)
    vals = [psi(a, r.normal(loc=m, size=5000)) for m in (0.0, 0.3, 0.6, 1.0)]
    assert vals == sorted(vals) and vals[-1] > 0.25


def test_psi_handles_zero_spike():
    r = np.random.default_rng(2)

    def sample(p_zero):
        return np.where(r.random(4000) < p_zero, 0.0, r.exponential(5, 4000))

    ref = sample(0.7)
    assert psi(ref, sample(0.7)) < 0.03
    assert psi(ref, sample(0.95)) > 0.25


def test_psi_binary_unseen_values_and_nans():
    ref = np.array([0] * 900 + [1] * 100)
    cur = np.array([0] * 600 + [1] * 400)
    assert psi(ref, cur) > 0.25
    ref2 = np.array([1, 2, 3] * 100, dtype=float)
    cur2 = np.array([1, 2, 3, 99, np.nan] * 50, dtype=float)
    assert np.isfinite(psi(ref2, cur2))
    assert np.isnan(psi([], [1.0]))


def test_report_flags_only_the_shifted_column():
    r = np.random.default_rng(3)
    ref = pd.DataFrame({'a': r.normal(size=3000), 'b': r.normal(size=3000)})
    cur = pd.DataFrame({'a': r.normal(size=3000), 'b': r.normal(loc=1.0, size=3000)})
    rep = drift_report(ref, cur, ['a', 'b']).set_index('feature')
    assert rep.loc['a', 'status'] == 'ok' and rep.loc['b', 'status'] == 'alert'


def test_aa_split_stays_quiet():
    r = np.random.default_rng(4)
    df = pd.DataFrame({f'f{i}': r.exponential(size=4000) for i in range(15)})
    rep = drift_report(df.iloc[:3000], df.iloc[3000:], list(df.columns))
    assert (rep['status'] == 'ok').all()


def test_binary_column_uses_chi2():
    ref = pd.DataFrame({'flag': [0] * 900 + [1] * 100})
    cur = pd.DataFrame({'flag': [0] * 600 + [1] * 400})
    row = drift_report(ref, cur, ['flag']).iloc[0]
    assert row['test'] == 'chi2' and row['p_value'] < 1e-6


def test_policy_levels():
    rep = pd.DataFrame({'feature': ['a', 'b', 'c'], 'psi': [0.4, 0.3, 0.05],
                        'status': ['alert', 'alert', 'ok']})
    assert drift_policy(rep, ['a', 'b'])['level'] == 'alert'          # 2 key features alert
    assert drift_policy(rep, ['a'])['level'] == 'watch'               # ek hi, score drift nahi
    assert drift_policy(rep, ['c'], score_psi=0.4)['level'] == 'alert'
    ok = rep.assign(status='ok')
    assert drift_policy(ok, ['a'])['level'] == 'ok'
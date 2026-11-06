"""Data drift metrics (Day 95). Evidently se independent: PSI, KS / chi-square, Wasserstein, alert policy."""
import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4
PSI_WATCH, PSI_ALERT = 0.10, 0.25      # rules of thumb (credit scoring se), statistical laws nahi


def _clean(x) -> np.ndarray:
    return pd.Series(x).dropna().to_numpy(dtype=float)


def _fractions(counts):
    p = np.asarray(counts, dtype=float)
    return np.clip(p / p.sum(), EPS, None)


def _prepare(ref: np.ndarray, bins: int):
    """Jo value reference ka >= 1/bins hissa hai (jaise 0 ka spike) use apni alag bin, baaki ko quantile bins."""
    vals, counts = np.unique(ref, return_counts=True)
    atoms = vals[counts / len(ref) >= 1.0 / bins]
    rest = ref[~np.isin(ref, atoms)]
    edges = np.array([])
    if len(rest):
        k = max(1, bins - len(atoms))
        edges = np.unique(np.quantile(rest, np.linspace(0, 1, k + 1)[1:-1]))
    return atoms, edges


def _bin_counts(x: np.ndarray, atoms, edges):
    idx = np.searchsorted(edges, x, side='right') + len(atoms)
    for i, a in enumerate(atoms):
        idx[x == a] = i
    return np.bincount(idx, minlength=len(atoms) + len(edges) + 1)


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index. Bins reference se bante hain; current ko usi par naapa jaata hai."""
    ref, cur = _clean(reference), _clean(current)
    if len(ref) == 0 or len(cur) == 0:
        return float('nan')
    atoms, edges = _prepare(ref, bins)
    pr = _fractions(_bin_counts(ref, atoms, edges))
    pc = _fractions(_bin_counts(cur, atoms, edges))
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def status_from_psi(v: float) -> str:
    if np.isnan(v):
        return 'n/a'
    return 'alert' if v > PSI_ALERT else 'watch' if v > PSI_WATCH else 'ok'


def chi2_p(ref: np.ndarray, cur: np.ndarray) -> float:
    cats = np.union1d(np.unique(ref), np.unique(cur))
    table = np.array([[(ref == c).sum() for c in cats], [(cur == c).sum() for c in cats]])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(table)[1])


def _row(name, reference, current) -> dict:
    ref, cur = _clean(reference), _clean(current)
    if len(ref) == 0 or len(cur) == 0:
        return {'feature': name, 'psi': np.nan, 'test': 'n/a', 'stat': np.nan, 'p_value': np.nan,
                'wasserstein_sd': np.nan, 'ref_mean': np.nan, 'cur_mean': np.nan, 'status': 'n/a'}
    p = psi(ref, cur)
    if len(np.unique(ref)) <= 2:                       # binary flags: chi-square
        test, stat, pv, wd = 'chi2', np.nan, chi2_p(ref, cur), np.nan
    else:
        ks = stats.ks_2samp(ref, cur)
        sd = float(np.std(ref))
        test, stat, pv = 'ks', float(ks.statistic), float(ks.pvalue)
        wd = float(stats.wasserstein_distance(ref, cur) / sd) if sd > 0 else np.nan
    return {'feature': name, 'psi': p, 'test': test, 'stat': stat, 'p_value': pv, 'wasserstein_sd': wd,
            'ref_mean': float(ref.mean()), 'cur_mean': float(cur.mean()), 'status': status_from_psi(p)}


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features) -> pd.DataFrame:
    rows = [_row(f, reference[f], current[f]) for f in features]
    return pd.DataFrame(rows).sort_values('psi', ascending=False).reset_index(drop=True)


def drift_policy(report: pd.DataFrame, key_features, score_psi=None, min_alerts: int = 2) -> dict:
    """Alert fatigue se bachne ke liye: sirf score drift ya important features ke drift par 'alert'."""
    key = report[report['feature'].isin(key_features)]
    n_key_alert = int((key['status'] == 'alert').sum())
    reasons = []
    if score_psi is not None and score_psi > PSI_ALERT:
        reasons.append(f'score PSI {score_psi:.2f} > {PSI_ALERT}')
    if n_key_alert >= min_alerts:
        reasons.append(f'{n_key_alert} key features in alert')
    if reasons:
        level = 'alert'
    elif (report['status'].isin(['watch', 'alert'])).any():
        level = 'watch'
    else:
        level = 'ok'
    return {'level': level, 'reasons': reasons}
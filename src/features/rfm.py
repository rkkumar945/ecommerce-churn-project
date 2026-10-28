"""RFM scoring and segmentation. Bin edges are fitted on training data only."""
import numpy as np
import pandas as pd

N_BINS = 5


def fit_rfm_bins(df: pd.DataFrame, n: int = N_BINS) -> dict:
    qs = np.linspace(0, 1, n + 1)[1:-1]
    return {c: np.quantile(df[c], qs) for c in ['Recency', 'Frequency', 'Monetary']}


def add_rfm_scores(df: pd.DataFrame, edges: dict, n: int = N_BINS) -> pd.DataFrame:
    out = df.copy()
    out['R_score'] = n - np.searchsorted(edges['Recency'], out['Recency'], side='left')
    out['F_score'] = 1 + np.searchsorted(edges['Frequency'], out['Frequency'], side='left')
    out['M_score'] = 1 + np.searchsorted(edges['Monetary'], out['Monetary'], side='left')
    out['RFM_score'] = out['R_score'] + out['F_score'] + out['M_score']
    return out


def assign_segment(r, f):
    return np.select(
        [(r >= 4) & (f >= 4), (r >= 3) & (f >= 3), (r >= 4) & (f <= 2),
         (r <= 2) & (f >= 3), (r <= 2) & (f <= 2)],
        ['Champions', 'Loyal', 'New / Promising', 'At Risk', 'Hibernating'],
        default='Needs Attention',
    )
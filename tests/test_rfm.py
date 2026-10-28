import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.rfm import fit_rfm_bins, add_rfm_scores


def make_df():
    return pd.DataFrame({'Recency': np.arange(1, 101),
                         'Frequency': np.arange(1, 101),
                         'Monetary': np.arange(1, 101) * 10.0})


def test_recent_customers_get_higher_r_score():
    df = make_df()
    out = add_rfm_scores(df, fit_rfm_bins(df))
    assert out.loc[0, 'R_score'] == 5      # Recency = 1 (sabse recent)
    assert out.loc[99, 'R_score'] == 1     # Recency = 100 (sabse purana)


def test_bins_do_not_change_when_extra_rows_scored():
    train = make_df()
    edges = fit_rfm_bins(train)
    extra = pd.DataFrame({'Recency': [5000], 'Frequency': [9999], 'Monetary': [1e9]})
    _ = add_rfm_scores(extra, edges)                       # scoring se edges nahi badalte
    assert np.array_equal(edges['Recency'], fit_rfm_bins(train)['Recency'])
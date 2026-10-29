import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # agar 'from
from src.features.rfm import rfm_scores, assign_segment


def test_scores_stay_in_range():
    ref = pd.Series(np.arange(100))
    s = rfm_scores(pd.Series([0, 50, 99, 500, -5]), ref)
    assert s.between(1, 5).all()


def test_recency_is_reversed():
    ref = pd.Series(np.arange(100))
    recent = rfm_scores(pd.Series([1]), ref, reverse=True).iloc[0]
    old = rfm_scores(pd.Series([99]), ref, reverse=True).iloc[0]
    assert recent == 5 and old == 1


def test_ties_get_same_score():
    ref = pd.Series([1] * 50 + list(range(2, 52)))
    s = rfm_scores(pd.Series([1, 1]), ref)
    assert s.nunique() == 1


def test_segment_rules():
    seg = assign_segment(pd.Series([5, 1]), pd.Series([5, 1]), pd.Series([5, 1]))
    assert list(seg) == ['Champions', 'Hibernating']
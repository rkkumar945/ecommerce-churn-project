"""RFM scoring (1-5) and rule-based segments. Reference distribution (train) se fit, kisi bhi data par apply."""

import numpy as np
import pandas as pd


# yeh function hai jo new data per bhi  training data ke jaisa hi score deta hai. 
def rfm_scores(
    values: pd.Series, reference: pd.Series, reverse: bool = False, k: int = 5
) -> pd.Series:
    """Mid-rank percentile ke basis par 1..k score. Barabar values ko same score milta hai."""
    ref = np.sort(reference.to_numpy())
    v = values.to_numpy()
    left = np.searchsorted(ref, v, side="left")
    right = np.searchsorted(ref, v, side="right")
    p = (left + right) / (2 * len(ref))
    score = np.clip(np.ceil(p * k), 1, k).astype(int)
    if reverse:  # Recency: kam din = behtar
        score = k + 1 - score
    return pd.Series(score, index=values.index)


# ye function R,F,M ke scores ke bisis per customer ko alag alag define karta hai , taki marketing team pata laga sake ki kis customer ke sath kya action lena hai.
def assign_segment(r: pd.Series, f: pd.Series, m: pd.Series) -> np.ndarray:
    conditions = [
        (r >= 4) & (f >= 4) & (m >= 4),
        (r >= 3) & (f >= 3),
        (r >= 4) & (f <= 2),
        (r <= 2) & (f >= 3),
        (r <= 2) & (f <= 2),
    ]
    names = ["Champions", "Loyal", "New / Promising", "At Risk", "Hibernating"]
    return np.select(conditions, names, default="Need Attention")


# ye pura code RFM customer segmentation pipeline hai , iske bsais per customer ke purchasing behaviour ke basis per unhe score karne or alag alag divide karne ke liye kiya jata hai.
# marketing team ke liye ek clean data prepare karta hai jis se team actions le sake.
'''ye python ko batata hai ki dusre project ke dusre folders kha hai taki hum waha ki files/modules import kar sake'''
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1])) #tesr file src/ folder  ke andar ki files ko easily dhund sake.


import numpy as np
import pandas as pd
import pytest

from src.features.split import make_split

APP_TABLE = Path(__file__).resolve().parents[1] / 'app_artifacts' / 'customer_features_app.csv'


# unit test ko test karne ke liye test data nahi milega or unhe run karne ke liye har baar real data file load karni hogi.
def _toy(n=400):
    return pd.DataFrame({'Churned': (np.arange(n) % 5 == 0).astype(int)}) # 0 se 399 tak we numbers jo 5 se divisible hai unko 0 or 1 me convert karti hai or dataframe unko 'Churned' column ke sath return kar deta hai.


# ye check karta hai ki train or test spilit disjoint(overlapping nahi) hai, total_rows = sari rows.
def test_split_is_disjoint_complete_and_stratified():
    feat = _toy()
    tr, te = make_split(feat)
    assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(feat)
    assert abs(feat.loc[tr, 'Churned'].mean() - feat.loc[te, 'Churned'].mean()) < 0.01 # trai,test ke andar 'Churned' column ka average(mean) nikata hai, unka difference(absolute value) claculate karke dekhta hai ki wo 1% se kam hai ya nahi.


# function check karta hai ki function deterministic hai ya nahi(sam input per same output deta hai ya nahi)
def test_split_is_deterministic():
    feat = _toy()
    assert np.array_equal(make_split(feat)[1], make_split(feat)[1])


# agar APP_TABLE csv file disk per exist nahi karti hai to is test ko skip ki jagah reason deta hai.
@pytest.mark.skipif(not APP_TABLE.exists(), reason='app_artifacts missing')
def test_make_split_reproduces_exported_split():
    table = pd.read_csv(APP_TABLE)
    _, te = make_split(table.rename(columns={'observed_churn': 'Churned'}))
    assert (table.loc[te, 'split'] == 'test').all()
    assert len(te) == int((table['split'] == 'test').sum())
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from src.features.behavioral import build_behavioral_features


def make_tx():
    rows = [
        (1, 'A1', '2011-01-01', 1, 10.0, 'S1'),
        (1, 'A2', '2011-01-11', 1, 10.0, 'S2'),   # gap = 10 din
        (2, 'A3', '2011-11-25', 1, 50.0, 'S1'),   # sirf future (cutoff = 2011-09-10) -> nahi aana chahiye
        (1, 'A4', '2011-12-09', 5, 100.0, 'S3'),  # future order -> features mein count nahi hona chahiye
    ]
    df = pd.DataFrame(rows, columns=['Customer ID', 'Invoice', 'InvoiceDate', 'Quantity', 'Price', 'StockCode'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df


def test_future_only_customers_excluded():
    out = build_behavioral_features(make_tx())
    assert set(out['Customer ID']) == {1}


def test_future_orders_do_not_leak_into_features():
    row = build_behavioral_features(make_tx()).set_index('Customer ID').loc[1]
    assert row['AvgGapDays'] == 10.0      # sirf A1 aur A2 se, future wale A4 se nahi
    assert row['Orders_recent'] == 0
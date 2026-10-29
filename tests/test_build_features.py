import sys
from pathlib import Path

import numpy as np
import pandas as pd

# plain `pytest` bhi chale (project root ko import path mein daalo)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.build_features import build_customer_dataset, region_bucket  # noqa: E402

COLS = ['Customer ID', 'Invoice', 'InvoiceDate', 'Quantity', 'Price', 'Country', 'StockCode']


def _df(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df


def make_tx():
    return _df([
        # (Customer ID, Invoice, InvoiceDate, Quantity, Price, Country, StockCode)
        (1, 'A1', '2011-01-01', 1, 10.0, 'United Kingdom', 'S1'),  # sirf purana -> churned
        (2, 'A2', '2011-01-05', 2, 5.0, 'Germany', 'S1'),          # past
        (2, 'A3', '2011-11-20', 1, 5.0, 'Germany', 'S2'),          # future -> not churned
        (3, 'A4', '2011-11-25', 1, 9.0, 'France', 'S3'),           # sirf future -> features mein nahi
        (2, 'A5', '2011-12-09', 1, 5.0, 'Germany', 'S2'),          # dataset ka aakhri din
    ])
# max date 2011-12-09 -> cutoff (90 din) = 2011-09-10


def test_future_only_customers_excluded():
    out = build_customer_dataset(make_tx())
    assert set(out['Customer ID']) == {1, 2}


def test_label_comes_from_future_window():
    out = build_customer_dataset(make_tx()).set_index('Customer ID')
    assert out.loc[1, 'Churned'] == 1
    assert out.loc[2, 'Churned'] == 0


def test_features_use_only_past_transactions():
    out = build_customer_dataset(make_tx()).set_index('Customer ID')
    assert out.loc[2, 'Frequency'] == 1      # sirf A2, future ke A3/A5 nahi
    assert out.loc[2, 'Monetary'] == 10.0    # 2 x 5.0
    assert out.loc[2, 'UniqueProducts'] == 1  # sirf S1


def test_recency_and_tenure_values():
    out = build_customer_dataset(make_tx()).set_index('Customer ID')
    # cutoff 2011-09-10: customer 1 -> 2011-01-01 (252 din), customer 2 -> 2011-01-05 (248 din)
    assert out.loc[1, 'Recency'] == 252 and out.loc[1, 'Tenure'] == 252
    assert out.loc[2, 'Recency'] == 248 and out.loc[2, 'Tenure'] == 248
    assert out.loc[2, 'AvgOrderValue'] == 10.0


def test_invoice_exactly_at_cutoff_belongs_to_future():
    df = _df([
        (1, 'A1', '2011-01-01', 1, 10.0, 'United Kingdom', 'S1'),
        (1, 'A2', '2011-09-10', 1, 10.0, 'United Kingdom', 'S1'),  # == cutoff -> future
        (2, 'A3', '2011-12-09', 1, 5.0, 'Germany', 'S1'),
    ])
    out = build_customer_dataset(df).set_index('Customer ID')
    assert out.loc[1, 'Frequency'] == 1      # A2 past mein nahi gina gaya
    assert out.loc[1, 'Churned'] == 0        # A2 future mein hai -> churn nahi


def test_horizon_days_changes_cutoff():
    # horizon 10 din -> cutoff 2011-11-29; customer 2 ke A3 (Nov 20) ab past mein aata hai
    out = build_customer_dataset(make_tx(), horizon_days=10).set_index('Customer ID')
    assert out.loc[2, 'Frequency'] == 2
    assert out.loc[2, 'Churned'] == 0        # A5 (Dec 9) future mein hai
    assert 3 in out.index                    # A4 (Nov 25) ab past mein -> customer 3 aata hai
    assert out.loc[3, 'Churned'] == 1        # aur future mein nahi aaya


def test_region_bucketing():
    assert region_bucket('United Kingdom') == 'UK'
    assert region_bucket('Germany') == 'Rest of Europe'
    assert region_bucket('Australia') == 'Rest of World'
    out = build_customer_dataset(make_tx()).set_index('Customer ID')
    assert out.loc[1, 'Region'] == 'UK' and out.loc[2, 'Region'] == 'Rest of Europe'


def test_negative_monetary_does_not_produce_nan():
    df = make_tx()
    df.loc[0, 'Quantity'] = -5               # customer 1: Monetary = -50
    out = build_customer_dataset(df).set_index('Customer ID')
    assert out.loc[1, 'Monetary'] == -50.0
    assert out.loc[1, 'Monetary_log'] == 0.0
    assert not out['Monetary_log'].isna().any()


def test_input_is_not_mutated():
    df = make_tx()
    before = df.copy()
    build_customer_dataset(df)
    pd.testing.assert_frame_equal(df, before)


def test_exclude_non_product_codes():
    df = make_tx()
    df.loc[1, 'StockCode'] = 'POST'          # customer 2 ki A2 row ab fee hai
    kept = build_customer_dataset(df).set_index('Customer ID')
    dropped = build_customer_dataset(df, exclude_non_product=True)
    assert kept.loc[2, 'Frequency'] == 1
    assert 2 not in set(dropped['Customer ID'])   # customer 2 ka past activity sab POST 









def test_no_nans_in_features():
    out = build_customer_dataset(make_tx())
    assert not out.drop(columns=['Churned']).isnull().any().any()


def test_features_invariant_to_future_rows():
    """Future transactions ko kuch bhi kar do, features nahi badalne chahiye (sirf label badal sakta hai)."""
    tx = make_tx()
    base = build_customer_dataset(tx).set_index('Customer ID')

    tx2 = tx.copy()
    future_mask = tx2['InvoiceDate'] >= pd.Timestamp('2011-09-10')
    tx2.loc[future_mask, ['Quantity', 'Price']] = [99, 99]
    changed = build_customer_dataset(tx2).set_index('Customer ID')

    cols = [c for c in base.columns if c != 'Churned']
    pd.testing.assert_frame_equal(base[cols], changed[cols])
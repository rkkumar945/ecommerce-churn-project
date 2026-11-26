''' ye test file ensure karti hai ki customer churn prediction pipeline bina data leakage ke past data se features or future window se sahi labels calculate kare'''

import sys
from pathlib import Path

import pandas as pd
import pytest

# plain `pytest` bhi chale (project root ko import path mein daalo)
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # isko use akrte hai taki src folder se koi bhi file ke function ya module ko bina error ke import kar sake.
from src.features.build_features import build_customer_dataset, region_bucket

COLS = [
    "Customer ID",
    "Invoice",
    "InvoiceDate",
    "Quantity",
    "Price",
    "Country",
    "StockCode",
]


def _df(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


# ek dummy transaction dataframe banata hai jisme alag alag dates, quantities,prices or countries ke sath 5 transactions hai. jo bina predictable data ke sath tests chalane liye hai.
def make_tx():
    return _df(
        [
            # (Customer ID, Invoice, InvoiceDate, Quantity, Price, Country, StockCode)
            (
                1,
                "A1",
                "2011-01-01",
                1,
                10.0,
                "United Kingdom",
                "S1",
            ),  # sirf purana -> churned
            (2, "A2", "2011-01-05", 2, 5.0, "Germany", "S1"),  # past
            (2, "A3", "2011-11-20", 1, 5.0, "Germany", "S2"),  # future -> not churned
            (
                3,
                "A4",
                "2011-11-25",
                1,
                9.0,
                "France",
                "S3",
            ),  # sirf future -> features mein nahi
            (2, "A5", "2011-12-09", 1, 5.0, "Germany", "S2"),  # dataset ka aakhri din
        ]
    )


# max date 2011-12-09 -> cutoff (90 din) = 2011-09-10


# jo customer future me activ ethe unko drop kiya hai ya nahi.
def test_future_only_customers_excluded():
    out = build_customer_dataset(make_tx())
    assert set(out["Customer ID"]) == {1, 2}


#churn label define karta hai.
def test_label_comes_from_future_window():
    out = build_customer_dataset(make_tx()).set_index("Customer ID")
    assert out.loc[1, "Churned"] == 1
    assert out.loc[2, "Churned"] == 0


# data leakage ko avoid karne ke liye.
def test_features_use_only_past_transactions():
    out = build_customer_dataset(make_tx()).set_index("Customer ID")
    assert out.loc[2, "Frequency"] == 1  # sirf A2, future ke A3/A5 nahi
    assert out.loc[2, "Monetary"] == 10.0  # 2 x 5.0
    assert out.loc[2, "UniqueProducts"] == 1  # sirf S1


# ye check karta hai recency time or tenure(pehli purchase se kitne din  ho gye).
def test_recency_and_tenure_values():
    out = build_customer_dataset(make_tx()).set_index("Customer ID")
    # cutoff 2011-09-10: customer 1 -> 2011-01-01 (252 din), customer 2 -> 2011-01-05 (248 din)
    assert out.loc[1, "Recency"] == 252 and out.loc[1, "Tenure"] == 252
    assert out.loc[2, "Recency"] == 248 and out.loc[2, "Tenure"] == 248
    assert out.loc[2, "AvgOrderValue"] == 10.0


# ye check karta hai ki exact date per koi order hai to wo future window me maan jaiga , past me nahi.
def test_invoice_exactly_at_cutoff_belongs_to_future():
    df = _df(
        [
            (1, "A1", "2011-01-01", 1, 10.0, "United Kingdom", "S1"),
            (
                1,
                "A2",
                "2011-09-10",
                1,
                10.0,
                "United Kingdom",
                "S1",
            ),  # == cutoff -> future
            (2, "A3", "2011-12-09", 1, 5.0, "Germany", "S1"),
        ]
    )
    out = build_customer_dataset(df).set_index("Customer ID")
    assert out.loc[1, "Frequency"] == 1  # A2 past mein nahi gina gaya
    assert out.loc[1, "Churned"] == 0  # A2 future mein hai -> churn nahi


# ye test karta hai ki agar hum horizon_days(jaise 90 din se 10 din) badal de to cutoff date shift ho jati hai, jis se kuch transactions jo pehle future me thi ab past me aa gyi.
def test_horizon_days_changes_cutoff():
    # horizon 10 din -> cutoff 2011-11-29; customer 2 ke A3 (Nov 20) ab past mein aata hai
    out = build_customer_dataset(make_tx(), horizon_days=10).set_index("Customer ID")
    assert out.loc[2, "Frequency"] == 2
    assert out.loc[2, "Churned"] == 0  # A5 (Dec 9) future mein hai
    assert 3 in out.index  # A4 (Nov 25) ab past mein -> customer 3 aata hai
    assert out.loc[3, "Churned"] == 1  # aur future mein nahi aaya



@pytest.mark.parametrize("horizon_days", [0, -1, 1.5, float("nan"), True]) #is line ka use karke check karte hai ki agar koi horizon_days me wrong value(0,-1,1.3,NaN, ya Boolean) de to ValueError de.
def test_horizon_days_must_be_a_positive_integer(horizon_days):
    with pytest.raises(ValueError, match="horizon_days must be a positive integer"):
        build_customer_dataset(make_tx(), horizon_days=horizon_days)

# ye verify karta hai agar hum horizon_days ki value na pass kare to function default 90 din hi le.
def test_default_horizon_days_is_90():
    pd.testing.assert_frame_equal(
        build_customer_dataset(make_tx()),
        build_customer_dataset(make_tx(), horizon_days=90),
    )


# ye check jarta hai alag alag countries ko sahi catagories me map kar raha hai kya.
def test_region_bucketing():
    assert region_bucket("United Kingdom") == "UK"
    assert region_bucket("Germany") == "Rest of Europe"
    assert region_bucket("Australia") == "Rest of World"
    out = build_customer_dataset(make_tx()).set_index("Customer ID")
    assert out.loc[1, "Region"] == "UK" and out.loc[2, "Region"] == "Rest of Europe"


#agar quantity negative ho(jaise refund/return case me) to Mintery_log me NaN de sakte hai.
def test_negative_monetary_does_not_produce_nan():
    df = make_tx()
    df.loc[0, "Quantity"] = -5  # customer 1: Monetary = -50
    out = build_customer_dataset(df).set_index("Customer ID")
    assert out.loc[1, "Monetary"] == -50.0
    assert out.loc[1, "Monetary_log"] == 0.0
    assert not out["Monetary_log"].isna().any()

# ye check karta hai ki build_customer_dataset function original Dataframe ko modify to nahi kar raha hai. iske liye hi hum .copy ka use karte hai taki orifignal data safe rahe.
def test_input_is_not_mutated():
    df = make_tx()
    before = df.copy()
    build_customer_dataset(df)
    pd.testing.assert_frame_equal(df, before)


# check karta hai agar exclude_non_product=True pass kare to non_product codes(jaise shipping fee"POST") drop ho jate hai.
def test_exclude_non_product_codes():
    df = make_tx()
    df.loc[1, "StockCode"] = "POST"  # customer 2 ki A2 row ab fee hai
    kept = build_customer_dataset(df).set_index("Customer ID")
    dropped = build_customer_dataset(df, exclude_non_product=True)
    assert kept.loc[2, "Frequency"] == 1
    assert 2 not in set(dropped["Customer ID"])  # customer 2 ka past activity sab POST


# ye check kart ha ki final features Dataframe me Churned ko chod kar kisi bhi features column me koi NaN value nahi hai.
def test_no_nans_in_features():
    out = build_customer_dataset(make_tx())
    assert not out.drop(columns=["Churned"]).isnull().any().any()


# ye ensure kart hai ki future or past ke data ka apas me koi calculation nahi hai.
def test_features_invariant_to_future_rows():
    """Future transactions ko kuch bhi kar do, features nahi badalne chahiye (sirf label badal sakta hai)."""
    tx = make_tx()
    base = build_customer_dataset(tx).set_index("Customer ID")

    tx2 = tx.copy()
    future_mask = tx2["InvoiceDate"] >= pd.Timestamp("2011-09-10")
    tx2.loc[future_mask, ["Quantity", "Price"]] = [99, 99]
    changed = build_customer_dataset(tx2).set_index("Customer ID")

    cols = [c for c in base.columns if c != "Churned"]
    pd.testing.assert_frame_equal(base[cols], changed[cols])




# ye chek karta ha ki agar as_of parameter explicitly pass na kare to automatically dataset ki sabse last date(max() date) ko hi as a reference leta hai.
def test_default_as_of_matches_explicit_max():
    tx = make_tx()
    pd.testing.assert_frame_equal(build_customer_dataset(tx),
                                  build_customer_dataset(tx, as_of=tx['InvoiceDate'].max()))


# agar hum as_of='2011-11-01' kar de to is date ke baad ki sari transactions hide ho jati hai or sare features isi dat tak ke data per bante hai.
def test_as_of_hides_later_transactions():
    out = build_customer_dataset(make_tx(), as_of='2011-11-01').set_index('Customer ID')
    assert set(out.index) == {1, 2}
    assert out['Churned'].tolist() == [1, 1]        # 2011-11-20 wala order ab dikhta hi nahi (default mein customer 2 = 0)


# ensure kart hai agar koi as_of date range se bahar ki di gyi to function Valuerror dega.
def test_as_of_beyond_data_or_before_data_is_rejected():
    with pytest.raises(ValueError):
        build_customer_dataset(make_tx(), as_of='2012-03-01')
    with pytest.raises(ValueError):
        build_customer_dataset(make_tx(), as_of='2010-01-01')


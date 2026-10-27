import pandas as pd
from src.features.build_features import build_customer_dataset


def make_tx():
    rows = [
        # (Customer ID, Invoice, InvoiceDate, Quantity, Price, Country, StockCode)
        (
            1,
            "A1",
            "2011-01-01",
            1,
            10.0,
            "United Kingdom",
            "S1",
        ),  # sirf purana, future mein nahi -> churned
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
        ),  # sirf future -> features mein nahi aana chahiye
        (2, "A5", "2011-12-09", 1, 5.0, "Germany", "S2"),  # dataset ka aakhri din
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "Customer ID",
            "Invoice",
            "InvoiceDate",
            "Quantity",
            "Price",
            "Country",
            "StockCode",
        ],
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def test_future_only_customers_excluded():
    out = build_customer_dataset(make_tx())
    assert set(out["Customer ID"]) == {1, 2}


def test_label_comes_from_future_window():
    out = build_customer_dataset(make_tx()).set_index("Customer ID")
    assert out.loc[1, "Churned"] == 1
    assert out.loc[2, "Churned"] == 0


def test_features_use_only_past_transactions():
    out = build_customer_dataset(make_tx()).set_index("Customer ID")
    assert out.loc[2, "Frequency"] == 1  # sirf A2, future ke A3/A5 nahi
    assert out.loc[2, "Monetary"] == 10.0  # 2 x 5.0

"""Transaction cleaning for Online Retail II (Days 31-35)."""

import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop guest checkouts, fill Description, drop duplicates and cancellations, fix dtypes."""
    df = df.dropna(subset=["Customer ID"]).copy()
    df["Description"] = df["Description"].fillna("Unknown Product")
    df = df.drop_duplicates()
    df = df[~df["Invoice"].astype(str).str.startswith("C")].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype(int)
    return df


import pandas as pd
from src.cleaning.clean_data import clean_transactions

raw = pd.read_csv(
    "data/raw/online_retail_II.csv", dtype={"Invoice": str, "StockCode": str}
)
ref = pd.read_csv(
    "data/processed/step4_datatypes_fixed.csv", dtype={"Invoice": str, "StockCode": str}
)

out = clean_transactions(raw)
print(len(out), len(ref))
print(out["Customer ID"].nunique(), ref["Customer ID"].nunique())

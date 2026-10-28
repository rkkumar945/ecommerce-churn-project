"""Transaction cleaning for Online Retail II (Days 31-35)."""
import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop guest checkouts, fill Description, drop duplicates and cancellations, fix dtypes."""
    df = df.dropna(subset=['Customer ID']).copy()
    df['Description'] = df['Description'].fillna('Unknown Product')
    df = df.drop_duplicates()
    df = df[~df['Invoice'].astype(str).str.startswith('C')].copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Customer ID'] = df['Customer ID'].astype(int)
    return df
















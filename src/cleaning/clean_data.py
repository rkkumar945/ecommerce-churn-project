"""Transaction cleaning for Online Retail II (Days 31-35)."""
import pandas as pd

REQUIRED_COLUMNS = ['Invoice', 'StockCode', 'Description', 'Quantity',
                    'InvoiceDate', 'Price', 'Customer ID', 'Country']

# Postage / fees / adjustments / test rows - not real products.
# (Original list + codes actually present in the data: C2 carriage, BANK CHARGES, PADS, D, TEST*)
NON_PRODUCT_CODES = ['POST', 'DOT', 'M', 'ADJUST', 'GIFT', 'AMAZONFEE',
                     'BANK CHARGES', 'C2', 'PADS', 'D', 'TEST001', 'TEST002']


def clean_transactions(df: pd.DataFrame, exclude_non_product: bool = False,
                       drop_nonpositive_price: bool = False) -> pd.DataFrame:
    """Drop guest checkouts, fill Description, drop duplicates and cancellations, fix dtypes.

    exclude_non_product:    True -> POST/DOT/M/ADJUST/... rows hata do (README mein "excluded" likha hai).
    drop_nonpositive_price: True -> Price <= 0 wali rows (free/adjustment lines) hata do.
    Dono default False hain taaki purane notebook results (aur README ke numbers) reproduce hon.
    Input DataFrame change nahi hota.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=['Customer ID']).copy()
    df['Description'] = df['Description'].fillna('Unknown Product')
    df = df.drop_duplicates()
    df = df[~df['Invoice'].astype(str).str.startswith('C')].copy()   # cancellations/returns
    if exclude_non_product:
        df = df[~df['StockCode'].astype(str).isin(NON_PRODUCT_CODES)].copy()
    if drop_nonpositive_price:
        df = df[df['Price'] > 0].copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Customer ID'] = df['Customer ID'].astype(int)
    return df
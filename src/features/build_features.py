"""Leakage-free customer feature table (observation window vs prediction window)."""
import numpy as np
import pandas as pd

EUROPE = ['Germany', 'France', 'Spain', 'Netherlands', 'Belgium', 'Switzerland', 'Portugal',
          'Italy', 'Austria', 'Denmark', 'Sweden', 'Norway', 'Finland', 'EIRE', 'Ireland',
          'Poland', 'Channel Islands', 'Cyprus', 'Greece']

# Postage / fees / adjustments in Online Retail II - not real products.
NON_PRODUCT_CODES = ['POST', 'DOT', 'M', 'ADJUST', 'GIFT', 'AMAZONFEE']


def region_bucket(country: str) -> str:
    if country == 'United Kingdom':
        return 'UK'
    return 'Rest of Europe' if country in EUROPE else 'Rest of World'


def build_customer_dataset(tx: pd.DataFrame, horizon_days: int = 90,
                           exclude_non_product: bool = False) -> pd.DataFrame:
    """Features sirf cutoff se pehle ke transactions se; label cutoff ke baad ke window se.

    tx must already be cleaned (see src/cleaning/clean_data.py).
    exclude_non_product: True karne par POST/DOT/M/ADJUST/GIFT/AMAZONFEE rows hata di jaati hain.
    Default False rakha hai taaki purane notebook results reproduce hote rahein.
    """
    tx = tx.copy()
    if exclude_non_product:
        tx = tx[~tx['StockCode'].astype(str).isin(NON_PRODUCT_CODES)]
    tx['LineTotal'] = tx['Quantity'] * tx['Price']

    cutoff = tx['InvoiceDate'].max() - pd.Timedelta(days=horizon_days)
    past = tx[tx['InvoiceDate'] < cutoff]
    future = tx[tx['InvoiceDate'] >= cutoff]

    f = past.groupby('Customer ID').agg(
        Frequency=('Invoice', 'nunique'),
        Monetary=('LineTotal', 'sum'),
        UniqueProducts=('StockCode', 'nunique'),
        FirstPurchase=('InvoiceDate', 'min'),
        LastPurchase=('InvoiceDate', 'max'),
        Country=('Country', lambda s: s.mode()[0]),
    ).reset_index()

    f['Recency'] = (cutoff - f['LastPurchase']).dt.days
    f['Tenure'] = (cutoff - f['FirstPurchase']).dt.days
    f['AvgOrderValue'] = f['Monetary'] / f['Frequency']
    # clip: net-negative Monetary (e.g. adjustment rows) par log1p NaN na de
    f['Monetary_log'] = np.log1p(f['Monetary'].clip(lower=0))
    f['Region'] = f['Country'].map(region_bucket)
    f['Churned'] = (~f['Customer ID'].isin(future['Customer ID'].unique())).astype(int)

    return f.drop(columns=['FirstPurchase', 'LastPurchase'])


if __name__ == '__main__':
    tx = pd.read_csv('data/processed/step4_datatypes_fixed.csv',
                     dtype={'Invoice': str, 'StockCode': str}, parse_dates=['InvoiceDate'])
    out = build_customer_dataset(tx)
    out.to_csv('data/processed/customer_ml_dataset_check.csv', index=False)
    print(out.shape, 'churn rate:', round(out['Churned'].mean(), 3))
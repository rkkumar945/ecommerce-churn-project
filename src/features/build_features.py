"""Leakage-free customer feature table (observation window vs prediction window)."""
import numpy as np
import pandas as pd

EUROPE = ['Germany', 'France', 'Spain', 'Netherlands', 'Belgium', 'Switzerland', 'Portugal',
          'Italy', 'Austria', 'Denmark', 'Sweden', 'Norway', 'Finland', 'EIRE', 'Ireland',
          'Poland', 'Channel Islands', 'Cyprus', 'Greece']
REGIONS = ['Rest of Europe', 'Rest of World', 'UK']

NUM_V1 = ['Recency', 'Frequency', 'Monetary_log', 'AvgOrderValue', 'Tenure', 'UniqueProducts']
NUM_NEW = ['AvgGapDays', 'LastGapDays', 'OverdueRatio', 'SingleOrder', 'OrdersPerMonth',
           'RunRateValue90_log', 'Orders_last90', 'Orders_prev90', 'OrderTrend',
           'Spend_last90_log', 'AvgUnitsPerOrder', 'AvgLinesPerOrder', 'MaxOrderValue_log']
NUM_V2 = NUM_V1 + NUM_NEW


def region_bucket(country: str) -> str:
    if country == 'United Kingdom':
        return 'UK'
    return 'Rest of Europe' if country in EUROPE else 'Rest of World'


def make_xy(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Numeric columns + one-hot Region (fixed categories, taaki columns hamesha same rahein)."""
    X = df[num_cols].copy()
    X['Region'] = pd.Categorical(df['Region'], categories=REGIONS)
    return pd.get_dummies(X, columns=['Region'], drop_first=True)


def build_customer_dataset(tx: pd.DataFrame, horizon_days: int = 90, exclude_non_product: bool = False) -> pd.DataFrame:
    """Features sirf cutoff se pehle ke transactions se; label cutoff ke baad ke window se."""
    tx = tx.copy()
    
    # Agar exclude_non_product True hai, toh 'POST' waale rows filter karein
    if exclude_non_product:
        tx = tx[tx['StockCode'] != 'POST']

    tx['LineTotal'] = tx['Quantity'] * tx['Price']

    cutoff = tx['InvoiceDate'].max() - pd.Timedelta(days=horizon_days)
    past = tx[tx['InvoiceDate'] < cutoff]
    future = tx[tx['InvoiceDate'] >= cutoff]

    # ---- Day 62: base features (unchanged) ----
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
    
    # Negative Monetary values ko 0 clip karke log1p calculate karein
    f['Monetary_log'] = np.log1p(f['Monetary'].clip(lower=0))
    
    f['Region'] = f['Country'].map(region_bucket)

    # ---- Day 82: behavioral features (sirf past) ----
    inv = (past.groupby(['Customer ID', 'Invoice'])
               .agg(date=('InvoiceDate', 'min'), value=('LineTotal', 'sum'),
                    units=('Quantity', 'sum'), lines=('StockCode', 'nunique'))
               .reset_index().sort_values(['Customer ID', 'date']))
    inv['gap'] = inv.groupby('Customer ID')['date'].diff().dt.days

    g = inv.groupby('Customer ID')
    beh = pd.DataFrame({
        'AvgGapDays': g['gap'].mean(),
        'LastGapDays': g['gap'].last(),
        'AvgUnitsPerOrder': g['units'].mean(),
        'AvgLinesPerOrder': g['lines'].mean(),
        'MaxOrderValue': g['value'].max(),
    })

    last90 = inv[inv['date'] >= cutoff - pd.Timedelta(days=90)]
    prev90 = inv[(inv['date'] >= cutoff - pd.Timedelta(days=180)) &
                 (inv['date'] < cutoff - pd.Timedelta(days=90))]
    beh['Orders_last90'] = last90.groupby('Customer ID').size()
    beh['Orders_prev90'] = prev90.groupby('Customer ID').size()
    beh['Spend_last90'] = last90.groupby('Customer ID')['value'].sum()

    f = f.merge(beh.reset_index(), on='Customer ID', how='left')
    for c in ['Orders_last90', 'Orders_prev90', 'Spend_last90', 'AvgGapDays', 'LastGapDays']:
        f[c] = f[c].fillna(0)                      # constant fill, data se seekha hua nahi

    f['SingleOrder'] = (f['Frequency'] == 1).astype(int)
    f['OverdueRatio'] = np.where(f['Frequency'] > 1,
                                 f['Recency'] / f['AvgGapDays'].clip(lower=1), 0.0)
    span = f['Tenure'].clip(lower=30)
    f['OrdersPerMonth'] = f['Frequency'] / (span / 30)
    f['RunRateValue90'] = (f['Frequency'] / span * 90) * f['AvgOrderValue']   # CLV-style heuristic
    f['OrderTrend'] = f['Orders_last90'] - f['Orders_prev90']

    for c in ['RunRateValue90', 'Spend_last90', 'MaxOrderValue']:
        f[f'{c}_log'] = np.log1p(f[c].clip(lower=0))

    f['Churned'] = (~f['Customer ID'].isin(future['Customer ID'].unique())).astype(int)
    return f.drop(columns=['FirstPurchase', 'LastPurchase'])


if __name__ == '__main__':
    tx = pd.read_csv('data/processed/step4_datatypes_fixed.csv',
                     dtype={'Invoice': str, 'StockCode': str}, parse_dates=['InvoiceDate'])
    out = build_customer_dataset(tx)
    out.to_csv('data/processed/customer_ml_dataset_v2.csv', index=False)
    print(out.shape, 'churn rate:', round(out['Churned'].mean(), 3))
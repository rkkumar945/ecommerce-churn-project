"""Pipeline ke data contracts. Hard problems ki list return hoti hai; khaali list = theek."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
import pandas as pd

from src.features.build_features import NUM_V2, REGIONS

RAW_COLS = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']


class DataCheckError(RuntimeError):
    pass


def fail_on(problems: list[str], stage: str):
    if problems:
        raise DataCheckError(f'{stage} checks failed: ' + '; '.join(problems))


def check_raw(df: pd.DataFrame) -> list[str]:
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        return [f'missing columns {missing}']
    problems = []
    if len(df) == 0:
        problems.append('raw data is empty')
    if not pd.api.types.is_string_dtype(df['Invoice']):
        problems.append('Invoice must be read as string (C-prefix cancellations)')
    for c in ('Quantity', 'Price'):
        if not pd.api.types.is_numeric_dtype(df[c]):
            problems.append(f'{c} must be numeric')
    return problems


def check_clean(df: pd.DataFrame) -> list[str]:
    problems = []
    if len(df) == 0:
        return ['empty after cleaning']
    if df['Customer ID'].isna().any():
        problems.append('null Customer ID remains')
    if df['Invoice'].astype(str).str.startswith('C').any():
        problems.append('cancellation invoices remain')
    if df.duplicated().any():
        problems.append('duplicate rows remain')
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        problems.append('InvoiceDate not datetime')
    return problems


def check_features(feat: pd.DataFrame, min_class_count: int = 50) -> list[str]:
    need = ['Customer ID', 'Churned', 'Region', *NUM_V2]
    missing = [c for c in need if c not in feat.columns]
    if missing:
        return [f'missing columns {missing}']
    problems = []
    if feat['Customer ID'].isna().any() or feat['Customer ID'].duplicated().any():
        problems.append('Customer ID must be unique and non-null')
    if not np.isfinite(feat[NUM_V2].to_numpy(dtype=float)).all():
        problems.append('non-finite values in numeric features')
    if not set(feat['Churned'].unique()) <= {0, 1}:
        problems.append('Churned must be 0/1')
    counts = feat['Churned'].value_counts()
    for cls in (0, 1):
        if counts.get(cls, 0) < min_class_count:
            problems.append(f'class {cls} has only {counts.get(cls, 0)} rows (< {min_class_count})')
    if not set(feat['Region'].unique()) <= set(REGIONS):
        problems.append('unknown Region value')
    if (feat['Frequency'] < 1).any():
        problems.append('Frequency < 1')
    if (feat['Recency'] < 0).any() or (feat['Tenure'] < 0).any():
        problems.append('negative Recency/Tenure')
    if ((feat['SingleOrder'] == 1) != (feat['Frequency'] == 1)).any():
        problems.append('SingleOrder inconsistent with Frequency')
    return problems
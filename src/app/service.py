"""App ka logic (Streamlit se alag, taaki test ho sake aur agent/API bhi reuse kar sake)."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'app_artifacts'
MANIFEST_PATH = ART / 'model_manifest.json'
MODEL_PATH = ART / 'final_model.joblib'
TABLE_PATH = ART / 'customer_features_app.csv'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_problems(manifest: dict) -> list[str]:
    """Training aur abhi ke library versions ka fark (joblib/pickle version-sensitive hote hain)."""
    problems = []
    for lib in ('scikit-learn', 'xgboost', 'numpy', 'joblib'):
        want = manifest['env'].get(lib)
        if want is None:
            continue
        try:
            have = md.version(lib)
        except md.PackageNotFoundError:
            have = None
        if have != want:
            problems.append(f'{lib}: trained with {want}, running {have}')
    return problems

def load_artifacts():
    """(model, manifest) return karta hai; load se PEHLE sha256 verify karta hai."""
    if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError('app_artifacts/ incomplete: python -m src.app.export_artifacts chalao')
    manifest = json.loads(MANIFEST_PATH.read_text())
    if _sha256(MODEL_PATH) != manifest['model_file']['sha256']:
        raise ValueError('Model file manifest se match nahi karti (badli hui ya purani)')

    model = joblib.load(MODEL_PATH)
    expected = getattr(model, 'feature_names_in_', None)
    if expected is not None and list(expected) != list(manifest['features']):
        raise ValueError('Model ke features aur manifest ke features alag hain')
    return model, manifest





def load_table() -> pd.DataFrame:
    return pd.read_csv(TABLE_PATH)


def add_scores(model, table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Har customer ka model score aur 'top_share' (0.01 = sabse risky 1%). Original table nahi badalta."""
    out = table.copy()
    out['score'] = model.predict_proba(out[features])[:, 1]
    out['top_share'] = out['score'].rank(ascending=False, method='min') / len(out)
    return out


def find_customer(table: pd.DataFrame, text):
    """(row, None) ya (None, error message). User input par bharosa nahi: sirf digits."""
    text = (text or '').strip()
    if not text:
        return None, 'Customer ID likho.'
    if not text.isdigit():
        return None, 'Customer ID sirf digits hona chahiye (jaise 12347).'
    hit = table[table['Customer ID'] == int(text)]
    if hit.empty:
        return None, (f'Customer {text} lookup table mein nahi hai. Ya ID galat hai, ya is customer ne '
                      'cutoff date se pehle kuch kharida hi nahi (model sirf unhe score karta hai jinki history hai).')
    return hit.iloc[0], None


def pick_random(table: pd.DataFrame, flagged_only: bool, threshold: float, seed=None):
    pool = table[table['score'] >= threshold] if flagged_only else table
    if pool.empty:
        return None
    return pool.sample(1, random_state=seed).iloc[0]


def readable_profile(row) -> dict[str, str]:
    """Model ke inputs insaani bhasha mein (log-scale columns wapas £ mein)."""
    single = bool(row['SingleOrder'])
    return {
        'Days since last order': f"{row['Recency']:.0f}",
        'Orders (before cutoff)': f"{row['Frequency']:.0f}",
        'Total spend': f"£{np.expm1(row['Monetary_log']):,.0f}",
        'Avg order value': f"£{row['AvgOrderValue']:,.0f}",
        'Customer age': f"{row['Tenure']:.0f} days",
        'Distinct products': f"{row['UniqueProducts']:.0f}",
        'Usual gap between orders': 'n/a (1 order)' if single else f"{row['AvgGapDays']:.0f} days",
        'Overdue ratio': 'n/a (1 order)' if single else f"{row['OverdueRatio']:.1f}x",
        'Orders (last 90d)': f"{row['Orders_last90']:.0f}",
        'Orders (prior 90d)': f"{row['Orders_prev90']:.0f}",
    }
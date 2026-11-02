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





    # ---------------- Day 89: real-time prediction, what-if, explanations ----------------
WHATIF = ('Recency', 'Orders_last90', 'Orders_prev90')


def row_to_frame(row, features: list[str]) -> pd.DataFrame:
    """Ek customer ko model ke exact column order mein 1-row DataFrame banao."""
    return pd.DataFrame([{f: float(row[f]) for f in features}], columns=features)


def top_share_for(reference_scores, score: float) -> float:
    """Sabse risky ka share (0.01 = top 1%). add_scores ke rank(method='min') jaisa hi formula."""
    s = np.asarray(reference_scores, dtype=float)
    return float((1 + (s > score).sum()) / len(s))


def predict_row(model, row, features, threshold, reference_scores=None) -> dict:
    """Real-time prediction: ek customer ke features -> score, alert flag, risk rank."""
    score = float(model.predict_proba(row_to_frame(row, features))[0, 1])
    out = {'score': score, 'flagged': score >= threshold}
    if reference_scores is not None:
        out['top_share'] = top_share_for(reference_scores, score)
    return out


def whatif_problems(row, changes: dict) -> list[str]:
    """Impossible combinations ki list (khaali list = scenario possible hai).
    Windows cutoff se peeche naapi jaati hain; 90/180 din ki seema par 1-2 din ka margin rakha hai."""
    r = {k: float(changes.get(k, row[k])) for k in WHATIF}
    msgs = []
    if r['Recency'] > row['Tenure']:
        msgs.append(f"Days since last order cannot exceed customer age ({row['Tenure']:.0f} days).")
    if r['Orders_last90'] + r['Orders_prev90'] > row['Frequency']:
        msgs.append(f"Orders in the two 90-day windows cannot exceed total orders ({row['Frequency']:.0f}).")
    if r['Recency'] >= 91 and r['Orders_last90'] > 0:
        msgs.append('Last order was more than 90 days before the cutoff, so orders in the last 90 days must be 0.')
    if r['Recency'] <= 88 and r['Orders_last90'] < 1:
        msgs.append('Last order was within 90 days of the cutoff, so there must be at least 1 order in that window.')
    if r['Recency'] >= 181 and r['Orders_prev90'] > 0:
        msgs.append('Last order was more than 180 days before the cutoff, so orders in the prior 90 days must be 0.')
    if 92 <= r['Recency'] <= 178 and r['Orders_prev90'] < 1:
        msgs.append('Last order fell in the prior 90-day window, so there must be at least 1 order there.')
    return msgs


def apply_whatif(row, changes: dict):
    """Chuni hui values badlo, aur dependent features build_features.py ke formulas se dobara nikaalo."""
    new = row.copy()
    for k, v in changes.items():
        if k not in WHATIF:
            raise KeyError(f'{k} what-if ke liye allowed nahi hai')
        new[k] = float(v)

    new['OverdueRatio'] = (0.0 if int(new['SingleOrder']) == 1
                           else float(new['Recency']) / max(float(new['AvgGapDays']), 1.0))
    new['OrderTrend'] = float(new['Orders_last90']) - float(new['Orders_prev90'])

    old_n, new_n = float(row['Orders_last90']), float(new['Orders_last90'])
    if new_n == 0:
        spend = 0.0
    else:   # order ka size wahi maano jo pehle tha (ya 0 orders the to average order value)
        per_order = np.expm1(float(row['Spend_last90_log'])) / old_n if old_n > 0 else float(row['AvgOrderValue'])
        spend = per_order * new_n
    new['Spend_last90_log'] = float(np.log1p(spend))
    return new


def _gbp(v) -> str:
    return f'£{np.expm1(v):,.0f}'


FEATURE_TEXT = {
    'Recency': lambda v: f'{v:.0f} days between the last order and the cutoff',
    'Frequency': lambda v: f'{v:.0f} orders so far',
    'Monetary_log': lambda v: f'{_gbp(v)} total spend',
    'AvgOrderValue': lambda v: f'£{v:,.0f} average order value',
    'Tenure': lambda v: f'customer for {v:.0f} days',
    'UniqueProducts': lambda v: f'{v:.0f} distinct products bought',
    'AvgGapDays': lambda v: f'{v:.0f} days between orders on average',
    'LastGapDays': lambda v: f'{v:.0f} days between the last two orders',
    'OverdueRatio': lambda v: f'{v:.1f}x past their usual order gap',
    'SingleOrder': lambda v: 'only one order so far' if v >= 0.5 else 'repeat customer',
    'OrdersPerMonth': lambda v: f'{v:.2f} orders per month',
    'RunRateValue90_log': lambda v: f'{_gbp(v)} expected 90-day spend at the past pace',
    'Orders_last90': lambda v: f'{v:.0f} orders in the 90 days before the cutoff',
    'Orders_prev90': lambda v: f'{v:.0f} orders in the 90 days before that',
    'OrderTrend': lambda v: f'order count {v:+.0f} vs the previous 90 days',
    'Spend_last90_log': lambda v: f'{_gbp(v)} spent in the 90 days before the cutoff',
    'AvgUnitsPerOrder': lambda v: f'{v:.0f} units per order',
    'AvgLinesPerOrder': lambda v: f'{v:.1f} product lines per order',
    'MaxOrderValue_log': lambda v: f'{_gbp(v)} largest single order',
    'Region_UK': lambda v: 'UK customer' if v >= 0.5 else 'non-UK customer',
    'Region_Rest of World': lambda v: 'customer outside UK/Europe' if v >= 0.5 else 'UK or Europe customer',
}


def describe_feature(name: str, value) -> str:
    fmt = FEATURE_TEXT.get(name)
    return fmt(float(value)) if fmt else f'{name} = {float(value):.2f}'


# ---------------- Day 90: presentation helpers ----------------
def wilson_interval(successes: float, n: int, z: float = 1.96):
    """Proportion ka 95% (Wilson) interval. n=0 par (nan, nan)."""
    if n == 0:
        return float('nan'), float('nan')
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def top_k_summary(pool: pd.DataFrame, frac: float) -> dict:
    """Pool ko score se rank karke top frac ka haal: precision, lift, churners ka kitna hissa reach hua."""
    n = len(pool)
    k = max(1, round(n * frac))
    top = pool.sort_values('score', ascending=False).head(k)
    base_rate = float(pool['observed_churn'].mean())
    precision = float(top['observed_churn'].mean())
    churners = float(pool['observed_churn'].sum())
    return {
        'k': k, 'n': n, 'precision': precision, 'base_rate': base_rate,
        'lift': precision / base_rate if base_rate > 0 else float('nan'),
        'recall': float(top['observed_churn'].sum() / churners) if churners > 0 else float('nan'),
        'min_score': float(top['score'].min()),
    }


def driver_frame(model, row, features, background):
    """Har feature ka SHAP asar, |asar| ke hisaab se sorted: (DataFrame[feature, value, shap, text], space)."""
    from src.models.explain import explain  # shap bhaari hai, zaroorat par hi import

    x = row_to_frame(row, features)
    r = explain(model, x, background=background)
    if r['gap'] > 1e-3:
        raise ValueError(f"SHAP additivity check fail (gap {r['gap']:.2e})")
    df = pd.DataFrame({'feature': features, 'value': x.iloc[0].to_numpy(),
                       'shap': r['values'].iloc[0].to_numpy()})
    df['text'] = [describe_feature(f, v) for f, v in zip(df['feature'], df['value'])]
    df = df.reindex(df['shap'].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df, r['space']


def explain_row(model, row, features, background, k: int = 3) -> dict:
    """Top-k 'score badhane wale' (up) aur 'ghatane wale' (down): [(feature, value, shap)]."""
    df, space = driver_frame(model, row, features, background)
    triples = list(zip(df['feature'], df['value'], df['shap']))
    return {'up': [t for t in triples if t[2] > 0][:k],
            'down': [t for t in triples if t[2] < 0][:k],
            'space': space}


def top_reasons(model, rows: pd.DataFrame, features, background) -> list[str]:
    """Har row ka sabse bada 'risk badhane wala' reason, ek hi SHAP call mein."""
    from src.models.explain import explain

    X = rows[features].astype(float).reset_index(drop=True)
    r = explain(model, X, background=background)
    if r['gap'] > 1e-3:
        raise ValueError(f"SHAP additivity check fail (gap {r['gap']:.2e})")
    vals = r['values'].to_numpy()
    out = []
    for i in range(len(X)):
        j = int(np.argmax(vals[i]))
        out.append(describe_feature(features[j], X.iloc[i, j]) if vals[i, j] > 0 else '-')
    return out
"""App ka logic (Streamlit se alag, taaki test ho sake aur agent/API bhi reuse kar sake)."""


import hashlib
import importlib.metadata as md
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd 

from src.models.topk import top_k_count

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "app_artifacts"
MANIFEST_PATH = ART / "model_manifest.json"
MODEL_PATH = ART / "final_model.joblib"
TABLE_PATH = ART / "customer_features_app.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_problems(manifest: dict) -> list[str]:
    """Training aur abhi ke library versions ka fark (joblib/pickle version-sensitive hote hain)."""
    problems = []
    for lib in ("scikit-learn", "xgboost", "numpy", "joblib"):
        want = manifest["env"].get(lib)
        if want is None:
            continue
        try:
            have = md.version(lib)
        except md.PackageNotFoundError:
            have = None
        if have != want:
            problems.append(f"{lib}: trained with {want}, running {have}")
    return problems


def load_artifacts():
    """(model, manifest) return karta hai; load se PEHLE sha256 verify karta hai."""
    if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "app_artifacts/ is incomplete: run python -m src.app.export_artifacts"
        )
    manifest = json.loads(MANIFEST_PATH.read_text())
    if _sha256(MODEL_PATH) != manifest["model_file"]["sha256"]:
        raise ValueError(
            "Model file does not match the manifest (it may be changed or outdated)"
        )

    model = joblib.load(MODEL_PATH)
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None and list(expected) != list(manifest["features"]):
        raise ValueError("Model features do not match the manifest features")
    return model, manifest


def load_table() -> pd.DataFrame:
    return pd.read_csv(TABLE_PATH)


def add_scores(model, table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Har customer ka model score aur 'top_share' (0.01 = sabse risky 1%). Original table nahi badalta."""
    out = table.copy()
    out["score"] = model.predict_proba(out[features])[:, 1]
    out["top_share"] = out["score"].rank(ascending=False, method="min") / len(out)
    return out


def select_contact_ids(
    table: pd.DataFrame, budget_pct: float, *, population: str = "test"
) -> set[int]:
    """Deterministic contact-list membership for a chosen active population.

    The default operational population is the held-out (test) split, because it matches the
    honest ranking view in the app. Selection uses one exact ordering: score desc, Customer ID asc,
    then head(k) with k = round_half_up(n * budget_pct / 100).
    """
    if not 0.0 <= float(budget_pct) <= 100.0:
        raise ValueError("budget_pct must be between 0 and 100")
    pool = table.copy()
    if population == "test":
        pool = table[table["split"] == "test"].copy() if "split" in table.columns else pool
    elif population == "all":
        pool = table.copy()
    elif population not in {"test", "all"}:
        raise ValueError("population must be 'test' or 'all'")
    if pool.empty:
        return set()
    k = top_k_count(len(pool), float(budget_pct) / 100.0)
    sort_cols = ["score"]
    ascending = [False]
    if "Customer ID" in pool.columns:
        sort_cols.append("Customer ID")
        ascending.append(True)
    ranked = pool.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ids = ranked["Customer ID"].tolist() if "Customer ID" in ranked.columns else ranked.index.tolist()
    return set(ids[:k])


def find_customer(table: pd.DataFrame, text):
    """(row, None) ya (None, error message). User input par bharosa nahi: sirf digits."""
    text = (text or "").strip()
    if not text:
        return None, "Enter a customer ID."
    if not text.isdigit():
        return None, "Customer ID must contain digits only (for example, 12347)."
    hit = table[table["Customer ID"] == int(text)]
    if hit.empty:
        return None, (
            f"Customer {text} was not found in the lookup table. The ID may be incorrect, or the customer may have no "
            "purchases before the cutoff date (the model scores customers with purchase history only)."
        )
    return hit.iloc[0], None


def pick_random(
    table: pd.DataFrame,
    flagged_only: bool,
    threshold: float,
    seed=None,
    top_share_max: float | None = None,
    selected_ids: set[int] | None = None,
):
    if not flagged_only:
        pool = table
    elif selected_ids is not None:
        pool = table[table["Customer ID"].isin(selected_ids)]
    elif top_share_max is not None:
        pool = table[table["top_share"] <= top_share_max]
    else:
        pool = table[table["score"] >= threshold]
    if pool.empty:
        return None
    return pool.sample(1, random_state=seed).iloc[0]


def readable_profile(row) -> dict[str, str]:
    """Model ke inputs insaani bhasha mein (log-scale columns wapas £ mein)."""
    single = bool(row["SingleOrder"])
    return {
        "Days since last order": f"{row['Recency']:.0f}",
        "Orders (before cutoff)": f"{row['Frequency']:.0f}",
        "Total spend": f"£{np.expm1(row['Monetary_log']):,.0f}",
        "Avg order value": f"£{row['AvgOrderValue']:,.0f}",
        "Customer age": f"{row['Tenure']:.0f} days",
        "Distinct products": f"{row['UniqueProducts']:.0f}",
        "Usual gap between orders": "n/a (1 order)"
        if single
        else f"{row['AvgGapDays']:.0f} days",
        "Overdue ratio": "n/a (1 order)" if single else f"{row['OverdueRatio']:.1f}x",
        "Orders (last 90d)": f"{row['Orders_last90']:.0f}",
        "Orders (prior 90d)": f"{row['Orders_prev90']:.0f}",
    }

    # ---------------- Day 89: real-time prediction, what-if, explanations ----------------


WHATIF = ("Recency", "Orders_last90", "Orders_prev90")


def row_to_frame(row, features: list[str]) -> pd.DataFrame:
    """Ek customer ko model ke exact column order mein 1-row DataFrame banao."""
    return pd.DataFrame([{f: float(row[f]) for f in features}], columns=features)


def top_share_for(reference_scores, score: float) -> float:
    """Sabse risky ka share (0.01 = top 1%). add_scores ke rank(method='min') jaisa hi formula."""
    s = np.asarray(reference_scores, dtype=float)
    if s.size == 0:
        raise ValueError("reference_scores must not be empty")
    if not np.isfinite(score):
        raise ValueError("score must be finite")
    share = (1 + (s > score).sum()) / len(s)
    return float(np.clip(share, 0.0, 1.0))


def predict_row(model, row, features, threshold, reference_scores=None) -> dict:
    """Real-time prediction: ek customer ke features -> score, alert flag, risk rank."""
    score = float(model.predict_proba(row_to_frame(row, features))[0, 1])
    out = {"score": score, "flagged": score >= threshold}
    if reference_scores is not None:
        out["top_share"] = top_share_for(reference_scores, score)
    return out


def whatif_problems(row, changes: dict) -> list[str]:
    """Impossible what-if combinations: return clear user-facing messages or an empty list."""
    msgs: list[str] = []
    numeric: dict[str, float] = {}
    for key, value in changes.items():
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            msgs.append(f"{key} must be numeric.")
            continue
        if not np.isfinite(number):
            msgs.append(f"{key} must be finite.")
            continue
        numeric[key] = number
        if key not in WHATIF:
            msgs.append(f"{key} cannot be changed in a what-if scenario.")

    counts = ("Frequency", "Orders_last90", "Orders_prev90", "UniqueProducts")
    for key in counts:
        if key in numeric:
            value = numeric[key]
        else:
            try:
                value = float(row[key])
            except (TypeError, ValueError, OverflowError):
                msgs.append(f"{key} must be a numeric count.")
                continue
            if not np.isfinite(value):
                msgs.append(f"{key} must be a finite count.")
                continue
        if value < 0 or not float(value).is_integer():
            msgs.append(f"{key} must be a non-negative integer count.")

    r: dict[str, float] = {}
    for key in WHATIF:
        if key in numeric:
            r[key] = numeric[key]
        else:
            try:
                value = float(row[key])
            except (TypeError, ValueError, OverflowError):
                msgs.append(f"{key} must be numeric.")
                continue
            if not np.isfinite(value):
                msgs.append(f"{key} must be finite.")
                continue
            r[key] = value

    if msgs:
        return msgs

    if r["Recency"] < 0 or not float(r["Recency"]).is_integer():
        msgs.append("Recency must be a non-negative integer number of days.")
    if r["Orders_last90"] < 0 or not float(r["Orders_last90"]).is_integer():
        msgs.append("Orders_last90 must be a non-negative integer count.")
    if r["Orders_prev90"] < 0 or not float(r["Orders_prev90"]).is_integer():
        msgs.append("Orders_prev90 must be a non-negative integer count.")

    if r["Recency"] > float(row["Tenure"]):
        msgs.append(
            f"Recency ({r['Recency']:.0f} days) cannot exceed customer age ({float(row['Tenure']):.0f} days)."
        )
    if int(row["SingleOrder"]) == 1 and not np.isclose(r["Recency"], float(row["Tenure"])):
        msgs.append("For a single-order customer, Recency must equal customer age.")
    if r["Orders_last90"] + r["Orders_prev90"] > float(row["Frequency"]) + 1e-9:
        msgs.append(
            f"Orders in the two 90-day windows cannot exceed total orders (Frequency={float(row['Frequency']):.0f})."
        )
    if r["Recency"] >= 91 and r["Orders_last90"] > 0:
        msgs.append(
            "If the last order was more than 90 days before the cutoff, the last-90 window must be 0 orders."
        )
    if r["Recency"] <= 88 and r["Orders_last90"] < 1:
        msgs.append(
            "If the last order was within 90 days of the cutoff, the last-90 window must include at least 1 order."
        )
    if r["Recency"] >= 181 and r["Orders_prev90"] > 0:
        msgs.append(
            "If the last order was more than 180 days before the cutoff, the prior-90 window must be 0 orders."
        )
    if 92 <= r["Recency"] <= 178 and r["Orders_prev90"] < 1:
        msgs.append(
            "If the last order fell in the prior 90-day window, that window must include at least 1 order."
        )
    if float(row["Tenure"]) < 90 and r["Orders_prev90"] > 0:
        msgs.append(
            "A customer with tenure under 90 days cannot have an order in the 90-to-180 day window."
        )
    return msgs


def apply_whatif(row, changes: dict):
    """Chuni hui values badlo, aur dependent features build_features.py ke formulas se dobara nikaalo."""
    for key in changes:
        if key not in WHATIF:
            raise KeyError(f"{key} is not allowed in a what-if scenario")
    problems = whatif_problems(row, changes)
    if problems:
        raise ValueError("Invalid what-if scenario: " + " ".join(problems))
    new = row.copy()
    for k, v in changes.items():
        if k not in WHATIF:
            raise KeyError(f"{k} is not allowed in a what-if scenario")
        new[k] = float(v)

    new["OverdueRatio"] = (
        0.0
        if int(new["SingleOrder"]) == 1
        else float(new["Recency"]) / max(float(new["AvgGapDays"]), 1.0)
    )
    new["OrderTrend"] = float(new["Orders_last90"]) - float(new["Orders_prev90"])

    old_n, new_n = float(row["Orders_last90"]), float(new["Orders_last90"])
    if new_n == 0:
        spend = 0.0
    else:  # order ka size wahi maano jo pehle tha (ya 0 orders the to average order value)
        per_order = (
            np.expm1(float(row["Spend_last90_log"])) / old_n
            if old_n > 0
            else float(row["AvgOrderValue"])
        )
        spend = per_order * new_n
    new["Spend_last90_log"] = float(np.log1p(spend))
    return new


def _gbp(v) -> str:
    return f"£{np.expm1(v):,.0f}"


FEATURE_TEXT = {
    "Recency": lambda v: f"{v:.0f} days between the last order and the cutoff",
    "Frequency": lambda v: f"{v:.0f} orders so far",
    "Monetary_log": lambda v: f"{_gbp(v)} total spend",
    "AvgOrderValue": lambda v: f"£{v:,.0f} average order value",
    "Tenure": lambda v: f"customer for {v:.0f} days",
    "UniqueProducts": lambda v: f"{v:.0f} distinct products bought",
    "AvgGapDays": lambda v: f"{v:.0f} days between orders on average",
    "LastGapDays": lambda v: f"{v:.0f} days between the last two orders",
    "OverdueRatio": lambda v: f"{v:.1f}x past their usual order gap",
    "SingleOrder": lambda v: "only one order so far" if v >= 0.5 else "repeat customer",
    "OrdersPerMonth": lambda v: f"{v:.2f} orders per month",
    "RunRateValue90_log": lambda v: f"{_gbp(v)} expected 90-day spend at the past pace",
    "Orders_last90": lambda v: f"{v:.0f} orders in the 90 days before the cutoff",
    "Orders_prev90": lambda v: f"{v:.0f} orders in the 90 days before that",
    "OrderTrend": lambda v: f"order count {v:+.0f} vs the previous 90 days",
    "Spend_last90_log": lambda v: f"{_gbp(v)} spent in the 90 days before the cutoff",
    "AvgUnitsPerOrder": lambda v: f"{v:.0f} units per order",
    "AvgLinesPerOrder": lambda v: f"{v:.1f} product lines per order",
    "MaxOrderValue_log": lambda v: f"{_gbp(v)} largest single order",
    "Region_UK": lambda v: "UK customer" if v >= 0.5 else "non-UK customer",
    "Region_Rest of World": lambda v: (
        "customer outside UK/Europe" if v >= 0.5 else "UK or Europe customer"
    ),
}


def describe_feature(name: str, value) -> str:
    fmt = FEATURE_TEXT.get(name)
    return fmt(float(value)) if fmt else f"{name} = {float(value):.2f}"


# ---------------- Day 90: presentation helpers ----------------
def wilson_interval(successes: float, n: int, z: float = 1.96):
    """Proportion ka 95% (Wilson) interval. n=0 par (nan, nan)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def top_k_summary(pool: pd.DataFrame, frac: float) -> dict:
    """Pool ko score se rank karke top frac ka haal: precision, lift, churners ka kitna hissa reach hua."""
    n = len(pool)
    k = top_k_count(n, frac)
    sort_cols = ["score"]
    ascending = [False]
    if "Customer ID" in pool.columns:
        sort_cols.append("Customer ID")
        ascending.append(True)
    top = pool.sort_values(sort_cols, ascending=ascending, kind="mergesort").head(k)
    base_rate = float(pool["observed_churn"].mean())
    precision = float(top["observed_churn"].mean())
    churners = float(pool["observed_churn"].sum())
    return {
        "k": k,
        "n": n,
        "precision": precision,
        "base_rate": base_rate,
        "lift": precision / base_rate if base_rate > 0 else float("nan"),
        "recall": float(top["observed_churn"].sum() / churners)
        if churners > 0
        else float("nan"),
        "min_score": float(top["score"].min()),
    }


def driver_frame(model, row, features, background):
    """Har feature ka SHAP asar, |asar| ke hisaab se sorted: (DataFrame[feature, value, shap, text], space)."""
    from src.models.explain import explain  # shap bhaari hai, zaroorat par hi import

    x = row_to_frame(row, features)
    r = explain(model, x, background=background)
    if r["gap"] > 1e-3:
        raise ValueError(f"SHAP additivity check fail (gap {r['gap']:.2e})")
    df = pd.DataFrame(
        {
            "feature": features,
            "value": x.iloc[0].to_numpy(),
            "shap": r["values"].iloc[0].to_numpy(),
        }
    )
    df["text"] = [describe_feature(f, v) for f, v in zip(df["feature"], df["value"])]
    df = df.reindex(df["shap"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )
    return df, r["space"]


def explain_row(model, row, features, background, k: int = 3) -> dict:
    """Top-k 'score badhane wale' (up) aur 'ghatane wale' (down): [(feature, value, shap)]."""
    df, space = driver_frame(model, row, features, background)
    triples = list(zip(df["feature"], df["value"], df["shap"]))
    return {
        "up": [t for t in triples if t[2] > 0][:k],
        "down": [t for t in triples if t[2] < 0][:k],
        "space": space,
    }


def top_reasons(model, rows: pd.DataFrame, features, background) -> list[str]:
    """Har row ka sabse bada 'risk badhane wala' reason, ek hi SHAP call mein."""
    from src.models.explain import explain

    X = rows[features].astype(float).reset_index(drop=True)
    r = explain(model, X, background=background)
    if r["gap"] > 1e-3:
        raise ValueError(f"SHAP additivity check fail (gap {r['gap']:.2e})")
    vals = r["values"].to_numpy()
    out = []
    for i in range(len(X)):
        j = int(np.argmax(vals[i]))
        out.append(
            describe_feature(features[j], X.iloc[i, j]) if vals[i, j] > 0 else "-"
        )
    return out





def runtime_env() -> dict:
    """Abhi jis environment mein app chal rahi hai (cloud par yahi dikhega)."""
    out = {'python': platform.python_version()}
    for lib in ('scikit-learn', 'xgboost', 'numpy', 'pandas', 'joblib', 'shap', 'streamlit'):
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            pass
    return out

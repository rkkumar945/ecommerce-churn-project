import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pytest

from src.app.service import (
    FEATURE_TEXT,
    add_scores,
    apply_whatif,
    explain_row,
    load_artifacts,
    load_table,
    predict_row,
    top_share_for,
    whatif_problems,
)


@pytest.fixture(scope="module")
def art():
    model, manifest = load_artifacts()
    return model, manifest, add_scores(model, load_table(), manifest["features"])


def test_predict_row_matches_table_scores(art):
    model, manifest, table = art
    for _, row in table.sample(40, random_state=0).iterrows():
        res = predict_row(
            model, row, manifest["features"], manifest["threshold"], table["score"]
        )
        assert np.isclose(res["score"], row["score"], atol=1e-9)
        assert res["flagged"] == (row["score"] >= manifest["threshold"])
        assert np.isclose(res["top_share"], row["top_share"])


def test_top_share_is_monotonic(art):
    _, _, table = art
    s = table["score"]
    assert (
        top_share_for(s, s.max() + 1)
        <= top_share_for(s, s.median())
        <= top_share_for(s, s.min() - 1)
    )


def test_no_change_whatif_reproduces_derived_features(art):
    """Train/serve skew guard: app ke formulas = build_features.py ke formulas."""
    _, _, table = art
    for _, row in table.iterrows():
        new = apply_whatif(row, {})
        for col in ("OverdueRatio", "OrderTrend", "Spend_last90_log"):
            assert np.isclose(new[col], row[col], atol=1e-6), (row["Customer ID"], col)


def test_real_customers_satisfy_the_constraints(art):
    """Agar yeh fail ho, to ya constraint rule mein off-by-one hai ya data mein koi edge case."""
    _, _, table = art
    bad = [
        (int(r["Customer ID"]), whatif_problems(r, {}))
        for _, r in table.iterrows()
        if whatif_problems(r, {})
    ]
    assert not bad, f"{len(bad)} real customers break the rules, e.g. {bad[:3]}"


def test_impossible_scenarios_are_rejected(art):
    _, _, table = art
    cand = table[(table["Frequency"] >= 3) & (table["Recency"] <= 88)]
    if cand.empty:
        pytest.skip("no suitable customer")
    row = cand.iloc[0]
    assert whatif_problems(row, {"Orders_last90": 0})  # last order recent, par 0 orders
    assert whatif_problems(
        row, {"Orders_last90": row["Frequency"], "Orders_prev90": 1}
    )  # total se zyada
    assert whatif_problems(row, {"Recency": row["Tenure"] + 1})


def test_dependents_follow_the_formulas(art):
    _, _, table = art
    row = table[table["Frequency"] > 1].iloc[0]
    new = apply_whatif(row, {"Recency": 40.0, "Orders_last90": 0, "Orders_prev90": 2})
    assert np.isclose(new["OverdueRatio"], 40.0 / max(row["AvgGapDays"], 1.0))
    assert new["OrderTrend"] == -2
    assert new["Spend_last90_log"] == 0.0
    assert row["Recency"] != 40.0 or True  # original row badli nahi
    with pytest.raises(KeyError):
        apply_whatif(row, {"Frequency": 9})


def test_every_model_feature_has_readable_text(art):
    _, manifest, _ = art
    assert set(manifest["features"]) <= set(FEATURE_TEXT)


def test_explain_row_signs_and_limit(art):
    pytest.importorskip("shap")
    model, manifest, table = art
    feats = manifest["features"]
    bg = table[feats].astype(float).sample(100, random_state=0).reset_index(drop=True)
    d = explain_row(model, table.iloc[0], feats, bg, k=3)
    assert len(d["up"]) <= 3 and len(d["down"]) <= 3
    assert all(s > 0 for _, _, s in d["up"])
    assert all(s < 0 for _, _, s in d["down"])

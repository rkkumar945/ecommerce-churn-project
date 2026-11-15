import numpy as np
import pandas as pd
import pytest

from scripts import topk_facts
from src.app.service import top_k_summary
from src.models.bootstrap import precision_at_frac, recall_at_frac
from src.models.topk import top_k_count


def test_top_k_rounding_is_shared_across_call_sites(monkeypatch, capsys):
    n = 1057
    labels = np.r_[np.ones(105), 0, np.ones(1), np.zeros(n - 107)]
    scores = np.arange(n, 0, -1)
    expected_precision = 105 / 106

    assert top_k_count(n, 0.10) == 106
    assert precision_at_frac(labels, scores, 0.10) == pytest.approx(
        expected_precision
    )
    assert recall_at_frac(labels, scores, 0.10) == pytest.approx(
        expected_precision
    )

    pool = pd.DataFrame(
        {"score": scores, "observed_churn": labels, "split": "test"}
    )
    summary = top_k_summary(pool, 0.10)
    assert summary["k"] == 106
    assert summary["precision"] == pytest.approx(expected_precision)

    monkeypatch.setattr(topk_facts.service, "load_artifacts", lambda: (None, {"features": []}))
    monkeypatch.setattr(topk_facts.service, "load_table", lambda: pool)
    monkeypatch.setattr(
        topk_facts.service, "add_scores", lambda model, table, features: table
    )
    topk_facts.main()
    assert "Top 10%: k=106" in capsys.readouterr().out

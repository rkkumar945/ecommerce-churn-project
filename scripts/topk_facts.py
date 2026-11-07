"""Reproduce held-out top-k targeting facts from the committed app artifacts."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import service


def main() -> None:
    model, manifest = service.load_artifacts()
    test = service.load_table()
    test = test.loc[test["split"].eq("test")].copy()
    if test.empty:
        raise ValueError("The app table contains no held-out test rows.")

    scored = service.add_scores(model, test, manifest["features"])
    scored = scored.sort_values("score", ascending=False, kind="mergesort")
    labels = scored["observed_churn"].to_numpy(dtype=int)
    scores = scored["score"].to_numpy(dtype=float)
    base_rate = float(labels.mean())

    for fraction in (0.10, 0.20):
        k = round(fraction * len(scored))
        top_labels = labels[:k]
        precision = float(top_labels.mean())
        lift = precision / base_rate
        reach = float(top_labels.sum() / labels.sum())
        print(
            f"Top {fraction:.0%}: k={k}, precision={precision:.3f}, "
            f"lift={lift:.3f}x, reach={reach:.1%}"
        )

        if fraction == 0.10:
            rng = np.random.default_rng(42)
            bootstrap_precision = np.empty(1000)
            for i in range(len(bootstrap_precision)):
                sample = rng.integers(0, len(labels), size=len(labels))
                order = np.argsort(-scores[sample], kind="stable")
                bootstrap_precision[i] = labels[sample[order[:k]]].mean()
            low, high = np.quantile(bootstrap_precision, [0.025, 0.975])
            print(
                f"Top 10% precision 95% bootstrap CI: "
                f"{low:.1%} to {high:.1%} (1000 resamples, seed=42)"
            )


if __name__ == "__main__":
    main()

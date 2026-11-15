"""Shared customer-count rounding for top-k targeting."""

import math


def top_k_count(n: int, frac: float) -> int:
    """Return the nearest whole-customer count, with half values rounded up."""
    return max(1, int(math.floor(n * frac + 0.5)))  # noqa: RUF046

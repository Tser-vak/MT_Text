"""Acceptance checks for the db_tools/metrics.py::bootstrap_ci hole. These FAIL
with NotImplementedError until you fill the hole; when they pass, the CI is
trustworthy enough to report alongside a ROUGE/BERTScore delta. Run via
`pytest` or `python test_metrics.py`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_tools.metrics import bootstrap_ci

VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_bootstrap_ci_brackets_mean():
    mean, lo, hi = bootstrap_ci(VALUES, B=500, seed=42)
    assert lo <= mean <= hi, f"CI [{lo}, {hi}] must bracket the mean {mean}"
    assert abs(mean - sum(VALUES) / len(VALUES)) < 1e-9, "mean must be the plain mean of `values`"


def test_bootstrap_ci_deterministic_under_seed():
    a = bootstrap_ci(VALUES, B=500, seed=42)
    b = bootstrap_ci(VALUES, B=500, seed=42)
    assert a == b, "same seed must reproduce the exact same CI"


def test_bootstrap_ci_narrows_as_b_grows():
    """Not the CI width itself (that need not shrink monotonically for one
    draw) -- the run-to-run VARIABILITY of the estimated width across
    different seeds must shrink as B grows, since more resamples means less
    Monte Carlo noise in the estimated percentiles."""
    values = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.55]
    small_widths = [bootstrap_ci(values, B=20, seed=s)[2] - bootstrap_ci(values, B=20, seed=s)[1]
                     for s in range(15)]
    big_widths = [bootstrap_ci(values, B=1500, seed=s)[2] - bootstrap_ci(values, B=1500, seed=s)[1]
                   for s in range(15)]
    assert np.std(big_widths) < np.std(small_widths), \
        "width estimate should stabilize (lower run-to-run variance) as B grows"


if __name__ == "__main__":
    test_bootstrap_ci_brackets_mean()
    test_bootstrap_ci_deterministic_under_seed()
    test_bootstrap_ci_narrows_as_b_grows()
    print("All metrics checks passed.")

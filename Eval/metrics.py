"""Day-10: pure scoring functions for the generate+score harness. CPU-testable
(no GPU, no model loading needed to run these) and function-level, no classes
-- `evaluate_model.py` is the only caller.

Per family ALWAYS, never pooled: ACI (full notes) and MTS (sections) are
different genres/lengths, so `summarize` is meant to be called once per
family by the caller, not once over a concatenated pool.

Read before implementing:
  - rouge_score package (pure python, no network): https://github.com/google-research/rouge_score
  - bert_score package: https://github.com/Tiiiger/bert_score
  - Bootstrap CI, case resampling: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Case_resampling
  - numpy percentile: https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
  - numpy Generator.integers (resampling with replacement):
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
"""
import sys
from pathlib import Path

import numpy as np

# ponytail: lets `python db_tools/metrics.py` resolve `db_tools.seed` when run
# directly (its own dir, not the repo root, is on sys.path in that case) --
# same shim `tests/*.py` already uses; a no-op when imported normally.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_tools.seed import SEED


def rouge_scores(preds: list[str], refs: list[str]) -> dict[str, list[float]]:
    """Per-example ROUGE-1/2/L F-measure via `rouge_score` (pure python, no
    network/model download). Returns one list per metric, same order/length
    as `preds`/`refs`, so `bootstrap_ci` can resample the per-example scores.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    out = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(preds, refs):
        scored = scorer.score(ref, pred)
        for key in out:
            out[key].append(scored[key].fmeasure)
    return out


def bertscore(preds: list[str], refs: list[str], lang: str = "en") -> list[float]:
    """Per-example BERTScore F1 via `bert_score.score`. Import stays inside
    this function (not module scope) because the first call downloads a
    scoring model over the network -- keeping that off the module's import
    path is what lets `metrics.py` still import on an offline CPU box.
    """
    import bert_score as _bert_score

    _, _, f1 = _bert_score.score(preds, refs, lang=lang, verbose=False)
    return f1.tolist()


def bootstrap_ci(values: list[float], B: int = 2000, alpha: float = 0.05, seed: int = SEED
                  ) -> tuple[float, float, float]:
    """Percentile bootstrap CI over `values` (e.g. one metric's per-example
    scores for one family). Resample `values` WITH replacement `B` times
    (same size as `values` each draw), take each resample's mean, and report
    the `(alpha/2, 1 - alpha/2)` percentiles of that distribution of means.

    The `seed` argument alone must fully determine the resample draws (e.g.
    via a local `np.random.default_rng(seed)`) so two calls with the same
    `seed` reproduce the same CI regardless of global RNG state.

    Returns: `(mean, lo, hi)` -- `mean` is `values`'s plain mean, `lo`/`hi`
    are the bootstrap percentile bounds (`lo <= mean <= hi` in expectation).
    """
    # ─────────────────────────────────────────────────────────────
    # YOUR CODE — body metrics.bootstrap_ci
    # Goal: resample `values` with replacement `B` times, take the mean of
    #       each resample, and return (mean(values), lo_percentile, hi_percentile)
    #       of that distribution of resample means.
    # Why:  n=40 (ACI test) makes a bare "ROUGE +1.3" indistinguishable from
    #       resampling noise -- the CI is what turns a raw metric delta into a
    #       claim you can defend in review.
    # Read: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Case_resampling
    # Done when: tests/test_metrics.py::test_bootstrap_ci passes -- the CI
    #       brackets the mean, is deterministic for a fixed seed, and its
    #       run-to-run variability shrinks as B grows.
    raise NotImplementedError("metrics.bootstrap_ci")
    # ─────────────────────────────────────────────────────────────


def summarize(metric_values: dict[str, list[float]], B: int = 2000, seed: int = SEED
              ) -> dict[str, tuple[float, float, float]]:
    """Run `bootstrap_ci` independently per metric key (e.g. rouge1/rouge2/
    rougeL/bertscore). Call this ONCE PER FAMILY -- pooling ACI and MTS scores
    together would blur a family-specific regression behind the other
    family's numbers. Returns `{metric_name: (mean, lo, hi)}`."""
    return {name: bootstrap_ci(values, B=B, seed=seed) for name, values in metric_values.items()}


if __name__ == "__main__":
    preds = ["the patient reports chest pain", "no medications currently"]
    refs = ["patient reports chest pain today", "no current medications"]
    scores = rouge_scores(preds, refs)
    assert set(scores) == {"rouge1", "rouge2", "rougeL"}
    assert all(len(v) == len(preds) for v in scores.values())
    assert all(0.0 <= s <= 1.0 for v in scores.values() for s in v)
    print("metrics.rouge_scores OK")
    # ponytail: bertscore's self-check is skipped here -- first call downloads
    # a scoring model over the network, which doesn't belong in a CPU smoke
    # test. Exercised for real in evaluate_model.py on the VM.

    mean, lo, hi = bootstrap_ci(scores["rouge1"], B=200, seed=SEED)
    assert lo <= mean <= hi
    print("metrics.bootstrap_ci OK")

    summary = summarize(scores, B=200, seed=SEED)
    assert set(summary) == set(scores)
    print("metrics.summarize OK")

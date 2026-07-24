"""Wiring smoke test for the Eval/ scripts (evaluate_model.py, metrics.py):
confirms they IMPORT on a CPU box and expose the expected public API. Structural
check, NOT a hole-acceptance check -- stays green regardless of whether the
render_prompt/bootstrap_ci holes are filled (test_metrics.py covers bootstrap_ci
behaviour; real generation is VM-only). Run via `pytest` or
`python test_evaluate_model.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # data_hand root

from Eval import evaluate_model, metrics


def test_evaluate_model_api():
    for name in ("render_prompt", "generate_batch", "run_eval", "main"):
        assert callable(getattr(evaluate_model, name)), f"evaluate_model.{name} must exist and be callable"


def test_metrics_api():
    for name in ("rouge_scores", "bertscore", "bootstrap_ci", "summarize"):
        assert callable(getattr(metrics, name)), f"metrics.{name} must exist and be callable"


if __name__ == "__main__":
    test_evaluate_model_api()
    test_metrics_api()
    print("All Eval wiring checks passed.")
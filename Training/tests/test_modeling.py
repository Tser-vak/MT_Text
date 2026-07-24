"""Wiring smoke test for the Training/ scripts (modeling.py, train.py): confirms
they IMPORT on a CPU box and expose the expected public API. Structural check,
NOT a hole-acceptance check -- it stays green whether or not the modeling/train
holes are filled (the real 4-bit load + LoRA behaviour is VM-only). Run via
`pytest` or `python test_modeling.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # data_hand root

from Training import modeling, train


def test_modeling_api():
    assert modeling.MODEL_ID == "google/medgemma-4b-it", "MODEL_ID is the single source of the model id"
    for name in ("load_processor", "load_base_model", "attach_lora", "load_for_inference"):
        assert callable(getattr(modeling, name)), f"modeling.{name} must exist and be callable"


def test_train_api():
    for name in ("build_sft_config", "inspect_loss_mask", "main"):
        assert callable(getattr(train, name)), f"train.{name} must exist and be callable"


if __name__ == "__main__":
    test_modeling_api()
    test_train_api()
    print("All Training wiring checks passed.")
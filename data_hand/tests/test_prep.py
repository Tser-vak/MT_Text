"""Checks for Day-3 Stage-A data prep. Run via `pytest` or `python test_prep.py`.

Runs Data_main.run() if data_hand/processed/*.csv aren't there yet, then
checks the written outputs against the expected shape from the Day-3 spec.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Data_main import OUT_DIR, run
from db_tools.dedup import TextDeduplicator

EXPECTED_ROWS = {
    "aci_train.csv": 67,
    "aci_taskB_test1.csv": 40,
    "aci_taskC_test2.csv": 40,
    "aci_valid.csv": 20,
    "aci_test.csv": 40,
    "mts_train.csv": 1195,  # 1197 - 2 train/test hash collisions dropped (see _dedup_mts)
    "mts_valid.csv": 98,
    "mts_test1.csv": 196,
    "mts_test2.csv": 198,
}
ACI_FILES = ["aci_train.csv", "aci_taskB_test1.csv", "aci_taskC_test2.csv", "aci_valid.csv", "aci_test.csv"]
MTS_FILES = ["mts_train.csv", "mts_valid.csv", "mts_test1.csv", "mts_test2.csv"]


def _ensure_outputs():
    if not all((OUT_DIR / name).exists() for name in EXPECTED_ROWS):
        run()


def test_row_counts():
    _ensure_outputs()
    for name, expected in EXPECTED_ROWS.items():
        df = pd.read_csv(OUT_DIR / name)
        assert len(df) == expected, f"{name}: expected {expected} rows, got {len(df)}"


def test_no_dead_sections_in_mts():
    _ensure_outputs()
    for name in MTS_FILES:
        df = pd.read_csv(OUT_DIR / name)
        headers = df["section_header"].astype(str).str.lower()
        assert not headers.isin({"labs", "other_history"}).any(), f"{name}: dead section survived"


def test_mts_headers_are_full_names():
    _ensure_outputs()
    for name in MTS_FILES:
        df = pd.read_csv(OUT_DIR / name)
        assert df["section_header"].notna().all(), f"{name}: unmapped (NaN) section_header found"

    train = pd.read_csv(OUT_DIR / "mts_train.csv")
    assert (train["section_header"] == "HISTORY of PRESENT ILLNESS").any(), \
        "expected GENHX -> 'HISTORY of PRESENT ILLNESS' mapping to be present in mts_train.csv"
    assert not (train["section_header"] == "GENHX").any(), "bare 'GENHX' header leaked through unmapped"


def test_aci_columns():
    _ensure_outputs()
    for name in ACI_FILES:
        df = pd.read_csv(OUT_DIR / name)
        assert list(df.columns) == ["encounter_id", "dialogue", "note"], \
            f"{name}: expected [encounter_id, dialogue, note], got {list(df.columns)}"


def test_no_train_test_collisions():
    """The 2 known MTS train<->test hash collisions (dialogue+section_text) must
    already be dropped from the test splits -- this is the dedup wiring's whole
    point (see Data_main._dedup_mts). Zero overlap expected."""
    _ensure_outputs()
    dedup = TextDeduplicator()
    cols = ["dialogue", "section_text"]
    train = pd.read_csv(OUT_DIR / "mts_train.csv")
    train_hashes = set(dedup.hash_rows(train, cols))
    for name in ("mts_test1.csv", "mts_test2.csv"):
        test = pd.read_csv(OUT_DIR / name)
        overlap = dedup.hash_rows(test, cols).isin(train_hashes).sum()
        assert overlap == 0, f"{name}: {overlap} MTS train/test hash collisions still present"


if __name__ == "__main__":
    test_row_counts()
    test_no_dead_sections_in_mts()
    test_mts_headers_are_full_names()
    test_aci_columns()
    test_no_train_test_collisions()
    print("All Day-3 Stage-A prep checks passed.")

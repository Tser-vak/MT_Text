"""Acceptance checks for db_tools/dedup.py. hash_rows is exercised directly
(complete, no hole); drop_collisions is a hole and is expected to FAIL with
NotImplementedError until filled -- an untested contamination check is the
worst kind of untested code. Run via `pytest` or `python test_dedup.py`.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_tools.dedup import TextDeduplicator


def test_hash_rows_matches_on_normalized_duplicate_content():
    dedup = TextDeduplicator()
    df = pd.DataFrame({
        "dialogue": ["Doctor: hi\nPatient: hello", "DOCTOR:  HI\nPATIENT:  HELLO"],
        "section_text": ["fine", "fine"],
    })
    hashes = dedup.hash_rows(df, ["dialogue", "section_text"])
    assert hashes.iloc[0] == hashes.iloc[1], "case/whitespace-only differences must hash identically"


def test_drop_collisions_flags_known_duplicate():
    dedup = TextDeduplicator()
    keep_df = pd.DataFrame({
        "dialogue": ["Doctor: hi\nPatient: hello", "Doctor: bye\nPatient: bye"],
        "section_text": ["fine", "ok"],
    })
    drop_df = pd.DataFrame({
        "dialogue": ["Doctor: hi\nPatient: hello", "Doctor: unique\nPatient: unique"],
        "section_text": ["fine", "unique"],
    })
    out = dedup.drop_collisions(keep_df, drop_df, ["dialogue", "section_text"])
    assert len(out) == 1, "the row duplicated in keep_df must be dropped"
    assert out.iloc[0]["section_text"] == "unique", "the non-duplicate row must survive"


if __name__ == "__main__":
    test_hash_rows_matches_on_normalized_duplicate_content()
    test_drop_collisions_flags_known_duplicate()
    print("All dedup checks passed.")
"""Day-2 Q4 scenario-overlap check (one-shot, detection only, writes nothing).

ACI-BENCH is built from templated patient scenarios, so two encounters can
describe the same underlying case with different wording -- overlap the
exact-content hash in dedup.py cannot see. This fingerprints each encounter on
cc + age + gender and reports any fingerprint shared across the pool/frozen-test
boundary.

Result (2026-07-23): 2 pairs share a fingerprint, both false positives --
`follow up|nan|male` (D2N084/D2N189) and `back pain|nan|male` (D2N152/D2N198).
Both collapsed only because a generic two-word `cc` met a missing age; reading
the four dialogues confirmed four different patients. Combined with dedup.py's
0 exact-hash collisions, the ACI test-split repurposing is clean.

A fuzzy rapidfuzz pass was run once at thresholds 90 and 80 and added nothing:
every near-match was a shared chief complaint with a different patient. Removed
rather than kept -- re-add only if the corpus changes.

Reads the raw *_metadata.csv files directly (NOT processed/, which drops metadata).
"""
from pathlib import Path

import pandas as pd

CHALLENGE_DIR = Path(__file__).parent.parent / "DB" / "aci" / "aci-bench-corpus" / "challenge_data"

POOL_FILES = [
    "train_metadata.csv",
    "valid_metadata.csv",
    "clinicalnlp_taskB_test1_metadata.csv",
    "clinicalnlp_taskC_test2_metadata.csv",
]
TEST_FILE = "clef_taskC_test3_metadata.csv"


def fingerprint(row: pd.Series) -> str:
    """One encounter's metadata -> a normalized cc|age|gender key.

    Normalize EACH field before joining: a trailing space inside a field would
    otherwise survive next to the separator (19 ACI rows carry one on `cc`) and
    fingerprint the same scenario two different ways.
    """
    cc_info = str(row.cc).lower().strip()
    patient_info = str(row.patient_age).lower().strip()
    gender_info = str(row.patient_gender).lower().strip()

    return cc_info + "|" + patient_info + "|" + gender_info


def _load_fps(filenames: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(CHALLENGE_DIR / name) for name in filenames]
    df = pd.concat(frames, ignore_index=True)
    df["fp"] = df.apply(fingerprint, axis=1)
    return df[["encounter_id", "fp"]]


def main() -> None:
    pool_fps = _load_fps(POOL_FILES)
    test_fps = _load_fps([TEST_FILE])
    print(f"pool encounters: {len(pool_fps)}  |  frozen test encounters: {len(test_fps)}")

    # ponytail: exact fingerprint match, no fuzzy scoring -- see module docstring.
    shared = pool_fps.merge(test_fps, on="fp", suffixes=("_pool", "_test"))
    print(f"{len(shared)} pair(s) sharing a fingerprint across the split boundary:")
    for row in shared.itertuples():
        print(f"  {row.encounter_id_pool} <-> {row.encounter_id_test}   {row.fp}")


if __name__ == "__main__":
    main()
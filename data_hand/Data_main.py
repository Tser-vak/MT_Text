"""Day-3 Stage-A data prep: raw ACI-bench / MTS-Dialog CSVs -> cleaned per-split CSVs.

Only column selection + MTS section-header expansion + dead-section drop.
No text cleaning, no speaker normalization, no JSONL schema (later steps).

STALE OUTPUT -- processed/*.csv currently holds the OLD two-role speaker
normalization (bare "Doctor"/"Patient", with Guest_family/Guest_clinician/
patient_guest left raw). data_tools.SPEAKER_TAGS now emits four roles with a
colon. To bring the CSVs back in sync, in this order:
  1. Fill the db_tools/dedup.py::drop_collisions hole -- run() calls it via
     _dedup_mts() and raises NotImplementedError until it exists.
  2. Re-run this script. Check the report_unmapped() line reads "all mapped".
  3. Re-run db_tools/measure_lengths.py -- the tag text changed, so token
     counts (and the max_length >= 4608 cap) need re-measuring.
"""
from pathlib import Path

import pandas as pd

from db_tools.data_tools import load_section_map, prep_aci, prep_mts, report_unmapped
from db_tools.dedup import TextDeduplicator

DB_DIR = Path(__file__).parent / "DB"
OUT_DIR = Path(__file__).parent / "processed"
SECTION_MAP_PATH = DB_DIR / "Norm_sec_head.txt"

ACI_JOBS = [
    ("training/aci/train.csv", "aci_train.csv"),
    ("training/aci/clinicalnlp_taskB_test1.csv", "aci_taskB_test1.csv"),
    ("training/aci/clinicalnlp_taskC_test2.csv", "aci_taskC_test2.csv"),
    ("valid/aci/valid.csv", "aci_valid.csv"),
    ("testing/aci/clef_taskC_test3.csv", "aci_test.csv"),
]

MTS_JOBS = [
    ("training/MTS/MTS-Dialog-TrainingSet.csv", "mts_train.csv"),
    ("valid/MTS/MTS-Dialog-ValidationSet.csv", "mts_valid.csv"),
    ("testing/MTS/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv", "mts_test1.csv"),
    ("testing/MTS/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv", "mts_test2.csv"),
]


def run() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    section_map = load_section_map(SECTION_MAP_PATH)

    jobs = [(prep_aci, src, out) for src, out in ACI_JOBS] + \
           [(lambda df: prep_mts(df, section_map), src, out) for src, out in MTS_JOBS]

    for prep, src_rel, out_name in jobs:
        df = pd.read_csv(DB_DIR / src_rel)
        cleaned = prep(df)
        out_path = OUT_DIR / out_name
        # newline="" -- pandas defaults lineterminator to os.linesep when given
        # a path, so on Windows rows end \r\n. Cosmetic (read_csv handles both),
        # but LF keeps the files stable across boxes and diffs.
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            cleaned.to_csv(f, index=False)
        print(f"{src_rel} -> {out_path.name}  ({len(cleaned)} rows)")

    report_unmapped()
    _dedup_mts()


def _dedup_mts() -> None:
    """Post-pass: drop MTS train<->test hash collisions from TEST, never train.

    The 2 known collisions are an eval-integrity problem (secondary MTS
    section eval would otherwise partly score on memorized rows), so only
    mts_test1.csv / mts_test2.csv are rewritten here -- mts_train.csv is
    read-only in this function.
    """
    dedup = TextDeduplicator()
    train = pd.read_csv(OUT_DIR / "mts_train.csv")
    for name in ("mts_test1.csv", "mts_test2.csv"):
        test_path = OUT_DIR / name
        test = pd.read_csv(test_path)
        deduped = dedup.drop_collisions(train, test, ["dialogue", "section_text"])
        with open(test_path, "w", newline="", encoding="utf-8") as f:
            deduped.to_csv(f, index=False)
        print(f"{name}: {len(test)} -> {len(deduped)} rows "
              f"(dropped {len(test) - len(deduped)} train/test collisions)")


if __name__ == "__main__":
    run()
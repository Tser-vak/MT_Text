"""Day-3 Stage-A data prep: raw ACI-bench / MTS-Dialog CSVs -> cleaned per-split CSVs.

Column selection + MTS section-header expansion + dead-section drop + text
cleaning + speaker normalization + train/held-out dedup. No JSONL schema --
prompts are built at train time via prompts.build (amendment 2026-07-20 #3).

processed/*.csv is in sync as of 2026-07-23 (four-role speaker normalization,
report_unmapped() reads "all mapped"). Still pending: re-run
db_tools/measure_lengths.py -- the tag text changed, so token counts (and the
max_length >= 4608 cap) need re-measuring on the GCP VM.
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

# Dedup sides for _dedup_aci: held-out splits are read-only, the train pool shrinks.
ACI_KEEP_FILES = ("aci_test.csv", "aci_valid.csv")
ACI_DROP_FILES = ("aci_train.csv", "aci_taskB_test1.csv", "aci_taskC_test2.csv")

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
    _dedup_aci()


def _dedup_mts() -> None:
    """Post-pass: drop MTS train<->test hash collisions from TRAIN, never test.

    The 2 known collisions are a contamination problem: the fine-tune sees
    MTS fresh, so removing the overlapping rows from TRAIN prevents the model
    ever learning a row it will later be scored on, while keeping the test
    splits byte-identical to the published benchmark. Only mts_train.csv is
    rewritten here -- mts_test1.csv / mts_test2.csv are read-only.
    """
    dedup = TextDeduplicator()
    cols = ["dialogue", "section_text"]
    train_path = OUT_DIR / "mts_train.csv"
    train = pd.read_csv(train_path)
    test = pd.concat(
        [pd.read_csv(OUT_DIR / name) for name in ("mts_test1.csv", "mts_test2.csv")],
        ignore_index=True,
    )
    deduped = dedup.drop_collisions(test, train, cols)
    with open(train_path, "w", newline="", encoding="utf-8") as f:
        deduped.to_csv(f, index=False)
    print(f"mts_train.csv: {len(train)} -> {len(deduped)} rows "
          f"(dropped {len(train) - len(deduped)} train/test collisions)")


def _dedup_aci() -> None:
    """Post-pass: drop ACI held-out<->train-pool hash collisions from the TRAIN pool.

    Same direction as _dedup_mts, for the same reason: aci_test.csv is the frozen
    clef_taskC_test3 benchmark split and stays byte-identical, so reported numbers
    stay comparable. aci_valid.csv joins the keep side because it is the held-out
    full-note validation set used for checkpoint selection (see plan amendment
    2026-07-21 #5) -- a train row duplicated there inflates the selection signal
    just as badly as one duplicated in test.

    keep is concatenated (it only ever becomes a set of hashes); the three train
    files are looped, not concatenated, because each is rewritten in place.
    """
    dedup = TextDeduplicator()
    cols = ["dialogue", "note"]
    keep = pd.concat(
        [pd.read_csv(OUT_DIR / name) for name in ACI_KEEP_FILES],
        ignore_index=True,
    )
    for name in ACI_DROP_FILES:
        path = OUT_DIR / name
        df = pd.read_csv(path)
        deduped = dedup.drop_collisions(keep, df, cols)
        with open(path, "w", newline="", encoding="utf-8") as f:
            deduped.to_csv(f, index=False)
        print(f"{name}: {len(df)} -> {len(deduped)} rows "
              f"(dropped {len(df) - len(deduped)} held-out/train collisions)")


if __name__ == "__main__":
    run()
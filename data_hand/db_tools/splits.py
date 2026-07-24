"""Day-4: processed/*.csv -> the two `Dataset` objects the trainer consumes,
plus the interleave/mixing knob (Q1) that combines them at train time.

Two families, kept as SEPARATE `Dataset` objects (never a pre-shuffled single
list): the 147-row ACI full-note train pool, and the 1195-row MTS section
train set. `prompts.build`/`prompts.build_train` are the only prompt paths --
this module never re-implements prompt assembly, it only wires rows into them
via `.map()`.

Day-6 addition: `load_families`/`make_train_mix` (messages-shaped, via
`prompts.build`) now feed eval-loss/inspection only -- TRL 1.8.0 does NOT mask
the prompt on a `messages` dataset (see prompts.py's module docstring). The
actual TRAIN mix goes through `load_families_train`/`make_train_mix_pc`
instead (prompt/completion-shaped, via `prompts.build_train`), which is what
`completion_only_loss=True` actually masks.

`aci_valid.csv` (20 rows) is held OUT of the train pool -- it's the full-note
validation split (see `EVAL_FILES`/`load_eval`), needed for a legal checkpoint
signal on the target task without touching the frozen test set.
"""
import argparse
import sys

import datasets
import pandas as pd


from db_tools import prompts
from Data_main import OUT_DIR

ACI_POOL_FILES = ["aci_train.csv", "aci_taskB_test1.csv", "aci_taskC_test2.csv"]
MTS_TRAIN_FILE = "mts_train.csv"

# Held-out eval/test splits, keyed by the logical name `load_eval` accepts.
# Each entry is (family, filename-or-list-of-filenames) so `prompts.build*`
# gets the right family per row.
EVAL_FILES = {
    "aci_val": ("ACI", "aci_valid.csv"),
    "mts_val": ("MTS", "mts_valid.csv"),
    "aci_test": ("ACI", "aci_test.csv"),
    "mts_test": ("MTS", ["mts_test1.csv", "mts_test2.csv"]),
}


def load_families() -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load the ACI train-pool (147 rows) and MTS train (1195 rows) CSVs into
    two `Dataset` objects, each with a `messages` column built via
    `prompts.build`. `.map()`'s row is dict-like, so it's wrapped in
    `pd.Series` to satisfy `prompts.build`'s attribute-style row access."""
    aci_paths = [str(OUT_DIR / name) for name in ACI_POOL_FILES]
    full_note_ds = datasets.load_dataset("csv", data_files=aci_paths, split="train")
    section_ds = datasets.load_dataset("csv", data_files=str(OUT_DIR / MTS_TRAIN_FILE), split="train")

    full_note_ds = full_note_ds.map(lambda row: {"messages": prompts.build("ACI", pd.Series(row))})
    section_ds = section_ds.map(lambda row: {"messages": prompts.build("MTS", pd.Series(row))})
    return full_note_ds, section_ds


def load_eval(name: str) -> datasets.Dataset:
    """Load one of the four held-out eval/test splits (aci_val=20, mts_val=98,
    aci_test=40, mts_test=394) into a `Dataset` with `messages` (full
    conversation, for eval loss), `prompt_messages` (user turn only, for
    generation), and `reference` (gold target string) columns. The latter two
    come from `prompts.build_prompt`."""
    if name not in EVAL_FILES:
        raise ValueError(f"unknown eval split {name!r}, expected one of {sorted(EVAL_FILES)}")

    if name == "aci_test":
        print("WARNING: loading FROZEN ACI test split (aci_test.csv) -- Day 10 only", file=sys.stderr)

    family, filenames = EVAL_FILES[name]
    if isinstance(filenames, str):
        filenames = [filenames]
    paths = [str(OUT_DIR / fname) for fname in filenames]
    ds = datasets.load_dataset("csv", data_files=paths, split="train")

    def _add_columns(row):
        series = pd.Series(row)
        prompt_messages, reference = prompts.build_prompt(family, series)
        return {
            "messages": prompts.build(family, series),
            "prompt_messages": prompt_messages,
            "reference": reference,
        }

    return ds.map(_add_columns)


def make_train_mix( p :float, seed_num:int ) -> datasets.Dataset:
    """The Q1 mixing decision: one stream the trainer draws batches from, at
    ACI exposure `p`, instead of concatenating + shuffling (which would freeze
    ACI at its natural ~11% and let the ~8x row imbalance dominate).

    `p` binds to ACI because ACI is FIRST in the list below. Chosen start:
    p=0.4 -> ~1990 rows, ACI replayed ~5.4x (see `repeat_report`).

    Not shuffled first: HF `Trainer` wraps a map-style `Dataset` in a
    `RandomSampler`, so this order never reaches the model. Revisit only if
    this ever becomes a streaming `IterableDataset`.
    """
    #load the training data
    aci_dt , mts_dt =load_families()

    #Data shuffling
    dataset = datasets.interleave_datasets([aci_dt, mts_dt],probabilities=[p, 1-p],seed=seed_num,stopping_strategy="all_exhausted")

    return dataset

def repeat_report(p: float) -> None:
    """Print how many times each family is replayed at draw probability `p`,
    WITHOUT materializing the mix -- this is how you choose `p` before paying
    to build it (day-by-day plan Q1 / gap #7: "epoch" is meaningless under
    `stopping_strategy="all_exhausted"`, so the step budget comes from here).

    Expectation only. The materialized mix lands within ~10% of these numbers
    (sampling noise plus the partial cycle at the stopping boundary)."""
    n_aci, n_mts = (len(d) for d in load_families())
    total = max(n_aci / p, n_mts / (1 - p))
    print(f"p={p:<5} draws={total:7.0f}  "
          f"ACI x{total * p / n_aci:5.2f} ({n_aci} rows) |Aci data occurrence:{total * p :5.2f}| "
          f"MTS x{total * (1 - p) / n_mts:5.2f} ({n_mts} rows)")


def load_families_train() -> tuple[datasets.Dataset, datasets.Dataset]:
    """Prompt/completion-shaped counterpart to `load_families`: the same two
    pools (ACI train-pool, MTS train), but each row carries `prompt`/
    `completion` columns via `prompts.build_train` instead of a `messages`
    column -- this is the shape TRL 1.8.0 needs to mask the prompt (see
    prompts.py's module docstring). Family is resolved HERE, per source CSV,
    before any interleaving -- `make_train_mix_pc` must never try to recover
    family after the streams are interleaved."""
    aci_paths = [str(OUT_DIR / name) for name in ACI_POOL_FILES]
    full_note_ds = datasets.load_dataset("csv", data_files=aci_paths, split="train")
    section_ds = datasets.load_dataset("csv", data_files=str(OUT_DIR / MTS_TRAIN_FILE), split="train")

    full_note_ds = full_note_ds.map(lambda row: prompts.build_train("ACI", pd.Series(row)))
    section_ds = section_ds.map(lambda row: prompts.build_train("MTS", pd.Series(row)))
    return full_note_ds, section_ds


def make_train_mix_pc(p: float, seed_num: int) -> datasets.Dataset:
    """Prompt/completion counterpart to `make_train_mix` -- same interleave
    knob and rationale (see that docstring for the p/seed discussion), built
    from `load_families_train` so the trainer actually gets a maskable
    dataset. THIS is the function `train.py` feeds `SFTTrainer`."""
    aci_dt, mts_dt = load_families_train()
    dataset = datasets.interleave_datasets([aci_dt, mts_dt], probabilities=[p, 1 - p],
                                            seed=seed_num, stopping_strategy="all_exhausted")
    return dataset


def load_eval_loss(name: str) -> datasets.Dataset:
    """Prompt/completion-shaped eval-loss split for `SFTTrainer`'s
    `eval_dataset` dict (only `aci_val`/`mts_val` are meaningful here -- the
    test splits are for Day-10 generation scoring via `load_eval`, not eval
    loss). Reuses `EVAL_FILES` + `prompts.build_train` the same way `load_eval`
    reuses `prompts.build_prompt`."""
    if name not in EVAL_FILES:
        raise ValueError(f"unknown eval split {name!r}, expected one of {sorted(EVAL_FILES)}")

    family, filenames = EVAL_FILES[name]
    if isinstance(filenames, str):
        filenames = [filenames]
    paths = [str(OUT_DIR / fname) for fname in filenames]
    ds = datasets.load_dataset("csv", data_files=paths, split="train")

    return ds.map(lambda row: prompts.build_train(family, pd.Series(row)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    for p in (0.3, 0.4, 0.5,0.65, 0.75):
        repeat_report(p)
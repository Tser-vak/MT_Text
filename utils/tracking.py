"""W&B run tracking for the EVAL stage only. `evaluate_model.py` computes
ROUGE/BERTScore/CI OUTSIDE any Trainer, so nothing auto-logs those -- this
module is the one place an eval run is started and its per-family metrics are
logged from.

TRAINING DOES NOT USE THIS MODULE. `SFTTrainer` logs train loss, BOTH family
eval losses, LR, and GPU memory to W&B by itself once `report_to="wandb"` is
set in the SFTConfig (H6, `build_sft_config`) -- reproducing that here would
just duplicate the Trainer, so `train.py` deliberately never imports this.

CRITICAL: no `import wandb` at module scope -- keep it inside the functions so
this module imports on any box (CPU smoke test) and honours a disabled/offline
run (`WANDB_MODE=offline` or `WANDB_DISABLED=true`). Same deferred-import
pattern as modeling.py's bitsandbytes and metrics.py's bert_score.

Read before implementing:
  - wandb.init (project / name / config / mode): https://docs.wandb.ai/ref/python/init
  - wandb.log (custom scalar metrics): https://docs.wandb.ai/ref/python/log
  - HF Trainer W&B integration (what you must NOT re-log):
    https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.integrations.WandbCallback
"""

import random
import sys
from pathlib import Path

try:
    from db_tools.seed import SEED
except ImportError:
    # ponytail: standalone `python utils/tracking.py` doesn't get evaluate_model.py's
    # sys.path bootstrap -- mirror it here so both entry points work.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_hand"))
    from db_tools.seed import SEED

WANDB_PROJECT = "medgemma-clinical-notes"


def start_run(stage: str, config: dict, run_name: str | None = None):
    """Start and return the W&B run for this eval invocation.

    `stage`     -- "eval" | "baseline" (drives the run name / group).
    `config`    -- dict of the eval's knobs to record (e.g. adapter path,
                   eval split, few-shot k).
    `run_name`  -- optional explicit name; when None, derive a stable one.

    Returns the `wandb.Run`, so callers can guard `if run is not None:` before
    logging. No explicit off-switch: `WANDB_MODE=offline` / `WANDB_MODE=disabled`
    already make `wandb.init` return a (no-op) run, and a second mechanism would
    just be a second thing to check when logging goes quiet.
    """
    import wandb

    if run_name is None:
        # a derived name beats "misty-sunset-42" -- these runs are only useful
        # side by side, so the knobs that differ have to be IN the name.
        parts = [stage, str(config.get("eval", "?"))]
        if config.get("fewshot"):
            parts.append(f"k{config['fewshot']}")
        if config.get("adapter"):
            parts.append(Path(config["adapter"]).name)
        run_name = "-".join(parts)

    return wandb.init(project=WANDB_PROJECT, name=run_name, config=config, group=stage)


def log_eval_summary(summary: dict, family: str, step: int | None = None) -> None:
    """Log one family's eval metrics to the active W&B run.

    `summary` -- exactly what `metrics.summarize` returns:
                 `{metric_name: (mean, lo, hi)}`.
    `family`  -- "ACI" or "MTS" (from `splits.EVAL_FILES[name][0]`).
    """
    import wandb
    if wandb.run is None:
        return

    # keys namespaced by family -- that prefix is what makes W&B render ACI and
    # MTS as two separate panel groups instead of one pooled (== MTS) number.
    flat = {}
    for name, (mean, lo, hi) in summary.items():
        flat[f"{family}/{name}"] = mean
        flat[f"{family}/{name}_lo"] = lo
        flat[f"{family}/{name}_hi"] = hi

    wandb.log(flat, step=step)  # step=None -> wandb auto-increments


def log_samples(prompts: list[str], preds: list[str], refs: list[str],
                 family: str, n: int = 10) -> None:
    """Log a qualitative sample table to the active W&B run.

    `prompts`, `preds`, `refs` -- three parallel lists (same length, same
    order as `run_eval`'s `preds`/`refs`).
    `family` -- "ACI" | "MTS" (from `splits.EVAL_FILES[name][0]`).
    `n`      -- number of rows to sample into the table.
    """
    import wandb
    if wandb.run is None:
        return

    # SEED, not a fresh RNG: baseline and fine-tuned runs MUST draw the same
    # row indices, or the two tables show different patients and can't be read
    # side by side.
    idxs = random.Random(SEED).sample(range(len(preds)), min(n, len(preds)))

    table = wandb.Table(columns=["dialogue", "reference", "prediction"])
    for i in idxs:
        # ponytail: flat 2000-char truncation on the dialogue only -- W&B cells
        # get unreadable past that. Widen if a real note gets clipped.
        table.add_data(prompts[i][:2000], refs[i], preds[i])

    wandb.log({f"{family}/samples": table})


def finish() -> None:
    """Close the active run, if any. Plumbing (not a hole) -- call after the
    last `run_eval` returns."""
    import wandb
    if wandb.run is not None:
        wandb.run.finish()


if __name__ == "__main__":
    # ponytail: no real wandb.init here -- that needs a W&B login + network,
    # neither of which the CPU smoke test should require. Just check wiring.
    assert WANDB_PROJECT
    for fn in (start_run, log_eval_summary, log_samples, finish):
        assert callable(fn)

    # the one real check: with no active run, the loggers must no-op rather
    # than raise -- that guard is what lets evaluate_model.py run on a box with
    # no W&B login. Skipped entirely if wandb isn't installed locally.
    try:
        import wandb
    except ImportError:
        print("wandb not installed locally -- no-op guard check skipped")
    else:
        assert wandb.run is None, "expected no active run in the smoke test"
        log_eval_summary({"rougeL": (0.4, 0.3, 0.5)}, "ACI")
        log_samples(["dialogue"], ["pred"], ["ref"], "ACI")
        finish()
    print("tracking.py wiring OK (all bodies filled -- real check is on the VM "
          "with `wandb login` done)")
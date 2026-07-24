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

WANDB_PROJECT = "medgemma-clinical-notes"


def start_run(stage: str, config: dict, run_name: str | None = None):
    """Start and return the W&B run for this eval invocation.

    `stage`     -- "eval" | "baseline" (drives the run name / group).
    `config`    -- dict of the eval's knobs to record (e.g. adapter path,
                   eval split, few-shot k).
    `run_name`  -- optional explicit name; when None, derive a stable one.

    Returns the `wandb.Run`, or None when tracking is disabled, so callers can
    guard `if run is not None:` before logging.
    """
    # ─────────────────────────────────────────────────────────────
    # YOUR CODE — body tracking.start_run
    # Goal: `import wandb` (inside the fn), call `wandb.init(project=
    #       WANDB_PROJECT, name=<run_name or a name you derive from stage+config
    #       so runs are legible, e.g. f"{stage}-{config['eval']}">,
    #       config=config, group=stage)` and return the run. Respect a disabled
    #       run: if the user set WANDB_MODE=offline / WANDB_DISABLED, wandb.init
    #       still returns a (no-op) run, so returning it as-is is fine -- but
    #       decide whether you also want an explicit off-switch (e.g. a
    #       `config`/env flag) that returns None instead.
    # Why:  you run this script several times -- baseline vs fine-tuned, per
    #       eval split, per selected checkpoint; if every run is auto-named
    #       "misty-sunset-42" you can't tell them apart. A name derived from the
    #       knobs is what makes the comparison table readable.
    # Read: https://docs.wandb.ai/ref/python/init
    # Done when (VM): `wandb.init` succeeds and the run shows your config dict
    #       and a legible name in the W&B UI; a baseline run and a fine-tuned
    #       run are visibly distinct and overlay-comparable.
    raise NotImplementedError("tracking.start_run")
    # ─────────────────────────────────────────────────────────────


def log_eval_summary(summary: dict, family: str, step: int | None = None) -> None:
    """Log one family's eval metrics to the active W&B run.

    `summary` -- exactly what `metrics.summarize` returns:
                 `{metric_name: (mean, lo, hi)}`.
    `family`  -- "ACI" or "MTS" (from `splits.EVAL_FILES[name][0]`).
    """
    # ─────────────────────────────────────────────────────────────
    # YOUR CODE — body tracking.log_eval_summary
    # Goal: `import wandb`, flatten `summary` into a flat metric dict keyed
    #       BY FAMILY (e.g. {"ACI/rougeL": mean, "ACI/rougeL_lo": lo,
    #       "ACI/rougeL_hi": hi, ...}) and `wandb.log(...)` it. No-op safely if
    #       there is no active run (`wandb.run is None`).
    # Why:  metrics are reported PER FAMILY, never pooled (Std/PROGRESS.md
    #       Day-10 #4) -- at ~8:1 MTS:ACI a pooled number is an MTS number in
    #       disguise. Namespacing the keys by family is what keeps the two
    #       curves separable in the W&B UI, the same discipline as the two
    #       separate training eval-losses.
    # Read: https://docs.wandb.ai/ref/python/log
    # Done when (VM): both "ACI/*" and "MTS/*" metric groups appear as distinct
    #       series in the run, and a baseline run vs a fine-tuned run are
    #       directly overlay-comparable per family.
    raise NotImplementedError("tracking.log_eval_summary")
    # ─────────────────────────────────────────────────────────────


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
    for fn in (start_run, log_eval_summary, finish):
        assert callable(fn)
    print("tracking.py wiring OK (start_run/log_eval_summary are holes -- fill + "
          "run on the VM with `wandb login` done)")
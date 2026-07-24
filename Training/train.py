"""Day 7-8: QLoRA SFT training entrypoint. Loads MedGemma 4-bit through
`db_tools/modeling.py`, attaches LoRA, builds the prompt/completion train mix
+ dict eval loss through `db_tools/splits.py`, and drives `SFTTrainer`.

`--overfit` is the Day-7 sanity gate (do not skip, per Std/PROGRESS.md): train
on a handful of rows only, no eval, to confirm the trainer/masking wiring can
actually drive loss to ~0 before spending a real run's compute on Day 8.

CRITICAL: no `bitsandbytes` import at module scope -- the actual 4-bit load
happens inside `modeling.load_base_model`, called from `main()`, so this
module still IMPORTS on a CPU box with no CUDA/bitsandbytes installed.

Read before implementing:
  - TRL SFTConfig: https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTConfig
  - TRL SFT dataset formats (prompt-completion): https://huggingface.co/docs/trl/main/en/dataset_formats
  - HF TrainingArguments: https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
  - completion_only_loss / DataCollatorForLanguageModeling labels: sft_trainer.py:1558-1568 (installed trl==1.8.0)
"""
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from peft import prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from Training import modeling
from db_tools import splits
from db_tools.seed import SEED, seed_everything

#load the tokens for HF and Wandb
load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent   # repo root: Eval/, Training/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data_hand"))     # db_tools/ lives here

def inspect_loss_mask(*args, **kwargs):
    """
    YOUR CODE — full train.inspect_loss_mask
    Goal: design a utility that pulls ONE collated training batch (e.g. from
          a trainer's dataloader), decodes the tokens where `labels != -100`
          separately from the tokens where `labels == -100`, and asserts only
          the completion (assistant turn) is unmasked while the prompt is
          masked. You choose the signature and what it needs access to
          (trainer, dataloader, tokenizer, ...).
    Why:  THIS is the check that proves the prompt/completion fix
          (`prompts.build_train` + `completion_only_loss=True`) actually
          works. If it fails, the model is training on its own input and
          every downstream loss/metric number is uninterpretable
          (Std/PROGRESS.md Day-7 #3).
    Read: https://huggingface.co/docs/trl/main/en/dataset_formats#which-formats-are-relevant-for-me
          and sft_trainer.py:1558-1568 (installed trl==1.8.0)
    Done when (VM): decoding shows the dialogue/instruction masked and the
          note/section unmasked; switching the trainer back onto a
          `messages`-shaped dataset makes this assertion FAIL.
    """
    raise NotImplementedError("train.inspect_loss_mask")


def build_sft_config(*args, **kwargs):
    """
    YOUR CODE — full train.build_sft_config
    Goal: assemble the full `SFTConfig` for the run, using the VERIFIED
          param names (not the deprecated ones): `eval_strategy="steps"`,
          `train_sampling_strategy="group_by_length"`, `warmup_steps=N`,
          `max_length` set >= 4608, `completion_only_loss=True`,
          `metric_for_best_model="eval_aci_val_loss"` (a SINGLE eval-loss
          key -- the two family losses can't be blended into one), plus
          `output_dir`/batch/accum/optimizer/LR/logging fields. Also compute
          `max_steps` from `splits.repeat_report(p)`'s draw count divided by
          the effective batch size (`per_device_train_batch_size *
          gradient_accumulation_steps`), times however many passes over the
          mixed stream you want -- "epoch" is meaningless under
          `stopping_strategy="all_exhausted"` (Std/PROGRESS.md Day-8 #2). You
          choose this function's signature (what args/CLI knobs it needs).
    Why:  LoRA needs a much higher LR than full fine-tuning (full FT ~1e-5,
          LoRA ~1e-4-2e-4 territory -- go read why); the deprecated param
          names (`evaluation_strategy`, `group_by_length=True`,
          `warmup_ratio`) either raise or silently no-op on trl==1.8.0; the
          default `max_length=1024` silently truncates ACI notes (measured
          max ~4551 tokens).
    Read: https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTConfig
          https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    Done when (VM): a short run logs `eval_aci_val_loss` and `eval_mts_val_loss`
          SEPARATELY, saves the best checkpoint by `metric_for_best_model`,
          and no max-length truncation warning appears.
    """
    raise NotImplementedError("train.build_sft_config")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.4, help="ACI draw probability for the train mix (see splits.py)")
    parser.add_argument("--output-dir", default="checkpoints/medgemma-qlora")
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--passes", type=float, default=3.0,
                         help="how many times to replay the full mixed stream (feeds max_steps arithmetic)")
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--optim", default="paged_adamw_8bit",
                         help="paged_adamw_8bit on the VM (needs bnb+CUDA); adamw_torch for a CPU wiring smoke test")
    parser.add_argument("--overfit", action="store_true",
                         help="Day-7 sanity check: train on a handful of rows only, eval disabled")
    parser.add_argument("--inspect-mask-only", action="store_true",
                         help="run the Day-7 loss-mask inspector on one batch, then exit -- no training")
    args = parser.parse_args()

    seed_everything()
    splits.repeat_report(args.p)

    # ── W&B tracking ────────────────────────────────────────────────────────
    # Training logs to W&B through SFTTrainer directly — set report_to="wandb"
    # (+ a sweep-legible run_name=...) in build_sft_config (H6). SFTTrainer then
    # streams train loss, both eval losses, LR, and GPU memory on its own.
    # utils/tracking.py is EVAL-ONLY and is deliberately NOT imported here.
    # ────────────────────────────────────────────────────────────────────────

    processor = modeling.load_processor()
    base_model = modeling.load_base_model(quantize=True)
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()
    model = modeling.attach_lora(base_model)

    train_ds = splits.make_train_mix_pc(args.p, SEED)
    eval_ds = {"aci_val": splits.load_eval_loss("aci_val"), "mts_val": splits.load_eval_loss("mts_val")}

    if args.overfit:
        n = min(8, len(train_ds))
        train_ds = train_ds.select(range(n))
        eval_ds = None
        print(f"--overfit: training on {n} rows only, eval disabled")

    sft_config = build_sft_config(args)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
    )

    if args.inspect_mask_only:
        inspect_loss_mask(trainer)
        return

    inspect_loss_mask(trainer)  # Day-7 gate -- must pass before a real run
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

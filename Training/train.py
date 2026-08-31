"""Day 7-8: QLoRA SFT training entrypoint. Loads MedGemma 4-bit through
`Training/modeling.py`, attaches LoRA, builds the prompt/completion train mix
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
import os
import sys
from pathlib import Path

# Path setup MUST come before the Training/db_tools imports below -- otherwise
# `python Training/train.py` puts only Training/ on sys.path and both fail.
_ROOT = Path(__file__).resolve().parent.parent   # repo root: Eval/, Training/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data_hand"))     # db_tools/ lives here

from dotenv import load_dotenv
from peft import prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from Training import modeling
from db_tools import splits
from db_tools.prompts import INSTRUCTION
from db_tools.seed import SEED, seed_everything
from utils.tracking import WANDB_PROJECT  # the CONSTANT only -- see below

#load the tokens for HF and Wandb
load_dotenv()

# HF's WandbCallback reads the project from the ENVIRONMENT and defaults to
# "huggingface" (integration_utils.py: os.getenv("WANDB_PROJECT", "huggingface")),
# so without this the training runs land in a different W&B project than the eval
# runs and the two can never be compared. setdefault, so an explicit env var wins.
# This imports tracking's CONSTANT, not its logging -- training still logs only
# through SFTTrainer.
os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)


def inspect_loss_mask(trainer: SFTTrainer, n_rows: int = 2) -> None:
    #Print and assert what the model is actually trained on, for `n_rows`
    #rows of ONE training batch. Raises AssertionError if the prompt leaked
    #into the loss. Returns nothing -- this is a gate, not a data source.

    #WHY THIS EXISTS: `"label"s == -100` means "this token contributes nothing to
    #the loss". We want the dialogue+instruction masked and only the note /
    #section unmasked. If that is wrong, the model is trained to reproduce its
    #own input, and every loss and metric downstream is uninterpretable -- but
    #NOTHING CRASHES. That is why this is a hand-written check and not a unit
    #test you can skip.

    #Step 1:Create the batchs for the Training Like the gpu reasds it (get the batch and the tokenizer )

    batch = next(iter(trainer.get_train_dataloader()))

    tok = trainer.processing_class.tokenizer

    #Step.2 : split one row into trained vs. read

    for i in range (min(n_rows,len(batch["input_ids"]))):

        ids = batch["input_ids"][i]
        keep = batch["labels"][i] != -100
        trained = tok.decode(ids[keep], skip_special_tokens=False)
        read = tok.decode(ids[~keep], skip_special_tokens=False)

        #Step.3 : print BEFORE asserting -- on the VM the traceback alone
        #won't tell you WHAT leaked.
        print(f"[row {i}] kept {int(keep.sum())}/{len(keep)} tokens")
        print(f"[TRAINED] {trained[:300]!r}")
        print(f"[READ]    {read[:300]!r}")

        #Step.4 : the three gates
        assert keep.any(), f"row {i}: nothing is trained on -- loss will be NaN/zero"
        assert (~keep).any(), f"row {i}: NOTHING is masked -- labels == input_ids, training on the dialogue"

        # (c) the instruction preamble must be on the READ side only. MTS is a
        # format string, so match on the literal prefix before "{section}" --
        # the rendered text carries the real header, never the placeholder.
        for family, template in INSTRUCTION.items():
            needle = template.split("{")[0]
            if needle in read:                      # this row's family
                assert needle not in trained, (
                    f"row {i}: {family} instruction leaked into the loss -- {trained[:200]!r}"
                )


def build_sft_config(args, n_train_rows: int) -> SFTConfig:
    #Build and return the `SFTConfig` for this run.

    # create the batch of steps count so it can say how many steps for 3 epoch or the number you want
    effective_batch = args.per_device_batch * args.grad_accum
    steps_per_pass = n_train_rows // effective_batch
    max_step = int(steps_per_pass * args.passes)

    #Step 2 SFTconfig
    # --overfit: main() sets eval_ds=None, so eval/save/best-model must all be off.
    # grad_accum FORCED to 1 here (not args.grad_accum): the gate asks "can loss
    # reach ~0 on 8 rows", and memorization comes from the NUMBER of optimizer
    # steps, not from batch size. At grad_accum=8 those 100 steps would consume
    # 1600 samples (200 replays of the 8 rows) to take the same 100 updates --
    # ~8x the forward/backward passes for no extra learning.
    if args.overfit:
        return SFTConfig( output_dir = args.output_dir ,max_steps = 100, per_device_train_batch_size= args.per_device_batch,
        gradient_accumulation_steps = 1, per_device_eval_batch_size= 1, max_length= 4608 ,
        completion_only_loss=True,learning_rate= 2e-4,lr_scheduler_type= "cosine",
        warmup_steps= 0, optim= args.optim, bf16= True, eval_strategy= "no",
        save_strategy= "no", load_best_model_at_end=False,
        logging_steps=5, report_to= "none", seed= SEED)

    return SFTConfig( output_dir = args.output_dir ,max_steps = max_step, per_device_train_batch_size= args.per_device_batch,
    gradient_accumulation_steps = args.grad_accum, per_device_eval_batch_size= 1, max_length= 4608 ,
    completion_only_loss=True,learning_rate= 2e-4,lr_scheduler_type= "cosine",
    warmup_steps= int(0.03 * max_step), optim= args.optim, bf16= True, eval_strategy= "steps", eval_steps= args.eval_steps,
    save_strategy= "steps", save_steps= args.eval_steps, save_total_limit= 2, load_best_model_at_end=True,
    # Selection on ACI, deliberately -- not an oversight of the ~8:1 imbalance.
    # ACI (full notes) is the TARGET task and what aci_test reports on; MTS is
    # AUXILIARY (extra volume + clinical vocabulary), and you never select a
    # checkpoint on an auxiliary objective. Not the mean of the two: they are
    # different genres on different loss scales, and averaging hides the case
    # that matters -- MTS still falling while ACI has begun to overfit.
    # The 20-row split is less noisy than the row count suggests: eval loss is
    # averaged per TOKEN, and 20 full notes (~599 tok each) outweigh the 98
    # short MTS sections. Confirm with measure_lengths.py on the VM.
    # MTS is not ignored -- its eval loss is logged every eval_steps. If it is
    # climbing at the selected step, that means `--p` is over-replaying ACI;
    # fix the mixing knob, not this metric.
    metric_for_best_model= "eval_aci_val_loss" ,logging_steps=5, report_to= "wandb",
    run_name=f"medgemma-qlora-p{args.p}-lr2e-4-r16", seed= SEED)



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
    # prepare_model_for_kbit_training already turns gradient checkpointing on --
    # calling gradient_checkpointing_enable() again after it re-enables it with
    # the default use_reentrant=True, which breaks grad flow through the LoRA
    # adapters on some peft/transformers combos. One call, non-reentrant.
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = modeling.attach_lora(base_model)

    train_ds = splits.make_train_mix_pc(args.p, SEED)
    eval_ds = {"aci_val": splits.load_eval_loss("aci_val"), "mts_val": splits.load_eval_loss("mts_val")}

    if args.overfit:
        n = min(8, len(train_ds))
        train_ds = train_ds.select(range(n))
        eval_ds = None
        print(f"--overfit: training on {n} rows only, eval disabled")

    sft_config = build_sft_config(args, len(train_ds))
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

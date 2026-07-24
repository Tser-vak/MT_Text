"""Day 5 + Day 10: ONE generate+score harness for both the zero-shot/few-shot
baseline and the fine-tuned (LoRA-adapter) model -- same code path end to end,
so a baseline-vs-fine-tuned comparison only ever differs by the weights, never
by the harness. `--adapter` absent -> baseline (optionally `--fewshot k`);
`--adapter <path>` -> fine-tuned (no few-shot -- the tuned model was trained
single-turn, see `Std/PROGRESS.md` Day-5 #3).

Consumes `splits.load_eval(name)` unchanged (`prompt_messages` + `reference`
columns); scores with `db_tools/metrics.py`, reported per family, never
pooled (Day-10 #4).

Read before implementing:
  - HF apply_chat_template / chat templating: https://huggingface.co/docs/transformers/main/en/chat_templating
  - MedGemma model card (prompt format notes): https://huggingface.co/google/medgemma-4b-it
"""
import argparse
import random
import sys

import pandas as pd
import torch

from db_tools import metrics, modeling, splits
from db_tools.decoding import GENERATION
from db_tools.seed import SEED, seed_everything

BATCH_SIZE = 4  # ponytail: fixed batch, bucket/bump further only if OOM or throughput demands it


def render_prompt(*args, **kwargs):
    """
    YOUR CODE — full evaluate_model.render_prompt
    Goal: design the function that turns ONE `prompt_messages` row (plus,
          for the baseline, an optional list of few-shot exemplar `messages`
          conversations) into a single model-ready STRING via
          `processor.apply_chat_template(..., add_generation_prompt=True,
          tokenize=False)`. You choose the signature and how exemplars fold
          into the final rendered conversation.
    Why:  `apply_chat_template` + `add_generation_prompt=True` is the exact
          single-turn discipline MedGemma is sensitive to (Std/PROGRESS.md
          Day-5 #3, #5) -- a train/inference prompt-format drift here is the
          single most common silent quality killer in a project like this.
    Read: https://huggingface.co/docs/transformers/main/en/chat_templating
    Done when: eyeball 3 rendered strings -- no assistant turn present, the
          generation prompt IS present at the end; the baseline then runs
          end-to-end on the 40-row `aci_test` split without error.
    """
    raise NotImplementedError("evaluate_model.render_prompt")


def _sample_fewshot(family: str, k: int, seed: int = SEED) -> list[list[dict]]:
    """Pick `k` example `messages` conversations (user+assistant) from that
    family's TRAIN pool -- never eval/test -- to use as few-shot exemplars.
    Non-hole plumbing: `render_prompt` (the hole above) decides how these
    actually get folded into the final rendered prompt."""
    aci_ds, mts_ds = splits.load_families()
    pool = aci_ds if family == "ACI" else mts_ds
    idxs = random.Random(seed).sample(range(len(pool)), k)
    return [pool[i]["messages"] for i in idxs]


def generate_batch(model, processor, rows: list[dict], fewshot_examples: list | None = None) -> list[str]:
    """Render each row (via `render_prompt`), left-pad the batch, run ONE
    greedy `model.generate` call using the frozen `decoding.GENERATION`
    kwargs, and decode only the newly generated tokens."""
    texts = [render_prompt(row, processor, fewshot_examples=fewshot_examples) for row in rows]
    inputs = processor.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    inputs = inputs.to(model.device)

    eos_ids = {processor.tokenizer.eos_token_id}
    end_of_turn_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id is not None:
        eos_ids.add(end_of_turn_id)

    with torch.inference_mode():
        out = model.generate(
            **inputs, **GENERATION,
            eos_token_id=list(eos_ids),
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
        )

    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def run_eval(model, processor, eval_name: str, fewshot_examples: list | None = None,
             batch_size: int = BATCH_SIZE, out_csv: str | None = None) -> dict:
    """Load `splits.load_eval(eval_name)`, batch it (bucketed by rendered
    prompt length so padding waste stays low), generate, score with ROUGE +
    BERTScore, and bootstrap a CI per metric. Prints and returns the summary.
    """
    ds = splits.load_eval(eval_name)

    order = sorted(range(len(ds)), key=lambda i: len(ds[i]["prompt_messages"][0]["content"]))
    preds: list[str | None] = [None] * len(ds)
    refs = [ds[i]["reference"] for i in range(len(ds))]

    for start in range(0, len(order), batch_size):
        idx_batch = order[start:start + batch_size]
        rows = [ds[i] for i in idx_batch]
        outputs = generate_batch(model, processor, rows, fewshot_examples=fewshot_examples)
        for i, text in zip(idx_batch, outputs):
            preds[i] = text

    scores = metrics.rouge_scores(preds, refs)
    scores["bertscore"] = metrics.bertscore(preds, refs)
    summary = metrics.summarize(scores)

    print(f"\n=== {eval_name} (n={len(ds)}) ===")
    for name, (mean, lo, hi) in summary.items():
        print(f"  {name:<10} {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

    if out_csv:
        pd.DataFrame({"prediction": preds, "reference": refs}).to_csv(out_csv, index=False)
        print(f"per-example predictions written to {out_csv}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=None,
                         help="path to a trained LoRA adapter; omit for the zero-shot/few-shot baseline")
    parser.add_argument("--fewshot", type=int, default=0,
                         help="number of few-shot exemplars to prepend (baseline only, incompatible with --adapter)")
    parser.add_argument("--eval", choices=sorted(splits.EVAL_FILES), default="aci_val",
                         help="which held-out split to score (aci_test is FROZEN -- Day 10 only)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--out-csv", default=None, help="optional path to dump per-example predictions/scores")
    args = parser.parse_args()

    if args.fewshot and args.adapter:
        sys.exit("--fewshot is a baseline-only knob; drop --adapter or drop --fewshot")

    seed_everything()
    processor, model = modeling.load_for_inference(adapter_path=args.adapter)

    fewshot_examples = None
    if args.fewshot:
        family = splits.EVAL_FILES[args.eval][0]
        fewshot_examples = _sample_fewshot(family, args.fewshot)

    run_eval(model, processor, args.eval, fewshot_examples=fewshot_examples,
              batch_size=args.batch_size, out_csv=args.out_csv)


if __name__ == "__main__":
    main()

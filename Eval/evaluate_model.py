"""Day 5 + Day 10: ONE generate+score harness for both the zero-shot/few-shot
baseline and the fine-tuned (LoRA-adapter) model -- same code path end to end,
so a baseline-vs-fine-tuned comparison only ever differs by the weights, never
by the harness. `--adapter` absent -> baseline (optionally `--fewshot k`);
`--adapter <path>` -> fine-tuned (no few-shot -- the tuned model was trained
single-turn, see `Std/PROGRESS.md` Day-5 #3).

Consumes `splits.load_eval(name)` unchanged (`prompt_messages` + `reference`
columns); scores with `Eval/metrics.py`, reported per family, never
pooled (Day-10 #4).

Read before implementing:
  - HF apply_chat_template / chat templating: https://huggingface.co/docs/transformers/main/en/chat_templating
  - MedGemma model card (prompt format notes): https://huggingface.co/google/medgemma-4b-it
"""
import argparse
import itertools
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # repo root: Eval/, Training/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data_hand"))     # db_tools/ lives here

import pandas as pd
import torch
from dotenv import load_dotenv

from Eval import metrics
from Training import modeling
from db_tools import splits
from db_tools.decoding import GENERATION
from db_tools.seed import SEED, seed_everything
from utils import tracking

# same as train.py: MedGemma is a GATED repo, so the HF token has to be in the
# environment before the first from_pretrained -- and W&B needs its own key.
load_dotenv()

BATCH_SIZE = 4  # ponytail: fixed batch, bucket/bump further only if OOM or throughput demands it


def render_prompt( row: dict , processor : object, fewshot_examples: list[list[dict]] | None = None ) -> str:

    #1. flatten fewshot_examples (list of 2-msg conversations) into a flat message list — empty when there are none

    flat_messages = list(itertools.chain.from_iterable(fewshot_examples)) if fewshot_examples else []

    #2. Append the row's user turn LAST. row["prompt_messages"] is a 1-element LIST, so concatenate it.
    # don't nest it inside the conversation.

    conversation = flat_messages + row["prompt_messages"]

    #3. render to a STRING (generate_batch tokenizes it at line 74).
    #    tokenize=False, add_generation_prompt=True — the second one has no
    #    error case if you forget it, only a silently wrong prompt.

    render = processor.apply_chat_template(conversation, tokenize = False ,add_generation_prompt = True)

    #4. the gate: the gold answer must never appear in the prompt.
    assert row["reference"] not in render, "Reference leaked in the prompt"
    return render



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

    # convert_tokens_to_ids returns the UNK id (not None) for a token the
    # vocab doesn't have, so compare against unk_token_id -- adding unk to the
    # eos set would halt generation on the first unknown token.
    eos_ids = {processor.tokenizer.eos_token_id}
    end_of_turn_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id is not None and end_of_turn_id != processor.tokenizer.unk_token_id:
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

    from tqdm import tqdm
    for start in tqdm(range(0, len(order), batch_size), desc=f"Generating {eval_name}"):
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

    # ── W&B tracking ────────────────────────────────────────────────────────
    # There is NO Trainer here, so nothing auto-logs eval — these two calls are
    # the only way these numbers reach W&B. Both no-op when no run is active, so
    # calling run_eval() directly (tests, a notebook) still works. The samples
    # table is a 10-row qualitative skim, NOT a replacement for --out-csv.
    family = splits.EVAL_FILES[eval_name][0]   # "ACI" | "MTS"
    tracking.log_eval_summary(summary, family)
    tracking.log_samples([ds[i]["prompt_messages"][0]["content"] for i in range(len(ds))],
                          preds, refs, family)
    # ────────────────────────────────────────────────────────────────────────

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
    parser.add_argument("--out-csv", default=None,
                         help="optional path to dump per-example predictions + references")
    args = parser.parse_args()

    if args.fewshot and args.adapter:
        sys.exit("--fewshot is a baseline-only knob; drop --adapter or drop --fewshot")

    seed_everything()

    # one run per invocation; run_eval does the per-family logging inside it.
    tracking.start_run("eval" if args.adapter else "baseline",
                        config={"adapter": args.adapter, "eval": args.eval,
                                "fewshot": args.fewshot})
    try:
        processor, model = modeling.load_for_inference(adapter_path=args.adapter)

        fewshot_examples = None
        if args.fewshot:
            family = splits.EVAL_FILES[args.eval][0]
            fewshot_examples = _sample_fewshot(family, args.fewshot)

        run_eval(model, processor, args.eval, fewshot_examples=fewshot_examples,
                  batch_size=args.batch_size, out_csv=args.out_csv)
    finally:
        # finally, not a trailing call: a CUDA OOM halfway through generation
        # would otherwise leave the run "running" in the W&B UI forever.
        tracking.finish()


if __name__ == "__main__":
    main()

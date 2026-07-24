"""Re-measure token lengths on the CLEANED data in processed/, via the real
MedGemma processor + `prompts.build` + `apply_chat_template`.

The day_by_day plan's max_length >= 4608 cap (ACI combined max 4551) was
measured on RAW data, before clean_text collapsed whitespace -- this script
re-measures on processed/ to confirm the cap still holds. Read-only: it
writes no files.

Needs the MedGemma HF gate cleared (Health AI Developer Foundations terms
accepted + a token-authenticated `huggingface-cli login`), so this likely has
to run on the GCP VM rather than this local box -- that gated-access failure
is EXPECTED here, not a bug to work around.
"""
import sys

import pandas as pd

from db_tools import prompts
from Data_main import OUT_DIR

MODEL_ID = "google/medgemma-4b-it"

ACI_FILES = ["aci_train.csv", "aci_valid.csv", "aci_taskB_test1.csv", "aci_taskC_test2.csv", "aci_test.csv"]
MTS_FILES = ["mts_train.csv", "mts_valid.csv", "mts_test1.csv", "mts_test2.csv"]


def _load_processor():
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained(MODEL_ID)
    except Exception as exc:
        sys.exit(
            f"Could not load '{MODEL_ID}': {exc}\n\n"
            "This is the EXPECTED failure on a local box without the MedGemma "
            "gate cleared. Fix by, on the machine you actually run this on:\n"
            "  1. Log into Hugging Face and accept the Health AI Developer "
            "Foundations (HAI-DEF) terms on https://huggingface.co/google/medgemma-4b-it\n"
            "  2. `huggingface-cli login` with a token that has access\n"
            "Do not attempt to work around the gate -- run this on the GCP VM instead."
        )


def _token_lens(processor, family: str, row: pd.Series) -> tuple[int, int, int]:
    messages = prompts.build(family, row)
    user_msg, assistant_msg = messages
    input_ids = processor.tokenizer(user_msg["content"], add_special_tokens=False)["input_ids"]
    output_ids = processor.tokenizer(assistant_msg["content"], add_special_tokens=False)["input_ids"]
    combined_ids = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return len(input_ids), len(output_ids), len(combined_ids)


def _stats(values: pd.Series) -> dict:
    return {
        "mean": values.mean(),
        "p50": values.quantile(0.50),
        "p95": values.quantile(0.95),
        "p99": values.quantile(0.99),
        "max": values.max(),
    }


def _print_stats(family: str, part: str, values: pd.Series) -> None:
    s = _stats(values)
    print(f"{family:>3} / {part:<8}  mean={s['mean']:.0f}  p50={s['p50']:.0f}  "
          f"p95={s['p95']:.0f}  p99={s['p99']:.0f}  max={s['max']:.0f}")


def main() -> None:
    processor = _load_processor()

    for family, filenames in (("ACI", ACI_FILES), ("MTS", MTS_FILES)):
        input_lens, output_lens, combined_lens = [], [], []
        for name in filenames:
            df = pd.read_csv(OUT_DIR / name)
            for _, row in df.iterrows():
                i, o, c = _token_lens(processor, family, row)
                input_lens.append(i)
                output_lens.append(o)
                combined_lens.append(c)

        _print_stats(family, "input", pd.Series(input_lens))
        _print_stats(family, "output", pd.Series(output_lens))
        _print_stats(family, "combined", pd.Series(combined_lens))


if __name__ == "__main__":
    main()
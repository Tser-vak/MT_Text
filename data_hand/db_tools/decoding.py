"""Frozen decoding config -- resolves plan gap #4. ONE source of generation
kwargs so the baseline (Day 5), the fine-tuned eval (Day 10), and faithfulness
generation (Day 11) all decode IDENTICALLY. A baseline-vs-fine-tuned claim is
only valid if the sole thing that changed is the weights, not the sampler.

GREEDY on purpose: deterministic, reproducible, and the standard choice for a
summarization baseline you must compare against. No `temperature`/`top_p` keys
-- `do_sample=False` ignores them, so setting them would misrepresent what runs.

`max_new_tokens` is a SAFETY CEILING, not a target length: greedy stops at
<end_of_turn>/EOS on its own, so this only bounds a runaway generation. Sized
well above ACI's measured output (mean ~599 tokens). CONFIRM against
measure_lengths' output-token p99 after the post-cleaning re-run (Day-2 #13)
before trusting long full notes not to clip.

Stop token: pass the processor's `eos_token_id` at call time -- Gemma ends a
turn with <end_of_turn>, which must be in the eos set or greedy never halts.
Usage: `model.generate(**inputs, **GENERATION, eos_token_id=<ids>,
pad_token_id=processor.tokenizer.pad_token_id)`.
"""

GENERATION = {
    "do_sample": False,      # greedy -> deterministic, reproducible baseline
    "num_beams": 1,          # plain greedy decode, no beam search
    "max_new_tokens": 1536,  # safety ceiling over ACI output p99, NOT a target
}


if __name__ == "__main__":
    assert GENERATION["do_sample"] is False, "baseline must be deterministic"
    assert "temperature" not in GENERATION and "top_p" not in GENERATION
    assert GENERATION["max_new_tokens"] > 0
    print("decoding.GENERATION OK")
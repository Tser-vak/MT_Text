"""Per-family prompt assembly: cleaned CSV row -> chat prompt data for SFT.

Pure string work, NO tokenizer. Two shapes come out of this module:
  - `build`/`build_prompt` -> a `messages` two-turn conversation, used for
    eval-time generation and eval-loss inspection.
  - `build_train` -> `{"prompt": [...], "completion": [...]}`, used for the
    actual TRAIN mix.
On the installed TRL 1.8.0, `completion_only_loss=True` on a `messages`-shaped
dataset does NOT mask the prompt -- it silently trains on the dialogue turn
too (verified: sft_trainer.py:1042-1046, 1481-1520). `assistant_only_loss=True`
raises for vision-language models, and MedGemma (loaded via
`AutoModelForImageTextToText` + `AutoProcessor`) is one (sft_trainer.py:1191).
So the only path that actually masks the prompt is prompt/completion data +
`completion_only_loss=True`, which is what `build_train` feeds `splits`'s
train-mix loaders. Set max_length >= 4608 so nothing truncates (ACI max
combined is 4551) -- then no custom tokenization/truncation is needed.

Row schemas (from data_hand/processed/, produced by Data_main.py):
  ACI: encounter_id, dialogue, note
  MTS: section_header (full name), section_text, dialogue

If throughput/target-clipping ever forces per-family head-truncation, switch to
SeqTokenizer.encode (still stubbed in tokenization.py) and use the 3-part build
variant preserved in the comment at the bottom of this file.
"""
import pandas as pd

# Per-family instruction preamble (task text, WITHOUT the dialogue). MTS is a
# format string because section_header selects WHICH section to summarize --
# without it the dialogue->section_text mapping is ambiguous. Tune the wording;
# the {section} slot for MTS is required.
INSTRUCTION = {
    "ACI": "Generate a clinical note from the following doctor-patient dialogue. "
    "Do NOT invent or hallucinate physical examination findings "
    "if they were not explicitly discussed in the dialogue.",
    "MTS": "Summarize the {section} section from the following doctor-patient dialogue:",
}


def build(family: str, row: pd.Series) -> list[dict[str, str]]:

    if family == "MTS":
        preamble = INSTRUCTION[family].format(section = row.section_header)
        target = row.section_text
    else:
        preamble = INSTRUCTION[family]
        target = row.note
    # Joined text
    content = preamble + "\n\n" + row.dialogue

    return [{"role": "user" , "content": content}, {"role": "assistant" , "content": target}]


def build_prompt(family: str, row: pd.Series) -> tuple[list[dict[str, str]], str]:
    """
       Goal: `build` renders the full two-turn conversation SFTTrainer trains on,
       but generation-time eval must never see the assistant turn -- handing the
       model the answer would make every eval number meaningless. This function
       is the single place that derives the inference-time rendering from the
       same row, so the train and eval prompt paths can never drift apart.
       """

    #call the build to get the list for the data
    chat_prompt = build(family, row)

    #Get the Eval prompt for the model
    prompt_message = [chat_prompt[0]]

    #Get the answer for the scorer
    reference = chat_prompt[-1]["content"]

    return prompt_message,reference


def build_train(family: str, row: pd.Series) -> dict[str, list[dict[str, str]]]:
    """Reuse `build(family, row)`'s two turns to produce the prompt/completion
    shape TRL 1.8.0 needs to build its `-100` completion mask (see the module
    docstring). Returns `{"prompt": [<user turn>], "completion": [<assistant
    turn>]}` -- each value is a ONE-message list; the "prompt" side must match
    `build_prompt`'s `prompt_messages` exactly, do not re-assemble the strings
    by hand.
    """
    # Slice build()'s own output -- never re-assemble the strings. That is what
    # keeps the train prompt byte-identical to build_prompt's eval prompt.
    conversation = build(family, row)
    return {"prompt": conversation[:1], "completion": conversation[-1:]}


if __name__ == "__main__":
    # ponytail: no-tokenizer self-check -- fill in once build() is implemented.
    aci_row = pd.Series({"dialogue": "[doctor] hi", "note": "CHIEF COMPLAINT\n..."})
    mts_row = pd.Series({"section_header": "MEDICATIONS",
                         "section_text": "None.", "dialogue": "Doctor: meds?"})
    msgs = build("MTS", mts_row)
    assert len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
    assert "MEDICATIONS" in msgs[0]["content"], msgs[0]
    assert msgs[1]["content"] == "None.", msgs[1]
    msgs = build("ACI", aci_row)
    assert msgs[1]["content"].startswith("CHIEF COMPLAINT"), msgs[1]
    print("prompts.build OK")

    for family, row in (("MTS", mts_row), ("ACI", aci_row)):
        prompt_messages, reference = build_prompt(family, row)
        assert len(prompt_messages) == 1 and prompt_messages[0]["role"] == "user"
        assert all(m["role"] != "assistant" for m in prompt_messages)
        assert prompt_messages == build(family, row)[:1]
        assert reference == build(family, row)[-1]["content"]
    print("prompts.build_prompt OK")

    for family, row in (("MTS", mts_row), ("ACI", aci_row)):
        train_row = build_train(family, row)
        assert set(train_row) == {"prompt", "completion"}
        assert train_row["prompt"] == build(family, row)[:1]
        assert train_row["completion"] == build(family, row)[-1:]
    print("prompts.build_train OK")

# =========== IF THE CLASSIC SFTrain HAS BAD OUTCOME =========
# --- 3-part variant (only if you switch to SeqTokenizer.encode custom path) ---
# def build_parts(family: str, row: pd.Series) -> tuple[str, str, str]:
#     """Return (preamble, dialogue, target) as three SEPARATE strings so encode
#     can head-truncate the dialogue independently without corrupting the chat
#     template. preamble = INSTRUCTION[family] ({section} filled for MTS);
#     dialogue = row.dialogue; target = row.note (ACI) / row.section_text (MTS).
#     Do NOT pre-join preamble+dialogue -- the separation is what makes safe
#     head-truncation possible. See SeqTokenizer.encode's Goal/Why/Done."""
#     raise NotImplementedError("prompts.build_parts")
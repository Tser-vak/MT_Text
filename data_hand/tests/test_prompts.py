"""Acceptance checks for the db_tools/prompts.py::build_prompt hole. These FAIL
with NotImplementedError until you fill the hole; when they pass, the
train-vs-inference rendering split holds. Run via `pytest` or `python test_prompts.py`.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_tools.prompts import build, build_prompt

ACI_ROW = pd.Series({"dialogue": "[doctor] hi", "note": "CHIEF COMPLAINT\n..."})
MTS_ROW = pd.Series({"section_header": "MEDICATIONS",
                     "section_text": "None.", "dialogue": "Doctor: meds?"})
ROWS = {"ACI": ACI_ROW, "MTS": MTS_ROW}


def test_build_prompt_has_no_assistant_turn():
    for family, row in ROWS.items():
        prompt_messages, _reference = build_prompt(family, row)
        assert len(prompt_messages) == 1, f"{family}: expected a single user turn"
        assert prompt_messages[0]["role"] == "user", f"{family}: expected role 'user'"
        assert all(m["role"] != "assistant" for m in prompt_messages), \
            f"{family}: assistant turn leaked into prompt_messages"


def test_build_prompt_matches_build():
    for family, row in ROWS.items():
        prompt_messages, reference = build_prompt(family, row)
        full = build(family, row)
        assert prompt_messages == full[:1], f"{family}: prompt_messages must match build()'s user turn"
        assert reference == full[-1]["content"], f"{family}: reference must match build()'s assistant content"


if __name__ == "__main__":
    test_build_prompt_has_no_assistant_turn()
    test_build_prompt_matches_build()
    print("All prompts checks passed.")
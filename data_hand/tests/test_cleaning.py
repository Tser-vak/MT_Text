"""Acceptance checks for the Day-3 text-cleaning holes in db_tools/data_tools.py
(clean_text, normalize_speakers, drop_degenerate_mts). These FAIL with
NotImplementedError until you fill the holes; when they pass, the cleaning meets
its contract. Run via `pytest` or `python test_cleaning.py`.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_tools.data_tools import (
    UNMAPPED, clean_text, drop_degenerate_mts, normalize_speakers,
)


def test_clean_text_repairs_mojibake():
    assert "•" in clean_text("a â€¢ b")  # â€¢ -> • bullet


def test_clean_text_normalizes_whitespace_but_keeps_turns():
    out = clean_text("Doctor: hi  there\r\nPatient: hello")
    assert "\r" not in out                 # CRLF gone
    assert "  " not in out                 # collapsed double space
    assert out.count("\n") == 1            # turn boundary preserved, not flattened


def test_clean_text_strips_ends():
    assert clean_text("  hi \n ") == clean_text("hi")  # leading/trailing stripped


def test_normalize_speakers_unifies_both_conventions():
    aci = normalize_speakers("[doctor] how are you\n[patient] fine")
    mts = normalize_speakers("Doctor: how are you\nPatient: fine")
    assert aci == mts, "ACI [doctor]/[patient] and MTS Doctor:/Patient: must normalize identically"


def test_normalize_speakers_unifies_guest_roles_across_sources():
    aci = normalize_speakers("[patient_guest] she fell")   # ACI convention
    mts = normalize_speakers("Guest_family: she fell")     # MTS convention
    assert aci == mts, "the same role must render identically in both sources"
    assert normalize_speakers("Gest_family: x") == aci.replace("she fell", "x")


def test_normalize_speakers_keeps_roles_distinct():
    # collapsing Caregiver/Clinician into Patient/Doctor would delete info the
    # gold notes need ("her daughter says ...") -- four roles, not two.
    rendered = {normalize_speakers(f"{tag} x")
                for tag in ("[doctor]", "[patient]", "Guest_family:", "Guest_clinician:")}
    assert len(rendered) == 4, rendered


def test_normalize_speakers_tolerates_leading_space():
    # 7 MTS turns start with a space; a bare ^ anchor silently skipped them.
    assert normalize_speakers(" Doctor: hi") == normalize_speakers("Doctor: hi")


def test_normalize_speakers_reports_unknown_tag_instead_of_dropping_it():
    before = UNMAPPED["nurse"]
    out = normalize_speakers("Nurse: vitals are stable")
    assert "Nurse:" in out, "an unmapped speaker must survive verbatim, not vanish"
    assert UNMAPPED["nurse"] == before + 1, "an unmapped speaker must be counted"


def test_normalize_speakers_ignores_transcription_artifacts():
    assert normalize_speakers("[ inaudible 00:09:25 ]") == "[ inaudible 00:09:25 ]"


def test_drop_degenerate_drops_null_keeps_short_valid():
    df = pd.DataFrame({
        "section_header": ["ALLERGY", "ALLERGY", "MEDICATIONS"],
        "section_text": ["n/a", "No known drug allergies.", "1. Prilosec. 2. Tramadol."],
        "dialogue": ["d1", "d2", "d3"],
    })
    out = drop_degenerate_mts(df)
    kept = set(out["section_text"])
    assert "n/a" not in kept                          # pure-null dropped
    assert "No known drug allergies." in kept         # short-but-valid survives
    assert "1. Prilosec. 2. Tramadol." in kept


if __name__ == "__main__":
    test_clean_text_repairs_mojibake()
    test_clean_text_normalizes_whitespace_but_keeps_turns()
    test_clean_text_strips_ends()
    test_normalize_speakers_unifies_both_conventions()
    test_normalize_speakers_unifies_guest_roles_across_sources()
    test_normalize_speakers_keeps_roles_distinct()
    test_normalize_speakers_tolerates_leading_space()
    test_normalize_speakers_reports_unknown_tag_instead_of_dropping_it()
    test_normalize_speakers_ignores_transcription_artifacts()
    test_drop_degenerate_drops_null_keeps_short_valid()
    print("All cleaning checks passed.")
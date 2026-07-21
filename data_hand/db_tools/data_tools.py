"""Cleaning helpers for the raw ACI-bench / MTS-Dialog CSVs."""
import re
from pathlib import Path
import ftfy
import pandas as pd

DROP_HEADERS = {"labs", "other_history"}
# pure-null placeholders that carry no clinical content (matched on
# lowercased+stripped section_text). NOT a length filter -- short-but-valid
# rows like "No known drug allergies." must survive (see drop_degenerate_mts).
NULL_PLACEHOLDERS = {"", "n/a", "na", "nil", "not applicable", "no", "-", "."}

SPEAKER_TAGS = {"doctor" : "Doctor" , "patient" : "Patient"}
SPEAKER_RE = re.compile(r"^(?:\[(doctor|patient)\]|(doctor|patient):)",re.IGNORECASE | re.MULTILINE)

def clean_text(text: str) -> str:
    # Clean ASCI formar
    cl_tx = ftfy.fix_text(text).replace("\r\n", "\n")

    #cleaned the text from the excessive  spaces and /t , /n
    rd_tx= re.sub(r"[^\S\n]+"," ",cl_tx).strip()

    return rd_tx

def normalize_speakers(dialogue: str) -> str:
    def _tag(m: re.Match ) -> str :
        return SPEAKER_TAGS[((m.group(1) or m.group(2)).lower())]
    return SPEAKER_RE.sub(_tag, dialogue)


def drop_degenerate_mts(df: pd.DataFrame) -> pd.DataFrame:
    #strip the /n in the df cause it has specs we dont need
    sc_dt = df["section_text"].str.lower().str.strip()
    #Creat a mask with the NULL ,we striped the none cause there is None in the section_text and if it undentifies it will get rid /
    # we need it
    flag = sc_dt.isin(NULL_PLACEHOLDERS)
    # the flagged data
    flagged = df[flag]
    if not flagged.empty :
        print(f"So the Length of the flagged is {len(flagged)}")
        print(flagged)
    return  df[~flag].copy()

def load_section_map(path: str | Path) -> dict[str, str]:
    """Parse 'key [FULL NAME]' or bare 'key' lines into {key: full_name} (lowercase keys)."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\S+)(?:\s+\[(.+)\])?$", line)
            if not m:
                continue
            key, bracket = m.group(1), m.group(2)
            mapping[key.lower()] = (bracket or key).strip()
    return mapping


def prep_aci(df: pd.DataFrame) -> pd.DataFrame:
    """Keep encounter_id/dialogue/note, drop 'dataset', then clean text + normalize speakers."""
    df = df[["encounter_id", "dialogue", "note"]].copy()
    df["dialogue"] = df["dialogue"].map(clean_text).map(normalize_speakers)
    df["note"] = df["note"].map(clean_text)  # no speaker tags in the target
    return df


def prep_mts(df: pd.DataFrame, section_map: dict[str, str]) -> pd.DataFrame:
    """ Drop sections, expand section_header, drop ID, then clean text + drop degenerate rows."""
    headers = df["section_header"].str.lower()
    df = df[~headers.isin(DROP_HEADERS)].copy()
    df["section_header"] = df["section_header"].str.lower().map(section_map)
    assert df["section_header"].notna().all(), "unmapped section_header found after mapping"
    df = df[["section_header", "section_text", "dialogue"]].copy()
    df["dialogue"] = df["dialogue"].map(clean_text).map(normalize_speakers)
    df["section_text"] = df["section_text"].map(clean_text)  # no speaker tags in the target
    return drop_degenerate_mts(df)
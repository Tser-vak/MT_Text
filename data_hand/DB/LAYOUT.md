# Repo layout — `data_hand/`

Mapped from the working tree. Raw data lives under `DB/`; pipeline code and
outputs live one level up under `data_hand/`.

```
data_hand/
├── Data_main.py            # pipeline entry point / orchestrator (empty — to build)
├── requirements.txt        # pinned deps, frozen from Python 3.11.9 venv
├── venv/                   # Python 3.11.9 env with full HF stack (USE THIS, not 3.14)
├── db_tools/
│   └── data_tools.py       # reusable cleaning/dedup helpers (empty — to build)
├── tests/                  # parser + dedup tests (empty — to build)
├── processed/              # pipeline output JSONL (empty — to build)
├── Splits/                 # (empty)
├── templates/              # prompt/chat template (empty — awaiting template file)
├── docs/                   # (empty)
├── logs/                   # (empty)
├── Std/
│   └── day_by_day(1).md    # project plan / day-by-day brief
└── DB/                     # RAW DATA — read-only
    ├── data_look.ipynb     # EDA notebook (source of truth for norm/hash/section-map)
    ├── Norm_sec_head.txt   # MTS section_header -> full-name mapping
    ├── LAYOUT.md           # this file
    ├── training/
    │   ├── MTS/MTS-Dialog-TrainingSet.csv
    │   └── aci/{train,clinicalnlp_taskB_test1,clinicalnlp_taskC_test2}.csv
    ├── valid/
    │   ├── MTS/MTS-Dialog-ValidationSet.csv
    │   └── aci/valid.csv
    ├── testing/
    │   ├── MTS/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv
    │   ├── MTS/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv
    │   └── aci/clef_taskC_test3.csv
    └── aci/aci-bench-corpus/challenge_data/
        └── *_metadata.csv  # per-ACI-split metadata, joined on encounter_id
```

## Split → role mapping

| Source | Files | Rows | Role |
|--------|-------|------|------|
| ACI train pool | train + taskB_test1 + taskC_test2 + valid | 167 | full_note training |
| ACI frozen test | clef_taskC_test3 | 40 | final eval, untouched |
| MTS train | MTS-Dialog-TrainingSet | 1201 | section training |
| MTS valid | MTS-Dialog-ValidationSet | 100 | validation |
| MTS test | TestSet-1 + TestSet-2 | 400 | secondary section eval |

Schemas differ: MTS = `ID, section_header, section_text, dialogue`;
ACI = `dataset, encounter_id, dialogue, note` (+ metadata join). The two
sources share no keys — unioned, never joined to each other.

## Planned pipeline outputs (`processed/`)

- `train_full_note.jsonl` — 167 (ACI)
- `train_section.jsonl` — 1199 (MTS train, minus 2 contamination IDs)
- `valid_section.jsonl` — 100 (MTS valid)
- `eval_section.jsonl` — 400 (MTS test, untouched)
- `frozen_test_full_note.jsonl` — 40 (ACI test)

Record schema: `{id, source, task_type, instruction, input, output, meta}`.

---

# Spec — MTS + ACI data pipeline

Stage 1: parse → clean → dedup → schema JSONL. Stage 2: template + tokenize.

## Goal
Turn the raw MTS-Dialog and ACI-bench CSVs into clean, deduplicated,
schema-uniform JSONL splits ready for QLoRA SFT — one record per training
example, with identical cleaning across both sources so the model never learns
spurious format differences. Split into **Stage A** (buildable now) and
**Stage B** (gated on the prompt template).

## Files

**Create `data_hand/db_tools/data_tools.py`** (canonical reusable helpers — lift
the proven ones from `DB/data_look.ipynb`, don't reinvent):
- `norm_text(s)` / `hash_text(s)` — **exactly** the notebook versions incl. the
  zero-width strip (`ZERO_WIDTH = {0x200B,0x200C,0x200D,0xFEFF}`).
  **Dedup/degenerate-detection only** — destructive flatten, never written to output.
- `clean_text(s, preserve_structure: bool)` — the *output-facing* cleaner:
  `ftfy.fix_text` → strip zero-width → normalize `\xa0`/` ` to plain space →
  `\r\n`→`\n` → strip trailing whitespace per line. When `preserve_structure=True`
  (ACI notes, MTS section_text) **keep newlines and `•` bullets**; when `False`
  collapse intra-line whitespace only. **Must never flatten a note to one line
  and never strip bullets.**
- `normalize_speakers(dialogue)` — map line-leading `Doctor:`→`[doctor]`,
  `Patient:`→`[patient]`; any other `Name:` prefix → `[name_lower]`; ACI
  `[doctor]`/`[patient]` pass through. **Assert-flag any unmapped speaker prefix**
  so a new speaker fails loud, not silent.
- `load_section_map(path)` — notebook version; maps `section_header`→full name
  via `DB/Norm_sec_head.txt`.
- `is_pure_null(section_text)` — the `JUNK` set check from the degenerate-scan
  cell (`none/n/a/nil/...`). Length is **not** a drop signal.

**Create `data_hand/Data_main.py`** (entry point / orchestrator):
- Load the 9 CSVs (paths from the notebook's `FILES` dict) with pandas.
- **ACI parser** → `task_type="full_note"`, `input=clean_text(dialogue, preserve=False)`
  (speaker-normalized), `output=clean_text(note, preserve=True)`, `meta`=metadata
  dict joined on `encounter_id`.
- **MTS parser** → `task_type="section"`, `input=clean_text(dialogue, preserve=False)`
  (speaker-normalized), `output=clean_text(section_text, preserve=True)`,
  `instruction` carries the mapped full section name, `meta={"section_header":...}`.
- **Drop rules:** MTS train — drop the 2 contamination IDs (recompute via
  `hash_text` against MTS test, don't hardcode) and any `is_pure_null` row
  (currently 0). ACI/MTS eval + frozen test — **no drops** (identical text
  cleaning only).
- **Emit to `processed/`:** `train_full_note.jsonl` (167),
  `train_section.jsonl` (1199 = 1201−2), `valid_section.jsonl` (100, MTS valid),
  `eval_section.jsonl` (400, MTS test, untouched),
  `frozen_test_full_note.jsonl` (40, ACI test). Sort by id for deterministic output.
- Schema per line: `{id, source, task_type, instruction, input, output, meta}`.

**Create `data_hand/tests/test_data_prep.py`** (the parsers are where silent bugs
corrupt everything):
- Schema conformance: every record, both sources — all 7 fields present,
  non-null, correctly typed; `task_type` matches source.
- Split counts exactly 167 / 1199 / 100 / 400 / 40.
- Speaker normalization: assert no raw `Doctor:`/`Patient:` survives in any
  `input`, and no unmapped-speaker escape.
- Dedup self-test: feed a known duplicate pair, assert `hash_text` collides
  (the check whose own correctness must be tested).
- Decontam applied: assert the 2 collision IDs are absent from
  `train_section.jsonl` and **present** in `eval_section.jsonl`.
- Fixture correctness: 2–3 hand-verified raw rows per source parse to expected
  known-good records.
- `clean_text` structure: assert an ACI note keeps its `\n` and `•`; assert a
  section_text like `"No known drug allergies."` survives untouched.

**Stage B — gated on `data_hand/templates/<prompt_template>`:**
- `format_example(record)` assembles system/user/assistant via
  `tokenizer.apply_chat_template`, pulling instruction text from the template
  file (**never hand-typed per record** — test byte-identical instruction per
  task_type).
- Tokenize with `google/medgemma-4b-it` tokenizer; enforce `max_seq_length=4096`,
  truncate **dialogue head only**, assert output/target is never truncated; log
  combined p99 stays ≤4096.

## Constraints & conventions
- **Run under `data_hand/venv` (Python 3.11.9)** — not the 3.14 interpreter.
- Reuse the notebook's already-validated logic verbatim where it exists
  (`norm_text`, `hash_text`, `load_section_map`, `FILES`, `ACI_ROLE`); the
  notebook is the source of truth for these.
- pandas for all CSV reads (embedded newlines). ftfy for cleaning. No new
  dependencies — everything needed is in `requirements.txt`.
- Deterministic: stable sort, so re-runs and test hashes are reproducible.
- **ponytail:** one helpers module + one orchestrator + one test file. No config
  framework for ~4 constants — put `MAX_SEQ_LENGTH`, drop-rule params, and paths
  at the top of `Data_main.py`.

## Definition of done
- **Stage A:** `python Data_main.py` under the venv writes the 5 JSONL files with
  exact counts; `pytest data_hand/tests/test_data_prep.py` green.
- **Stage B:** every record renders + tokenizes; truncation asserts hold; length
  report printed. Only startable once the template file lands.

## Do NOT touch
- `data_hand/DB/**` raw CSVs and `Norm_sec_head.txt` — read-only.
- `data_hand/DB/data_look.ipynb` — EDA stays as-is (lift functions *from* it,
  don't edit it).
- The 400 MTS eval rows and 40 ACI frozen-test rows — **no
  contamination/degenerate drops**, cleaning only. Never drop the collision rows
  from eval.
- `requirements.txt` and `venv/**`.

## Open decision (blocks Stage B only)
Speaker-tag target: spec normalizes both sources to lowercase
`[doctor]`/`[patient]`. Change to `Doctor:`/`Patient:` if preferred — one-line change.
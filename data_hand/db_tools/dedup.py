"""Exact-hash text deduplication across dataset splits.

Read before implementing:
  - hashlib:      https://docs.python.org/3/library/hashlib.html
  - str.translate: https://docs.python.org/3/library/stdtypes.html#str.translate
"""
import pandas as pd


class TextDeduplicator:
    """Normalizes then SHA-256-hashes text so identical rows can be found across splits."""

    ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0xFEFF}  # ZWSP, ZWNJ, ZWJ, BOM
    ZERO_CP= {cp: None for cp in ZERO_WIDTH}

    def norm_text(self, text:str) -> str:
        import re

        text = str(text).lower().translate(self.ZERO_CP)
        return re.sub(r"\s+"," ",text).strip()

    def hash_text(self, text:str) -> str:
        import hashlib
        return hashlib.sha256(self.norm_text(text).encode("utf-8")).hexdigest()

    def hash_rows(self, df: pd.DataFrame, cols: list[str]) -> pd.Series:
        """Per-row hash of `cols` concatenated (joined, then through hash_text)."""
        joined = df[cols].astype(str).agg("||".join, axis=1)
        return joined.map(self.hash_text)

    def drop_collisions(self, keep_df: pd.DataFrame, drop_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
                Goal: Remove rows from `drop_df` (the TRAINING split) whose `cols`
                content also appears in `keep_df` (the held-out TEST/benchmark split),
                so the fine-tune never trains on a row it will later be scored on.
                `keep_df` is never modified -- direction is test-keeps / train-drops.
                The model sees MTS/ACI fresh in this fine-tune (no prior exposure), so
                we decontaminate on the training side and leave the test split
                byte-identical to the published benchmark, keeping reported numbers
                comparable. Only a handful of rows overlap, so train loses nothing
                that matters.
                """

        # Creating the hash list of both
        hash_drop = self.hash_rows(drop_df, cols)
        hash_keep = self.hash_rows(keep_df, cols)

        #Create a set of hash in the keep_df
        keep_set = set(hash_keep)

        #the collided hash
        collided = hash_drop.isin(keep_set)

        # Cleaned drop_df
        clean_df = drop_df[~collided].copy()

        # Which rows left, not just how many -- the caller already logs the count.
        # Silent when nothing collided so the clean splits don't add noise.
        if collided.any():
            for idx, h in hash_drop[collided].items():
                print(f"  dropped row {idx} (hash {h[:12]})")

        return clean_df







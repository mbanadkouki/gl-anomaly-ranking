"""Sentence construction and sentence embeddings for journal entries (notebook 04_encoding_embeddings).

Decision D16 (results/feature_engineering_plan.md):
  - The embedding replaces only the "account + debit/credit per line" block of the one-hot
    encoding; all other one-hot columns stay identical and are appended later (week 3, D17).
  - Main template T-C: Bakumenko et al. (2025) list structure with three changes:
    account NAME instead of code, the words Debit / Credit instead of _D / _C without a
    repeated prefix, and "; " as separator (account names contain commas).
  - Ablation T-Bak: the exact Bakumenko format with account codes.
  - Lines in the D9 order (src/encode.sort_lines), so both encodings see the same order.

The whole ledger has only 29 distinct sentences, so only the unique sentences are embedded
and stored, plus an index that maps every entry (meta.parquet order) to its sentence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .encode import sort_lines

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEMPLATES = ("tc", "bak")
SIDE_WORD = {"D": "Debit", "C": "Credit"}


def line_part(lines: pd.DataFrame, template: str) -> pd.Series:
    """Text for each posting line."""
    if template == "tc":       # "Debit Trade payables"
        return lines["cd_flag"].map(SIDE_WORD) + " " + lines["gl_account_name"]
    if template == "bak":      # "Account_DC: 1600_C" (Bakumenko et al. 2025, Eq. 2, without Source)
        return "Account_DC: " + lines["gl_account"].astype(str) + "_" + lines["cd_flag"]
    raise ValueError(f"unknown template {template}")


def build_sentences(lines: pd.DataFrame, meta: pd.DataFrame, template: str) -> pd.Series:
    """One sentence per entry, in meta.parquet row order."""
    sl = sort_lines(lines)                                  # D9 order
    parts = line_part(sl, template)
    if template == "tc":
        s = parts.groupby(sl["posting_id"], sort=False).agg("; ".join) + "."
    else:
        s = parts.groupby(sl["posting_id"], sort=False).agg(", ".join)
    return s.reindex(meta["posting_id"].to_numpy()).rename("sentence")


def unique_index(sentences: pd.Series) -> tuple[list[str], np.ndarray]:
    """Unique sentences (most frequent first) and, per entry, the position of its sentence."""
    counts = sentences.value_counts()
    texts = counts.index.tolist()
    pos = {t: i for i, t in enumerate(texts)}
    index = sentences.map(pos).to_numpy().astype(np.int32)
    return texts, index


def encode(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """L2-normalised sentence embeddings (needs sentence-transformers, laptop / Colab)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def save(out_dir: Path | str, template: str, texts: list[str], index: np.ndarray,
         emb: np.ndarray, info: dict) -> dict:
    """emb_<template>.npy (unique x dim), emb_<template>_index.npy (one int per entry), emb_<template>.json."""
    out_dir = Path(out_dir)
    np.save(out_dir / f"emb_{template}.npy", emb)
    np.save(out_dir / f"emb_{template}_index.npy", index)
    counts = np.bincount(index, minlength=len(texts)).tolist()
    meta = {"template": template, "texts": texts, "entries_per_text": counts, **info}
    with open(out_dir / f"emb_{template}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    return meta


def load(out_dir: Path | str, template: str, expand: bool = True):
    """Returns (matrix, info). expand=True -> one row per entry (meta order); False -> unique rows."""
    out_dir = Path(out_dir)
    emb = np.load(out_dir / f"emb_{template}.npy")
    index = np.load(out_dir / f"emb_{template}_index.npy")
    with open(out_dir / f"emb_{template}.json", encoding="utf-8") as f:
        info = json.load(f)
    info["index"] = index
    return (emb[index] if expand else emb), info


def non_account_columns(feature_info: list[dict], groups: dict) -> list[int]:
    """Columns of the one-hot encoding that T-C keeps next to the embedding (D16):
    every column of the active groups except the per-slot account and side one-hots."""
    keep = []
    for i, f in enumerate(feature_info):
        if not groups.get(f["group"], False):
            continue
        is_slot_cat = f["slot"] is not None and ("_acct_" in f["name"] or "_cd_" in f["name"])
        if not is_slot_cat:
            keep.append(i)
    return keep

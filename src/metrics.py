"""Budget-based ranking metrics for journal entry testing (RQ1-RQ3).

An auditor reviews the top k entries of a ranking (the review budget). For every
method the metrics answer: how many of the k reviewed entries are anomalies, and
how many anomalies of the whole fiscal year were found.

Conventions (D14, decided 23 Nov 2026, budgets changed 25 Nov 2026):
  - Higher score = more suspicious. NaN score = entry is not on the method's list
    (e.g. outside the JET prefilter); it is never reviewed.
  - Ties are broken by a seeded random permutation, never by row order
    (row order = CSV order and may carry generator structure). Report the mean
    over several seeds.
  - Recall and AP use ALL anomalies of the evaluated year as denominator, so an
    anomaly that a method cannot reach (e.g. outside the JET list) counts as missed.
  - Budgets are absolute numbers of reviews (auditor capacity), the same set as
    Gronewald, Scherer et al. (2026): 25, 50, 100, 250, 500, 1,000, 2,000; main
    budgets in the text are k = 100 and k = 1,000.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUDGETS_ABS = (25, 50, 100, 250, 500, 1000, 2000)   # D14: same as the preprint
MAIN_BUDGETS = ("k100", "k1000")
SEEDS = (0, 1, 2, 3, 4)


def budgets() -> dict[str, int]:
    """Budget label -> number of reviewed entries: {'k25': 25, ..., 'k2000': 2000}."""
    return {f"k{k}": k for k in BUDGETS_ABS}


def review_order(scores, seed: int) -> np.ndarray:
    """Indices of the entries in review order: score descending, ties in random order.

    Entries with a NaN score are not on the list and are left out.
    """
    s = np.asarray(scores, dtype=float)
    on_list = np.flatnonzero(~np.isnan(s))
    tie_key = np.random.default_rng(seed).random(on_list.size)
    order = np.lexsort((tie_key, -s[on_list]))   # last key sorts first: score desc, then random
    return on_list[order]


def metrics_at_k(order: np.ndarray, labels, k: int) -> dict:
    """Hits, precision, recall and false positives when the top k of `order` are reviewed.

    If the list is shorter than k, only the listed entries are reviewed (n_reviewed < k).
    """
    y = np.asarray(labels).astype(bool)
    reviewed = order[:k]
    n_rev = reviewed.size
    hits = int(y[reviewed].sum())
    n_pos = int(y.sum())
    return {
        "k": k,
        "n_reviewed": n_rev,
        "hits": hits,
        "fp": n_rev - hits,
        "precision": hits / n_rev if n_rev else 0.0,
        "recall": hits / n_pos if n_pos else 0.0,
    }


def average_precision(order: np.ndarray, labels) -> float:
    """Mean of the precision at each hit, divided by ALL anomalies of the year.

    Anomalies that are not on the list contribute 0 (they are never found).
    """
    y = np.asarray(labels).astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    hit = y[order]
    if not hit.any():
        return 0.0
    ranks = np.flatnonzero(hit) + 1                  # 1-based positions of the hits
    precision_at_hits = np.arange(1, ranks.size + 1) / ranks
    return float(precision_at_hits.sum() / n_pos)


def evaluate(scores, labels, seeds=SEEDS, budget_map: dict | None = None) -> pd.DataFrame:
    """One row per seed and budget: k, n_reviewed, hits, fp, precision, recall, ap, list_size."""
    y = np.asarray(labels)
    budget_map = budget_map or budgets()
    rows = []
    for seed in seeds:
        order = review_order(scores, seed)
        ap = average_precision(order, y)
        for name, k in budget_map.items():
            r = metrics_at_k(order, y, k)
            rows.append({"seed": seed, "budget": name, **r, "ap": ap, "list_size": order.size})
    return pd.DataFrame(rows)


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds, one row per budget (in budget order)."""
    cols = ["n_reviewed", "hits", "fp", "precision", "recall", "ap"]
    g = per_seed.groupby("budget", sort=False)
    out = g[cols].mean().add_suffix("_mean").join(g[["precision", "recall"]].std(ddof=0).add_suffix("_sd"))
    out.insert(0, "k", g["k"].first())
    return out.reset_index()


def tag(per_seed: pd.DataFrame, **labels) -> pd.DataFrame:
    """Put descriptive columns (method, detector, setting, fy, ...) in front of the per-seed
    metrics, so that the rows of several runs can be stacked into one result table."""
    out = per_seed.copy()
    for i, (name, value) in enumerate(labels.items()):
        out.insert(i, name, value)
    return out

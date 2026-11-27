"""Unsupervised detectors (Isolation Forest, LOF) and a per-seed runner.

All detectors return a score where higher = more suspicious, so the output plugs
directly into src/metrics.py. Models are fitted on one fiscal year at a time,
without labels (D6); hyperparameters are chosen on FY1 = 2023 by mean AP (D15).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from . import metrics

# D15: small grids, chosen by mean AP on FY1 (2023) only; edge rule in edge_extension().
GRIDS = {
    "iforest": [
        {"n_estimators": 100, "max_samples": 256},
        {"n_estimators": 300, "max_samples": 256},
        {"n_estimators": 300, "max_samples": 2048},
    ],
    # k must be larger than the largest group of identical entries, otherwise the
    # k-distance is 0 and LOF is unstable (Campos et al. 2016). Main setting: largest
    # group = 36 in both years (notebook 02, section 8), so every k here is allowed.
    "lof": [
        {"n_neighbors": 50},
        {"n_neighbors": 100},
        {"n_neighbors": 200},
    ],
}


def edge_extension(detector: str, best: dict, X=None) -> dict | None:
    """D15 edge rule: if the FY1-best setting lies on the edge of its grid, return one
    extra setting one step further out (to be evaluated on FY1 like the others), else None.

    LOF: one step below the smallest k (k - 10), only if it still exceeds the largest
    duplicate group of X. IF: twice the largest max_samples."""
    grid = GRIDS[detector]
    if detector == "lof":
        ks = [g["n_neighbors"] for g in grid]
        if best["n_neighbors"] == min(ks):
            k = best["n_neighbors"] - 10
            if X is not None and k <= max_duplicate_group(X):
                return None
            return {"n_neighbors": k}
        if best["n_neighbors"] == max(ks):
            return {"n_neighbors": 2 * best["n_neighbors"]}
    if detector == "iforest":
        if best["max_samples"] == max(g["max_samples"] for g in grid):
            return {"n_estimators": best["n_estimators"], "max_samples": 2 * best["max_samples"]}
    return None


def max_duplicate_group(X) -> int:
    """Size of the largest group of identical rows (dense array)."""
    _, counts = np.unique(np.asarray(X), axis=0, return_counts=True)
    return int(counts.max())


def iforest_scores(X, n_estimators: int, max_samples: int, seed: int) -> np.ndarray:
    """Isolation Forest: short average path = easy to isolate = suspicious."""
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            random_state=seed, n_jobs=-1)
    model.fit(X)
    return -model.score_samples(X)          # sklearn: lower = more abnormal -> flip sign


def lof_scores(X, n_neighbors: int) -> np.ndarray:
    """LOF: local density lower than that of the neighbours = suspicious (deterministic)."""
    model = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm="brute", n_jobs=-1)
    model.fit(X)
    return -model.negative_outlier_factor_  # higher = more abnormal


def run(detector: str, X, labels, params: dict, seeds=metrics.SEEDS,
        budget_map: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Fit + score + evaluate for every seed. Returns (per-seed metrics, scores of the first seed).

    Isolation Forest is refitted per seed (the seed changes the trees and the tie order);
    LOF is deterministic, so it is fitted once and only the tie order changes per seed.
    """
    t0 = time.perf_counter()
    frames, first_scores = [], None
    lof_cache = None
    for seed in seeds:
        if detector == "iforest":
            s = iforest_scores(X, seed=seed, **params)
        elif detector == "lof":
            if lof_cache is None and params["n_neighbors"] <= max_duplicate_group(X):
                raise ValueError("n_neighbors must exceed the largest group of identical rows (LOF instability)")
            lof_cache = lof_cache if lof_cache is not None else lof_scores(X, **params)
            s = lof_cache
        else:
            raise ValueError(f"unknown detector {detector}")
        if first_scores is None:
            first_scores = s
        frames.append(metrics.evaluate(s, labels, seeds=(seed,), budget_map=budget_map))
    per_seed = pd.concat(frames, ignore_index=True)
    info = {"detector": detector, "params": params, "seconds": round(time.perf_counter() - t0, 1)}
    return per_seed, {"scores": first_scores, **info}


def params_label(params: dict) -> str:
    """Short text for the result tables, e.g. 'n_estimators=300;max_samples=256'."""
    return ";".join(f"{k}={v}" for k, v in params.items())

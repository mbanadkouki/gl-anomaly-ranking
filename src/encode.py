"""One-hot + padding encoding of journal entries (notebook 02_encoding_onehot).

Design decisions are documented in the Decisions sheet of Thesis_Execution_Plan.xlsx
(review of 25 Nov 2026):
  D9  line order inside the padding (debit first, then account, then amount desc)
  D10 scalers fitted per fiscal year, on the same entries the detector is fitted on (follows D6)
  D4  time features = lag_days + weekend + nwh; promptly is NOT a feature (binning of lag_days)
  D13 user once per entry; tax_rate and n_lines are NOT features (determined by the account
      pattern; n_lines is also visible in the PAD slots) - redundancy check of 26 Nov
  D11 nwh_dist, hour sin/cos, weekday sin/cos checked and dropped (built only for the data checks)

Everything is aligned to the row order of data/derived/meta.parquet (posting_id).
Fitted parameters are stored as JSON (no pickle).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import sparse

from .load import WORK_START, WORK_END   # working window of the dataset flag nwh (08:00-17:59)

MAX_LINES = 4   # longest entry in the ledger (1 entry has 4 lines)
PAD = "PAD"

# Main setting (D4, D11, D13; 25 Nov 2026). Switch groups on/off per run with select_columns().
FEATURE_GROUPS_MAIN = {
    "lines": True,           # per-slot account / side / amount, user
    "calendar_flags": True,  # weekend, nwh
    "lag_days": True,        # continuous entry lag (D4)
    "promptly": False,       # dropped (D4): deterministic binning of lag_days
    "tax_rate": False,       # dropped (D13): deterministic function of the accounts
    "n_lines": False,        # dropped (D13, 26 Nov): determined by the account pattern, already in the PAD slots
    "nwh_dist": False,       # dropped (D11): no gradient in the data
    "hour_cyc": False,       # dropped (D11)
    "wday_cyc": False,       # dropped (D11)
}
# Proposal variant for the ablation without time features (D4): lines only.
FEATURE_GROUPS_NO_TIME = {g: (g == "lines") for g in FEATURE_GROUPS_MAIN}
ALL_GROUPS = list(FEATURE_GROUPS_MAIN)


# ---------------------------------------------------------------- line order (D9)
def sort_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Sort lines inside each entry: debit first, then account number, then amount
    (largest first), then original CSV position. Adds `csv_pos` and `slot` (0-based)."""
    out = lines.copy()
    out["csv_pos"] = out.groupby("posting_id", sort=False).cumcount()
    out["_side"] = (out["cd_flag"] == "C").astype(int)        # D = 0 comes first
    out = out.sort_values(
        ["posting_id", "_side", "gl_account", "amount", "csv_pos"],
        ascending=[True, True, True, False, True],
        kind="mergesort",  # stable sort: same result on every run (csv_pos already breaks all ties)
    )
    out["slot"] = out.groupby("posting_id", sort=False).cumcount()
    return out.drop(columns="_side")


def order_changes(sorted_lines: pd.DataFrame) -> pd.Series:
    """True per posting_id if the CSV line order differs from the D9 order."""
    changed = sorted_lines["csv_pos"] != sorted_lines["slot"]
    return changed.groupby(sorted_lines["posting_id"]).any()


# ---------------------------------------------------------------- scaling (D10)
@dataclass
class Scaler:
    """optional log1p -> robust (median / IQR) -> min-max.

    D10 (25 Nov 2026): the scaler is part of the fitted pipeline and is fitted on the
    same fiscal year as the detector, so every year is scaled with its own parameters
    and ends in [0, 1]."""
    log1p: bool
    robust: bool
    median: float = 0.0
    iqr: float = 1.0
    zmin: float = 0.0
    zmax: float = 1.0

    def _pre(self, x):
        x = np.asarray(x, dtype=float)
        if self.log1p:
            x = np.log1p(x)
        if self.robust:
            x = (x - self.median) / self.iqr
        return x

    def fit(self, x_year):
        x = np.asarray(x_year, dtype=float)
        if self.log1p:
            x = np.log1p(x)
        if self.robust:
            q1, q3 = np.percentile(x, [25, 75])
            self.median = float(np.median(x))
            self.iqr = float(q3 - q1) if q3 > q1 else 1.0
        z = self._pre(x_year)
        self.zmin, self.zmax = float(z.min()), float(z.max())
        if self.zmax == self.zmin:
            self.zmax = self.zmin + 1.0
        return self

    def transform(self, x):
        return (self._pre(x) - self.zmin) / (self.zmax - self.zmin)


# ---------------------------------------------------------------- entry table
ENTRY_COLS = ["document_date", "entered_date", "user", "tax_rate",
              "promptly", "weekend", "nwh"]


def entry_table(lines: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per entry in meta order. The ENTRY_COLS are constant inside every
    posting_id (checked in notebook 02), so the first line is taken."""
    g = lines.groupby("posting_id", sort=False)
    # D13 guard: .first() is only correct if the field has one value per entry. On a ledger
    # with, e.g., several tax rates in one document this stops instead of keeping line 1.
    varying = [c for c in ENTRY_COLS if (g[c].nunique(dropna=False) > 1).any()]
    if varying:
        raise ValueError(f"not constant inside every entry (encode per line slot instead): {varying}")
    e = g[ENTRY_COLS].first()
    e = e.reindex(meta["posting_id"].to_numpy())
    e["fy"] = meta["fy"].to_numpy()
    e["n_lines"] = meta["n_lines"].to_numpy()

    ent = pd.to_datetime(e["entered_date"])
    doc = pd.to_datetime(e["document_date"])
    e["lag_days_raw"] = (ent.dt.normalize() - doc).dt.days.astype(float)
    t = ent.dt.hour + ent.dt.minute / 60 + ent.dt.second / 3600
    e["hour_t"] = t
    e["wday"] = ent.dt.dayofweek                              # Monday = 0
    e["nwh_dist_raw"] = np.maximum(0, WORK_START - t) + np.maximum(0, t - WORK_END)
    return e


# ---------------------------------------------------------------- encoding
def _onehot(values, categories):
    idx = pd.Categorical(values, categories=categories).codes
    if (idx < 0).any():
        raise ValueError("value outside the category list")
    n = len(values)
    return sparse.csr_matrix((np.ones(n), (np.arange(n), idx)), shape=(n, len(categories)))


def build_onehot(lines: pd.DataFrame, meta: pd.DataFrame):
    """Return (X csr float32, feature_info list, params dict, sorted lines, entry table).

    lines: raw posting lines (CSV), meta: meta.parquet. Numeric columns are scaled per
    fiscal year (D10); category lists come from the whole ledger so both years have the
    same columns (no labels involved).
    """
    sl = sort_lines(lines)
    ent = entry_table(lines, meta)
    n = len(meta)
    row_of = pd.Series(np.arange(n), index=meta["posting_id"].to_numpy())
    sl["row"] = row_of.reindex(sl["posting_id"]).to_numpy()
    if sl["row"].isna().any():
        raise ValueError("posting_id in CSV but not in meta")
    sl["row"] = sl["row"].astype(int)
    sl["fy"] = ent["fy"].to_numpy()[sl["row"]]

    accounts = sorted(lines["gl_account"].unique().tolist())
    users = sorted(lines["user"].unique().tolist())
    taxes = sorted(lines["tax_rate"].unique().tolist())

    # per-year scaling (D10): fit and transform inside each fiscal year
    years = sorted(ent["fy"].unique().tolist())
    amount_scaled = np.zeros(len(sl))
    lag_scaled = np.zeros(n)
    dist_scaled = np.zeros(n)
    scaler_params = {}
    for fy in years:
        m_line = (sl["fy"] == fy).to_numpy()
        m_ent = (ent["fy"] == fy).to_numpy()
        a_sc = Scaler(log1p=True, robust=True).fit(sl.loc[m_line, "amount"])
        l_sc = Scaler(log1p=True, robust=True).fit(ent.loc[m_ent, "lag_days_raw"])
        d_max = float(ent.loc[m_ent, "nwh_dist_raw"].max()) or 1.0
        amount_scaled[m_line] = a_sc.transform(sl.loc[m_line, "amount"].to_numpy())
        lag_scaled[m_ent] = l_sc.transform(ent.loc[m_ent, "lag_days_raw"].to_numpy())
        dist_scaled[m_ent] = ent.loc[m_ent, "nwh_dist_raw"].to_numpy() / d_max
        scaler_params[int(fy)] = {"amount": asdict(a_sc), "lag_days": asdict(l_sc),
                                  "nwh_dist": {"divide_by_year_max": d_max}}
    sl["amount_scaled"] = amount_scaled

    blocks, info = [], []

    def add(block, names, group, slot=None):
        blocks.append(sparse.csr_matrix(block))
        for nm in names:
            info.append({"name": nm, "group": group, "slot": slot})

    # --- per-slot block (group "lines")
    for s in range(MAX_LINES):
        part = sl[sl["slot"] == s]
        rows = part["row"].to_numpy()
        acct = np.full(n, PAD, dtype=object)
        acct[rows] = part["gl_account"].astype(str).to_numpy()
        side = np.full(n, PAD, dtype=object)
        side[rows] = part["cd_flag"].to_numpy()
        amt = np.zeros(n)
        amt[rows] = part["amount_scaled"].to_numpy()

        acct_cats = [str(a) for a in accounts] + [PAD]
        add(_onehot(acct, acct_cats), [f"s{s+1}_acct_{a}" for a in acct_cats], "lines", s + 1)
        add(_onehot(side, ["D", "C", PAD]), [f"s{s+1}_cd_{c}" for c in ["D", "C", PAD]], "lines", s + 1)
        add(amt.reshape(-1, 1), [f"s{s+1}_amount"], "lines", s + 1)

    # --- per-entry attributes (group "lines", D13: once per entry)
    # D13 (26 Nov): n_lines is determined by the account pattern and already visible in the PAD slots
    add(((ent["n_lines"].to_numpy() - 2) / (MAX_LINES - 2)).reshape(-1, 1), ["n_lines"], "n_lines")
    add(_onehot(ent["user"].to_numpy(), users), [f"user_{u}" for u in users], "lines")
    # D13: tax_rate is a deterministic function of the accounts -> own group, off
    add(_onehot(ent["tax_rate"].to_numpy(), taxes), [f"tax_{t:g}" for t in taxes], "tax_rate")

    # --- calendar flags (D4); promptly is a binning of lag_days -> own group, off
    add(((ent["promptly"].to_numpy() - 1) / 2).reshape(-1, 1), ["promptly"], "promptly")
    add(_onehot(ent["weekend"].to_numpy(), [0, 1, 2]),
        ["weekend_weekday", "weekend_sat", "weekend_sun"], "calendar_flags")
    add(ent["nwh"].to_numpy().reshape(-1, 1).astype(float), ["nwh"], "calendar_flags")

    # --- lag_days (D4)
    add(lag_scaled.reshape(-1, 1), ["lag_days"], "lag_days")

    # --- D11: checked and dropped (kept only so the data checks can be reproduced)
    add(dist_scaled.reshape(-1, 1), ["nwh_dist"], "nwh_dist")
    a = 2 * np.pi * ent["hour_t"].to_numpy() / 24
    add(np.c_[np.sin(a), np.cos(a)], ["hour_sin", "hour_cos"], "hour_cyc")
    w = 2 * np.pi * ent["wday"].to_numpy() / 7
    add(np.c_[np.sin(w), np.cos(w)], ["wday_sin", "wday_cos"], "wday_cyc")

    X = sparse.hstack(blocks, format="csr", dtype=np.float32)
    X.eliminate_zeros()
    params = {
        "scaling": "per fiscal year (D10)",
        "max_lines": MAX_LINES,
        "line_order": "D before C, then gl_account asc, then amount desc, then CSV position (D9)",
        "categories": {"gl_account": accounts, "user": users, "tax_rate": taxes},
        "scalers": scaler_params,
        "working_window": [WORK_START, WORK_END],
    }
    return X, info, params, sl, ent


# ---------------------------------------------------------------- selection / IO
def select_columns(X, feature_info, groups: dict):
    """Keep only the columns whose group is switched on. Returns (X_sel, names)."""
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"unknown groups: {unknown}")
    keep = [i for i, f in enumerate(feature_info) if groups.get(f["group"], False)]
    return X[:, keep], [feature_info[i]["name"] for i in keep]


# Feature settings used in the experiments (main setting + ablations, Ch. 4). Defined once here
# so that notebooks 02-10 and the result tables use the same names.
ABLATIONS = {
    "main": FEATURE_GROUPS_MAIN,
    "main+nwh_dist": {**FEATURE_GROUPS_MAIN, "nwh_dist": True},
    "main+hour_cyc": {**FEATURE_GROUPS_MAIN, "hour_cyc": True},
    "main+wday_cyc": {**FEATURE_GROUPS_MAIN, "wday_cyc": True},
    "main+promptly": {**FEATURE_GROUPS_MAIN, "promptly": True},
    "no_time": FEATURE_GROUPS_NO_TIME,
}


# Encoding version written into results/runs.csv. v2 = decisions review of 25 Nov 2026
# (promptly and tax_rate dropped, per-year scalers); v3 = redundancy check of 26 Nov
# (n_lines dropped); rows without a version are v1.
ENCODING_VERSION = "v3"


def groups_label(groups: dict) -> str:
    """Short text for the feature_groups column in results/runs.csv, e.g. 'lines+calendar_flags+lag_days@v3'."""
    return "+".join(g for g in ALL_GROUPS if groups.get(g, False)) + f"@{ENCODING_VERSION}"


def save_encoding(X, feature_info, params, x_path, json_path, posting_ids):
    """Write the matrix (npz) and, in the JSON, the columns, fitted parameters, the encoding
    version and the row order (posting_id of every row), so that later notebooks can check
    that they use the file they expect (C19, review of 26 Nov 2026)."""
    posting_ids = [str(p) for p in posting_ids]
    if len(posting_ids) != X.shape[0]:
        raise ValueError("one posting_id per row is needed")
    sparse.save_npz(x_path, X)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"encoding_version": ENCODING_VERSION, "n_rows": X.shape[0],
                   "features": feature_info, "params": params, "posting_ids": posting_ids},
                  f, indent=1)


def load_encoding(x_path, json_path, meta: pd.DataFrame | None = None):
    """Read the encoding. Stops if the file was written by another encoding version, or if
    `meta` is given and its posting_id order differs from the row order of the matrix."""
    X = sparse.load_npz(x_path).tocsr()
    with open(json_path, encoding="utf-8") as f:
        info = json.load(f)
    version = info.get("encoding_version", "v1/v2 (not stored)")
    if version != ENCODING_VERSION:
        raise ValueError(f"{x_path} is encoding {version}, the code is {ENCODING_VERSION}: rerun notebook 02")
    if X.shape[0] != info["n_rows"]:
        raise ValueError("matrix and JSON do not belong together")
    if meta is not None and list(meta["posting_id"].astype(str)) != info["posting_ids"]:
        raise ValueError("row order of the encoding differs from meta.parquet")
    return X, info["features"], info["params"]

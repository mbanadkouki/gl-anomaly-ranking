"""Figures for the feature-engineering checks (notebook 02, Ch. 3 of the thesis).

Every figure supports one decision in the Decisions sheet of Thesis_Execution_Plan.xlsx.
Figures that use labels (anomaly rates) are drawn on FY1 only, so that no label of the
reporting year (FY2) enters a feature decision (D6). Figures without labels (redundancy
check, duplicates) may use the whole ledger.

Style: one sequential blue for magnitude, grey for context, thin bars, recessive grid,
one y-axis per panel. Files are written as PNG (300 dpi) and PDF to results/figures/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

BLUE = "#257de0"        # main series
GREY = "#b4b4b4"        # context / comparison
INK = "#191919"
INK2 = "#666666"
GRID = "#e6e6e6"

matplotlib.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name: str, out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    return out / f"{name}.png"


def _pct_axis(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=None))


def _base_line(ax, base: float, where: str = "left"):
    ax.axhline(base, color=INK2, lw=0.8, ls="--")
    x, ha, dx = (0, "left", 2) if where == "left" else (1, "right", -2)
    ax.annotate(f"base rate {base:.2%}", xy=(x, base), xycoords=("axes fraction", "data"),
                xytext=(dx, 3), textcoords="offset points", ha=ha, va="bottom",
                fontsize=8, color=INK2)


# ------------------------------------------------------------------ tables behind the figures
def entry_frame(lines: pd.DataFrame, meta: pd.DataFrame, ent: pd.DataFrame,
                sorted_lines: pd.DataFrame) -> pd.DataFrame:
    """One row per entry (meta order) with the fields the checks need."""
    pat = (sorted_lines.assign(tok=sorted_lines["cd_flag"] + sorted_lines["gl_account"].astype(str))
           .groupby("posting_id", sort=False)["tok"].agg(" ".join))
    g = lines.groupby("posting_id", sort=False)
    out = pd.DataFrame({
        "posting_id": meta["posting_id"].to_numpy(),
        "fy": meta["fy"].to_numpy(),
        "label": meta["label"].to_numpy(),
        "pattern": pat.reindex(meta["posting_id"]).to_numpy(),
        "n_lines": meta["n_lines"].to_numpy(),
        "user": ent["user"].to_numpy(),
        "tax_rate": ent["tax_rate"].to_numpy(),
        "promptly": ent["promptly"].to_numpy(),
        "weekend": ent["weekend"].to_numpy(),
        "nwh": ent["nwh"].to_numpy(),
        "lag_days": ent["lag_days_raw"].to_numpy(),
        "hour": np.floor(ent["hour_t"].to_numpy()).astype(int),
        "weekday": ent["wday"].to_numpy(),
        "nwh_dist": ent["nwh_dist_raw"].to_numpy(),
        "amount": g["amount"].max().reindex(meta["posting_id"]).to_numpy(),
        "text": g["text"].first().reindex(meta["posting_id"]).to_numpy(),
    })
    return out


def rate_table(df: pd.DataFrame, by) -> pd.DataFrame:
    t = df.groupby(by, observed=True)["label"].agg(entries="size", anomalies="sum")
    t["rate"] = t["anomalies"] / t["entries"]
    return t


def functional_dependencies(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pairs A -> B where every value of A has exactly one value of B (B not constant)."""
    rows = []
    for a in cols:
        for b in cols:
            if a != b and df[b].nunique() > 1 and df.groupby(a)[b].nunique().max() == 1:
                rows.append({"determines": a, "determined": b,
                             "values of A": df[a].nunique(), "values of B": df[b].nunique()})
    return pd.DataFrame(rows)


def cramers_v(x, y) -> float:
    t = pd.crosstab(x, y)
    if min(t.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))


def association_matrix(df: pd.DataFrame, cols: list[str], numeric_bins: dict[str, int]) -> pd.DataFrame:
    """Cramér's V for every pair; numeric columns are cut into quantile bins first."""
    d = df[cols].copy()
    for c, q in numeric_bins.items():
        d[c] = pd.qcut(d[c], q=q, duplicates="drop")
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            m.loc[a, b] = m.loc[b, a] = cramers_v(d[a], d[b])
    return m


# ------------------------------------------------------------------ figures
def fig_text_vs_price(df: pd.DataFrame, items: list[str]):
    """D1: anomaly rate per item text next to the item's median entry amount (two panels, one axis each)."""
    d = df[df["text"].isin(items)]
    t = rate_table(d, "text")
    t["median_amount"] = d.groupby("text")["amount"].median()
    t = t.sort_values("median_amount")
    base = df["label"].mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    y = np.arange(len(t))
    a1.barh(y, t["median_amount"], color=GREY, height=0.6)
    a1.set_yticks(y, t.index)
    a1.set_xlabel("median entry amount (EUR)")
    a1.set_title("Price of the item", loc="left")
    a1.grid(axis="x"); a1.grid(axis="y", visible=False)
    a2.barh(y, t["rate"], color=BLUE, height=0.6)
    a2.axvline(base, color=INK2, lw=0.8, ls="--")
    a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    a2.set_xlabel("anomaly rate")
    a2.set_title("Anomaly rate", loc="left")
    a2.grid(axis="x"); a2.grid(axis="y", visible=False)
    a2.annotate("base rate", xy=(base, -0.75), xytext=(3, 0), textcoords="offset points",
                fontsize=8, color=INK2, va="center")
    a2.set_ylim(-1.1, len(t) - 0.4)
    fig.suptitle("The item text follows the price: the most expensive item has the highest anomaly rate",
                 x=0.0, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    return fig, t


def fig_rate_by_lag(df: pd.DataFrame):
    """D4: anomaly rate by delay between document and entry date."""
    edges = [-1, 4, 9, 14, 19, 29, 44, 59, 89, 10_000]
    labels = ["0–4", "5–9", "10–14", "15–19", "20–29", "30–44", "45–59", "60–89", "90+"]
    t = rate_table(df.assign(lag_bin=pd.cut(df["lag_days"], edges, labels=labels)), "lag_bin")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(t))
    ax.bar(x, t["rate"], color=BLUE, width=0.6)
    ax.set_xticks(x, t.index)
    ax.set_xlabel("delay between document date and entry date (days)")
    ax.set_ylabel("anomaly rate")
    _pct_axis(ax)
    top = t["rate"].max() * 1.18
    ax.set_ylim(0, top)
    for xi, txt in ((1.5, "10 days"), (4.5, "30 days")):     # bin edges at 10 and 30 days
        ax.axvline(xi, color=INK2, lw=0.8, ls=":")
        ax.annotate(txt, xy=(xi, top), xytext=(-3, -10), textcoords="offset points",
                    ha="right", fontsize=8, color=INK2)
    _base_line(ax, df["label"].mean(), "left")
    ax.set_title("Risk rises with the delay, in steps at 10 and 30 days", loc="left")
    fig.tight_layout()
    return fig, t


def fig_rate_by_hour(df: pd.DataFrame, work_start: int = 8, work_end: int = 18):
    """D11: anomaly rate per entry hour; inside vs outside the working window."""
    t = rate_table(df, "hour")
    inside = (t.index >= work_start) & (t.index < work_end)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(t.index[inside], t["rate"][inside], color=GREY, width=0.7, label="working hours (08:00–17:59)")
    ax.bar(t.index[~inside], t["rate"][~inside], color=BLUE, width=0.7, label="outside working hours (nwh = 1)")
    ax.set_xticks(t.index, [f"{h:02d}" for h in t.index])
    ax.set_xlabel("hour of entry")
    ax.set_ylabel("anomaly rate")
    _pct_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    ax.set_title("Flat inside working hours, high outside", loc="left")
    fig.tight_layout()
    return fig, t


def fig_rate_by_distance(df: pd.DataFrame):
    """D11: inside nwh = 1, the anomaly rate does not grow with the distance from working hours."""
    d = df[df["nwh"] == 1]
    edges = [-0.01, 0.5, 1, 2, 3, 4, 6]
    labels = ["0–0.5", "0.5–1", "1–2", "2–3", "3–4", "4–6"]
    t = rate_table(d.assign(dist=pd.cut(d["nwh_dist"], edges, labels=labels)), "dist")
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    x = np.arange(len(t))
    ax.bar(x, t["rate"], color=BLUE, width=0.6)
    ax.set_xticks(x, t.index)
    ax.set_xlabel("hours outside the working window (entries with nwh = 1)")
    ax.set_ylabel("anomaly rate")
    _pct_axis(ax)
    ax.set_title("No gradient with the distance from working hours", loc="left")
    fig.tight_layout()
    return fig, t


def fig_rate_by_weekday(df: pd.DataFrame):
    """D11: anomaly rate per weekday (weekend flag in blue)."""
    t = rate_table(df, "weekday")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    cols = [GREY if d < 5 else BLUE for d in t.index]
    ax.bar(t.index, t["rate"], color=cols, width=0.6)
    ax.set_xticks(t.index, [names[d] for d in t.index])
    ax.set_ylabel("anomaly rate")
    _pct_axis(ax)
    _base_line(ax, df["label"].mean())
    ax.set_title("Weekdays alike; the weekend is what matters", loc="left")
    fig.tight_layout()
    return fig, t


def fig_association_heatmap(m: pd.DataFrame, title: str):
    """Redundancy check: Cramér's V between entry-level fields (no labels)."""
    n = len(m)
    fig, ax = plt.subplots(figsize=(0.62 * n + 1.6, 0.55 * n + 1.0))
    vals = m.values.astype(float).copy()
    np.fill_diagonal(vals, np.nan)                      # the diagonal is 1 by definition: leave it blank
    cmap = matplotlib.colormaps["Blues"].copy(); cmap.set_bad("#f2f2f2")
    im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n), m.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), m.index)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if i != j:
                v = m.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Cramér's V", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def fig_duplicate_groups(X_dense: np.ndarray, k_marks=(40, 50), title: str = ""):
    """D15: size of the groups of identical encoded vectors (log scale)."""
    _, counts = np.unique(X_dense, axis=0, return_counts=True)
    sizes, freq = np.unique(counts, return_counts=True)
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.bar(sizes, freq, color=BLUE, width=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("size of a group of identical vectors")
    ax.set_ylabel("number of groups (log)")
    for k in k_marks:
        ax.axvline(k, color=INK2, lw=0.8, ls=":")
        ax.annotate(f"k = {k}", xy=(k, ax.get_ylim()[1]), xytext=(3, -10), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.set_title(title or f"Largest group = {counts.max()}: LOF needs k above it", loc="left")
    fig.tight_layout()
    return fig, pd.Series(freq, index=sizes, name="groups")

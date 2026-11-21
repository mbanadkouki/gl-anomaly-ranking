"""Load the DIB synthetic general ledger and build the journal-entry table.

Unit of analysis = journal entry (posting_id). Labels, markings and all rule
columns are constant within an entry (checked in validate_jets.py).
"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "journal_entries_4203.csv"

# Precomputed columns of the dataset. top_n / high_cash define the JET flags only;
# promptly / weekend / nwh are also model features (decision D4, src/encode.py).
PRECOMPUTED_RULE_COLS = ["promptly", "weekend", "nwh", "top_n", "high_cash"]
LABEL_COLS = ["label", "marking"]
CASH_ACCOUNT = 1000          # Cash-in-hand
WORK_START, WORK_END = 8, 18  # working hours 08:00-17:59 (matches dataset nwh)


def load_lines(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=["document_date", "posting_date", "entered_date"])
    df["fy"] = df["posting_date"].dt.year
    df["line_no"] = df.groupby("posting_id").cumcount()
    return df


def build_entries(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per journal entry with the raw-derived fields the JETs and features need."""
    g = lines.groupby("posting_id", sort=False)
    first = g.first()
    e = pd.DataFrame({
        "fy": first["fy"],
        "document_date": first["document_date"],
        "entered_date": first["entered_date"],
        "user": first["user"],
        "n_lines": g.size(),
        "max_amount": g["amount"].max(),
        "label": g["label"].max(),
        "marking": g["marking"].max(),
    })
    e["lag_days"] = (e["entered_date"].dt.normalize() - e["document_date"]).dt.days
    e["hour"] = e["entered_date"].dt.hour
    e["weekday"] = e["entered_date"].dt.dayofweek          # 0=Mon ... 6=Sun
    cash_out = lines[(lines.gl_account == CASH_ACCOUNT) & (lines.cd_flag == "C")]
    e["cash_out_amount"] = cash_out.groupby("posting_id")["amount"].max().reindex(e.index).fillna(0.0)
    return e

"""Tests for the notebook-02 review changes of 26 Nov 2026 (entry amount, entry-level guard,
versioned save / load with row order, Wilson interval)."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src import encode as fe
from src.figures import wilson_interval
from src.load import entry_amount


def _lines():
    # e1: payroll-like entry with two debits and two credits (max line != entry amount)
    return pd.DataFrame({
        "posting_id": ["e1"] * 4 + ["e2"] * 2,
        "cd_flag": ["D", "D", "C", "C", "D", "C"],
        "amount": [60.0, 40.0, 70.0, 30.0, 5.0, 5.0],
    })


def test_entry_amount_is_sum_of_debits():
    s = entry_amount(_lines())
    assert s["e1"] == 100.0          # not the largest line (70)
    assert s["e2"] == 5.0


def test_entry_table_stops_on_mixed_entry_fields():
    n = 4
    lines = pd.DataFrame({
        "posting_id": ["a", "a", "b", "b"],
        "document_date": pd.to_datetime(["2023-01-02"] * n),
        "entered_date": pd.to_datetime(["2023-01-03 09:00"] * n),
        "user": ["Alice"] * n,
        "tax_rate": [0.19, 0.07, 0.19, 0.19],      # two rates in entry a
        "promptly": [1] * n, "weekend": [0] * n, "nwh": [0] * n,
    })
    meta = pd.DataFrame({"posting_id": ["a", "b"], "fy": [2023, 2023], "n_lines": [2, 2]})
    with pytest.raises(ValueError, match="tax_rate"):
        fe.entry_table(lines, meta)


def test_save_load_checks_version_and_row_order(tmp_path):
    X = sparse.csr_matrix(np.eye(3, dtype=np.float32))
    info = [{"name": f"c{i}", "group": "lines", "slot": None} for i in range(3)]
    xp, jp = tmp_path / "X.npz", tmp_path / "X.json"
    fe.save_encoding(X, info, {}, xp, jp, posting_ids=["a", "b", "c"])
    X2, info2, _ = fe.load_encoding(xp, jp, meta=pd.DataFrame({"posting_id": ["a", "b", "c"]}))
    assert (X2 != X).nnz == 0 and info2 == info
    with pytest.raises(ValueError, match="row order"):
        fe.load_encoding(xp, jp, meta=pd.DataFrame({"posting_id": ["b", "a", "c"]}))


def test_wilson_interval_contains_rate_and_handles_zero():
    lo, hi = wilson_interval([0, 5, 50], [25, 100, 100])
    p = np.array([0, 0.05, 0.5])
    assert (lo <= p).all() and (p <= hi).all() and lo[0] == 0 and hi[0] > 0

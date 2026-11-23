"""Tests for the sentence construction in src/embed.py (no model needed).

Run from the project root:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import embed  # noqa: E402

# entry a_1 as written in the CSV (3400 before 1576) and a second entry e_2
LINES = pd.DataFrame({
    "posting_id": ["a_1", "a_1", "a_1", "e_2", "e_2"],
    "gl_account": [3400, 1576, 1600, 1600, 1100],
    "gl_account_name": ["Cost of merchandise, 19% input tax", "Deductible input tax, 19%", "Trade payables",
                        "Trade payables", "Bank (Postbank)"],
    "amount": [112.42, 21.36, 133.78, 50.0, 50.0],
    "cd_flag": ["D", "D", "C", "D", "C"],
})
META = pd.DataFrame({"posting_id": ["e_2", "a_1"]})        # meta order differs from CSV order


def test_tc_sentence_uses_d9_order_names_and_semicolons():
    s = embed.build_sentences(LINES, META, "tc")
    assert s.iloc[1] == ("Debit Deductible input tax, 19%; Debit Cost of merchandise, 19% input tax; "
                         "Credit Trade payables.")
    assert s.iloc[0] == "Debit Trade payables; Credit Bank (Postbank)."


def test_bak_sentence_is_bakumenko_format_with_codes():
    s = embed.build_sentences(LINES, META, "bak")
    assert s.iloc[1] == "Account_DC: 1576_D, Account_DC: 3400_D, Account_DC: 1600_C"


def test_unique_index_maps_every_entry_back():
    s = pd.Series(["x", "y", "x", "x"])
    texts, idx = embed.unique_index(s)
    assert texts == ["x", "y"]                              # most frequent first
    assert [texts[i] for i in idx] == s.tolist()


def test_non_account_columns_drop_only_slot_account_and_side():
    info = [{"name": "s1_acct_1600", "group": "lines", "slot": 1},
            {"name": "s1_cd_D", "group": "lines", "slot": 1},
            {"name": "s1_amount", "group": "lines", "slot": 1},
            {"name": "user_Max", "group": "lines", "slot": None},
            {"name": "nwh_dist", "group": "nwh_dist", "slot": None}]
    groups = {"lines": True, "nwh_dist": False}
    assert embed.non_account_columns(info, groups) == [2, 3]

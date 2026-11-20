"""The four journal entry tests (JETs) of Gronewald et al. (2024), implemented from raw fields.

J1 top-n amounts       : entry max line amount >= the n-th highest per fiscal year (ties included)
J2 high cash disbursement: credit on cash-in-hand (1000) above EUR 1,500
J3 delayed entry       : entered date - document date >= 10 days
J4 weekend / non-working hours: entered on Sat/Sun or outside 08:00-17:59
"""
import pandas as pd
from .load import WORK_START, WORK_END

TOP_N = 1000
HIGH_CASH_EUR = 1500
DELAY_DAYS = 10
JET_COLS = ["J1", "J2", "J3", "J4"]


def apply_jets(e: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    out = pd.DataFrame(index=e.index)
    nth = e.groupby("fy")["max_amount"].transform(lambda s: s.nlargest(top_n).iloc[-1])
    out["J1"] = (e["max_amount"] >= nth).astype(int)
    out["J2"] = (e["cash_out_amount"] > HIGH_CASH_EUR).astype(int)
    out["J3"] = (e["lag_days"] >= DELAY_DAYS).astype(int)
    out["J4"] = ((e["weekday"] >= 5) | (e["hour"] < WORK_START) | (e["hour"] >= WORK_END)).astype(int)
    out["h"] = out[JET_COLS].sum(axis=1)          # number of rules hit
    out["in_jet"] = (out["h"] > 0).astype(int)
    return out

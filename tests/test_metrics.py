"""Hand-checkable tests for src/metrics.py.

Run from the project root:  python -m pytest tests -q

Toy ledger: 10 entries, 3 anomalies. Scores are all different, so the review order is fixed:

    review position   1    2    3    4    5    6    7    8    9    10
    entry index       0    1    2    3    4    5    6    7    8    9
    score            .95  .90  .80  .70  .60  .50  .40  .30  .20  .10
    anomaly?          Y    .    .    Y    .    .    Y    .    .    .

By hand:
    P@1 = 1/1 = 1.0      P@4 = 2/4 = 0.5      P@5 = 2/5 = 0.4      P@10 = 3/10 = 0.3
    R@4 = 2/3            FP@4 = 2
    AP  = (1/1 + 2/4 + 3/7) / 3 = 0.642857...
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as m  # noqa: E402

SCORES = np.array([.95, .90, .80, .70, .60, .50, .40, .30, .20, .10])
LABELS = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0])


def test_review_order_is_score_descending():
    assert m.review_order(SCORES, seed=0).tolist() == list(range(10))


@pytest.mark.parametrize("k, precision, recall, fp", [
    (1, 1.0, 1 / 3, 0),
    (4, 0.5, 2 / 3, 2),
    (5, 0.4, 2 / 3, 3),
    (10, 0.3, 1.0, 7),
])
def test_metrics_at_k_by_hand(k, precision, recall, fp):
    r = m.metrics_at_k(m.review_order(SCORES, 0), LABELS, k)
    assert r["precision"] == pytest.approx(precision)
    assert r["recall"] == pytest.approx(recall)
    assert r["fp"] == fp


def test_average_precision_by_hand():
    ap = m.average_precision(m.review_order(SCORES, 0), LABELS)
    assert ap == pytest.approx((1 / 1 + 2 / 4 + 3 / 7) / 3)


def test_entries_not_on_the_list_are_never_reviewed():
    # entries 6-9 are outside the list (NaN); anomaly 6 can no longer be found
    s = SCORES.copy()
    s[6:] = np.nan
    order = m.review_order(s, 0)
    assert order.tolist() == [0, 1, 2, 3, 4, 5]
    r = m.metrics_at_k(order, LABELS, 10)          # budget larger than the list
    assert r["n_reviewed"] == 6
    assert r["recall"] == pytest.approx(2 / 3)       # denominator = all 3 anomalies
    assert m.average_precision(order, LABELS) == pytest.approx((1 / 1 + 2 / 4) / 3)


def test_ties_are_random_but_reproducible():
    s = np.array([1, 1, 1, 1, 0.5])                  # four entries tied at the top
    orders = {tuple(m.review_order(s, seed)) for seed in range(20)}
    assert len(orders) > 1                           # different seeds -> different tie order
    assert m.review_order(s, 7).tolist() == m.review_order(s, 7).tolist()
    assert all(o[-1] == 4 for o in orders)           # the lower score is always last


def test_ties_are_not_broken_by_row_order():
    s = np.ones(1000)
    first = [m.review_order(s, seed)[0] for seed in range(5)]
    assert first != [0] * 5


def test_budgets_are_the_preprint_set():
    b = m.budgets()
    assert b == {"k25": 25, "k50": 50, "k100": 100, "k250": 250, "k500": 500,
                 "k1000": 1000, "k2000": 2000}


def test_evaluate_and_summarise_shapes():
    per_seed = m.evaluate(SCORES, LABELS, seeds=(0, 1), budget_map={"k4": 4})
    assert len(per_seed) == 2
    summ = m.summarise(per_seed)
    assert summ.loc[0, "precision_mean"] == pytest.approx(0.5)
    assert summ.loc[0, "precision_sd"] == pytest.approx(0.0)   # no ties -> same for every seed


def test_tag_puts_labels_first():
    per_seed = m.evaluate(SCORES, LABELS, seeds=(0,), budget_map={"k4": 4})
    t = m.tag(per_seed, method="x", fy=2024)
    assert list(t.columns[:2]) == ["method", "fy"] and (t["method"] == "x").all()

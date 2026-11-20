"""Week 1: profile entries and validate J1-J4 against the dataset's precomputed columns."""
import pandas as pd
from src.load import load_lines, build_entries, PRECOMPUTED_RULE_COLS
from src.jets import apply_jets, JET_COLS

lines = load_lines()
g = lines.groupby("posting_id")
assert (g[PRECOMPUTED_RULE_COLS + ["label", "marking"]].nunique() == 1).all().all()
e = build_entries(lines)
j = apply_jets(e)
ref = g[PRECOMPUTED_RULE_COLS].first().reindex(e.index)
ref_jets = pd.DataFrame({"J1": ref.top_n, "J2": ref.high_cash, "J3": (ref.promptly > 1).astype(int),
                         "J4": ((ref.weekend > 0) | (ref.nwh == 1)).astype(int)})

anom = e.label == 1
rows = []
for r in JET_COLS:
    pop = j[r] == 1
    rows.append({"rule": r, "population": int(pop.sum()), "dataset_population": int(ref_jets[r].sum()),
                 "disagreements": int((j[r] != ref_jets[r]).sum()),
                 "anomalies_in_pop": int((pop & anom).sum()),
                 "precision_%": round(100 * (pop & anom).sum() / pop.sum(), 2)})
u = j.in_jet == 1
rows.append({"rule": "Union", "population": int(u.sum()), "dataset_population": int((ref_jets.sum(axis=1) > 0).sum()),
             "disagreements": int((u.astype(int) != (ref_jets.sum(axis=1) > 0).astype(int)).sum()),
             "anomalies_in_pop": int((u & anom).sum()), "precision_%": round(100 * (u & anom).sum() / u.sum(), 2)})
val = pd.DataFrame(rows)

by_mark = pd.DataFrame({"n": e[anom].groupby("marking").size()})
for r in JET_COLS + ["in_jet"]:
    by_mark[r] = j[anom].groupby(e.marking[anom])[r].sum()
by_mark["recall_union_%"] = (100 * by_mark.in_jet / by_mark.n).round(1)

prof = pd.DataFrame({
    "entries": e.groupby("fy").size(), "anomalies": e.groupby("fy").label.sum(),
    "jet_union": j.groupby(e.fy).in_jet.sum(),
    "anoms_in_jet": (j.in_jet & e.label).groupby(e.fy).sum()})
hist = j[u].h.value_counts().sort_index()

with open("results/week01_jet_validation.md", "w") as f:
    f.write("# Week 01 - JET validation\n\n## Rule populations (entries)\n\n" + val.to_markdown(index=False))
    f.write("\n\n## Anomalies by marking captured by each rule\n\n" + by_mark.to_markdown())
    f.write("\n\n## Per fiscal year\n\n" + prof.to_markdown())
    f.write("\n\n## Rules hit inside the JET union\n\n" + hist.to_frame("entries").to_markdown())
e.join(j).to_pickle("data/entries.pkl")
print(open("results/week01_jet_validation.md").read())

> **ARCHIVED 27 Nov 2026** — copied from the Thesis project (claude/notebook02_review_changes.md) before it was deleted there. The rewrite is committed (repo commit 424b74d). Cell numbers C1–C21 refer to the notebook **before** the rewrite (commit b08ec5e); e.g. old 02_18 (duplicates) is cell_02_24 now. Only X1–X5 (Excel rows) were still open on 27 Nov.

> **Last updated: 2026-11-26 19:30 (Berlin)** — **applied** in the rewrite (local branch `nb02-rewrite`, commit 22662e0, patch sent in chat); Mohsen runs 01 + 02 on the laptop and commits

# Notebook 02 review — collected changes (temporary working list)

## Status (26 Nov, 19:30)
- Applied: C1–C8, C10–C21, N1–N4, N6 (code cells run as scripts here; 19 tests pass; notebook 03 run on a copy still works: LOF AP FY2024 0.144, p_2 rank LOF 212 / IF 1,918).
- Not applied (optional): C7 robust step kept for comparability (text W6 says it has no effect); N5 category lists per year.
- Still open for Mohsen: run 01 and 02 on the laptop (outputs are not stored in the notebooks — no Jupyter / pyarrow in the Claude workspace), commit, rerun 03 (results now v3 + new save format); Excel rows X1–X5; notebook-03 / 04 points listed in chat (duplicate check only in 02, path to the plan in 04 — done in the first cell).

Collected during the cell-by-cell review of `notebooks/02_encoding_onehot.ipynb` (repo commit b08ec5e, 26 Nov 2026). Applied in **one rewrite** after the review is finished (new narrative order agreed on 26 Nov, see bottom). Delete this file after the rewrite is committed.

Rule (Mohsen, 26 Nov): results matter for the thesis, run time does not — no changes for speed only.

Labels: **[code]** change in `src/`, **[nb]** change in the notebook, **[text]** affects thesis text (see `thesis_writing_snippets.md`), **[xlsx]** Mohsen updates `Thesis_Execution_Plan.xlsx`.

## Per cell

| # | Cell | Change | Type |
|---|---|---|---|
| C1 | 02_01 | Markdown: say where to download the CSV and that it goes to `data/raw/`; notebook 01 must run first (`meta.parquet`). Optional: `pip install -e .` instead of `sys.path.insert`. | nb |
| C2 | 02_02 | Comment: the set check does not check row order; row order is guaranteed by `build_onehot` (`row_of.reindex`). Optional assert: `fy` of lines and meta agree per entry. | nb |
| C3 | 02_03 | Comment and Series name: "entries with more than one distinct value"; use `nunique(dropna=False)`; build the column list from `fe.ENTRY_COLS + ["gl_account", "cd_flag", "amount"]`; add `posting_date`, `text`, `label`, `marking` (`text` is taken with `.first()` in `figures.entry_frame`). | nb |
| C4 | 02_04 | Move the constant-inside-entry check into `src/encode.py` `entry_table` (raise ValueError before `.first()`); keep the notebook cell as display. Shorten comment to "expected: 0 for all entry-level fields". | code, nb |
| C5 | 02_05 | Comment "descriptive only (whole ledger); no feature decision uses this table" (D6); rename variable `pattern` → `cd_sequence` (also in 02_08). | nb, text (W2, W3) |
| C6 | 02_06 | Show `gl_account_name` and `text`, drop constant flag columns; comment with the corrected reading of p_2 (accrual settlement, unique account 2020). | nb, text (W5) |
| C7 | 02_07 | Robust step in `Scaler` is redundant before min-max (W6): keep for comparability or remove; text must say log1p does the work. Note `weekend` (3 one-hot columns, distance √2) weighs more than `nwh` (distance 1). `mergesort` comment: reproducibility comes from `csv_pos`, not from the sort algorithm. | code (optional), text (W6) |
| C8 | 02_08 / 02_09 | Markdown: the 50 % order changes do **not** mean that identical content gets different vectors in this ledger (02_09: each of 29 patterns stored in one order); D9 aligns slots across patterns (the 60 credit-first entries are reversals of purchases) and is not neutral (purchase a_1 vs reversal r_1: d 2.45 → 2.83 in the account+side block). | nb, text (W9) |
| C10 | 02_10 | **Agreed:** one cell showing purchase `a_1` and its reversal `r_1` side by side, CSV order and D9 order, plus the account+side distance in both orders. Pick the entries by id. | nb, text (W9) |
| C11 | 02_11 | Printed text says "entries" but the code counts (entry, side, account) groups; count `posting_id`s instead. Result 0 either way. | nb |
| C12 | 02_12 | Split into **code checks** (a–d) and **data checks** (e: `promptly` = binning of `lag_days` → label-free evidence for D4; f: weekend vs weekday); loop (c) over all four slot amounts; add NaN/inf check, one `user_*` / `weekend_*` per row, row-alignment spot check (p_2 only row with slot 4 filled; a_1 has `s1_acct_1576`); markdown: amount 0 = PAD and also the year minimum, PAD columns keep them apart. | nb |
| C13 | 02_13 | **Agreed (26 Nov): entry amount = sum of debit lines** (standard definition; equal to max line for all 102,429 entries here, so no number changes). Change `figures.entry_frame` (`amount`) and `load.build_entries` (`max_amount` → rename `entry_amount`, update `jets.py` J1 own implementation and notebook 01 if it shows the column). Markdown in 02_13: raw values for figures; `pattern` = booking pattern (W2). In the new order, build `ent` / `sorted_lines` with `fe.entry_table` / `fe.sort_lines` before `build_onehot`. | code, nb, text (W10) |
| C14 | 02_14 | (1) Markdown: `pattern` stands for the slot block (one-to-one after D9); the **dependent** field is dropped, not the determining one. (2) Add `hour` → `nwh` and `weekday` → `weekend` to the dependency list (both hold); if `hour_cyc` / `wday_cyc` are switched on in an ablation, `nwh` / `weekend` become partly redundant. (3) Markdown: heatmap is for redundancy only; low V ≠ useless (`user` vs pattern V = 0.14, but 21 of 29 patterns single-user, 638 entries). (4) `weekend` vs pattern / `n_lines` V = 0.30 = generator structure (30 % of 2-line entries on weekends). | nb, text (W12) |
| C15 | 02_15 | (1) Title → "The expensive items carry the higher anomaly rates" (not monotonic: Gardening Gloves 0.60 % > mid-priced 0.29–0.48 %). (2) New small cell after it: item × amount check (0 of 845 anomalies outside their item's normal amount range; W11). (3) Median entry amount = sum of debit lines (C13; same numbers). | nb, text (W11) |
| C16 | 02_16 | (1) Title / markdown: the steps at 10 and 30 days come from the generator (M3: 60 of 95 FY2023 anomalies with 10–29 days; M4: 60 of 62 with 30+ days) — new small cell printing the M-type counts per delay class. (2) Add Wilson 95 % intervals (or entries per bar) — several bins have only 32–196 entries; the rate is not monotonic (0–4 > 5–9; 15–19 > 20–29). (3) D4 text: drop or weaken "bin edges are arbitrary for a model" — in this dataset the edges are the generator's own; the reason to drop `promptly` is the functional dependency (W12). | nb, text (W13), xlsx |
| C17 | 02_17 | (1) The cell prints only the distance table; print the hour and weekday tables too (the thesis quotes them: FY2023 hour 08–16 0.54–0.94 %, hours 05–07 and 18–23 4.5–15.9 %; hour 17: 25 entries, 0 anomalies; weekday Mon–Fri 0.39–0.80 %, Sat 1.77 %, Sun 2.35 %). (2) Wilson intervals here too — the distance bins have 34–128 entries; the closest bin (0–0.5 h) has the highest rate (16.7 %, 84 entries), which is the opposite of a "further = riskier" gradient, but do not over-read it (wide interval). (3) Markdown: Sat and Sun differ (1.8 % vs 2.4 %) → the 3-category `weekend` one-hot keeps them apart; weekday sin/cos adds nothing beyond it. | nb, text |
| C18 | 02_18 | (1) Printed "groups of size > 1: 9280" counts groups, not entries — print distinct vectors, entries sharing their vector (36,137 = 71 %) and the largest group. (2) Run for **both** years (no labels; LOF also runs on FY2) and for the three settings main / no-time / all (largest group 36 / 187 / 6) — the D15 constraint on k depends on the setting. (3) `k_marks=(40, 50)`: **corrected** — 40 is the edge step that notebook 03 tests next to the best grid value (`detectors.edge_extension`, chosen on FY2023 in the v3 run); the figure now marks 40, 50, 100, 200. (4) Markdown: largest group = recurring payroll (wage tax); "all groups" has almost no duplicates because the exact entry time makes entries unique. (5) Markdown: dropping the dependent fields (v2 → v3) leaves 24,207 distinct vectors unchanged → evidence for W12. | nb, text (W15, W12) |
| C19 | 02_19 / save | (1) Store `encoding_version` (`fe.ENCODING_VERSION`) and the number of rows in `onehot_features.json`; `load_encoding` should check the version against the code — today `groups_label()` writes the *code's* version into runs.csv, so an old `X_onehot.npz` could be run and labelled with a newer version. (2) Store the row order: save the `posting_id` array (e.g. `onehot_rows.npy` or a list in the JSON) and assert in later notebooks that it equals `meta["posting_id"]` — today the row order is only an agreement between notebooks. (3) Optional: source CSV size / hash in the JSON (provenance). | code, nb |
| C20 | 02_20 | (1) Also show the per-year slice (`X_main[meta["fy"] == 2023]`) — detectors are fitted per fiscal year (D6), so this is how notebooks 03/04 use the matrix. (2) Add the row-order assert from C19. (3) Define the ablation settings once in `src/encode.py` (e.g. `ABLATIONS = {"main": …, "main+nwh_dist": …, "main+hour_cyc": …, "main+wday_cyc": …, "no_time": …}`) and loop over them here, printing columns and the LOF minimum k (largest duplicate group + 1) per setting — so notebooks 03/04 and the Ch. 4 table use the same names. (4) Keep the `assert X_main.shape[1] == 131` guard. | code, nb |
| C21 | final markdown "Notes for the thesis" | Out of date in parts: (a) p_2 described as normal and "a false positive caused by the encoding" — replace by W5 (accrual settlement, unique account 2020, two sources of rarity); (b) D9 note — add slot alignment and the purchase / reversal cost (W9); (c) "unlike Gronewald et al. 2024, who repeat it per slot" — **verify in the paper** before keeping it. In the rewrite, this section should list the W-numbers of `thesis_writing_snippets.md` instead of repeating text. | nb, text |

(C9 — speed of `groupby().apply` in 02_09 — dropped, 26 Nov.)

## Excel updates for Mohsen (Thesis_Execution_Plan.xlsx)

| # | Sheet / row | Update |
|---|---|---|
| X1 | FE-DataChecks (new row) | 26 Nov · journal_entries_4203.csv · item × amount · FY2023 (notebook 02 §7.1): 0 of 420 anomalies with an item text lie outside the normal amount range of their own item (whole ledger 0 of 845) · D1; L5 · 3.3 |
| X2 | Decisions D4, column "Why" | "scalers fitted on FY1 (D10)" is out of date → "scalers fitted per fiscal year (D10)". Weaken "its bin edges are arbitrary for a model" (the generator uses these edges, W13); keep the functional-dependency reason first. |
| X3 | FE-DataChecks (new row) | 26 Nov · notebook 02 §6.3 · delay classes vs M-types (FY2023) · 60 of 62 anomalies with lag ≥ 30 days are M4; 60 of 95 with 10–29 days are M3 · D4; L8 · 3.3 |
| X4 | FE-DataChecks (new row) | 26 Nov · notebook 02 §6.1 · user vs booking pattern · Cramér's V 0.14, but 21 of 29 patterns are posted by one user only (638 entries) · D13; RQ1 · 3.3 |
| X5 | FE-DataChecks (new row) | 26 Nov · one-off run, FY2023 · lag_days vs promptly · LOF AP 0.116 (lag_days) vs 0.046 (promptly instead, k = 171) vs 0.101 (both); LOF R@1000 M4 0.87 / 0.23 / 0.00, M3 0.00 / 0.00 / 0.23; IF AP 0.057 / 0.068 / 0.066 · D4 · 3.3 / 4 (W14) |

## New cells / sections (agreed 26 Nov)

| # | What | Type |
|---|---|---|
| N1 | Numeric fields: histogram raw vs log1p, skewness, zeros, p99 vs max, per fiscal year — `amount`, `lag_days`, `nwh_dist` (zero-inflated, bounded → no log) (W7) | nb |
| N2 | Small table cell that prints the W7 numbers so the thesis numbers come from the notebook | nb |
| N3 | Narrative markdown between cells: question → evidence → decision (D#). **Every code cell gets a markdown before it with its goal and one after it (or in the same markdown) with its result** — in particular cells 02_18 (duplicates, D15) and 02_19 (save), which had no goal / result text at all (Mohsen, 26 Nov). | nb |
| N4 | Rule in the narrative: on/off decisions only from label-free checks (redundancy, dimensions, duplicates); label views are descriptive, FY1 only (D6). Wording: "association (Cramér's V) / functional dependency", "anomaly rate per bin" — not "correlation". | nb, text |
| N5 | Optional: category lists per fiscal year (W8) | code (optional) |
| N6 | Helper in `src/figures.py`: Wilson 95 % interval for `rate_table` and error bars in all rate figures (C16, C17) | code, nb |

## New order of the notebook (agreed 26 Nov)

1 Load → 2 line vs entry level → 3 line order (D9, `fe.sort_lines`) → 4 categorical one-hot per slot with PAD → 5 numeric fields: distribution → transformation → new features (`lag_days`, `nwh_dist`, sin/cos) → 6 redundancy without labels (on/off decisions) → 7 descriptive label views on FY1 → 8 build matrix and groups (`build_onehot`), sanity checks, duplicates → 9 save and use.

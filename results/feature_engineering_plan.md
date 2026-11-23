> **Last updated: 2026-11-23 16:50 (Berlin)** — D14 metric conventions (src/metrics.py) + first real-data check of the JET-only and random baselines

# Feature engineering plan (draft, 19 Nov 2026; updated 23 Nov 2026: D13 dataset-vs-real-ledger note, section F Limitations)

To discuss with supervisors before implementation (week 2). Rule: the raw CSV is never modified; everything is stored in derived files keyed by `posting_id`, in the row order of `meta.parquet`.

**Writing rule:** every decision below has options, a rationale and, where useful, an example. These go into the thesis (Ch. 3, Methodology: feature engineering / experimental setup), not only the final choice. Every point where a choice depends on a property of the synthetic dataset rather than on real accounting practice is also listed in section F (Limitations).

## A. Entry-level meta table — `data/derived/meta.parquet` (built in notebook 01, NOT a model input)

| Feature | How it is computed | Used in |
|---|---|---|
| `posting_id` | entry key | joins, row order of all matrices |
| `fy` | year of `posting_date` (= year of `document_date`; never differs) | FY1 tuning / FY2 reporting split |
| `n_lines` | number of posting lines (2–4) | info; also a model feature (B) |
| `label` | max of line `label` | evaluation only |
| `marking` | max of line `marking` (0–8) | type-aware metrics, RQ3 |
| `J1` | `top_n` of the dataset (2,000 entries, 196 anomalies — same as preprint Fig. 5) | JET-only, H1, H2, prefilter, combined score |
| `J2` | `high_cash` (402) | same |
| `J3` | `promptly ≥ 2` (8,207) | same |
| `J4` | `weekend > 0` or `nwh = 1` (16,493) | same |
| `h` | J1 + J2 + J3 + J4 | JET-only baseline order; combined score s = α·h/4 + (1−α)·r |
| `jet_any` | h > 0 (24,183 entries, 798 anomalies) | prefilter 4(a), H1, inside/outside split, RQ3 |

Decision (19 Nov): J1 follows the dataset `top_n` for comparability. Own raw implementation in `src/jets.py` (ties included: 2,209) is kept as validation only (used in notebook 01).

## B. One-hot + padding encoding (week 2) — `X_onehot.npz` (sparse) + `onehot_features.json`

Per line, padded to 4 lines (PAD category / 0 for missing lines). Lines are sorted before padding (D9).

| Feature | How it is computed | Group |
|---|---|---|
| `gl_account` | one-hot (25 accounts + PAD) | `lines` |
| `cd_flag` | one-hot D / C / PAD | `lines` |
| `amount` | log1p(amount) → RobustScaler → MinMax [0,1] (as Gronewald et al. 2024); scalers fitted on FY1 lines (D10); PAD = 0 | `lines` |

Per entry:

| Feature | How it is computed | Group |
|---|---|---|
| `n_lines` | from meta, (n − 2) / 2 | `lines` |
| `user` | one-hot (6 users), once per entry (D13) | `lines` |
| `tax_rate` | one-hot (0.19 / 0.00), once per entry (D13 — dataset property, see F) | `lines` |
| `promptly` | dataset column, ordinal 1 / 2 / 3 (lag 0–9 / 10–29 / 30+ days), as (p − 1) / 2 | `calendar_flags` (D4) |
| `weekend` | dataset column: 0 weekday, 1 Sat, 2 Sun (one-hot) | `calendar_flags` (D4) |
| `nwh` | dataset column: 1 = entered outside 08:00–17:59 (checked in the data, see D11) | `calendar_flags` (D4) |
| `lag_days` | NEW, from raw fields: date(`entered_date`) − `document_date` in days (0–181, median 5, skewed); log1p → RobustScaler → MinMax, fitted on FY1 (D10) | `lag_days` (D4) |
| `nwh_dist` | NEW: hours outside the working window 08:00–18:00; t = hour + minute/60; `nwh_dist = max(0, 8 − t) + max(0, t − 18)`; 0 inside working hours; divided by the FY1 maximum | `nwh_dist` (D11, switchable) |
| `hour_sin`, `hour_cos` | t on a 24 h circle: sin(2π·t/24), cos(2π·t/24) | `hour_cyc` (D11, switchable) |
| `wday_sin`, `wday_cos` | weekday of `entered_date` (0–6) on a 7-day circle | `wday_cyc` (D11, switchable) |

H2 input = selected feature groups + [J1, J2, J3, J4] columns from meta.

**Built in notebook `02_encoding_onehot.ipynb` with `src/encode.py` (21 Nov 2026; run with outputs, committed 21 Nov):** 140 columns in total — `lines` 129 (4 slots × (26 account + 3 side + 1 amount) + n_lines + 6 users + 2 tax rates), `calendar_flags` 5, `lag_days` 1, `nwh_dist` 1, `hour_cyc` 2, `wday_cyc` 2. Main setting = 135 columns; all groups = 140; proposal variant (`lines` only) = 129. Fitted parameters (categories, scaler values) are stored in `onehot_features.json`.

**Data checks from notebook 02 (numbers for Ch. 3):**
- `user`, `tax_rate`, `entered_date`, `document_date`, `promptly`, `weekend`, `nwh` are constant inside every entry (0 exceptions) → D13.
- Line patterns as written in the CSV: DC 51,056 entries (542 anomalies); DDC 33,312 (217); DCC 18,000 (91); CCD 60 (0); DDDC 1 (0).
- **Only one entry has 4 lines** (`p_2`, a normal utilities invoice, Feb 2024). It alone fills slot 4, so the one-hot encoding makes it very isolated. → check where it ranks in the one-hot runs; if near the top, it is a false positive caused by the encoding (one sentence in Ch. 4/5).
- `promptly` agrees with `lag_days` bins (0 mismatches); `weekend` agrees with the weekday of `entered_date` (0 mismatches); `nwh_dist` > 0 exactly when `nwh` = 1.
- Scaled values: FY1 in [0, 1]; FY2 `lag_days` max 0.80, `nwh_dist` max 1.05 (not clipped, D10).

### Decision D13 — entry-level fields encoded once per entry (21 Nov 2026, from the data check; real-ledger note added 23 Nov)

**Original plan:** `user` and `tax_rate` one-hot in every line slot (with PAD).

**Change:** both are constant inside every entry in this dataset, so they are encoded once per entry. Repeating them in 4 slots would count the same fact up to 4 times in every distance (LOF, kNN), give `user` more weight than the accounts, and make 2-line and 3-line entries differ in user columns only because of padding. No information is lost on this dataset.

**Important: the two fields are entry-level for different reasons (Mohsen, 23 Nov).**

| Field | Entry-level in this dataset | Entry-level in a real ledger? |
|---|---|---|
| `user`, dates (`document_date`, `posting_date`, `entered_date`) | yes | **Yes** — header data of the document. In SAP they are stored once per document in the header table BKPF (user name, document / posting / entry date); line items are in BSEG, linked by client, company code, fiscal year and document number. |
| `tax_rate` | yes | **No** — the tax code is a line-item field (SAP BSEG-MWSKZ, "Tax on sales/purchases code"). One document can carry several rates (e.g. an invoice with 7% and 19% items in Germany), and vendor / customer lines and bank lines normally carry no tax code at all (SAP shows "**" in the vendor line of an invoice, SAP KBA 1882126). |

**Data check that shows `tax_rate` is a generator property (23 Nov, journal_entries_4203.csv):** the rate is set per transaction and copied to every line, including lines that would have no tax code in a real system: 0.19 on 65,843 Trade payables (1600) lines, 38,919 Bank (1100) lines and 35,949 Trade receivables (1400) lines. Rate 0.00 appears only in 389 entries (payroll, social security, rent and their bank lines). So `tax_rate` here means "taxable at 19% / tax-exempt transaction type", not a line-level tax code.

**`tax_rate` adds no information beyond the accounts (23 Nov):** it is fully determined by the set of accounts in an entry (0 of 27 account combinations have more than one rate); the SKR03 accounts already carry the rate in their names (e.g. 3400 "Cost of merchandise, 19% input tax", 1576 "Deductible input tax, 19%", 1776 "VAT, 19%"). So the simplification has no effect on the results: the models do not learn anything from `tax_rate` that they do not already see in the accounts. Asked as Q4 in `questions_for_dataset_authors.md`.

**Consequences:**
- For this dataset, encoding `tax_rate` once per entry is correct and loses nothing.
- The code is safe for other data: notebook 02 asserts that every entry-level field has one value per entry; on a ledger with mixed rates it stops with an error instead of silently taking the first line's rate. For a real ledger, `tax_rate` would move into the line slots (per-slot one-hot with PAD).
- Mixed tax rates inside one entry, or a rate that does not fit the account (e.g. 7% on an account that should be 19%), could themselves be anomaly signals in real data; this dataset cannot test that → Limitations / Future Work (section F).

**Thesis sentence (draft, Ch. 3):** "In the synthetic ledger, the user, the dates and the tax rate are identical on all lines of an entry and are therefore encoded once per entry. For the user and the dates this matches real ERP systems, where they are document header fields; the tax rate, however, is a line-item attribute in real ledgers (e.g. SAP BSEG-MWSKZ) and would have to be encoded per line slot there."

### Feature groups and switches (decided 21 Nov 2026, Mohsen)

All candidate features are computed once and stored together in `X_onehot.npz`. `onehot_features.json` records, for every column, its name and its **group**. A model run selects columns by group, so features can be switched on or off without rebuilding the encoding.

| Group | Columns | Main setting | Why a switch |
|---|---|---|---|
| `lines` | per-slot one-hot + amount, `n_lines`, `user`, `tax_rate` | always on | core encoding (Gronewald et al. 2024) |
| `calendar_flags` | `promptly`, `weekend`, `nwh` | on (D4, to confirm) | proposal originally excluded dataset rule attributes; switch allows the "proposal variant" |
| `lag_days` | `lag_days` | on (D4, to confirm) | continuous version of `promptly` |
| `nwh_dist` | `nwh_dist` | off (see D11) | own feature, no empirical support yet; ablation |
| `hour_cyc` | `hour_sin`, `hour_cos` | off (see D11) | standard technique; ablation |
| `wday_cyc` | `wday_sin`, `wday_cos` | off (see D11) | standard technique; ablation |

In code (`src/encode.py`): `FEATURE_GROUPS_MAIN`, `select_columns(X, feature_info, groups)`, `groups_label(groups)`.

Every row in `results/runs.csv` stores the active groups (column `feature_groups`, e.g. `lines+calendar_flags+lag_days`), so each result can be traced to its feature set.

**Rule against tuning on test labels (same as D6):** the main feature set is fixed before any FY2 result is looked at. If groups are compared to choose the main set, the comparison uses FY1 only; FY2 ablation results are reported as a table but do not change the main setting.

**Scaling note:** `nwh_dist` and `lag_days` are scaled with values fitted on FY1 (D10). sin/cos are already in [−1, 1] and are not scaled.

**Thesis use:** the switches give a small ablation table for Ch. 4 (main set vs + `nwh_dist` vs + `hour_cyc` vs + `wday_cyc` vs proposal variant without `calendar_flags`/`lag_days`). Keep it small: one detector (IF) and one encoding is enough if time is short (Cut List).

### Decision D9 — line order inside the 4-line padding (21 Nov 2026, Mohsen accepts the default; to confirm with Smitha and Nazneen)

**Question.** The one-hot + padding encoding gives every entry 4 line slots. In which order are the lines put into the slots?

**Why the order matters (example for the thesis).** Two entries with the same accounting content:

- Entry A: slot 1 = Debit, cash account 1000, EUR 500; slot 2 = Credit, revenue account, EUR 500
- Entry B: slot 1 = Credit, revenue account, EUR 500; slot 2 = Debit, cash account 1000, EUR 500

For an accountant these are the same booking, and they should be treated the same. If the lines are kept in the original CSV order, A and B get completely different vectors: the slot-1 columns say "cash, debit" in A and "revenue, credit" in B. Isolation Forest, LOF and the autoencoder would then see B as a rare pattern only because of line position, not because of its content. This creates artificial rarity, i.e. extra false positives, which is exactly what the thesis tries to reduce. A fixed sorting rule gives every entry with the same content the same vector.

**Options considered:**

| Option | What it is | Assessment |
|---|---|---|
| 1. Original CSV order | lines as they appear in the file | Not safe: the meaning of a slot depends on how the generator wrote the lines; same content can give different vectors (example above). |
| 2. **Debit first, then account number, then amount (largest first)** | deterministic sort | **Chosen.** Follows the accounting convention (debit before credit), easy to explain, fully deterministic. |
| 3. Credit first, then account number | mirror image of option 2 | Works equally well; the model learns the same structure. Not chosen only because debit-first is the usual convention. |
| 4. Amount, largest first | sort by amount only | Fixed, but the order flips when two amounts are close or equal, so similar entries can still get different slot orders. |
| 5. No slots: aggregate per account × debit/credit (e.g. summed amount per account side) | order-free "bag of lines" | Removes the order problem completely, but loses the line structure and differs from the encoding of Gronewald et al. (2024), so results would be less comparable. → mention as future work. |

**Key point for the thesis:** debit-first vs credit-first makes almost no difference; what matters is that the rule is **fixed and deterministic**.

**Tie-break.** If two lines have the same side and the same account, sort by amount (largest first); if still equal, by original line position. Then every entry has exactly one order.

**Same order in the embeddings.** `entry_text` (section C) lists the lines in the same order, so both encodings see the same line order and the RQ2 comparison stays fair.

**Data check (notebook 02, 21 Nov 2026): the problem is real in this dataset.** 51,253 of 102,429 entries (50.0%) are written in the CSV in a different order than the D9 order: all 18,000 DCC entries, 33,192 of 33,312 DDC entries, all 60 CCD entries (credit lines first) and the single DDDC entry; none of the DC entries. So the generator does not write lines in a fixed order by account; without sorting, entries with the same content could get different vectors.

Example from the data (entry `a_1`, purchase of goods): CSV order = D 3400 Cost of merchandise 112.42, D 1576 Deductible input tax 21.36, C 1600 Trade payables 133.78. D9 order = D 1576, D 3400, C 1600.

The tie-break (same side and same account twice) is never needed in this ledger (0 cases), but it stays in the code so the rule is complete. In real ledgers it is needed (e.g. several expense lines on the same account), see F.

**Thesis sentence (draft):** "Before padding, the lines of each entry are sorted by debit/credit side, account number and amount, so that each slot has the same meaning in every entry and two entries with the same content always receive the same vector."

### Decision D10 — what the scalers are fitted on (21 Nov 2026, default; to confirm)

**Question.** `amount` (log1p → RobustScaler → MinMax), `lag_days` and `nwh_dist` need scaling. Are the scalers fitted on FY1 only or on the whole ledger?

| Option | Assessment |
|---|---|
| A. **FY1 only** (chosen) | Nothing from the test year (FY2) is used in preprocessing. Same procedure an auditor could apply in practice: scale this year's data with what was learned last year. |
| B. Whole ledger | No labels involved, so not a label leak; but the FY2 maximum and quartiles would shape the scale, i.e. test-year information enters preprocessing. |

**Example.** MinMax maps the largest value to 1. Fitted on the whole ledger, one very large FY2 amount would compress all FY1 amounts; fitted on FY1, that FY2 amount simply gets a value above 1.

**Implemented:** FY2 values outside [0, 1] are kept (not clipped), so extreme values stay extreme for the detectors. Category lists (accounts, users, tax rates) come from the whole ledger; no labels are involved. Note both in Ch. 3.

### Decision D4 — time features from the entry timestamp (21 Nov 2026, agreed with Shravan; to present to Smitha and Nazneen)

**Original default (19 Nov):** include lag, hour sin/cos and weekday sin/cos, and never use the precomputed dataset columns `promptly`, `weekend`, `nwh`.

**New decision:** keep the dataset columns `promptly`, `weekend` and `nwh` as model features, and add a continuous `lag_days` next to `promptly`. How the time of day is encoded is decided separately in D11.

**Rationale (for the thesis, feature engineering section):**
- `promptly` groups the entry lag into three bins (0–9, 10–29, 30+ days). The bin edges are arbitrary for a model: an entry entered 28 days late and one entered 31 days late fall into different groups although their delay is almost the same, while 10 and 29 days fall into the same group. Distance-based and tree-based detectors therefore see a large jump at 30 days and no difference inside a bin. `lag_days` keeps the real distance between entries; `promptly` is kept because it is the dataset's own definition and the basis of J3, which keeps results comparable with Gronewald et al.
- `weekend` and `nwh` describe clear calendar facts with a direct audit meaning (posting on a weekend or outside working hours); they are kept as they are.
- `lag_days` is computed from raw fields only; the raw CSV is not changed.

**Points to state or check in the thesis:**
- Proposal sentence "The precomputed rule attributes of the dataset are not used as model features" must be rewritten: `promptly`, `weekend` and `nwh` are model features; `top_n` and `high_cash` are used only as JET flags (meta, H2).
- H2 adds the J1–J4 flags to the features. Since J3 and J4 are built from `promptly`, `weekend` and `nwh`, they are already close to the base features, so the difference between H2 and the plain model comes mainly from J1 and J2. Mention this when interpreting H2.
- `nwh` has the same edge problem as `promptly` (17:59 vs 18:00); see D11.
- The "proposal variant" (without `calendar_flags` and `lag_days`) stays available via the switches.

### Decision D11 — time-of-day and weekday encoding (21 Nov 2026, updated: all variants built, switchable; main setting = flags only; to present to Smitha and Nazneen together with D4)

**Question.** With D4, `nwh` and `weekend` are already features. Should the model also see the time of day and the weekday, and in which form?

**Options considered (all three are built as switchable groups; see "Feature groups and switches"):**

| Option | What it is | Assessment |
|---|---|---|
| A. Cyclical sin/cos encoding (group `hour_cyc`, `wday_cyc`) | `hour_sin = sin(2π·t/24)`, `hour_cos = cos(2π·t/24)`, same with weekday on a 7-day circle; 4 columns. Standard technique, e.g. scikit-learn example "Time-related feature engineering" (cite in Ch. 3). Both sine and cosine are needed because sine alone maps two hours to the same value (03:00 and 09:00 both give 0.71); "sin/cos" means the pair, not a division (sin ÷ cos = tan would go to infinity at 06:00 and 18:00 and repeat every 12 hours). | Places each time on a circle so 23:00 and 00:00 are close. It describes *where* an hour is on the clock, not *how unusual* it is. 4 extra dimensions, which weigh on distance-based detectors (kNN, LOF). The wrap 23:00 → 00:00 does not occur in this ledger (no entries 00:00–04:59). |
| B. No time-of-day feature beyond `nwh` / `weekend` | only the binary flags | **Main setting (proposed).** Simplest, no extra dimensions, matches how auditors test (weekend / after-hours JET). Keeps the hard 17:59 / 18:00 edge. |
| C. `nwh_dist` (group `nwh_dist`, own idea) | distance in hours from the working window 08:00–18:00: 0 inside; `max(0, 8 − t) + max(0, t − 18)` with t = hour + minute/60. Examples: 07:30 → 0.5, 18:30 → 0.5, 23:00 → 5, 05:00 → 3. | Removes the edge of `nwh` and expresses "how far outside working hours". But see the counter-arguments below → ablation only. |

**Why `nwh_dist` is not in the main setting (for the thesis):**
1. **No empirical support.** Audit standards ask auditors to look at items processed outside the normal course of business (PCAOB AU 316.61; AICPA Practice Alert 2003-02), which motivates the binary after-hours test, but no published study was found that tests on real ledger data whether fraud becomes more likely the further an entry is from working hours. Real labelled journal-entry data is rarely available (Debreceny & Gray, 2010, IJAIS 11(3), note that published research on journal-entry data mining was almost non-existent for this reason).
2. **Adversarial argument (Mohsen, 21 Nov).** An informed perpetrator who knows that after-hours postings are tested would post within or close to working hours (e.g. 17:30) to blend in. `nwh_dist` would rank such an entry as *less* suspicious, i.e. it rewards the concealment. Rule-based tests are known to be circumvented by fraudsters who adapt (Schreyer et al., 2017, arXiv:1709.05254), and Schreyer et al. (2019, arXiv:1910.03810) show that entries can even be generated deliberately to camouflage anomalies.
3. **False-positive risk in practice.** After-hours flags also hit legitimate scheduled batch postings (practitioner report: a scheduled Saturday 20:00 batch is flagged the same as an unauthorised manual posting). A distance score would push such regular batch jobs further up the list.
4. **Synthetic data shows no gradient** (data check below).

**More realistic signal (Future Work):** "unusual time for this user or account" (a contextual anomaly, e.g. a user who never posts in the evening suddenly does). IF / LOF can partly capture this because they see `user` and `nwh` together; an explicit per-user time baseline is future work.

**Weekday:** `weekend` carries the weekday fact that matters for an auditor; `wday_sin`/`wday_cos` are built but off in the main setting.

**Data check (21 Nov 2026, journal_entries_4203.csv, 102,429 entries):**
- Entry times only between 05:00 and 23:59. `nwh` = 1 exactly for hours 05–07 and 18–23; hour 17 (52 entries) has `nwh` = 0. So the flag's window is 08:00–17:59.
- The generator description (Data in Brief, Table 3, T1) draws non-working hours from 05–07 and 17–23, so it starts non-working time at 17:00 while the flag counts 17:xx as working. `nwh_dist` follows the flag. → add to `questions_for_dataset_authors.md`.
- Anomaly rate per hour: 0.65–0.79% in hours 08–16, 1.9% at 17, 5.6–14.9% in hours 05–07 and 18–23.
- Inside `nwh` = 1 there is **no trend with distance** (about 8–14% anomalies in every 1-hour bin of `nwh_dist`). The generator draws non-working times uniformly, so on this synthetic ledger `nwh_dist` is expected to add little beyond `nwh`.

**What to write in the thesis:** present B as the main setting, A and C as considered alternatives with the arguments above, and the ablation (FY2 table) as evidence. State in Limitations that synthetic data cannot show whether a distance effect exists in real ledgers.

## C. Sentence embeddings (week 2, laptop) — `emb_minilm.npy` (102,429 × 384)

| Feature | How it is computed | Used in |
|---|---|---|
| `entry_text` | one string per entry from raw fields, e.g. user, weekday + time, lag, then per line "D/C account name amount tax" in the D9 line order (no `text` field, see D1) | input to the model only |
| embedding | all-MiniLM-L6-v2, normalised, 384-d | IF, LOF, AE (embedding runs), RQ2 |

**Note on switches:** the embedding cannot switch single features off after encoding; the content of `entry_text` is fixed when the embeddings are computed. To keep RQ2 fair, `entry_text` should contain the same information as the main one-hot setting (e.g. "weekend: no, after hours: yes, lag 12 days" rather than the exact clock time). Decide the exact template in the embedding notebook and write it down here; a second embedding file with exact time is optional (Cut List).

### Decision D1 — free-text `text` field (proposed 21 Nov 2026, Mohsen; NOT yet confirmed by Smitha and Nazneen — an earlier note saying 'confirmed 19 Nov' was wrong)

Exclude `text` from both encodings (B and C).

**Rationale (for the thesis, feature engineering section; data check in `week01_notes.md`):**
1. **No added information.** `text` is one value per entry with only 325 values: 10 garden item names (~10,000 entries each) and ~315 monthly recurring texts. Anomalies reuse the normal texts (only 2 of 850 have a text that no normal entry has), so there is no direct label leak. The higher anomaly rate of "Lawn Mower" (3.0% vs 0.83% overall) comes from price: it is the most expensive item, and 118 of 120 M1 entries and all `top_n` entries are Lawn Mower. `text` is therefore a proxy of the amount level, which the model already sees via `amount`.
2. **Fair comparison of the encodings (RQ2).** Only the embedding can use free text. Adding it to the embedding alone would mix the effect of the encoding with an extra information channel.
3. **Comparability with prior work.** Gronewald et al. (2024) drop `text` in their preprocessing, and Gronewald, Scherer & Mehdiyev (2026 preprint, Table S3) also exclude the free text on this public ledger.
4. **External validity.** The synthetic text (10 item names) is far simpler than real posting texts, so a result with `text` would say little about real ledgers (Limitations).
5. **False-positive risk.** Near-unique monthly texts (e.g. "Rent July") would look novel in the embedding space for reasons unrelated to anomalies.

Optional if time allows: ablation of the embedding with vs without `text`.

## D. Never used as model features

`top_n`, `high_cash` (except as JET flags in meta / H2), `label`, `marking`, raw dates, `posting_id`, `ref_id`, `text` (D1), `gl_account_name` (redundant with `gl_account`). (`nwh_dist`, `hour_sin/cos`, `wday_sin/cos` are built but off in the main setting — ablation only, D11.)

## E. Experimental-setup decisions (not features, but written in the same Ch. 3)

### Decision D6 — fitting / evaluation split (21 Nov 2026, Mohsen accepts the default; to confirm)

**Choice.** Unsupervised models are fitted on each fiscal year's data without labels. Hyperparameters (e.g. number of trees, k for LOF, AE size) and the main feature set are chosen with FY1 (2023) labels; all reported results are on FY2 (2024).

**Rationale.** Each fiscal year has enough entries to fit and test on its own (51,064 in 2023, 51,365 in 2024; ~12k JET entries per year). Labels of the reported year are never used for any choice, so the reported numbers are not optimistic. It also mirrors practice: an auditor sets up the method with last year's knowledge and applies it to the new year.

**Example.** If k for LOF were chosen by the best AP on 2024 labels and then AP on 2024 were reported, the number would show the best case that no auditor could reach without knowing the answers.

**Options not chosen:** random train/test split of entries (mixes both years and breaks the yearly audit logic; also JET J1 is defined per year); cross-validation (not needed with two large years; more runs).

### Decision D7 — weight α in the combined score (21 Nov 2026, Mohsen accepts the default; to confirm)

**Choice.** s = α·h/4 + (1−α)·r with α = 0.5 fixed in advance. A sweep α ∈ {0, 0.25, 0.5, 0.75, 1} is reported only as a sensitivity analysis.

**Rationale.** α = 1 gives the pure JET order and α = 0 the pure ML order, so 0.5 is the neutral midpoint that gives rules and model equal weight without looking at labels. Choosing α from test labels would be tuning on the answers (same logic as D6).

**What to write.** The sweep shows how stable the result is around 0.5; if a different α is clearly better on FY1, it can be noted, but the main result stays at the pre-set 0.5.

### Decision D14 — metric conventions (23 Nov 2026, implemented in `src/metrics.py`)

**Choice.**
- Budgets k = 50, 100, 250, 500 reviewed entries, plus 1 % and 5 % of the ledger of the evaluated year (FY2024: 514 / 2,568). Budgets are absolute because they stand for auditor capacity, which does not depend on the method's list length.
- P@k = anomalies in the top k / reviewed entries; FP@k = reviewed − anomalies; R@k = anomalies in the top k / **all** anomalies of the year; AP = mean precision at each hit / **all** anomalies of the year.
- Ties are broken by a seeded random order; results are the mean (and SD) over 5 seeds. Never by row order.
- An entry with NaN score is not on the method's list and is never reviewed; if the list is shorter than k, n_reviewed < k.
- Every run row in `results/runs.csv` stores the git commit (`+dirty` if uncommitted changes).

**Options considered and why.**

| Question | Options | Chosen | Why |
|---|---|---|---|
| Recall denominator | all anomalies of the year / only anomalies inside the JET list | all | Anomalies a prefilter cannot reach must count as missed; the 93.9 % ceiling then shows directly in the numbers (RQ3). Example: a prefilter that finds all 396 in-JET anomalies of FY2024 still has recall 396 / 427 = 92.7 %. |
| Tie-break | row order / random with seeds | random, 5 seeds | Row order is CSV order and can carry generator structure, which would make a method look better or worse by accident. 21,387 JET entries share h = 1, so ties are the normal case for JET-only. |
| Budget | absolute / share of each method's list | absolute (+ 1 % / 5 % of the ledger) | The auditor's capacity is the same for every method. |

**Tests.** `tests/test_metrics.py`, 11 tests on a toy ranking computed by hand (10 entries, 3 anomalies at positions 1, 4, 7: P@4 = 0.5, R@4 = 2/3, AP = (1/1 + 2/4 + 3/7) / 3 = 0.643), plus NaN entries, ties and budgets.

**Real-data check (23 Nov, FY2024, 51,365 entries, 427 anomalies; mean of 5 seeds):**

| Method | P@50 | P@500 | R@5 % | AP | Expected |
|---|---|---|---|---|---|
| Random, whole ledger | 0.016 | 0.011 | 0.053 | 0.009 | base rate 0.83 % |
| Random inside JET union | 0.044 | 0.030 | 0.172 | 0.030 | union precision 3.3 % |
| JET-only (h, random ties) | 0.480 | 0.112 | 0.358 | 0.092 | h ≥ 3: 56 entries, 27 anomalies (48 %) |

All three match the expected rates, so the metrics are trusted. JET-only is strong only in its top ~56 entries; after that it is close to random inside the union, which is the gap the ML ranking must close (RQ1).

### Decision D3 — H1 threshold (default; to confirm; needed from week 3)

**Choice.** H1 flags an entry when it is in a JET AND in the top 5% of ML scores; the flagged entries are ranked by ML score. The 5% is fixed in advance.

**Rationale.** H1 (Gronewald et al., 2024) needs a yes/no ML flag. A fixed threshold avoids tuning on labels and is easy to explain to an auditor.

## F. Limitations: dataset properties vs real accounting practice (collect here; for Ch. 5 Discussion / Limitations)

Every choice that is valid only because of how the synthetic ledger was generated. Each entry: what the dataset does → what a real ledger does → consequence for the thesis.

| # | Topic | In this dataset | In a real ledger | Consequence / what to write |
|---|---|---|---|---|
| L1 | Tax rate level (D13) | one rate per entry, copied to every line (also payables, receivables, bank lines); only 0.19 and 0.00 | tax code per line item (SAP BSEG-MWSKZ); mixed rates in one document; vendor / customer / bank lines without tax code | encoded once per entry here; per line slot on real data. Fully determined by the accounts here, so no effect on results. Mixed or mismatched tax codes as an anomaly signal cannot be tested (Future Work). Dataset authors: Q4. |
| L2 | Number of lines | 2–3 lines, only one entry with 4 lines | documents with many lines are common (e.g. payroll, collective invoices) | padding to a fixed 4 slots does not scale; a long-tail line count needs another encoding (aggregation per account × side, or sequence / set models) → Future Work. The single 4-line entry may become an encoding-made false positive. |
| L3 | Tie-break in line order (D9) | never needed (no entry has two lines with the same side and account) | common (several lines on the same expense account) | rule kept complete in code; untested here. |
| L4 | Distance outside working hours (D11) | non-working times drawn uniformly, no gradient | unknown; adversarial argument suggests fraudsters stay near working hours | `nwh_dist` only as ablation; synthetic data cannot show a distance effect. |
| L5 | Free text (D1) | 10 item names + monthly texts | rich, varied posting texts | excluded; results say nothing about text-based detection on real ledgers. |
| L6 | Relational anomalies (M8, `ref_id`) | M8 linked via `ref_id` | linked documents (invoice ↔ payment, reversals) are central in audits | out of scope; entry-level encodings cannot see them. |
| L7 | Working-hour window | flag window 08:00–17:59; generator description starts non-working time at 17:00 | company-specific (shifts, time zones, batch jobs) | follow the dataset flag; question to dataset authors. |
| L9 | Transfer to a real company (Mohsen, 23 Nov) | 6 users, 25 accounts; M1–M6 are built so that an unusual user for the context is the signal | hundreds of accounts and users, new / leaving employees, system users for batch postings | **The method transfers, the fitted model does not:** unsupervised detectors are fitted on each company's own ledger (and each year, D6), so `user` and `gl_account` are learned as "what is normal here", e.g. which user usually posts to which account. Not transferable: the synthetic labels and the hyperparameters tuned on them. Open questions on real data: (a) one-hot dimension grows with many accounts / users → group accounts by chart class (e.g. SKR03 account class) or use frequency encoding; sentence embeddings can also place a new account by its name (supports RQ2); (b) new users and batch users look novel → false positives; (c) whether "unusual user for this account" really marks fraud is unknown on real data — the precision found here is an upper-bound estimate, not a prediction. |
| L8 | Labels | injected anomalies with known types M1–M8 | labels rarely known; anomalies ≠ fraud | evaluation only possible on synthetic data; external validity limited. |

## Open decisions for supervisors (status 23 Nov 2026)

1. D4 + D11: keep `promptly`, `weekend`, `nwh`, add `lag_days`; `nwh_dist` and sin/cos built as switchable groups, off in the main setting, shown as ablation — **real question** (changes the proposal).
2. D1, D3, D6, D7, D9, D10, D13: proposals / defaults above, Mohsen agrees (D6, D7, D9 explicitly) — send as "I will use these unless you object".
3. `ref_id`-based relational features for M8 — out of scope (proposal), mention as limitation (L6).

## Storage convention

- Tables: Parquet (`pyarrow`). Dense arrays: `.npy`. Sparse matrices: `.npz` (scipy). Results per run: `results/runs.csv` (with a `feature_groups` column).
- No pickle for shared outputs (depends on library versions and is unsafe to load from others).
- Notebooks are named by process step: `01_jets_meta`, `02_encoding_onehot`, `03_baselines`, … (not by week).

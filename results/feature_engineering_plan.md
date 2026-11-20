# Feature engineering plan (draft, 19 Nov 2026)

To discuss with supervisors before implementation (week 2). Rule: the raw CSV is never modified; everything is stored in derived files keyed by `posting_id`, in the row order of `meta.parquet`.

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

Decision (19 Nov): J1 follows the dataset `top_n` for comparability. Own raw implementation (ties included: 2,209) is kept as validation only. Proposal sentence "precomputed rule attributes are not used as model features" must be reworded: dataset rule attributes define the JET populations and the H2 flags; no other model uses them.

## B. One-hot + padding encoding (week 2) — `X_onehot.npz` (sparse) + `onehot_features.json`

Per line, padded to 4 lines (PAD category / 0 for missing lines):

| Feature | How it is computed | Used in |
|---|---|---|
| `gl_account` | one-hot (25 accounts + PAD) | IF, LOF, AE (one-hot runs) |
| `cd_flag` | one-hot D / C / PAD | same |
| `user` | one-hot (6 users + PAD) | same |
| `tax_rate` | one-hot (+ PAD) | same |
| `amount` | log1p(amount) → RobustScaler → MinMax [0,1] (as Gronewald et al. 2024); PAD = 0 | same |

Per entry (from raw fields, not from the dataset flags):

| Feature | How it is computed | Used in |
|---|---|---|
| `lag_days` | date(entered_date) − document_date in days, scaled | all one-hot runs |
| `hour_sin`, `hour_cos` | hour of entered_date (+ minute/60) on a 24 h circle | same |
| `wday_sin`, `wday_cos` | weekday of entered_date on a 7-day circle | same |
| `n_lines` | from meta | same |

H2 input = B + [J1, J2, J3, J4] columns from meta.

## C. Sentence embeddings (week 2, laptop) — `emb_minilm.npy` (102,429 × 384)

| Feature | How it is computed | Used in |
|---|---|---|
| `entry_text` | one string per entry from raw fields, e.g. user, weekday + time, lag, then per line "D/C account name amount tax" | input to the model only |
| embedding | all-MiniLM-L6-v2, normalised, 384-d | IF, LOF, AE (embedding runs), RQ2 |

## D. Never used as model features

`promptly`, `weekend`, `nwh`, `top_n`, `high_cash` (except as JET flags in meta / H2), `label`, `marking`, raw dates, `posting_id`, `ref_id`, `text`, `gl_account_name` (redundant with `gl_account`).

## Open decisions for the meeting

1. Line order inside the padding (e.g. debit first, then by account) — must be fixed and documented.
2. Scalers fitted on FY1 only or on the whole ledger (no labels involved either way).
3. Include the free-text field `text` (item name) in the embedding string or not? (Gronewald et al. 2024 dropped it.)
4. `ref_id`-based relational features for M8 — out of scope (proposal), mention as limitation.

## Storage convention

- Tables: Parquet (`pyarrow`). Dense arrays: `.npy`. Sparse matrices: `.npz` (scipy). Results per run: `results/runs.csv`.
- No pickle for shared outputs (depends on library versions and is unsafe to load from others).

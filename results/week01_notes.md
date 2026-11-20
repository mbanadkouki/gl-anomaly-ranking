# Week 01 notes (17 Nov 2026, updated 19 Nov 2026)

## JET implementation decisions
- Fiscal year = year of posting_date (51,064 entries in 2023, 51,365 in 2024; no entry spans two years).
  Year of document_date is always the same; year of entered_date differs for 336 entries (cross-period), so it is not used.
- **Main setting (decided 19 Nov): JET populations come from the dataset rule attributes**, for comparability with
  Gronewald, Scherer, et al. (2026): J1 = `top_n`, J2 = `high_cash`, J3 = `promptly >= 2`, J4 = `weekend > 0 or nwh = 1`.
  Stored in `data/derived/meta.parquet` (notebook 01).
- Own re-implementation from raw fields is kept as **validation only**:
  - J1: entry amount = max line amount; entries >= the 1,000th highest per year, ties included (RANK) → 2,209 entries.
    Dataset `top_n` takes exactly 1,000 per year (row order / LIMIT) → 2,000 entries, 196 anomalies
    (same as preprint Fig. 5). Difference = 209 entries, all at the exact threshold amount
    (2023: EUR 2,667.50, 122 tied, 33 taken; 2024: EUR 2,976.49, 143 tied, 23 taken). Dataset set ⊂ own set.
  - J2: credit on cash-in-hand (1000) > EUR 1,500 — identical to dataset (402 entries).
  - J3: date(entered_date) − document_date >= 10 days — identical (8,207). Matches `promptly` exactly
    (1 = 0–9 days, 2 = 10–29, 3 = 30+).
  - J4: Sat/Sun or hour outside 08:00–17:59 — identical (16,493). `weekend`: 0 weekday, 1 Sat, 2 Sun.
- With `top_n`, 2 of 120 M1 entries are outside J1 (a_2744, aa_774; both exactly at the threshold amount).
  Both are in J2, so union recall is unchanged. → question to dataset authors (by 15 Dec), see
  `questions_for_dataset_authors.md`; tie handling mentioned in the thesis.

## Findings that change the proposal text
1. Prefilter recall ceiling is 93.9% (798/850), not 90.6%: M7/M8 were injected without a rule, but 22 of 60 M7
   and 6 of 20 M8 entries fall inside the JET union by chance. Proposal assumed all 80 lie outside.
   → report 93.9% as ceiling; RQ3 "outside JET" set = 52 anomalies (38 M7, 14 M8). (Same with own J1 or `top_n`.)
2. JET union = 24,183 entries (23.6% of ledger, ~12k per year: 12,177 / 12,006), precision 3.30% → far above any
   budget, so ranking inside it is the main lever (supports RQ1). (Own J1 with ties: 24,329 entries, 3.28%.)
3. M1–M6 are each tied to one user that is unusual for that context (Max: M1, M5; Fred: M2, M3, M6;
   David: M4); M7 uses rare bank account 1200; M8 is relational (ref_id). → user and account must be in both
   encodings; M8 will be hard for any entry-level encoding (as the preprint reports).
4. M1 entries all post to cash (1000) and all fall in J2 as well as J1.
5. Rules hit inside the union (dataset flags): h=1: 21,387 entries / 543 anomalies (2.5%); h=2: 2,677 / 200 (7.5%);
   h=3: 115 / 52 (45%); h=4: 4 / 3. → h ranks only the top ~120 entries well; 68% of in-union anomalies sit in the
   h=1 block where h gives no order → ML ranking inside the union is needed (RQ1). JET-only baseline breaks ties randomly.
6. Proposal sentence "The precomputed rule attributes of the dataset are not used as model features" must be reworded:
   dataset rule attributes define the JET populations and the H2 flags only; no other model uses them as features.

## Working conventions (decided 19 Nov)
- Raw CSV never modified; derived files keyed by `posting_id`, row order of `meta.parquet`.
- Storage: tables Parquet (pyarrow), dense arrays `.npy`, sparse `.npz`, run results CSV. No pickle for shared outputs.
- Code: one notebook per step (01_jets_meta, 02_encoding, ...); shared logic moves to `src/*.py` from week 2;
  `requirements.txt` so the repo runs on Colab.
- Feature engineering plan: `feature_engineering_plan.md` (to discuss with supervisors before week 2).
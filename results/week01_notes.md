# Week 01 notes (17 Nov 2026, updated 21 Nov 2026)

Status: week 1 finished (21 Nov). Carried over to week 2: Ch. 2 (not started), src/ refactor, laptop installs.

## Decisions after the 19 Nov supervisor meeting (see Decisions sheet in Thesis_Execution_Plan.xlsx)
- D1 NOT yet discussed with Smitha & Nazneen (an earlier version of these notes wrongly said "confirmed 19 Nov";
  corrected 21 Nov). Proposal: free-text `text` field excluded from both encodings — rationale from a data check,
  see "D1 — data check on `text`" below. To present at the next supervisor meeting.
- D2 confirmed (19 Nov): main prefilter model is fitted on JET (prefiltered) entries only; rationale: Isolation Forest fits
  better on the prefiltered population (find and cite a source in Ch. 3.6). Whole-ledger ablation only if time allows.
- D4 changed (Shravan, 21 Nov; still to present to Smitha & Nazneen): keep dataset columns `promptly`, `weekend`, `nwh`
  as features and add continuous `lag_days` = date(entered_date) − document_date next to `promptly` (promptly bins
  0–9 / 10–29 / 30+ days). Consequences: reword the proposal sentence on rule attributes; H2 then adds mainly
  J1 / J2 beyond the base features. Done: feature_engineering_plan.md D4 updated (21 Nov).
- D11 proposal (Mohsen, 21 Nov; to present with D4): drop hour / weekday sin/cos; add `nwh_dist` = hours outside
  the 08:00–18:00 window (0 inside), keep `nwh` and `weekend`. Data check: entry times only 05:00–23:59 (no midnight
  wrap for sin/cos to handle); nwh = 1 exactly for hours 05–07 and 18–23; anomaly rate ~0.7% in 08–16 vs 5.6–14.9%
  per hour outside, but no trend with distance inside nwh = 1 (generator draws uniformly) → little gain expected on
  this ledger; audit rationale for real ledgers, note in Limitations. Options A/B/C + rationale: feature plan, D11.
- Decisions sheet now has the columns "My Decision" (before the meeting) and "Final Choice" (filled after Smitha &
  Nazneen confirm); the old "Final Choice / Notes" text moved to "Notes".
- Not discussed on 19 Nov (defaults in use until confirmed): D3 (H1 = top 5%), D6 (tune FY1 / report FY2),
  D7 (α = 0.5 fixed), D9 line order (debit first, then account), D10 scalers fitted on FY1.
- D12 (21 Nov): thesis text written directly in thesis_v0.docx (SRH template) from week 2; keep an AI-usage log
  (template requires a List of generative AI tool usages).

## D1 — data check on `text` (21 Nov 2026, journal_entries_4203.csv)
Facts:
- `text` is constant within an entry (0 entries with more than one value); 325 distinct values, no nulls.
- 10 garden item names (Lawn Mower, Rake, Shovel, ...) with ~10,000 entries each, spread almost evenly;
  ~315 monthly recurring texts (rent, salary, utilities, ...; `x_` / `xa_` entries), each used 1–2 times.
- Anomalies reuse the normal texts: only 2 of 850 anomalous entries have a text that no normal entry has
  ("Rent July", M1 — a text that occurs only twice in the whole ledger). → no direct label leak.
- Anomaly rate per text: Lawn Mower 3.0% (314 / 10,446), Wheelbarrow 1.1%, cheap items 0.3–0.4%; base rate 0.83%.
  Cause is price: Lawn Mower is the most expensive item (median line amount ~EUR 1,111 vs ~EUR 97 for Rake);
  118 of 120 M1 entries and all `top_n` entries are Lawn Mower. M6 uses only the four more expensive items
  (Lawn Mower, Wheelbarrow, Leaf Blower, Hedge Trimmer), while normal entries with the same accounts are spread
  evenly over all 10. → `text` acts as a proxy of the amount level, which the model already sees via `amount`.

Reasons for excluding `text` from both encodings (for Ch. 3, feature engineering):
1. No added information: no direct leak, and its link to anomalies comes from the amount (data check above).
2. Fair comparison of the encodings (RQ2): only the embedding can use free text; adding it to E2 alone would mix
   the effect of the encoding with an extra information channel.
3. Comparability with prior work on this data: Gronewald et al. (2024, ICAA, data preprocessing section) drop `text`
   ("not considered in the ML model"); Gronewald, Scherer & Mehdiyev (2026 preprint, Table S3) also exclude the free
   text on the public ledger.
4. External validity: the synthetic text (10 item names) is far simpler than real posting texts (free, noisy, often
   empty or multilingual), so a result with `text` would say little about real ledgers. → Limitations.
5. False-positive risk: near-unique monthly texts (e.g. "Rent July") would look novel in the embedding space for
   reasons unrelated to anomalies.
Optional if time allows: ablation E2 with vs without `text`.

## JET implementation decisions
- Fiscal year = year of posting_date (51,064 entries in 2023, 51,365 in 2024; no entry spans two years).
  Year of document_date is always the same; year of entered_date differs for 336 entries (cross-period), so it is not used.
- **Main setting (decided 19 Nov): JET populations come from the dataset rule attributes**, for comparability with
  Gronewald, Scherer, et al. (2026): J1 = `top_n`, J2 = `high_cash`, J3 = `promptly >= 2`, J4 = `weekend > 0 or nwh = 1`.
  Stored in `data/derived/meta.parquet` (notebook 01).
- Own re-implementation from raw fields is kept as **validation only**:
  - J1: entry amount = max line amount; entries >= the 1,000th highest per year, ties included (RANK) → 2,209 entries.
    Dataset `top_n` takes exactly 1,000 per year → 2,000 entries, 196 anomalies (same as preprint Fig. 5).
    Difference = 209 entries, all at the exact threshold amount
    (2023: EUR 2,667.50, 122 tied, 33 taken; 2024: EUR 2,976.49, 143 tied, 23 taken). Dataset set ⊂ own set.
    The dataset's choice among ties is not reproducible by file order or a simple sort (80–92 mismatches each);
    likely random or set at an earlier generator step → check `synthetic_data_4203.ipynb` on Zenodo.
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
   dataset rule attributes define the JET populations and the H2 flags (and, if D4 is confirmed, promptly / weekend / nwh
   are also model features).

## Working conventions (decided 19 Nov)
- Raw CSV never modified (`data/raw/`, `;`-separated); derived files in `data/derived/`, keyed by `posting_id`,
  row order of `meta.parquet`.
- Storage: tables Parquet (pyarrow), dense arrays `.npy`, sparse `.npz`, run results CSV. No pickle for shared outputs.
- Code: one notebook per step (01_jets_meta, 02_encoding, ...); shared logic moves to `src/*.py` from week 2;
  `requirements.txt` so the repo runs on Colab.
- Feature engineering plan: `feature_engineering_plan.md`.
> **Last updated: 2026-11-23 19:00 (Berlin)** — final template: Bakumenko et al. (2025) structure with three changes (names, Debit/Credit words, "; "); exact Bakumenko format as ablation T-Bak; 29 distinct sentences; M1 / M7 analysis

# How journal entries are turned into sentences before SBERT encoding (notes, 21 Nov 2026)

Purpose: reference material for the thesis (related work + methodology, encoding section). Describe how prior work built the input sentence, then justify our own template.

## 1. Bakumenko et al. (2025), IEEE Access 13 (same method in arXiv 2406.03614, 2024)

**Data**
- Real, confidential general ledger aggregated from several anonymised companies (same data as Bakumenko & Elragal, 2022). Not public.
- 32,100 transaction lines; 148 anomalies of eight types, injected by financial auditors.
- Entries with more than four transactions removed (as in the 2022 paper).
- After cleaning only three attributes remain: `SOURCE` (source system, remapped to generic codes), `ACCOUNT_DC` (account and debit/credit flag merged into one categorical value; 577 unique values) and a surrogate ID. No amounts, dates, users or text.

**Sentence construction (Section IV-A, Eq. 2)**
- Lines are grouped by journal-entry ID.
- Each line becomes `"Source: " + S_j + " Account_DC: " + A_j`.
- Lines of one entry are joined with ", " (comma and space); the whole string is treated as one sentence.
- Encoded with SBERT (all-MiniLM-L6-v2, all-distilroberta-v1, all-mpnet-base-v2; called E1, E2, E3 in the paper), mean pooling, no PCA, then standardised (zero mean, unit variance).
- Setting is supervised (LR, RF, SVM, XGBoost, NN; 80/20 stratified split; recall average macro).

**What the paper does NOT do (useful for our gap statement)**
- No reason is given for this particular template, and no alternative template is tested.
- The authors state this themselves in the Limitations section: changes in the input feature concatenation can alter the embeddings, and this was "not covered in this work".
- Also in Limitations: the method "focuses on categorical features" and needs extension for precise numerical analysis.
- Future work (IEEE version, Section VIII-C): explore unsupervised strategies, and investigate preprocessing strategies "including aggregation methods and prompt engineering".
- → Our thesis touches all three: unsupervised use, numerical attributes (amount, lag) as columns, and an ablation of the sentence template.

## 2. Gronewald et al. (2024, ICAA) — no sentence encoding
- One-hot encoding + concatenation of postings; they name LLM embeddings (Bakumenko et al., 2024) as a possible future improvement. `text` field dropped.

## 3. Gronewald, Scherer & Mehdiyev (2026 preprint)
- One-hot, frequency and learned entity encodings ("AE-emb" = learned embedding inside an autoencoder), not pre-trained sentence embeddings. Free text excluded (Table S3).

→ No prior work justifies a sentence template. Following Bakumenko et al. is itself the justification (comparability); our changes are small and each has a reason.

## 4. Decision (23 Nov 2026, D16 in feature_engineering_plan.md)

**Rule:** the sentence embedding replaces only the "account + debit/credit per line" block of the one-hot encoding; all other columns stay identical and are appended (amounts per slot, n_lines, user, tax rate, promptly, weekend, nwh, lag_days). RQ2 then compares a semantic account representation (names) with one-hot codes, everything else equal.

**Format: Bakumenko et al. (2025) structure with three changes**

| # | Bakumenko et al. (2025) | This thesis | Why |
|---|---|---|---|
| 1 | account code (`1576`) | account name, unchanged from the chart of accounts | names carry accounting meaning; codes are arbitrary tokens; no manual cleaning of names |
| 2 | `Account_DC:` prefix on every line, side as `_D` / `_C` suffix | words `Debit` / `Credit` before the name, no repeated prefix | `_D` / `_C` are nearly meaningless tokens (patterns that differ only in side would look the same); a repeated prefix adds identical tokens to every sentence and, with mean pooling, makes all sentences more alike |
| 3 | separator `, ` | separator `; ` and a final `.` | account names contain commas ("Deductible input tax, 19%") |
| — | `Source:` field | not used | no source-system field in the dataset |

Example (entry `a_1`): `Debit Deductible input tax, 19%; Debit Cost of merchandise, 19% input tax; Credit Trade payables.`

**Templates**

| ID | Template | Role |
|---|---|---|
| **T-C** | sentence above → embedding + the non-account one-hot columns | **main setting** |
| T-A | the same sentence only, no columns | ablation |
| T-Bak | exact Bakumenko et al. format with codes: `Account_DC: 1576_D, Account_DC: 3400_D, Account_DC: 1600_C` (+ the same columns) | ablation: names vs codes, comparison with the reference paper |
| T-B | everything in the text, numbers as tokens | **dropped** |

**Data check (23 Nov):** the whole ledger produces only **29 distinct sentences** (same with names or codes) — the number of account / side patterns. Computing all templates takes seconds. Only M1 (cash-in-hand instead of trade payables) and M7 (rare bank account 1200 "Bank") change the account pattern; M3–M6 and M8 sit inside the five most frequent normal patterns, so only the columns can reveal them.
- M1: "Cash-in-hand" vs "Trade payables" differ in meaning → embedding should separate them.
- M7 risk: "Bank" (1200) vs normal "Bank (Postbank)" (1100) mean almost the same → embedding may hide M7, one-hot does not. Hypothesis for T-Bak; report as a finding, do not edit names.

**Why numbers stay out of the text (sources for Ch. 3):**
- Wallace, E., Wang, Y., Li, S., Singh, S., & Gardner, M. (2019). Do NLP models know numbers? Probing numeracy in embeddings. EMNLP-IJCNLP 2019, 5307–5315. https://aclanthology.org/D19-1534/ — models treat numbers like other tokens; word embeddings encode magnitude for numbers up to about 1,000; BERT (sub-word units) is less exact.
- Thawani, A., Pujara, J., Szekely, P., & Ilievski, F. (2021). Representing numbers in NLP: A survey and a vision. NAACL-HLT 2021. https://arxiv.org/abs/2103.13136 — NLP systems rarely give special consideration to numbers in text.
- Bakumenko et al. (2025), Limitations: the encoding focuses on categorical features and needs extension for numerical analysis.
- Turning numbers into words ("on time / delayed") is binning with fixed edges (same problem as `promptly`, D4).

**Why the user is a column, not text:** the user is needed (M1–M6 are each tied to an unusual user), but a person's name has no accounting meaning for a language model.

**Why the tax rate is not in the text:** fully determined by the accounts (L1); the account names already contain "19%".

**Thesis text (draft, Ch. 3):** "The sentence template follows Bakumenko et al. (2025): the lines of an entry are listed one after another. Three changes are made: the account name replaces the account code, the side is written as the word Debit or Credit instead of a suffix and without a repeated field prefix, and lines are separated by a semicolon because account names contain commas. As in Bakumenko et al., other templates are not searched; the exact original format with account codes is reported as an ablation."

**Rules so the comparison stays clean:**
- Any choice between templates uses FY1 labels only (D6); FY2 results reported for all templates that were run.
- Main pipeline (prefilter, combined score, H1, H2) uses only T-C; the others appear as a sensitivity table.
- Open: weight of the embedding block vs the column block (D17; PCA of the embedding to ≤ 28 components is lossless because of the 29 sentences).
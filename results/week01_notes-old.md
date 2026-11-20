# Week 01 notes (17 Nov 2026)

## JET implementation decisions
- Fiscal year = year of posting_date (51,064 entries in 2023, 51,365 in 2024; no entry spans two years).
- J1: entry amount = max line amount; population = entries >= the 1,000th highest per year, ties included
  (deterministic; the dataset's own top_n breaks ties at the threshold arbitrarily: 88 entries differ, all at the
  exact threshold amount). With ties J1 has 2,209 entries and catches all 120 M1 (dataset top_n: 118).
- J2: credit on cash-in-hand (1000) > EUR 1,500 — identical to dataset (402 entries).
- J3: entered date − document date >= 10 days — identical (8,207).
- J4: Sat/Sun or hour outside 08:00–17:59 — identical (16,493).

## Findings that change the proposal text
1. Prefilter recall ceiling is 93.9% (798/850), not 90.6%: M7/M8 were injected without a rule, but 22 of 60 M7
   and 6 of 20 M8 entries fall inside the JET union by chance. Proposal assumed all 80 lie outside.
   → report 93.9% as ceiling; RQ3 "outside JET" set = 52 anomalies (38 M7, 14 M8).
2. JET union = 24,329 entries (23.8% of ledger, ~12k per year), precision 3.28% → far above any budget, so
   ranking inside it is the main lever (supports RQ1).
3. M1–M6 are each tied to one user that is unusual for that context (Max: M1, M5; Fred: M2, M3, M6;
   David: M4); M7 uses rare bank account 1200; M8 is relational (ref_id). → user and account must be in both
   encodings; M8 will be hard for any entry-level encoding (as the preprint reports).
4. M1 entries all post to cash (1000) and all fall in J2 as well as J1.

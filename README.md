# gl-anomaly-ranking

**Reducing false positives in journal entry testing: hybrid rule-based + unsupervised ranking of general-ledger anomalies under review budgets.**

Master's thesis, M.Sc. Big Data and AI, SRH University Berlin. Mohsen Banadkouki (expected March 2027).

> ⚠️ Work in progress. Code, structure and results will change until the thesis is submitted.

---

## Motivation

Auditors use rule-based **journal entry tests (JETs)** to select suspicious bookings from the general ledger, for example entries with very high amounts, large cash credits, late recording or postings on weekends and outside working hours. These rules are easy to explain, but they flag far more entries than an auditor can review. On the dataset used here, the union of four JETs selects about 24% of all journal entries, and only about 3% of those are actual anomalies.

An auditor can review only a fixed number of entries (the **review budget**). This project ranks entries by combining the JET rules with unsupervised anomaly detection models, so that more true anomalies, and fewer false positives, fall inside the budget.

## Research questions

- **RQ1:** Inside the JET population, does an unsupervised ranking find more anomalies within a fixed review budget than the JET rules alone?
- **RQ2:** How does the entry encoding (one-hot with padding vs. sentence embeddings) affect ranking quality?
- **RQ3:** How well are anomalies outside all JET rules detected, and what does using JETs as a prefilter cost?

## Approach

1. **JETs:** J1 top amounts, J2 high cash credits, J3 late recording, J4 weekend or outside working hours.
2. **Encodings:** one-hot + padding of posting lines plus time features; sentence embeddings (`all-MiniLM-L6-v2`).
3. **Detectors:** Isolation Forest, Local Outlier Factor, Autoencoder.
4. **Methods:** JET-only baseline, model-only, JET prefilter + model, combined rule/model score.
5. **Evaluation:** budget-based metrics (recall and precision at k reviewed entries), per anomaly type, with statistical tests.

## Data

Put `journal_entries_4203.csv` in `data/raw/`. It is not committed.

> Gronewald, Stephan et al. (2026). Zenodo. doi: [10.5281/zenodo.22014653](https://doi.org/10.5281/zenodo.22014653). License: CC BY 4.0.

The dataset is synthetic and public. No company or confidential data is used in this repository.

## Layout

```
gl-anomaly-ranking/
├── data/
│   ├── raw/          # original CSV (not committed, never modified)
│   └── derived/      # generated features, keyed by posting_id (not committed)
├── src/              # reusable modules (no week prefix)
├── configs/          # run configurations (planned)
├── results/          # week reports, runs.csv, figures
├── docs/
│   └── publications/ # thesis and related papers, once approved by SRH
├── week01_validate_jets.py
├── requirements.txt
└── README.md
```

**Current modules**
- `src/load.py`: load posting lines and build one row per journal entry (raw-derived fields only)
- `src/jets.py`: J1–J4 implemented from raw fields (validation against the dataset rule attributes)
- `week01_validate_jets.py`: week 01 profiling + JET validation → `results/week01_jet_validation.md`

**Planned:** `src/encode.py`, `src/detectors.py`, `src/methods.py`, `src/metrics.py`, `src/stats.py`, `run.py`, `configs/`

## Rules

- **FY2023 = tuning year** (labels may be used for hyperparameters). **FY2024 = test year.** Fiscal year = year of `posting_date`.
- The JET populations used in the experiments come from the dataset rule attributes (`top_n`, `high_cash`, `promptly`, `weekend`, `nwh`), for comparability with Gronewald et al. (2026). The raw-field implementation in `src/jets.py` is used for validation only.
- The precomputed columns `promptly`, `weekend`, `nwh`, `top_n`, `high_cash`, `label` and `marking` are **never model features**. The rule attributes define only the JET flags, and `label` and `marking` are used only for evaluation.
- The raw CSV is never modified. Storage formats: Parquet for tables, `.npy` for dense arrays, `.npz` for sparse matrices and CSV for run results. No pickle.

## Run

```bash
pip install -r requirements.txt
python week01_validate_jets.py
```

## Naming

Scripts and outputs that belong to one week carry a zero-padded prefix (`week01_`, `week02_`, … `week16_`) so they sort correctly. Reusable modules in `src/` have no prefix.

## Status

| Step | Status |
|---|---|
| Week 01: data profiling, JET validation | ✅ done |
| Encodings (one-hot, embeddings) | 🔄 next |
| Detectors and methods | ⏳ planned |
| Evaluation and statistics | ⏳ planned |
| Thesis write-up | ⏳ planned |

## Publications

The final thesis and any related papers will be published in [`docs/publications/`](docs/publications/), subject to approval by SRH University Berlin.

## Supervision

Smitha (primary supervisor), Nazneen Mansoor (second supervisor). SRH University Berlin.

## License

Code: MIT (see `LICENSE`). The dataset keeps its own license (CC BY 4.0). Thesis documents in `docs/publications/` may follow different terms, which will be stated there.
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
├── notebooks/        # one notebook per process step (01_jets_meta, 02_encoding_onehot, ...)
├── src/              # reusable modules (no week prefix)
├── configs/          # run configurations (planned)
├── results/          # week reports, runs.csv, figures
├── docs/
│   └── publications/ # thesis and related papers, once approved by SRH
├── requirements.txt
└── README.md
```

**Current modules**
- `src/load.py`: load posting lines and build one row per journal entry (raw-derived fields only)
- `src/jets.py`: J1–J4 implemented from raw fields (validation against the dataset rule attributes, used in notebook 01)
- `src/encode.py`: one-hot + padding encoding with switchable feature groups (`select_columns`)
- `src/metrics.py`: budget-based metrics (P@k, R@k, FP@k, AP; random tie-break over 5 seeds) and `append_runs` → `results/runs.csv`
- `src/detectors.py`: Isolation Forest and LOF (score: higher = more suspicious), tuning grids (D15), per-seed runner
- `tests/test_metrics.py`: hand-checkable tests for the metrics (toy ranking computed on paper)
- `notebooks/01_jets_meta.ipynb`: JET flags from the dataset rule attributes, validation against `src/jets.py`, `data/derived/meta.parquet` + `jet_validation.csv`
- `notebooks/02_encoding_onehot.ipynb`: one-hot + padding encoding → `data/derived/X_onehot.npz` + `onehot_features.json`
- `notebooks/03_baselines.ipynb`: random, JET-only, Isolation Forest, LOF; tuned on FY2023, reported on FY2024 → `results/runs.csv`

**Planned:** `src/methods.py`, `src/stats.py`, `run.py`, `configs/`

## Rules

- **FY2023 = tuning year** (labels may be used for hyperparameters). **FY2024 = test year.** Fiscal year = year of `posting_date`.
- The JET populations used in the experiments come from the dataset rule attributes (`top_n`, `high_cash`, `promptly`, `weekend`, `nwh`), for comparability with Gronewald et al. (2026). The raw-field implementation in `src/jets.py` is used for validation only.
- `top_n`, `high_cash`, `label` and `marking` are **never model features**: `top_n` and `high_cash` define only the JET flags, and `label` and `marking` are used only for evaluation.
- `promptly`, `weekend` and `nwh` are model features (feature group `calendar_flags`, decision D4, to be confirmed by the supervisors), together with a continuous `lag_days`. Each feature group can be switched off per run, so the proposal variant without them stays available.
- Design decisions (D1–D13) with options and rationale: `results/feature_engineering_plan.md`.
- The raw CSV is never modified. Storage formats: Parquet for tables, `.npy` for dense arrays, `.npz` for sparse matrices and CSV for run results. No pickle.

## Run

```bash
pip install -r requirements.txt
python -m pytest tests -q        # metric tests must pass before any result is trusted
```

Then run the notebooks in `notebooks/` in order (01, 02, …). Each notebook finds the project root itself, so it runs from `notebooks/`, the project root or Colab.

## Naming

Notebooks are named by process step, not by week: `NN_step_name.ipynb` (`01_jets_meta`, `02_encoding_onehot`, `03_baselines`, …). Every code cell starts with a label comment `# cell_NN_KK` (notebook NN, code cell KK, e.g. `cell_02_07` = 7th code cell of notebook 02); the same label is the cell's Jupyter id. Labels are fixed once given: a new cell inserted later gets a letter suffix (`cell_02_07a`) so that existing references stay valid. Reusable modules in `src/` have no prefix. Only weekly reports and notes in `results/` carry a week prefix (`week01_`, `week02_`, …).

## Status

| Step | Status |
|---|---|
| Week 01: data profiling, JET validation | ✅ done |
| One-hot + padding encoding | ✅ done |
| Metrics, baselines, sentence embeddings | 🔄 next |
| Detectors and methods | ⏳ planned |
| Evaluation and statistics | ⏳ planned |
| Thesis write-up | ⏳ planned |

## Publications

The final thesis and any related papers will be published in [`docs/publications/`](docs/publications/), subject to approval by SRH University Berlin.

## Supervision

Smitha (primary supervisor), Nazneen Mansoor (second supervisor). SRH University Berlin.

## License

Code: MIT (see `LICENSE`). The dataset keeps its own license (CC BY 4.0). Thesis documents in `docs/publications/` may follow different terms, which will be stated there.
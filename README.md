# thesis-jet

Reducing False Positives in Journal Entry Testing: hybrid rule-based + unsupervised ranking under review budgets.
M.Sc. Big Data and AI, SRH Berlin — Mohsen Banadkouki.

## Data
Put `journal_entries_4203.csv` (Gronewald, Stephan et al., 2026; Zenodo doi 10.5281/zenodo.22014653, CC BY 4.0)
in `data/raw/`. Not committed.

## Layout
- `src/load.py`   – load posting lines, build one row per journal entry (raw-derived fields only)
- `src/jets.py`   – J1–J4 implemented from raw fields
- `week01_validate_jets.py` – week 01: profiling + JET validation → `results/week01_jet_validation.md`
- planned: `src/encode.py`, `src/detectors.py`, `src/methods.py`, `src/metrics.py`, `src/stats.py`, `run.py`, `configs/`

## Rules
- FY2023 = tuning year (labels allowed for hyperparameters), FY2024 = test year.
- Precomputed columns `promptly, weekend, nwh, top_n, high_cash, label, marking` are never model features.

## Run
    pip install -r requirements.txt
    python week01_validate_jets.py

## Naming
Scripts and outputs that belong to one week carry a zero-padded prefix (`week01_`, `week02_`, … `week16_`)
so they sort correctly. Reusable modules in `src/` have no prefix.

# Automated Anomaly Detection Project

An automated anomaly detection pipeline that selects and tunes models from data shape, then ensembles scores—aiming to reduce manual configuration for new CSV-like datasets.

## Features

- **Input:** CSV upload via FastAPI; in Python, a `DataFrame`, CSV path, or dict sources supported by `advanced_system.InputLayer` (API, SQLite).
- **Preprocessing:** Numeric columns only, missing values filled with `0.0`, `StandardScaler`, and `PCA` retaining ~95% explained variance when multiple features exist.
- **Models:** Isolation Forest, One-Class SVM, Local Outlier Factor; for larger tabular data, PyTorch **Autoencoder** and **LSTM autoencoder** may be included.
- **Optimization:** Short Optuna search trials; ensemble score normalization and weighting; configurable percentile threshold with default at the **95th percentile** of combined scores.
- **UI:** **`http://127.0.0.1:8000/`** → `/ui/` — separate **EDA** CSV + **`POST /eda`**, optional synthetic **`POST /synthetic-preview`** / **`POST /synthetic-export`**, then independent **pipeline** CSV + **`POST /upload`**; anomaly summary, evaluation metrics when a binary label column is present, and charts show only after a successful upload run.
- **EDA quality metrics:** Pearson + Spearman correlation slices, top-N correlated pairs, per-numeric Tukey/|z|>3 outlier counts, kurtosis, duplicate-row count, categorical top-k frequencies, datetime-like detection, high-skew log-transform hints (see [docs/USAGE.md](docs/USAGE.md) `POST /eda`).
- **Synthetic evaluation:** `api/synthetic_injection.py` and dashboard **Synthetic anomaly (preview)** (`POST /synthetic-preview`, `POST /synthetic-export`) — eight scenarios covering numeric perturbations (`spike_single`, `joint_shift`, `scale_burst`, `dead_sensor`, `sign_flip`, `temporal_block`), **categorical** corruption (`categorical_flip`), and **missing-value** injection on any dtype (`missing_value`); plus `binary_score_metrics` (ROC-AUC, PR-AUC) for continuous scores. See [docs/USAGE.md](docs/USAGE.md) and [docs/SYNTHETIC_SCENARIOS.md](docs/SYNTHETIC_SCENARIOS.md).

- **Benchmarking:** `scripts/run_synthetic_benchmark.py` injects scenarios and writes `results/synthetic_benchmark_summary.csv` with ensemble + per-model metrics and best percentile threshold sweeps.

## Project layout

```
api/
  main.py                # FastAPI: GET / → /ui/, POST /upload, /eda, synthetic preview/export, static /ui/
  eda_report.py          # JSON EDA payload for POST /eda (no ML)
  advanced_system.py   # Pipeline layers and AdvancedAnomalySystem
  synthetic_injection.py # inject(), merged_params(), binary metrics, SCENARIO_DEFAULTS
ui/
  index.html             # Dashboard (analysis + synthetic preview)
data/
  test_data.txt          # Sample CSV (numeric metrics + optional label column)
  synthetic_examples/    # Small baseline CSVs (see docs/SYNTHETIC_SCENARIOS.md)
  external/              # Downloaded CSVs (gitignored except README.md)
docs/
  USAGE.md
  ROADMAP.md
  SYNTHETIC_SCENARIOS.md
  fetch_public_datasets.py
scripts/
  run_synthetic_benchmark.py
results/
  synthetic_benchmark_summary.csv
requirements.txt
LICENSE
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

1. Start the API from the `api` directory:

   ```bash
   cd api
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

2. Open `http://127.0.0.1:8000/docs` to try `POST /upload`, or open **`http://127.0.0.1:8000/`** in a browser for the dashboard (redirects to `/ui/`).

For detailed usage, example `curl` commands, and programmatic calls, see **[docs/USAGE.md](docs/USAGE.md)**. For the course completion plan (synthetic injection, robustness benchmarks, phased delivery), see **[docs/ROADMAP.md](docs/ROADMAP.md)**.

## Dependencies

`fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `numpy`, `python-multipart`, `optuna`, `torch` — consider pinning versions in `requirements.txt` for reproducibility.

## License

Use this project according to the `LICENSE` file in the repository.

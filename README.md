# Automated Anomaly Detection Project

An automated anomaly detection pipeline that selects and tunes models from data shape, then ensembles scores—aiming to reduce manual configuration for new CSV-like datasets.

## Features

- **Input:** CSV upload via FastAPI; in Python, a `DataFrame`, CSV path, or dict sources supported by `advanced_system.InputLayer` (API, SQLite).
- **Preprocessing:** Numeric columns only, missing values filled with `0.0`, `StandardScaler`, and `PCA` retaining ~95% explained variance when multiple features exist.
- **Models:** Isolation Forest, One-Class SVM; for larger tabular data, PyTorch **Autoencoder** and **LSTM autoencoder** may be included.
- **Optimization:** Short Optuna search trials; ensemble score normalization and weighting; default anomaly threshold at the **95th percentile** of combined scores.
- **UI:** **`http://127.0.0.1:8000/`** → `/ui/` — separate **EDA** CSV + **`POST /eda`**, optional synthetic **`POST /synthetic-preview`** / **`POST /synthetic-export`**, then independent **pipeline** CSV + **`POST /upload`**; anomaly summary and charts show only after a successful upload run.
- **Synthetic evaluation:** `api/synthetic_injection.py` and dashboard **Synthetic anomaly (preview)** (`POST /synthetic-preview`, `POST /synthetic-export`) — see [docs/USAGE.md](docs/USAGE.md) and [docs/SYNTHETIC_SCENARIOS.md](docs/SYNTHETIC_SCENARIOS.md).

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

# Automated Anomaly Detection Project

An automated anomaly detection pipeline that selects and tunes models from data shape, then ensembles scores—aiming to reduce manual configuration for new CSV-like datasets.

## Features

- **Input:** CSV upload via FastAPI; in Python, a `DataFrame`, CSV path, or dict sources supported by `advanced_system.InputLayer` (API, SQLite).
- **Preprocessing:** Numeric columns only, missing values filled with `0.0`, `StandardScaler`, and `PCA` retaining ~95% explained variance when multiple features exist.
- **Models:** Isolation Forest, One-Class SVM; for larger tabular data, PyTorch **Autoencoder** and **LSTM autoencoder** may be included.
- **Optimization:** Short Optuna search trials; ensemble score normalization and weighting; default anomaly threshold at the **95th percentile** of combined scores.
- **UI:** `ui/index.html` — file pick / drag-and-drop, summary, chart, and results table (calls `http://127.0.0.1:8000/upload`).

## Project layout

```
api/
  main.py              # FastAPI app, POST /upload
  advanced_system.py   # Pipeline layers and AdvancedAnomalySystem
ui/
  index.html           # Static dashboard
data/
  test_data.txt        # Sample CSV (numeric metrics + optional label column)
docs/
  USAGE.md             # Detailed usage (API, UI, Python, troubleshooting)
  ROADMAP.md           # Completion plan (synthetic anomalies, robustness, phases)
requirements.txt
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

2. Open `http://127.0.0.1:8000/docs` to try `POST /upload`, or open `ui/index.html` in a browser to upload a CSV.

For detailed usage, example `curl` commands, and programmatic calls, see **[docs/USAGE.md](docs/USAGE.md)**. For the course completion plan (synthetic injection, robustness benchmarks, phased delivery), see **[docs/ROADMAP.md](docs/ROADMAP.md)**.

## Dependencies

`fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `numpy`, `python-multipart`, `optuna`, `torch` — consider pinning versions in `requirements.txt` for reproducibility.

## License

Use this project according to the `LICENSE` file in the repository.

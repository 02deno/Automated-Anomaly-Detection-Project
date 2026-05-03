# Usage guide

This document covers the API, web UI, and programmatic usage. For a high-level overview, see the root [README.md](../README.md). For the completion roadmap (synthetic anomalies, robustness study, phased checklist), see [ROADMAP.md](ROADMAP.md).

## Prerequisites

- Python 3.10 or newer is recommended.
- Install dependencies: `pip install -r requirements.txt` (from the repository root).

## Running the server (FastAPI)

Because `main.py` lives under `api`, the most reliable approach is to run Uvicorn with `api` as the working directory:

```bash
cd api
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- Base URL: `http://127.0.0.1:8000`
- Interactive API docs: `http://127.0.0.1:8000/docs` (Swagger UI)

### `POST /upload`

- **Body:** `multipart/form-data` with field name `file` containing a CSV file.
- **Expected CSV:** At least one numeric column. Non-numeric columns are ignored during preprocessing. Missing numeric values are filled with `0.0`.

**Example (`curl`, single line — from inside `api`):**

```bash
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@../data/test_data.txt"
```

On Windows CMD you can break lines with `^` at the end of each line. In PowerShell, `curl` is often an alias for `Invoke-WebRequest`; use the real curl binary:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload" -F "file=@../data/test_data.txt"
```

**Example response (main fields):**

| Field | Description |
|--------|-------------|
| `anomaly_count` | Number of rows flagged as anomalies |
| `sample_scores` / `sample_anomalies` | Preview for the first 20 rows |
| `summary` | Includes `total_samples`, `anomaly_ratio`, `model_weights`, `top_scores`, etc. |
| `full_data` | Original rows plus `anomaly_score` and `is_anomaly` as returned table values |
| `full_scores` / `full_anomalies` | Per-row scores and binary labels |

## Web UI (`ui/index.html`)

1. Start the FastAPI server as above (`127.0.0.1:8000`).
2. Open `ui/index.html` in a browser (double-click or serve via a static file server).
3. Select or drag-and-drop a CSV, then click **Run Analysis**.

The UI posts to `http://127.0.0.1:8000/upload`. If you change host or port, update the `fetch` URL in `ui/index.html`.

## Programmatic usage (Python)

`AdvancedAnomalySystem` accepts a `DataFrame`, a CSV path, or any dict source supported by `InputLayer`:

```python
import pandas as pd
from advanced_system import AdvancedAnomalySystem

df = pd.read_csv("data/test_data.txt")
system = AdvancedAnomalySystem()
anomalies, scores, details = system.run(df)
print(details["report"])
```

To run the built-in sample from the shell:

```bash
cd api
python advanced_system.py
```

That script generates random sample data and prints the report.

## Data and labels

- The pipeline is **unsupervised** for labeling: it does not use training labels; it only scores numeric features.
- The `label` column in `data/test_data.txt` is for illustration or evaluation only; output `is_anomaly` is independent of it.

## Troubleshooting

| Issue | Suggestion |
|--------|------------|
| `ModuleNotFoundError` on import | Run commands from the `api` folder or add the project root to `PYTHONPATH`. |
| UI request fails | Confirm the server is on port 8000 and the browser can reach `http://127.0.0.1:8000` (not `file://` restrictions blocking fetch, depending on how you open the page). |
| Slow first run | PyTorch and Optuna startup plus trial loops add latency; test with a small CSV first. |

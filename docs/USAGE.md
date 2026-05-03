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
- Dashboard (static UI): `http://127.0.0.1:8000/` redirects to `http://127.0.0.1:8000/ui/` (`ui/index.html`). Use this URL instead of opening the HTML file from disk so `/upload` resolves on the same origin.

### `POST /synthetic-preview`

- **Body:** `multipart/form-data` with field name `file` (CSV) plus optional form fields:
  - `scenario`: `spike_single` | `joint_shift` | `scale_burst` (default `spike_single`)
  - `random_seed`: integer (default `42`)
  - `preview_rows`: integer, max 80 (default `20`) — how many leading rows to return for before/after tables
  - `contamination`, `magnitude_in_std`, `scale_factor`: optional strings parsed as floats; omit to use scenario defaults
  - `column`: optional single column name for `spike_single`
  - `columns`: optional comma-separated names for `joint_shift` / `scale_burst`
- **Response:** JSON with `before_preview`, `after_preview`, `y_true_preview`, `injected_row_indices`, `cell_changes_sample`, `params_effective`, `explanation`, etc. Used by the dashboard “Synthetic anomaly (preview)” card.

### `POST /synthetic-export`

- **Body:** Same `multipart/form-data` fields as **`/synthetic-preview`** except **`preview_rows`** is ignored (not present on this route). Same `file`, `scenario`, `random_seed`, and optional override fields.
- **Response:** `text/csv` attachment — the **full** table after injection (all rows), same column layout as the upload. Filename is like `synthetic_after_spike_single_seed42.csv`. Use this file as the input to **`POST /upload`** (or open it in Python) when you want the full pipeline on corrupted data, not just the preview window.

**Example (`curl`) — download corrupted CSV (from inside `api`; single line):**

```bash
curl.exe -L -o corrupted.csv -X POST "http://127.0.0.1:8000/synthetic-export" -F "file=@../data/test_data.txt" -F "scenario=spike_single" -F "random_seed=42"
```

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

**Example (`curl`) — synthetic preview only (from inside `api`; single line):**

```bash
curl.exe -X POST "http://127.0.0.1:8000/synthetic-preview" -F "file=@../data/test_data.txt" -F "scenario=spike_single" -F "random_seed=42" -F "preview_rows=10"
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
2. Open **`http://127.0.0.1:8000/`** (redirects to `/ui/`) or go directly to **`http://127.0.0.1:8000/ui/`**.
3. **Synthetic anomaly (preview)** (top card): choose a CSV for preview (can differ from analysis). Set scenario / seed / optional parameters, then **Preview synthetic injection** — **`POST /synthetic-preview`** (before/after tables; no full model). Use **Download full corrupted CSV** — **`POST /synthetic-export`** — to save the entire perturbed file with the same settings. If you skip the synthetic-zone CSV, preview and export fall back to the analysis CSV when that one is set.
4. **Full pipeline analysis** (second card): select CSV for **`POST /upload`** (ensemble + Optuna + optional deep models), then **Run Analysis**. To evaluate on injected anomalies, either upload the exported corrupted CSV here or run `inject()` in Python and pass the returned frame to `AdvancedAnomalySystem().run(...)`.

Opening `ui/index.html` via `file://` will not reach the API; always use the URLs in step 2.

## Programmatic usage (Python)

Run snippets from the **`api/`** directory (same as Uvicorn) so imports resolve.

`AdvancedAnomalySystem` accepts a `DataFrame`, a CSV path, or any dict source supported by `InputLayer`:

```python
import pandas as pd
from advanced_system import AdvancedAnomalySystem

df = pd.read_csv("../data/test_data.txt")
system = AdvancedAnomalySystem()
anomalies, scores, details = system.run(df)
print(details["report"])
```

From the **repository root**, you can instead set `PYTHONPATH=api` or `cd api` before importing.

To run the built-in random sample bundled in `advanced_system.py`:

```bash
cd api
python advanced_system.py
```

That script generates random sample data and prints the report.

## Synthetic evaluation (Phase 1)

Inject controlled anomalies, run the same `AdvancedAnomalySystem` used by the API, and score predictions against synthetic ground truth (`y_true`). This is for **benchmarking and development**, not a claim about real-world anomaly prevalence. For **what each scenario does**, **real-world analogies**, and **example CSVs**, see [SYNTHETIC_SCENARIOS.md](SYNTHETIC_SCENARIOS.md).

**API (Python):**

- Module: `api/synthetic_injection.py`
- `inject(df, scenario, random_seed=..., params=None)` returns `(corrupted_df, y_true)`; the input frame is not mutated.
- `merged_params(scenario, overrides)` returns the effective parameter dict (defaults + overrides).
- `binary_classification_metrics(y_true, y_pred)` returns precision, recall, and F1 (binary).
- Built-in scenarios: `spike_single`, `joint_shift`, `scale_burst` (see `SCENARIO_DEFAULTS` in that module). Use `list_scenarios()` to list ids.

To score the full pipeline on a corrupted frame in Python, call `inject()` then `AdvancedAnomalySystem().run(corrupted_df)` and compare `y_true` to `details["results"]["is_anomaly"]` (see `api/synthetic_injection.binary_classification_metrics`).

A full `AdvancedAnomalySystem` run is slow on first execution (Optuna + PyTorch); use a small CSV or the **synthetic preview** endpoint when you only need before/after tables.

## Public benchmark datasets (download)

`scripts/fetch_public_datasets.py` downloads public CSVs into `data/external/` (gitignored except `data/external/README.md`). Sources include UCI, scikit-learn’s KDD Cup 1999 helper, and **Annthyroid** from [mala-lab/ADBenchmarks-anomaly-detection-datasets](https://github.com/mala-lab/ADBenchmarks-anomaly-detection-datasets) (DevNet numerical folder).

```bash
python scripts/fetch_public_datasets.py --dataset adb_annthyroid
python scripts/fetch_public_datasets.py --dataset all
```

See `data/external/README.md` for output filenames, label column conventions (`ground_truth`), license notes, and which ADBenchmarks files are large enough that you should fetch them manually from GitHub instead of extending the script.

## Data and labels

- The pipeline is **unsupervised** for labeling: it does not use training labels; it only scores numeric features.
- The `label` column in `data/test_data.txt` is for illustration or evaluation only; output `is_anomaly` is independent of it.

## Troubleshooting

| Issue | Suggestion |
|--------|------------|
| `ModuleNotFoundError` on import | Run commands from the `api` folder or add the project root to `PYTHONPATH`. |
| UI request fails | Confirm the server is on port 8000 and the browser can reach `http://127.0.0.1:8000` (not `file://` restrictions blocking fetch, depending on how you open the page). |
| Slow first run | PyTorch and Optuna startup plus trial loops add latency; use a small CSV first. |
| `GET /` returns JSON “Not Found” | Use a current `main.py`: `/` should redirect to `/ui/`. Restart Uvicorn after pulling changes. |

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
  - `scenario`: `spike_single` | `joint_shift` | `scale_burst` | `dead_sensor` | `sign_flip` | `temporal_block` | `categorical_flip` | `missing_value` (default `spike_single`)
  - `random_seed`: integer (default `42`)
  - `preview_rows`: integer, max 80 (default `20`) — how many leading rows to return for before/after tables
  - `contamination`, `magnitude_in_std`, `scale_factor`, `constant`, `block_count`: optional strings parsed as numbers; omit to use scenario defaults. `constant` applies to `dead_sensor`; `block_count` to `temporal_block`.
  - `mode`: optional `swap` | `sentinel` for `categorical_flip`
  - `sentinel`: optional string (default `__UNKNOWN__`) for `categorical_flip` with `mode=sentinel`
  - `column`: optional single column name for `spike_single` and `categorical_flip`
  - `columns`: optional comma-separated names for `joint_shift` / `scale_burst` / `dead_sensor` / `sign_flip` / `temporal_block` / `missing_value`
- **Response:** JSON with `before_preview`, `after_preview`, `y_true_preview`, `injected_row_indices`, `cell_changes_sample`, `params_effective`, `explanation`, `numeric_columns`, `non_numeric_columns`, etc. Used by the dashboard “Synthetic anomaly (preview)” card.

### `POST /synthetic-export`

- **Body:** Same `multipart/form-data` fields as **`/synthetic-preview`** except **`preview_rows`** is ignored (not present on this route). Same `file`, `scenario`, `random_seed`, and optional override fields (`contamination`, `magnitude_in_std`, `scale_factor`, `column`, `columns`, `constant`, `mode`, `sentinel`, `block_count`).
- **Response:** `text/csv` attachment — the **full** table after injection (all rows), same column layout as the upload. Filename is like `synthetic_after_spike_single_seed42.csv`. Use this file as the input to **`POST /upload`** (or open it in Python) when you want the full pipeline on corrupted data, not just the preview window.

**Example (`curl`) — download corrupted CSV (from inside `api`; single line):**

```bash
curl.exe -L -o corrupted.csv -X POST "http://127.0.0.1:8000/synthetic-export" -F "file=@../data/test_data.txt" -F "scenario=spike_single" -F "random_seed=42"
```

### `POST /eda`

- **Body:** `multipart/form-data` with field name `file` (CSV).
- **Response:** JSON suitable for the dashboard EDA card. Top-level keys:
  - `row_count_raw`, `row_count_used`, `sampled` (true if the first 50,000 rows were used), `column_count`.
  - `duplicate_row_count`, `duplicate_row_pct` — exact-row duplicate detection on the working slice.
  - `columns` — per-column `name`, `dtype`, `missing_count`, `missing_pct`, `is_numeric`, `nunique`.
  - `numeric_column_names`, `numeric_summary` — per-column `count`, `mean`, `std`, `min`, `p25`, `median`, `p75`, `max`, `skew`, `kurtosis`, `tukey_outlier_count` (1.5·IQR fences), `z_outlier_count` (|z|>3 with population std).
  - `categorical_summary` — for up to 8 non-numeric columns: `column`, `nunique`, top-10 `top: [{value, count, pct}]`, and `rare_count` (categories appearing in <1% of rows).
  - `datetime_columns` — object-dtype columns where ≥80% of values parse as datetimes: `column`, `parsed_pct`, `min`, `max`, `range_days`.
  - `correlation` (Pearson) and `correlation_spearman` — same column slice (up to 14 highest-variance numerics): `columns` + `matrix`.
  - `top_correlations` — up to 10 numeric pairs ranked by `|pearson|`: `a`, `b`, `pearson`, `spearman`.
  - `scatter` — `x_column`, `y_column`, `pearson_r`, `x`/`y` arrays (up to 2,800 sampled rows) for the strongest |Pearson| pair.
  - `boxplots` — Tukey fence stats for up to six high-variance numerics (client-side box drawings).
  - `histograms` — up to six high-variance numerics with bin counts.
  - `warnings` — high missingness, near-constant columns, high-skew log-transform hint, rare categories, duplicate rows, etc.

**No** anomaly models or Optuna.

### `POST /upload`

- **Body:** `multipart/form-data` with field name `file` containing a CSV file, plus optional `threshold_percentile` (default `95.0`, clamped to `50.0`-`99.9`).
- **Expected CSV:** At least one numeric column. Non-numeric columns are ignored during preprocessing. Missing numeric values are filled with `0.0`.
- **Optional labels for visible evaluation:** if the CSV contains a binary column named `label`, `target`, `ground_truth`, `y_true`, `is_anomaly`, or `anomaly`, `/upload` excludes that column from model features and returns precision, recall, F1, accuracy, ROC-AUC, PR-AUC, and a confusion matrix under `evaluation`.

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
| `columns` | Column names for `full_data`, including returned `anomaly_score` and the prediction column |
| `full_data` | Original rows plus `anomaly_score` and `is_anomaly` as returned table values (`predicted_is_anomaly` if the input already had an `is_anomaly` label column) |
| `full_scores` / `full_anomalies` | Per-row scores and binary labels |
| `evaluation` | If a binary label-like column exists, visible metrics: precision, recall, F1, accuracy, ROC-AUC, PR-AUC, and `tp`/`fp`/`tn`/`fn`; otherwise a note explaining which columns were searched |
| `threshold` | Combined-score cutoff; default rule is the **95th percentile** (`PostProcessingLayer.threshold` in `advanced_system.py`) — expect roughly **~5%** of rows flagged in typical continuous-score settings, independent of synthetic contamination. |
| `models_used` | List of model ids run in that request (e.g. `iforest`, `ocsvm`, `lof`, and optionally `autoencoder`, `lstm`). |
| `meta` | Dataset profile from `AnalysisLayer.analyze` (sample count, numeric feature count, missing rate, variance, skewness, kurtosis, correlation, sparsity, entropy, numeric column names, etc.). |
| `threshold_rule` / `threshold_note` | Machine-readable rule id and a short human explanation for the UI. |

Pass `threshold_percentile` to `/upload` when you want a less strict or more strict detector. Lower values generally increase recall and false positives; higher values generally increase precision and miss more borderline anomalies.

## Web UI (`ui/index.html`)

1. Start the FastAPI server as above (`127.0.0.1:8000`).
2. Open **`http://127.0.0.1:8000/`** (redirects to `/ui/`) or go directly to **`http://127.0.0.1:8000/ui/`**.
3. **EDA** (first card): upload a CSV in the **EDA-only** drop zone (independent of the pipeline file), then **Run EDA** — **`POST /eda`** returns column overview, missingness, numeric summary, correlation heatmap, **scatter** for the strongest linear pair, **Tukey boxplot** stats (drawn in the UI), and histograms. No Optuna.
4. **Synthetic anomaly (preview)** (second card): optional separate CSV (or reuse the pipeline file). **Preview** / **Export** use **`POST /synthetic-preview`** / **`POST /synthetic-export`**. If the synthetic zone has no file, preview falls back to the pipeline CSV once it is selected in the pipeline card.
5. **Full pipeline analysis** (third card): same CSV as EDA, then **Run Analysis** — **`POST /upload`**. Summary, evaluation metrics when labels are present, score charts, profile card, weights table, and download appear **only after** a successful run (the block stays hidden until then).

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
- `binary_score_metrics(y_true, scores)` returns ROC-AUC and PR-AUC (Average Precision) for **continuous** anomaly scores — threshold-independent ranking metrics. Returns `0.0` safely when `y_true` has only one class or scores are constant.
- Built-in scenarios: `spike_single`, `joint_shift`, `scale_burst`, `dead_sensor`, `sign_flip`, `temporal_block`, `categorical_flip`, `missing_value` (see `SCENARIO_DEFAULTS` in that module). Use `list_scenarios()` to list ids.

To score the full pipeline on a corrupted frame in Python, call `inject()` then `AdvancedAnomalySystem().run(corrupted_df)` and compare `y_true` to `details["results"]["is_anomaly"]` (binary metrics) **or** to the continuous combined score for ROC-AUC / PR-AUC via `binary_score_metrics`.

A full `AdvancedAnomalySystem` run is slow on first execution (Optuna + PyTorch); use a small CSV or the **synthetic preview** endpoint when you only need before/after tables.

### Synthetic benchmark script

Run the reproducible benchmark from the repository root:

```bash
python scripts/run_synthetic_benchmark.py
```

The script injects common numeric scenarios into `data/synthetic_examples/baseline_server_metrics.csv`, runs the full pipeline once per scenario, and writes `results/synthetic_benchmark_summary.csv`.

The CSV contains one row for the ensemble and one row for each component model score source (`iforest`, `ocsvm`, `lof`, and any selected neural model). It reports precision, recall, F1, ROC-AUC, PR-AUC, injected/detected row indices, and a percentile sweep (`best_percentile`, `best_f1`) to show whether ranking is good but the fixed threshold is too strict.

## Public benchmark datasets (download)

`scripts/fetch_public_datasets.py` downloads public CSVs into `data/external/` (gitignored except `data/external/README.md`). Sources include UCI, scikit-learn’s KDD Cup 1999 helper, and **Annthyroid** from [mala-lab/ADBenchmarks-anomaly-detection-datasets](https://github.com/mala-lab/ADBenchmarks-anomaly-detection-datasets) (DevNet numerical folder).

```bash
python scripts/fetch_public_datasets.py --dataset adb_annthyroid
python scripts/fetch_public_datasets.py --dataset all
```

See `data/external/README.md` for output filenames, label column conventions (`ground_truth`), license notes, and which ADBenchmarks files are large enough that you should fetch them manually from GitHub instead of extending the script.

## Data and labels

- The pipeline is **unsupervised** for labeling: it does not train on labels; it only scores feature columns.
- During `POST /upload`, binary label-like columns (`label`, `target`, `ground_truth`, `y_true`, `is_anomaly`, `anomaly`) are excluded from model features and used only to calculate the visible `evaluation` metrics.
- The `label` column in `data/test_data.txt` is for illustration or evaluation only; output `is_anomaly` is independent of it.

## Troubleshooting

| Issue | Suggestion |
|--------|------------|
| `ModuleNotFoundError` on import | Run commands from the `api` folder or add the project root to `PYTHONPATH`. |
| UI request fails | Confirm the server is on port 8000 and the browser can reach `http://127.0.0.1:8000` (not `file://` restrictions blocking fetch, depending on how you open the page). |
| Slow first run | PyTorch and Optuna startup plus trial loops add latency; use a small CSV first. |
| `GET /` returns JSON “Not Found” | Use a current `main.py`: `/` should redirect to `/ui/`. Restart Uvicorn after pulling changes. |

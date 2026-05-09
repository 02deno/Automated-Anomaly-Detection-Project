# Project roadmap

This roadmap aligns the codebase with course expectations: **synthetic anomaly injection** for controlled evaluation and **comparative model robustness** as a clear contribution. For day-to-day usage of the API and UI, see [USAGE.md](USAGE.md).

---

## Goals (summary)

| Theme | Intent |
|--------|--------|
| Synthetic anomalies | Build reproducible ground-truth labels to benchmark the pipeline; real-world-only data makes iteration and measurement hard. |
| Robustness | Compare component models (and optionally the ensemble) under multiple injection scenarios and noise/contamination levels. |
| Delivery | Reproducible scripts, documented commands, figures/tables for the report and presentation. |

---

## Phase 1 — Synthetic anomaly injection (detailed steps)

Phase 1 delivers a **deterministic injector** and a **minimal evaluation hook** against the existing `AdvancedAnomalySystem`.

### Step 1.1 — Define anomaly scenario catalog

- [x] List **3+ distinct scenario types** (e.g. single-feature spike, multi-feature joint shift, optional time-series break if data is temporal).
- [x] For each scenario, specify **parameters**: fraction of injected rows, magnitude (e.g. in standard deviations or scale factor), which columns are affected, optional correlation across features.
- [x] Document parameter defaults in code docstrings or a small `configs/synthetic_defaults.yaml` (optional).

### Step 1.2 — Specify contract (inputs / outputs)

- [x] **Input:** `pd.DataFrame` (clean baseline), `random_seed: int`, scenario id + parameter dict.
- [x] **Output:** corrupted `DataFrame` (copy; do not mutate caller’s frame in place unless explicitly documented), `y_true: np.ndarray` of shape `(n_rows,)` with `1` = injected anomaly, `0` = normal.
- [x] Guarantee: injected row indices are a subset of rows; **no accidental duplicate** label logic (one row = one label).

### Step 1.3 — Implement core injector module

- [x] Add a dedicated module (e.g. `api/synthetic_injection.py` or `src/synthetic/`) with a class or functions: `inject(df, scenario, seed) -> tuple[pd.DataFrame, np.ndarray]`.
- [x] Start with **one scenario** (e.g. spike on one numeric column) end-to-end; add others incrementally.
- [x] Use only **numeric columns** from the frame for perturbation; align with how `AdvancedAnomalySystem` preprocesses data.

### Step 1.4 — Determinism and validation

- [x] Fix `numpy` / `random` seed at the start of each injection path; same seed + same config ⇒ **identical** output.
- [x] **Manual / visual checks:** `POST /synthetic-preview` (or `inject()` in Python) to confirm row count unchanged, non-injected rows match the baseline, and `y_true.sum()` matches the intended injection count when you need to verify behavior.

### Step 1.5 — Wire to the detection pipeline

- [x] Run `AdvancedAnomalySystem().run(corrupted_df)` and obtain `is_anomaly` (or per-model scores if exposed in a later phase) — **documented** in [USAGE.md](USAGE.md) as a short Python snippet.
- [x] **Precision / recall / F1** helpers exist (`binary_classification_metrics`); end-to-end F1 on the full pipeline is left to your notebook or future `experiments/` runner.
- [x] **HTTP + UI:** `POST /synthetic-preview` returns before/after previews and explanations without running the heavy ensemble.

### Step 1.6 — Documentation and reproducibility

- [x] Extend [USAGE.md](USAGE.md) with a short **“Synthetic evaluation”** subsection: injection API and UI preview.
- [x] Note limitations: synthetic labels are for **evaluation**, not a claim about real-world prevalence.

**Phase 1 exit criteria:** Deterministic `inject()` + three scenarios + **dashboard / `POST /synthetic-preview`** for human-readable before–after; optional full-pipeline F1 via local Python when you need numbers for the report.

---

## Phase 2 — Evaluation framework

- [x] Standardize metrics: precision, recall, F1, ROC-AUC, PR-AUC for synthetic ground truth.
- [x] Python benchmark entrypoint with dataset path, scenario, seed, and CSV output: `python scripts/run_synthetic_benchmark.py`.
- [x] Experiment config (YAML): contamination sweeps, multiple seeds, public dataset presets — `configs/experiments/{quick,robustness}.yaml` driven via `--config`.

---

## Phase 3 — Model robustness comparison

- [x] Compare component score sources (`iforest`, `ocsvm`, `lof`, and selected neural models) plus the ensemble on the same `y_true`.
- [x] Sweeps: **contamination** (e.g. 1%, 5%, 10%) and **magnitude_in_std** declared in `robustness.yaml`; **Gaussian noise** as a `noise_std` list applied via `add_feature_noise()` before injection.
- [x] Aggregation: `results/<...>_aggregated.csv` collapses across seeds (`f1_mean ± std`, `roc_auc_mean ± std`); `results/<...>_worst_case.csv` ranks score sources by worst-case F1 — the headline "robustness floor" table.
- [x] Produce heatmaps/figures and summarize one clear takeaway for the report — `scripts/plot_robustness.py` writes scenario × model heatmaps and per-parameter sweep curves under `results/figures/`.

---

## Phase 4 — Real-data sanity check (limited)

- [x] Labeled real datasets are downloaded into `data/external/` via `scripts/fetch_public_datasets.py` (Annthyroid, KDD'99 SMTP/HTTP, UCI Glass, Pen Digits).
- [x] `scripts/run_real_data_eval.py` runs the unsupervised pipeline, drops the label column, and reports ROC-AUC / PR-AUC per score source against `ground_truth`.
- [x] `--inject` flag emits `results/real_vs_synthetic_consistency.csv` plus a per-dataset top-1 agreement line, answering "does the model ranking from synthetic injection survive on real data?".

---

## Phase 5 — Course deliverables

- [x] `tests/test_synthetic_injection.py` and `tests/test_benchmark_aggregation.py` lock down determinism / no-mutation / aggregation contracts (`pytest -q`).
- [x] Update README / USAGE for any new commands; keep docs in **English** per project rules — done in this iteration.
- [ ] Presentation: add slides for synthetic protocol, robustness heatmap, and real-vs-synthetic agreement table.

---

## Suggested timeline (example)

| Week | Focus |
|------|--------|
| 1 | Phase 1.1–1.4 (catalog, contract, injector, validation via preview). |
| 2 | Phase 1.5–1.6 + Phase 2 skeleton (config + one full benchmark run). |
| 3 | Phase 3 matrices and plots; draft results section. |
| 4 | Phase 4–5, doc pass, demo recording or live checklist. |

---

## References in-repo

- Pipeline: `api/advanced_system.py`
- API + static UI: `api/main.py` (`/`, `/upload`, `/synthetic-preview`, `/ui/…`)
- Synthetic injection + metrics + Gaussian noise helper: `api/synthetic_injection.py`
- Benchmark orchestrator (single-run + YAML grid): `scripts/run_synthetic_benchmark.py`
- Robustness figures: `scripts/plot_robustness.py`
- Real-data evaluator: `scripts/run_real_data_eval.py`
- Experiment configs: `configs/experiments/{quick,robustness}.yaml`
- Tests: `tests/test_synthetic_injection.py`, `tests/test_benchmark_aggregation.py`
- Usage: [USAGE.md](USAGE.md)

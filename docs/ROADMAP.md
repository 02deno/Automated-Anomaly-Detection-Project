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
- [ ] Experiment config (YAML or richer CLI): contamination sweeps, multiple seeds, public dataset presets.

---

## Phase 3 — Model robustness comparison

- [x] Compare component score sources (`iforest`, `ocsvm`, `lof`, and selected neural models) plus the ensemble on the same `y_true`.
- [ ] Sweeps: **contamination** (e.g. 1%, 5%, 10%); optional **Gaussian noise** on all numeric features.
- [x] Produce a summary table in `results/synthetic_benchmark_summary.csv` with F1, ROC-AUC, PR-AUC, and best percentile threshold.
- [ ] Produce heatmaps/figures and summarize one clear takeaway for the report.

---

## Phase 4 — Real-data sanity check (limited)

- [ ] If a labeled subset exists (e.g. `label` in sample data), use it only as **secondary** validation; state in the report that primary metrics are synthetic-ground-truth.
- [ ] If no labels: qualitative review on a small slice (optional).

---

## Phase 5 — Course deliverables

- [ ] Polish API/UI only if required by the brief; prioritize **benchmark + figures** for grading narrative.
- [ ] Update README / USAGE for any new commands; keep docs in **English** per project rules.
- [ ] Presentation: add slides for synthetic protocol, robustness matrix, and one example figure.

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
- Synthetic injection + metrics: `api/synthetic_injection.py`
- Usage: [USAGE.md](USAGE.md)

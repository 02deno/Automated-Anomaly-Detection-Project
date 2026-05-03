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

- [ ] List **3+ distinct scenario types** (e.g. single-feature spike, multi-feature joint shift, optional time-series break if data is temporal).
- [ ] For each scenario, specify **parameters**: fraction of injected rows, magnitude (e.g. in standard deviations or scale factor), which columns are affected, optional correlation across features.
- [ ] Document parameter defaults in code docstrings or a small `configs/synthetic_defaults.yaml` (optional).

### Step 1.2 — Specify contract (inputs / outputs)

- [ ] **Input:** `pd.DataFrame` (clean baseline), `random_seed: int`, scenario id + parameter dict.
- [ ] **Output:** corrupted `DataFrame` (copy; do not mutate caller’s frame in place unless explicitly documented), `y_true: np.ndarray` of shape `(n_rows,)` with `1` = injected anomaly, `0` = normal.
- [ ] Guarantee: injected row indices are a subset of rows; **no accidental duplicate** label logic (one row = one label).

### Step 1.3 — Implement core injector module

- [ ] Add a dedicated module (e.g. `api/synthetic_injection.py` or `src/synthetic/`) with a class or functions: `inject(df, scenario, seed) -> tuple[pd.DataFrame, np.ndarray]`.
- [ ] Start with **one scenario** (e.g. spike on one numeric column) end-to-end; add others incrementally.
- [ ] Use only **numeric columns** from the frame for perturbation; align with how `AdvancedAnomalySystem` preprocesses data.

### Step 1.4 — Determinism and validation

- [ ] Fix `numpy` / `random` seed at the start of each injection path; same seed + same config ⇒ **identical** output.
- [ ] Add **unit tests** (or a tiny script): row count unchanged; non-injected rows equal to baseline (within float tolerance where applicable); `y_true.sum()` matches intended injection count.

### Step 1.5 — Wire to the detection pipeline

- [ ] Run `AdvancedAnomalySystem().run(corrupted_df)` and obtain `is_anomaly` (or per-model scores if exposed in a later phase).
- [ ] Compute **basic metrics** for Phase 1 closure: at least **precision, recall, F1** comparing `y_true` to predicted labels (binary).
- [ ] Log or print a one-line summary; optionally write `results/phase1_smoke.json` for CI or manual checks.

### Step 1.6 — Documentation and reproducibility

- [ ] Extend [USAGE.md](USAGE.md) with a short **“Synthetic evaluation”** subsection: how to run injection + smoke benchmark, expected artifacts.
- [ ] Note limitations: synthetic labels are for **evaluation**, not a claim about real-world prevalence.

**Phase 1 exit criteria:** At least one scenario + reproducible seed + F1 computed against `AdvancedAnomalySystem` labels on injected data; tests or scripted smoke pass.

---

## Phase 2 — Evaluation framework

- [ ] Standardize metrics: precision, recall, F1; optional AUROC/AUPRC if score vectors are exposed consistently.
- [ ] Experiment config (YAML or Python): dataset path or generator, scenario, seed, contamination list.
- [ ] Single entrypoint, e.g. `python -m experiments.run_benchmark --config ...`, writing `results/*.json` or CSV.

---

## Phase 3 — Model robustness comparison

- [ ] Run **each component model** separately (Isolation Forest, One-Class SVM, Autoencoder, LSTM when selected) on the same `y_true` where feasible; compare F1 across scenarios.
- [ ] Sweeps: **contamination** (e.g. 1%, 5%, 10%); optional **Gaussian noise** on all numeric features.
- [ ] Produce tables/heatmaps: rows = scenarios, columns = models, cells = F1 (or chosen primary metric).
- [ ] Summarize **one clear takeaway** for the report (e.g. which model wins under which regime).

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
| 1 | Phase 1.1–1.4 (catalog, contract, injector, tests). |
| 2 | Phase 1.5–1.6 + Phase 2 skeleton (config + one full benchmark run). |
| 3 | Phase 3 matrices and plots; draft results section. |
| 4 | Phase 4–5, doc pass, demo recording or live checklist. |

---

## References in-repo

- Pipeline: `api/advanced_system.py`
- API: `api/main.py`
- Usage: [USAGE.md](USAGE.md)

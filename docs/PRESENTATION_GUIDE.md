# Presentation Guide

This guide is a ready-to-use English presentation plan for a **20-25 minute talk** with a **live demo** for the **Automated Anomaly Detection Project**.

Recommended total duration:

- **Slides:** 14-17 minutes
- **Live demo:** 6-8 minutes
- **Q&A buffer:** 2-3 minutes if needed

---

## 1. Suggested presentation structure

### Slide 1 — Title
**Title:** Automated Anomaly Detection for CSV-like Data

**Subtitle idea:** Data-driven model selection, synthetic evaluation, and robustness analysis

**Say:**
- This project builds an anomaly detection pipeline for tabular data.
- The goal is to reduce manual model selection for new datasets.
- The system combines preprocessing, model selection, optimization, ensembling, and evaluation in one workflow.

**Time:** 1 minute

---

### Slide 2 — Problem motivation
**Title:** Why anomaly detection is difficult

**Key points:**
- Real anomaly labels are often rare, incomplete, or expensive to collect.
- Different datasets behave very differently.
- A single detector does not work best for every case.
- Manual tuning is slow and does not scale well.

**Say:**
- In real projects, we usually do not know which anomaly detector will work best before testing.
- That makes anomaly detection both a modeling problem and a workflow problem.

**Time:** 1.5 minutes

---

### Slide 3 — Project objective
**Title:** Project objective

**Key points:**
- Accept CSV-like tabular data through an API and web UI.
- Profile the dataset automatically.
- Run suitable anomaly detectors.
- Combine detector scores into a final anomaly score.
- Support evaluation with synthetic anomalies and labeled public datasets.

**Say:**
- The main idea is not just building one model.
- The contribution is building an end-to-end system that helps choose, compare, and evaluate anomaly detectors in a more systematic way.

**Time:** 1.5 minutes

---

### Slide 4 — System overview
**Title:** End-to-end workflow

**Visual recommendation:**
- A simple pipeline diagram:
  `CSV input -> EDA -> preprocessing -> model set -> score normalization -> ensemble -> thresholding -> results`

**Key points:**
- Input comes from CSV upload.
- EDA is separated from the heavy anomaly pipeline.
- Models generate anomaly scores.
- Scores are normalized and combined.
- A percentile threshold converts scores into anomaly flags.

**Time:** 2 minutes

---

### Slide 5 — Core pipeline components
**Title:** Pipeline architecture

**Key points:**
- Numeric feature extraction
- Missing value handling and scaling
- Optional PCA for dimensionality reduction
- Classical models: Isolation Forest, One-Class SVM, LOF
- Optional neural models: Autoencoder, LSTM autoencoder
- Ensemble and post-processing layer

**Say:**
- The project is implemented as layered components in `api/advanced_system.py`.
- This makes it easier to separate input handling, optimization, ensemble logic, and output generation.

**Time:** 2 minutes

---

### Slide 6 — Why synthetic anomalies were added
**Title:** Controlled evaluation with synthetic injection

**Key points:**
- Real-world labels are limited.
- Synthetic injection creates known ground truth without adding new rows.
- Existing rows are perturbed in controlled ways.
- This supports repeatable benchmarking.

**Scenarios to mention:**
- `spike_single`
- `joint_shift`
- `scale_burst`
- `dead_sensor`
- `sign_flip`
- `temporal_block`
- `categorical_flip`
- `missing_value`

**Time:** 2 minutes

---

### Slide 7 — Evaluation methodology
**Title:** How the system was evaluated

**Key points:**
- Synthetic benchmarks for controlled experiments
- Multi-scenario robustness analysis
- Public labeled datasets for sanity checks
- Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC

**Say:**
- Synthetic data tells us whether the ranking and thresholding logic behave reasonably.
- Real labeled datasets help test whether those insights transfer outside the synthetic setting.

**Time:** 2 minutes

---

### Slide 8 — Synthetic benchmark takeaway
**Title:** What happened on synthetic data

**Use these points:**
- On small controlled synthetic examples, the system often detected strong perturbations very well.
- In `results/synthetic_benchmark_summary.csv`, the ensemble reached **F1 = 1.0** on several scenarios such as `spike_single`, `joint_shift`, `scale_burst`, `temporal_block`, and `sign_flip`.
- The `dead_sensor` scenario was harder for some generic detectors, which is useful because it shows why scenario diversity matters.

**Visual recommendation:**
- Use `results/figures/baseline_server_metrics_noise0_heatmap_f1_mean.png`.

**Speaker note:**
- Do not oversell perfect synthetic scores.
- Explain that strong synthetic performance means the pipeline can separate clear controlled anomalies, but real data is still harder.

**Time:** 2 minutes

---

### Slide 9 — Real-data takeaway
**Title:** What happened on real labeled datasets

**Balanced message:**
- Results were mixed across datasets.
- No single model dominated everywhere.
- This supports the main motivation for flexible model comparison.

**Concrete examples from `results/real_data_multi_summary.csv`:**
- On **KDDCup99 HTTP**, `ocsvm` achieved **F1 = 0.858** and **ROC-AUC = 0.996**.
- On **Annthyroid**, `temporal_change` achieved **F1 = 0.264** and **ROC-AUC = 0.745**, outperforming the default ensemble.
- On **Pendigits**, `autoencoder` achieved **F1 = 0.274** and **ROC-AUC = 0.816**.

**Say:**
- The key conclusion is that anomaly behavior is dataset-dependent.
- That is exactly why a fixed one-model strategy is risky.

**Time:** 2.5 minutes

---

### Slide 10 — Robustness and overfitting
**Title:** Robustness and threshold sanity

**Key points:**
- The project includes robustness studies across scenarios, contamination levels, and noise.
- It also includes an overfitting diagnostic for threshold-selection sanity.
- In `results/overfitting_check_annthyroid.json`, the mean train/test F1 gap was about **0.034**, with interpretation: **no strong overfitting signal**.

**Say:**
- This does not prove full generalization.
- But it adds a practical stability check to the workflow.

**Time:** 1.5 minutes

---

### Slide 11 — Main contribution
**Title:** What this project contributes

**Key points:**
- An end-to-end anomaly detection workflow
- A separate EDA-first inspection path
- Synthetic anomaly injection for controlled benchmarking
- Robustness comparison across models and scenarios
- A web UI and API for practical use

**Good closing sentence for this slide:**
- The project is valuable not only because it detects anomalies, but because it makes anomaly-detection experiments more reproducible and explainable.

**Time:** 1 minute

---

### Slide 12 — Limitations and future work
**Title:** Limitations and next steps

**Key points:**
- Real-world anomaly datasets remain challenging.
- Thresholding at a fixed percentile is simple but not always optimal.
- Model selection could be more adaptive.
- More domain-specific features and better calibration could improve performance.

**Possible future work:**
- Learned meta-selection
- Better threshold calibration
- More real datasets
- Time-series-aware modeling

**Time:** 1.5 minutes

---

### Slide 13 — Demo transition
**Title:** Live demo

**One-line transition:**
- Now I will show how the system works from upload to anomaly results using the web dashboard.

**Time:** 0.5 minutes

---

## 2. Recommended live demo plan (6-8 minutes)

### Demo goal
Show that the system is not only a research prototype, but also a usable workflow:

1. inspect data,
2. create controlled anomalies,
3. run the pipeline,
4. interpret outputs.

### Demo flow

#### Step 1 — Open the dashboard
Use:

- `http://127.0.0.1:8000/`

Say:
- The dashboard separates EDA, synthetic preview, and the full pipeline so that exploratory analysis does not require running the full model stack immediately.

#### Step 2 — Run EDA on a sample CSV
Recommended file:

- `data/test_data.txt`

What to show:
- row and column counts
- missingness
- correlations
- outlier statistics
- warnings

Say:
- This stage is useful because many anomaly-detection problems are actually data-quality problems first.

#### Step 3 — Show synthetic anomaly preview
Use the same file and choose:

- scenario: `spike_single`

What to show:
- before/after preview
- injected rows
- explanation of what changed

Say:
- Synthetic preview is fast because it does not run the heavy ensemble yet.
- It is mainly for validation and benchmarking setup.

#### Step 4 — Export corrupted data
Use:

- **Synthetic export**

Say:
- Now we turn the preview into a full corrupted CSV and use that as input for the actual anomaly pipeline.

#### Step 5 — Run the full anomaly pipeline
Upload the exported CSV to the full pipeline section.

What to show:
- anomaly count
- anomaly scores
- anomaly flags
- model list
- summary table

If the dataset contains a label-like column, also show:
- evaluation metrics
- overfit hint

#### Step 6 — Close with interpretation
Say:
- The output is not just a binary label.
- The system also returns the score distribution, metadata, and model information needed for analysis.

---

## 3. Demo backup plan

If the live demo becomes slow or unstable, use this backup order:

1. Show the UI already running.
2. Use **EDA** only.
3. Use **synthetic preview** instead of full export + upload.
4. Show prepared figures from `results/figures/`.
5. Mention that the full pipeline can be slower because of Optuna and PyTorch startup.

This makes the presentation safer without changing the story.

---

## 4. Recommended commands before the presentation

Run from the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd api
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Useful URLs:

- API docs: `http://127.0.0.1:8000/docs`
- Dashboard: `http://127.0.0.1:8000/`

---

## 5. Suggested slide count

Recommended deck size:

- **12-13 slides total**
- **1 title slide**
- **9-10 content slides**
- **1 live demo slide**
- **1 final slide for conclusion / questions**

This is a good size for a 20-25 minute course presentation.

---

## 6. Short closing statement

You can end with:

> In summary, this project combines anomaly detection, synthetic evaluation, robustness analysis, and a practical web interface in one system. The most important lesson is that anomaly detection performance is highly dataset-dependent, so flexible evaluation and model comparison are essential.

---

## 7. If you want a stronger academic emphasis

If your instructor expects a more research-style talk, emphasize these three claims:

1. **Methodological contribution:** synthetic injection enables controlled benchmarking.
2. **Engineering contribution:** the project integrates API, UI, EDA, and model pipeline into one workflow.
3. **Empirical contribution:** different detectors win on different datasets, so automated comparison is more realistic than choosing one fixed detector.

---

## 8. Optional final Q&A slide

**Title:** Thank you

**Subtitle:** Questions?

You can place these small footer bullets:

- FastAPI + web dashboard
- synthetic anomaly injection
- robustness benchmarking
- real-data sanity checks

# AutoAD Slide-by-Slide Script

Target duration: about 14 minutes, maximum 15 minutes.

Before the demo:

```powershell
cd "C:\Users\LENOVO\Desktop\telecom\bahar 25\ai\anomaly_detection\Automated-Anomaly-Detection-Project\api"
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000/`

Demo file: `data/test_data.txt`

## Slide 1 - Title

Hello everyone, we are Team Anomaly. Our project is called Automated Anomaly Detection.

The goal is to build a system that can adapt to new CSV-like datasets without requiring the user to manually choose a model, tune parameters, or decide a threshold from scratch.

In short, the user uploads data, the system analyzes it, runs a suitable anomaly detection pipeline, and presents the results in a web dashboard.

## Slide 2 - Agenda Overview

Here is the structure of the presentation.

First, I will explain the workflow and system architecture. Then I will talk about datasets and synthetic anomaly injection. After that, I will cover the evaluation methodology, robustness results, and the threshold-overfitting issue. Finally, I will show the live demo and end with limitations and future work.

## Slide 3 - Workflow

The workflow is an end-to-end anomaly detection pipeline.

It starts with data ingestion. A user uploads a CSV file through the UI or API. Then the backend profiles the dataset, selects appropriate models, runs them, combines the anomaly scores, applies a threshold, and sends the results back to the dashboard.

The key point is that this is not a single fixed model. It is a modular pipeline that adapts based on the shape and statistics of the input data.

## Slide 4 - System Architecture

The system is built with a FastAPI backend and a static web dashboard.

In the backend, `api/main.py` exposes the main routes: `/eda`, `/synthetic-preview`, `/synthetic-export`, `/upload`, and `/overfit-check`.

The modules are separated by responsibility. `eda_report.py` handles exploratory analysis, `synthetic_injection.py` handles controlled anomaly injection, `advanced_system.py` contains the main anomaly detection pipeline, and `overfit_diagnostic.py` checks threshold stability.

This architecture makes the system easier to test, extend, and demo.

## Slide 5 - Datasets

We worked with both small demo data and real labeled datasets.

For the live demo, we use `data/test_data.txt`. It contains server-like metrics such as CPU usage, memory usage, network traffic, response time, and a label column.

For real-data evaluation, we used datasets such as Annthyroid and KDD'99 HTTP/SMTP. The important detail is that label columns are excluded from model features. They are used only after prediction to calculate evaluation metrics.

## Slide 6 - Synthetic Anomaly Injection

Synthetic anomaly injection helps us test the system when real labels are limited or unavailable.

The project supports eight scenarios: `spike_single`, `joint_shift`, `scale_burst`, `dead_sensor`, `sign_flip`, `temporal_block`, `categorical_flip`, and `missing_value`.

For example, `spike_single` creates a sudden spike in one numeric column, while `dead_sensor` simulates a sensor that gets stuck at a constant value. The injection function also creates `y_true`, but the detector does not use `y_true` as an input feature. It is only used for evaluation.

## Slide 7 - Evaluation Methodology

We evaluate the system in two ways.

First, we use threshold-dependent metrics: precision, recall, F1, accuracy, and the confusion matrix. These tell us how good the final binary anomaly decisions are.

Second, we use threshold-independent metrics: ROC-AUC and PR-AUC. These tell us whether the anomaly scores rank suspicious rows well, even before choosing a threshold.

This distinction matters because a model can produce good ranking scores but still perform poorly if the threshold is not selected carefully.

## Slide 8 - Real Data Evaluation

For real-data evaluation, the pipeline is tested on labeled datasets.

The model never trains on the label as a feature. Instead, the label is kept aside and used only after detection to compare predicted anomalies with ground truth.

This makes the evaluation label-safe and closer to the real unsupervised setting.

## Slide 9 - Unsupervised Pipeline and Threshold Validation

The core pipeline is unsupervised. It starts by selecting numeric features, imputing missing values, scaling the data, and optionally applying PCA for high-dimensional datasets.

Then the system runs selected models such as Isolation Forest, One-Class SVM, LOF, KNN distance, autoencoder, or LSTM, depending on dataset size and feature count.

The model scores are normalized and combined into an ensemble score. Then the threshold turns continuous scores into anomaly labels.

## Slide 10 - Real-Data Results and Metrics

The results show that different models perform better on different datasets.

For example, on some KDD'99 settings, One-Class SVM or specific score sources can be very strong. On Annthyroid, ensemble and temporal-change behavior can be more useful.

This supports the main motivation of the project: anomaly detection should not rely on one fixed model for all datasets. A more adaptive ensemble approach is more reliable.

## Slide 11 - Threshold Selection in Anomaly Detection

Threshold selection is one of the most important parts of anomaly detection.

The model produces continuous anomaly scores, but the user needs a binary decision: normal or anomalous.

If the threshold is too low, the system flags too many normal rows as anomalies. If it is too high, the system misses real anomalies. So thresholding controls the precision-recall tradeoff.

## Slide 12 - Overfitting Problem in Threshold Selection

When labels are available, it is tempting to choose the threshold that gives the best F1 on the same dataset.

But this can overfit. The threshold may look perfect on the current dataset but fail on new data.

That is why the project includes an `overfit_hint` in the `/upload` response and an optional `/overfit-check` endpoint. The deeper check uses subsampled train/test splits to see whether the threshold behavior is stable.

## Slide 13 - Label-Safe Anomaly Detection Pipeline

The pipeline is designed to be label-safe.

Columns like `label`, `target`, `ground_truth`, `y_true`, or `is_anomaly` are automatically excluded from the feature set.

They are only used after the model runs, in the evaluation layer. This prevents data leakage and makes the reported metrics more trustworthy.

## Slide 14 - Robustness Fix: Holdout-Based Threshold Validation

The robustness fix is to avoid trusting only in-sample threshold tuning.

When possible, the system can evaluate threshold behavior on holdout splits or subsampled train/test checks. This gives a better idea of whether the selected threshold generalizes.

In practice, this is especially important when the dataset is small or the anomaly ratio is very low.

## Slide 15 - Contribution Architecture: Modeling and Evaluation

Our contribution is not only one model. It is the combination of modeling, evaluation, synthetic testing, and UI interpretation.

The modeling contribution is the automated pipeline with model selection, Optuna-based tuning for several models, score normalization, and ensemble scoring.

The evaluation contribution is the combination of real labels, synthetic labels, ROC-AUC, PR-AUC, F1, and overfit diagnostics.

## Slide 16 - Future Work

There are several future improvements.

First, we can add more real datasets to better validate generalization.

Second, we can improve meta-selector calibration so the system chooses model weights and threshold strategies more reliably.

Third, we can make deep models faster and add interactive threshold adjustment in the UI, so users can explore the precision-recall tradeoff directly.

## Slide 17 - Live Demo

Now I will show the live demo.

The backend is running with FastAPI, and the dashboard is available at `http://127.0.0.1:8000/`.

First, I use the EDA card. I upload `data/test_data.txt` and click `Run EDA`. This shows column types, missing values, numeric summaries, outlier hints, correlation heatmap, and scatter plots. No anomaly model runs here; this is only data profiling.

Second, I use the synthetic anomaly card. I select the same file, keep the scenario as `spike_single`, set seed `42`, and click `Preview synthetic injection`. The UI shows before and after tables, changed cells, and injected row indices. This demonstrates controlled anomaly generation.

Third, I go to Full pipeline analysis. I upload `data/test_data.txt` again and click `Run Analysis`. Now the backend runs the full anomaly detection system.

In the results, I show the anomaly count, evaluation metrics, overfit hint, dataset profile, model choices, score chart, histogram, and final table. Red points or highlighted rows represent detected anomalies.

The main thing to notice is that the dashboard separates data understanding, synthetic testing, and the actual detection pipeline.

## Slide 18 - Conclusion

To conclude, this project makes anomaly detection on new CSV datasets more automatic and more explainable.

It combines dataset profiling, adaptive model selection, ensemble scoring, synthetic anomaly injection, evaluation metrics, and a live dashboard.

The main lesson is that one fixed model and one fixed threshold are not enough for every anomaly detection problem. A useful system should show both the predictions and the reasoning behind them.

## Slide 19 - Thank You

Thank you for listening.

We are happy to answer questions about the pipeline, synthetic scenarios, evaluation results, or the live dashboard.

## Slide 20 - Backup / Q&A

If asked why we use an ensemble: different anomaly types are captured better by different models, so combining score sources reduces dependence on a single model.

If asked whether labels affect the model: no. Label-like columns are excluded from features and used only for evaluation.

If asked whether synthetic anomalies replace real anomalies: no. Synthetic injection is for controlled testing and robustness analysis. Real-data evaluation is still necessary.

If asked why thresholding is difficult: anomaly scores are continuous, but decisions are binary. The threshold controls the precision-recall tradeoff and can overfit if selected only on the same labeled data.

## Fast Demo Checklist

1. Open `http://127.0.0.1:8000/`.
2. EDA card: select `data/test_data.txt`, click `Run EDA`.
3. Synthetic card: select same file, scenario `spike_single`, seed `42`, click `Preview synthetic injection`.
4. Full pipeline card: select `data/test_data.txt`, click `Run Analysis`.
5. Show Summary, Evaluation results, Overfitting and threshold sanity, Dataset profile, Score chart, Histogram, and Results table.

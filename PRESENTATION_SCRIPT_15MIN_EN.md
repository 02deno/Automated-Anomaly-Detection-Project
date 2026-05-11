# AutoAD 15-Minute Presentation Script

Target duration: 13:30-14:30 minutes, leaving 30-90 seconds for questions.

Before the demo:

```powershell
cd "C:\Users\LENOVO\Desktop\telecom\bahar 25\ai\anomaly_detection\Automated-Anomaly-Detection-Project\api"
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Browser: `http://127.0.0.1:8000/`

Fastest demo file: `data/test_data.txt`

## 0:00-0:45 - Opening

Hello, we are Team Anomaly. In this project, our goal was to build an automated anomaly detection system that reduces the need for manual model selection and manual threshold tuning on new CSV-like datasets.

The core idea is simple: the user uploads a table; the system analyzes the numeric columns, selects suitable models, runs them, normalizes and combines their scores, and then shows which rows are anomalous through a web UI.

## 0:45-1:20 - Agenda

I will keep the presentation in four parts: first the problem and architecture, then datasets and synthetic anomaly injection, then evaluation and robustness, and finally the live demo. I will spend extra time on the demo because the main contribution becomes clearer when the UI is running.

## 1:20-2:30 - Workflow

The workflow has three main steps: data ingestion, modeling, and decision making with visualization.

In the first step, the FastAPI backend receives a CSV file. In `api/main.py`, we have endpoints such as `/upload`, `/eda`, `/synthetic-preview`, `/synthetic-export`, and `/overfit-check`.

In the second step, `AdvancedAnomalySystem` starts the pipeline. It first analyzes the dataset: number of samples, number of features, missing rate, variance, skewness, kurtosis, correlation, sparsity, and other metadata.

Then, based on this profile, the system runs suitable models such as Isolation Forest, One-Class SVM, LOF, KNN distance, autoencoder, and LSTM. Each model produces an anomaly score. These scores are normalized to a common scale and combined into an ensemble score.

## 2:30-3:20 - Architecture

The architecture separates backend and frontend. The backend is FastAPI, and the UI is served as a static dashboard from `ui/index.html`.

At the code level, the system is modular:

`eda_report.py` only produces exploratory data analysis. It does not run ML models.

`synthetic_injection.py` creates controlled anomaly scenarios.

`advanced_system.py` contains the main pipeline: preprocessing, model selection, optimization, ensemble scoring, and thresholding.

`overfit_diagnostic.py` checks whether threshold selection may be too dependent on label information.

This separation is important because, during the demo, we will see these three workflows as separate cards in the UI.

## 3:20-4:10 - Datasets

We worked with two types of data.

The first type is small demo data. For example, `data/test_data.txt` contains CPU usage, memory usage, network traffic, response time, and a label column. This file is fast and convenient for the live demo.

The second type is real-data evaluation, using labeled datasets such as Annthyroid and KDD'99 HTTP/SMTP. In these datasets, the label column is not used as a model feature. It is used only after prediction to compute precision, recall, F1, ROC-AUC, and PR-AUC.

## 4:10-5:20 - Synthetic Anomaly Injection

Synthetic anomaly injection is the testability part of the project. Since clean labels are not always available in real data, we inject controlled corruptions and measure whether the system can detect them.

The code includes eight scenarios: `spike_single`, `joint_shift`, `scale_burst`, `dead_sensor`, `sign_flip`, `temporal_block`, `categorical_flip`, and `missing_value`.

For example, `spike_single` adds a large standard-deviation-based increase to one numeric column in selected rows. `dead_sensor` simulates a sensor that gets stuck at a constant value. `temporal_block` corrupts a consecutive time block.

The important point is that the injection process creates `y_true`, but the detector does not see that column as a feature. `y_true` is used only for evaluation.

## 5:20-6:40 - Evaluation Methodology

We evaluate the system at two levels.

The first level is threshold-dependent metrics: precision, recall, F1, and the confusion matrix. These answer the question: which rows did we actually mark as anomalies?

The second level is threshold-independent metrics: ROC-AUC and PR-AUC. These measure whether the anomaly scores rank suspicious rows well. This distinction matters because a model can rank anomalies well, but if the threshold is too strict, recall can still be low.

By default, the pipeline compares the anomaly score with a threshold. In the code, `PostProcessingLayer.label` is straightforward: if the score is greater than the threshold, the row is labeled as anomalous.

## 6:40-7:45 - Results and Robustness

The results support the main message of the project: one single model is not best for every dataset.

In the small synthetic benchmark, the ensemble works very well on clear scenarios such as spike and joint shift. On real datasets, however, model behavior changes depending on the data. For example, on KDD HTTP, OCSVM and freeze-like score sources can perform strongly, while on Annthyroid, ensemble and temporal-change scores can be more meaningful.

That is why the system uses ensemble and meta-selection ideas. The goal is not to hard-code one model, but to make a more adaptive decision based on the dataset profile and score behavior.

## 7:45-8:45 - Threshold and Overfitting Problem

Thresholding is critical in anomaly detection. If the threshold is too low, false positives increase. If it is too high, real anomalies may be missed.

When labeled data is available, choosing the threshold only by maximizing F1 on the same data can create an overfitting risk. In other words, the system may look good on that dataset but fail to generalize to new data.

For this reason, the code has two layers. The `/upload` response returns a fast `overfit_hint`, and the UI can optionally call `/overfit-check` for a subsampled train/test diagnostic. The second check is slower, but it gives a better view of threshold stability.

## 8:45-12:30 - Live Demo

Now I will run the system. The FastAPI backend is running, and the UI is available at `127.0.0.1:8000`.

The first card is EDA. Here, I upload `data/test_data.txt` and click Run EDA. This part does not run any anomaly detection model. It only profiles the data. We can see column types, missing values, numeric summaries, outlier hints, the correlation heatmap, and a scatter plot. This step answers the question: what does the data look like before modeling?

The second card is synthetic anomaly preview. I use the same file, keep the scenario as `spike_single`, and set the seed to 42. When I click Preview, the backend receives the file, injects controlled corruptions into selected rows, and the UI shows before/after tables. The highlighted rows are the rows that received synthetic perturbation. One key detail is that preview only shows the first N rows, while the export button downloads the full corrupted CSV.

The third card is full pipeline analysis. Here, I select `data/test_data.txt` as the pipeline upload and click Run Analysis. This time, the backend runs the full anomaly detection pipeline.

In the result, the summary shows how many rows were detected as anomalies. If a label column is present, the Evaluation card shows precision, recall, F1, accuracy, ROC-AUC, and PR-AUC. In the score-vs-row-index chart, red points are the rows marked as anomalies. The histogram shows the score distribution and where the threshold sits.

There is also a Dataset profile and decision rule card. It shows which numeric columns were used, which models were selected, what threshold strategy was applied, and whether PCA was used.

Finally, I look at the Overfitting and threshold sanity card. This card gives a quick warning about whether threshold selection may be too dependent on labels. If we were using a larger labeled file, the Run subsampled train/test check button could start a deeper train/test diagnostic.

This demo shows three things: data profiling is separate, synthetic testing is separate, and the real detection pipeline is separate. So the system is not only "upload a CSV and get a result"; it is also a dashboard that helps interpret the result.

## 12:30-13:30 - Limitations and Future Work

The main limitation is that in unsupervised anomaly detection, if there are no labels, we cannot always know the exact ground truth. That is why metrics are most meaningful either on labeled datasets or on synthetic injection experiments.

The second limitation is threshold selection. Finding the best F1 can look easy, but if it depends too much on labels, it can hurt generalization. That is why holdout-based validation and overfit diagnostics are important future work.

Future work includes adding more real datasets, improving meta-selector calibration, making deep model options faster, and adding interactive threshold adjustment in the UI.

## 13:30-14:20 - Conclusion

To summarize, this project aims to make anomaly detection on new CSV datasets more automated and more explainable.

Our contributions can be grouped into three parts: an automatic model selection and ensemble pipeline, controlled synthetic anomaly injection for testability, and a usable web dashboard that combines EDA, evaluation, and overfit checks.

The main message is that one model and one fixed threshold are not enough for every anomaly detection problem. A more reliable decision-support system should show the dataset profile, model scores, threshold behavior, and evaluation risk together.

Thank you. We are ready for your questions.

## Quick Demo Click Flow

1. Open `http://127.0.0.1:8000/`.
2. In the EDA card, select `data/test_data.txt`, then click `Run EDA`.
3. In the Synthetic card, select the same file or use the pipeline-file fallback, choose `spike_single`, seed `42`, then click `Preview synthetic injection`.
4. Optionally show `Download full corrupted CSV`, but do not wait on it during the demo.
5. In the Full pipeline card, select `data/test_data.txt`, then click `Run Analysis`.
6. Show Summary, Evaluation results, Overfitting and threshold sanity, Dataset profile, Score chart, and Results table.

## Short Answers

Question: Does the label column affect the model?

Answer: No. Columns such as `label`, `target`, `ground_truth`, and `is_anomaly` are excluded from the model features. They are used only for evaluation.

Question: How is the threshold selected?

Answer: If there is no label, the system uses a default or fallback percentile, or a meta-selected contamination rule. If labels are available, auto mode can choose a threshold based on F1, which is why we added the overfit hint and train/test diagnostic.

Question: Why use an ensemble?

Answer: Different anomaly types are detected better by different models. The ensemble reduces dependence on a single model by combining multiple score sources.

Question: Can synthetic injection replace real anomalies?

Answer: No, it cannot fully replace real anomalies. Its purpose is controlled testing and robustness analysis. It should be interpreted together with real-data evaluation.

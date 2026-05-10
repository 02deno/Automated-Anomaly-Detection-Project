const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "..");
const OUT_DIR = path.join(ROOT, "output", "presentations");
const PREVIEW_DIR = path.join(OUT_DIR, "previews");
const PPTX_PATH = path.join(OUT_DIR, "AutoAD_Project_Presentation_updated.pptx");

const C = {
  ink: "16213A",
  muted: "5E6B7A",
  light: "F4F7FB",
  line: "D9E2EF",
  blue: "2563EB",
  cyan: "0891B2",
  green: "16A34A",
  amber: "D97706",
  red: "DC2626",
  purple: "7C3AED",
  white: "FFFFFF",
};

const W = 13.333;
const H = 7.5;
const SHAPE = { rect: "rect", ellipse: "ellipse" };

const synthetic = [
  ["AutoAD", 0.552, 0.912, 0.690],
  ["Isolation Forest", 0.519, 0.815, 0.755],
  ["One-Class SVM", 0.500, 0.820, 0.695],
  ["LOF", 0.372, 0.776, 0.565],
];

const real = [
  ["One-Class SVM", 0.235, 0.690, 0.242],
  ["LOF", 0.074, 0.633, 0.091],
  ["Isolation Forest", 0.068, 0.756, 0.202],
  ["AutoAD", 0.034, 0.737, 0.116],
];

const realCalibrated = [
  ["One-Class SVM", 0.236, 0.695, 0.244],
  ["AutoAD calibrated", 0.119, 0.606, 0.114],
  ["LOF", 0.076, 0.619, 0.090],
  ["Isolation Forest", 0.071, 0.732, 0.199],
];

const selectorComparison = [
  ["Default AutoAD", 0.034, 0.737, 0.116],
  ["Calibrated weights", 0.119, 0.606, 0.114],
  ["Previous LODO", 0.133, 0.573, 0.114],
  ["Learned selector LODO", 0.066, 0.528, 0.082],
  ["Learned in-sample", 0.317, 0.823, 0.281],
];

const lodoLearned = [
  ["Annthyroid", "LOF", "Glass", 0.139, 0.693],
  ["KDD HTTP", "Ensemble", "Annthyroid", 0.021, 0.040],
  ["KDD SMTP", "OCSVM", "KDD HTTP", 0.011, 0.702],
  ["Glass", "Ensemble", "Annthyroid", 0.080, 0.694],
  ["Pendigits", "LOF", "Glass", 0.077, 0.512],
];

const scenario = [
  ["spike", 0.651, 0.775],
  ["joint", 0.632, 0.775],
  ["scale", 1.0, 0.775],
  ["temporal", 0.856, 0.775],
  ["dead", 0.353, 0.200],
  ["sign", 0.450, 0.775],
];

const slides = [
  {
    kind: "cover",
    title: "AutoAD+",
    kicker: "A self-configuring hybrid anomaly detection system",
    body: "Model pipeline, datasets, and benchmark evidence",
    note: "Open by framing AutoAD+ as an unsupervised anomaly detection project. The goal is not just to run one model, but to automatically analyze a dataset, choose detectors, combine their signals, and report anomalies with comparable evidence.",
  },
  {
    title: "The Problem",
    lead: "Unsupervised anomaly detection is fragile when the data shape changes.",
    bullets: ["Different anomalies need different detectors", "Thresholds are usually guessed", "Small demos can overstate performance"],
    note: "Explain that anomaly detection often looks good on one dataset and then fails on another. Point out that distance, density, tree, and temporal anomalies are different, so a single detector is rarely enough.",
  },
  {
    title: "Project Goal",
    lead: "Build a model that adapts before it predicts.",
    bullets: ["Analyze the uploaded table", "Select suitable detectors", "Optimize and normalize model scores", "Compare against known baselines"],
    note: "The project proposal is about a self-configuring hybrid system. Emphasize the word self-configuring: the pipeline should inspect the data and then decide how to process it.",
  },
  {
    title: "System View",
    lead: "AutoAD+ is a layered pipeline, not a single algorithm.",
    diagram: ["Input", "Analysis", "Optimization", "Core Models", "Ensemble", "Results"],
    note: "Walk from left to right. CSV or API data enters the system, numeric features are profiled, candidate models are selected, each model is tuned, scores are normalized, then the ensemble produces final anomaly scores and labels.",
  },
  {
    title: "Input Layer",
    lead: "The system accepts tabular data and keeps the model unsupervised.",
    bullets: ["CSV, API, or database source", "Numeric features are extracted", "Known label columns are excluded from training", "Labels are used only for evaluation"],
    note: "Clarify that labels are not used to train the model. In experiments, labels are held aside and only used to calculate F1, ROC-AUC, and PR-AUC.",
  },
  {
    title: "Analysis Layer",
    lead: "Before modeling, AutoAD+ measures the dataset.",
    bullets: ["Rows and numeric feature count", "Missing rate and zero rate", "Variance, skewness, kurtosis", "Correlation and entropy signals"],
    note: "This layer lets the system make decisions from data properties. For example, a dataset with enough rows may use LOF, while larger or higher-dimensional datasets can activate neural models.",
  },
  {
    title: "Model Selection",
    lead: "Candidate detectors are chosen from the data profile.",
    bullets: ["Isolation Forest for global outliers", "One-Class SVM for boundary learning", "LOF for local density anomalies", "Autoencoder and LSTM when data size supports them"],
    note: "Explain the model menu. Isolation Forest, OCSVM, and LOF are the popular baselines. Autoencoder and LSTM are optional components when the dataset is large enough.",
  },
  {
    title: "Optimization",
    lead: "Each detector is tuned automatically with Optuna.",
    bullets: ["Isolation Forest: tree count and sample fraction", "OCSVM: nu and gamma", "LOF: neighbor count and distance metric", "Neural models: hidden size and learning rate"],
    note: "Mention that optimization is unsupervised: the objective uses score spread or reconstruction loss, not ground-truth labels. This keeps the training process realistic.",
  },
  {
    title: "Core Detectors",
    lead: "The model combines complementary views of abnormality.",
    bullets: ["Tree isolation score", "Kernel boundary score", "Local density score", "Reconstruction error", "Sequence reconstruction error"],
    note: "The important point is complementarity. A global spike, a local density break, and a temporal block may each be easier for a different detector.",
  },
  {
    title: "Domain Detectors",
    lead: "The latest version adds targeted signals for sensor-style failures.",
    bullets: ["Flatline / dead-sensor score", "Temporal-change score", "Median-centrality and repeated-value evidence", "Detector gating to avoid false dominance"],
    note: "Explain why this was added: the original model missed dead-sensor cases because median-like values can look normal to distance models. The domain detector helps, but it is gated so it does not always dominate.",
  },
  {
    title: "Score Normalization",
    lead: "Raw detector scores are not directly comparable.",
    bullets: ["Scores are clipped at robust percentiles", "Each score stream is scaled to a common range", "The ensemble uses reliability-weighted contributions"],
    note: "Different algorithms output scores in different scales. If we simply average raw scores, one model can dominate by accident. Normalization and weighting make the ensemble more stable.",
  },
  {
    title: "Adaptive Thresholding",
    lead: "The final label is produced by an adaptive score-gap threshold.",
    bullets: ["Searches for separation near top scores", "Falls back to percentile behavior when separation is weak", "Reports threshold metadata to the API/UI"],
    note: "This slide explains why the ensemble may not be exactly F1 at the 95th percentile. AutoAD now uses the data distribution to find a gap when it exists.",
  },
  {
    title: "Meta-Selection Layer",
    lead: "AutoAD+ can switch from the ensemble to the detector that best fits a dataset.",
    bullets: ["Profiles are learned from labeled validation results", "Candidate sources include ensemble, IForest, OCSVM, LOF, and temporal-change", "The selected source also supplies expected-contamination thresholding", "The selector records which training dataset it matched"],
    note: "Explain that this is not supervised anomaly training. Labels are used only in validation to learn which score source is best for a type of dataset. At runtime, the model still scores the uploaded data without row labels.",
  },
  {
    title: "Learned Selector Upgrade",
    lead: "The newest version replaces nearest-profile matching with a small classifier.",
    bullets: ["Runtime trains a Random Forest selector from profile rows", "Feature vector combines dataset shape and score diagnostics", "Each detector contributes distribution statistics", "Fallback remains nearest-profile matching when learned vectors are unavailable"],
    note: "This is the latest implementation change. The older selector compared only dataset shape values. The new selector adds diagnostics from actual detector score distributions, then trains a Random Forest to predict which score source should be used.",
  },
  {
    title: "Selector Features",
    lead: "The learned selector uses richer evidence than row and column counts.",
    bullets: ["Dataset shape: rows, features, sparsity, correlations", "Score spread: mean, standard deviation, percentiles", "Tail behavior: top-one-percent gap and max-to-p99 gap", "Skewness: whether scores form a heavy anomaly tail"],
    note: "Describe why this should help in principle. If OCSVM creates a sharp high-score tail while the ensemble is flat, that is useful evidence for selecting OCSVM. The selector now has access to these score-shape clues.",
  },
  {
    title: "Output",
    lead: "The result is both a prediction and an explanation surface.",
    bullets: ["Anomaly score per row", "Binary anomaly flag", "Model weights", "Per-model normalized scores", "Evaluation metrics when labels exist"],
    note: "The system is designed to make results visible, not just return a black-box label. The API gives scores, labels, weights, and optional evaluation.",
  },
  {
    kind: "section",
    title: "Datasets",
    lead: "The project uses controlled synthetic tests and multiple real labeled benchmarks.",
    note: "Transition from model design to evaluation. Stress that synthetic tests show whether known anomaly types can be found, while multiple real datasets test whether the model generalizes beyond the designed scenarios.",
  },
  {
    title: "Synthetic Dataset",
    lead: "Controlled anomalies are injected into server-metric data.",
    bullets: ["Base: server metrics CSV", "Expanded to 120 rows with small jitter", "54 validation runs", "Three random seeds", "Multiple contamination levels"],
    note: "Explain that the larger synthetic benchmark is not just the tiny 24-row smoke test. It repeats and jitters rows to create a larger controlled set, then varies seeds and anomaly strengths.",
  },
  {
    title: "Synthetic Scenarios",
    lead: "The benchmark checks multiple anomaly families.",
    bullets: ["Single-feature spikes", "Joint shifts", "Scale bursts", "Temporal blocks", "Dead sensor variants", "Sign flips"],
    note: "List the scenarios and explain the purpose. Dead sensor has multiple modes: median, zero, previous value, and random constant. This makes the evaluation harder.",
  },
  {
    title: "Real-World Datasets",
    lead: "Five labeled datasets test generalization.",
    bullets: ["Annthyroid: 7,200 rows, 534 anomalies", "KDDCup99 HTTP: 10,000-row sample, 376 attacks", "KDDCup99 SMTP: 9,571 rows, 3 attacks", "Glass and Pendigits: derived rare-class labels"],
    note: "Explain that the real evaluation is now broader than Annthyroid. KDD has native attack labels. Glass and Pendigits are not native anomaly datasets, so rare-class or one-vs-rest labels are derived for benchmark-style evaluation.",
  },
  {
    title: "Baselines",
    lead: "AutoAD+ is compared with popular unsupervised models.",
    bullets: ["Isolation Forest", "One-Class SVM", "Local Outlier Factor", "AutoAD+ ensemble"],
    note: "These are the right comparisons because they are widely used unsupervised anomaly detection methods and are also components inside AutoAD.",
  },
  {
    kind: "bars",
    title: "Synthetic F1 Comparison",
    lead: "AutoAD+ ranks first on the larger synthetic validation benchmark.",
    data: synthetic.map((r) => [r[0], r[1]]),
    axis: "Average F1 across 54 runs",
    note: "AutoAD reaches average F1 of 0.552. Isolation Forest is close at 0.519, OCSVM at 0.500, and LOF trails at 0.372. This is a modest synthetic advantage, not a broad superiority claim.",
  },
  {
    kind: "bars",
    title: "Synthetic ROC-AUC",
    lead: "AutoAD+ separates positives well in controlled tests.",
    data: synthetic.map((r) => [r[0], r[2]]),
    axis: "Average ROC-AUC",
    note: "The synthetic ROC-AUC result is stronger: AutoAD scores 0.912, ahead of OCSVM, Isolation Forest, and LOF. This means ranking is good in controlled tests even when thresholded F1 is unstable.",
  },
  {
    kind: "bars",
    title: "Real-World F1 Comparison",
    lead: "Across five real datasets, default AutoAD+ ranks last by F1.",
    data: real.map((r) => [r[0], r[1]]),
    axis: "Average F1 across five real datasets",
    note: "This is the most important honesty slide. Default AutoAD averages only 0.034 F1 across the five real datasets. OCSVM leads because it performs extremely well on KDDCup99 HTTP. The synthetic win does not transfer directly.",
  },
  {
    kind: "bars",
    title: "Real-World ROC-AUC",
    lead: "AutoAD+ ranks reasonably, but thresholded F1 is weak.",
    data: real.map((r) => [r[0], r[2]]),
    axis: "Average ROC-AUC across five real datasets",
    note: "Default AutoAD has average ROC-AUC of 0.737, close to Isolation Forest and above LOF. That means ranking is not terrible, but the final thresholding and ensemble weights do not convert ranking into good F1.",
  },
  {
    kind: "bars",
    title: "Calibration Helps",
    lead: "Annthyroid-calibrated AutoAD+ improves, but still trails OCSVM overall.",
    data: realCalibrated.map((r) => [r[0], r[1]]),
    axis: "Average F1 after calibrated weights",
    note: "Using calibrated weights and expected-contamination thresholding improves AutoAD average F1 from 0.034 to 0.119. This is a real improvement, but OCSVM remains best overall at 0.236. Calibration helps but does not fully generalize.",
  },
  {
    title: "Where AutoAD+ Fails",
    lead: "Harder real and synthetic cases expose weakness.",
    bullets: ["Previous-value dead sensors remain difficult", "KDD HTTP strongly favors OCSVM", "Default thresholding hurts real-data F1", "Calibration helps but is dataset-specific"],
    note: "This is where you show maturity. The project does not hide failures. AutoAD is useful as a framework, but it still needs meta-selection and calibration that generalize across datasets.",
  },
  {
    title: "Overfitting Check",
    lead: "The earlier perfect score was too optimistic.",
    bullets: ["Small 24-row benchmark produced inflated scores", "Larger synthetic validation reduced F1 to 0.552", "Five real datasets drop default AutoAD F1 to 0.034", "Conclusion: no general superiority claim yet"],
    note: "Explain the overfitting concern directly. The model was improved after seeing failure cases, so broader validation was necessary. The broader results are more realistic and more defensible.",
  },
  {
    kind: "table",
    title: "Final Evidence Summary",
    lead: "Synthetic results are promising; real-world results require calibration.",
    note: "This table is the final evidence summary. It gives the audience a balanced conclusion: AutoAD wins the synthetic validation, calibrated AutoAD improves on real data, but OCSVM remains strongest overall across the five real datasets.",
  },
  {
    kind: "section",
    title: "Latest Validation",
    lead: "The learned selector was tested with leave-one-dataset-out validation.",
    note: "Transition to the newest experiment. Emphasize that leave-one-dataset-out is stricter than in-sample evaluation because each dataset is hidden while the selector is trained.",
  },
  {
    title: "Why LODO Matters",
    lead: "Leave-one-dataset-out checks whether meta-selection generalizes.",
    bullets: ["Hold out one real dataset completely", "Train selector profiles on the remaining datasets", "Run AutoAD+ on the hidden dataset", "Repeat until every dataset has been held out"],
    note: "Explain that this is the correct test for the meta-selector. In-sample results can look strong because the selector has already seen the answer for that dataset. LODO removes that shortcut.",
  },
  {
    kind: "bars",
    title: "Updated Real F1 Comparison",
    lead: "The learned selector is strong in-sample but weaker under LODO.",
    data: selectorComparison.map((r) => [r[0], r[1]]),
    axis: "Average F1 across five real datasets",
    note: "Read the bars carefully. Learned in-sample F1 is about 0.317, nearly the same as the previous meta-selected result. But under leave-one-dataset-out it falls to 0.066, below the previous LODO result of 0.133.",
  },
  {
    kind: "bars",
    title: "Updated Real ROC-AUC",
    lead: "The LODO drop appears in ranking quality too.",
    data: selectorComparison.map((r) => [r[0], r[2]]),
    axis: "Average ROC-AUC across five real datasets",
    note: "The learned selector also drops in ROC-AUC under LODO: about 0.528. This means the problem is not just thresholding. The selector is sometimes choosing the wrong score source for the hidden dataset.",
  },
  {
    kind: "lodoTable",
    title: "LODO Learned Selector Detail",
    lead: "The selector often matched the held-out dataset to the wrong training profile.",
    note: "Use this slide to explain the failure pattern. KDD HTTP is the clearest issue: it was matched to Annthyroid and selected the ensemble, but the dataset strongly favors OCSVM or freeze-style behavior. Pendigits also moved to LOF and lost performance.",
  },
  {
    title: "What The New Result Means",
    lead: "The implementation improved the pipeline, but the experiment disproved the improvement claim.",
    bullets: ["The learned selector is technically working", "Five real datasets are not enough to train a robust meta-model", "In-sample selector scores are optimistic", "LODO should be the headline real-data metric"],
    note: "Be direct here. The right scientific conclusion is not that the learned selector is useless. It is that the current training set is too small. The method needs more datasets before it can be trusted.",
  },
  {
    title: "Updated Engineering Decision",
    lead: "Keep the learned selector, but do not present it as the best model yet.",
    bullets: ["Use calibrated weights as the safer current default", "Report learned-selector results as an experiment", "Use LODO as the main generalization check", "Add more real datasets before retraining the selector"],
    note: "This is the practical recommendation. The code path is valuable, but the model selection policy should not switch to learned selector by default until it improves on LODO.",
  },
  {
    title: "How To Run",
    lead: "The repo includes reproducible scripts for both evaluation types.",
    code: [
      ".venv\\Scripts\\python.exe scripts\\run_synthetic_benchmark.py --config configs\\experiments\\larger_validation.yaml",
      ".venv\\Scripts\\python.exe scripts\\run_real_data_eval.py --dataset data\\external\\adb_annthyroid_21feat_normalised.csv::adb_annthyroid::ground_truth --max-rows 10000",
      ".venv\\Scripts\\python.exe scripts\\calibrate_ensemble_weights.py --dataset data\\external\\adb_annthyroid_21feat_normalised.csv --label-column ground_truth",
      ".venv\\Scripts\\python.exe scripts\\run_leave_one_dataset_out.py --summary results\\real_data_multi_summary_calibrated.csv --out results\\leave_one_dataset_out_learned_selector.csv",
    ],
    note: "Mention that these commands reproduce the main reported results. The CSV outputs are written under the results folder.",
  },
  {
    title: "Next Improvements",
    lead: "The next step is more data for the selector, not more complexity.",
    bullets: ["Add more labeled real anomaly datasets", "Group native-label and derived-label datasets separately", "Train the selector with cross-dataset validation only", "Optimize rare-positive thresholding with PR-AUC and recall-at-k"],
    note: "The previous next step was to build dataset-aware meta-selection. That is now implemented. The updated next step is to feed it enough real validation datasets and tune it against leave-one-dataset-out performance.",
  },
  {
    title: "Conclusion",
    lead: "AutoAD+ is a strong project prototype, but not a finished universal detector.",
    bullets: ["Pipeline is implemented end to end", "Synthetic evaluation shows a modest ensemble advantage", "Calibration improves real-data F1", "Learned selector needs more real datasets before it can generalize"],
    note: "Close with the updated balanced message. The project satisfies the core proposal as a self-configuring hybrid anomaly detector with evaluation. The new learned-selector experiment is valuable because it shows what still does not generalize.",
  },
  {
    kind: "close",
    title: "AutoAD+",
    lead: "Self-configuring anomaly detection with reproducible evidence",
    note: "End by inviting questions. If asked what matters most, answer: the pipeline works, synthetic results are promising, calibration helps, but the broader real-data test shows that generalization is the main challenge.",
  },
];

function ensureDirs() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.mkdirSync(PREVIEW_DIR, { recursive: true });
  for (const file of fs.readdirSync(PREVIEW_DIR)) {
    if (file.toLowerCase().endsWith(".png")) fs.unlinkSync(path.join(PREVIEW_DIR, file));
  }
}

function addTitle(slide, title, lead) {
  slide.addText(title, { x: 0.65, y: 0.38, w: 6.8, h: 0.4, fontFace: "Aptos Display", fontSize: 21, bold: true, color: C.ink, margin: 0 });
  if (lead) slide.addText(lead, { x: 0.65, y: 0.95, w: 8.4, h: 0.55, fontFace: "Aptos", fontSize: 16, color: C.muted, breakLine: false, fit: "shrink", margin: 0 });
}

function addFooter(slide, n) {
  slide.addText(`AutoAD+ project | ${String(n).padStart(2, "0")}`, { x: 0.65, y: 7.08, w: 3.5, h: 0.2, fontSize: 7, color: "8A95A5", margin: 0 });
}

function addBullets(slide, bullets, x = 0.9, y = 1.85, w = 7.8) {
  bullets.forEach((b, i) => {
    const yy = y + i * 0.72;
    slide.addShape(SHAPE.ellipse, { x, y: yy + 0.08, w: 0.14, h: 0.14, fill: { color: C.blue }, line: { color: C.blue } });
    slide.addText(b, { x: x + 0.32, y: yy, w, h: 0.38, fontSize: 15, color: C.ink, margin: 0, fit: "shrink" });
  });
}

function addBars(slide, data, x, y, w, h, color = C.blue) {
  const max = Math.max(...data.map((d) => d[1]), 1);
  data.forEach((d, i) => {
    const yy = y + i * (h / data.length);
    const bw = (w - 2.2) * (d[1] / max);
    slide.addText(d[0], { x, y: yy + 0.02, w: 1.9, h: 0.28, fontSize: 10.5, color: C.ink, margin: 0, fit: "shrink" });
    slide.addShape(SHAPE.rect, { x: x + 2.05, y: yy, w: w - 2.5, h: 0.34, fill: { color: C.light }, line: { color: C.line } });
    slide.addShape(SHAPE.rect, { x: x + 2.05, y: yy, w: bw, h: 0.34, fill: { color }, line: { color } });
    slide.addText(d[1].toFixed(3), { x: x + 2.12 + bw, y: yy + 0.03, w: 0.7, h: 0.2, fontSize: 9, color: C.ink, margin: 0 });
  });
}

function addDiagram(slide, labels) {
  const startX = 0.75;
  const gap = 0.18;
  const bw = 1.82;
  labels.forEach((label, i) => {
    const x = startX + i * (bw + gap);
    slide.addShape(SHAPE.rect, { x, y: 2.75, w: bw, h: 0.9, rectRadius: 0.08, fill: { color: i % 2 ? "E0F2FE" : "E8F0FF" }, line: { color: C.line } });
    slide.addText(label, { x: x + 0.12, y: 3.04, w: bw - 0.24, h: 0.24, align: "center", fontSize: 12, bold: true, color: C.ink, margin: 0, fit: "shrink" });
    if (i < labels.length - 1) slide.addText(">", { x: x + bw + 0.04, y: 3.02, w: 0.14, h: 0.2, fontSize: 14, color: C.muted, margin: 0 });
  });
}

function addLodoTable(slide) {
  const headers = ["Held-out", "Selected", "Matched", "F1", "ROC"];
  const xs = [0.75, 3.0, 5.0, 8.0, 9.2];
  const ws = [1.85, 1.55, 2.4, 0.8, 0.8];
  slide.addShape(SHAPE.rect, { x: 0.62, y: 1.78, w: 9.95, h: 0.55, fill: { color: C.ink }, line: { color: C.ink } });
  headers.forEach((h, i) => {
    slide.addText(h, { x: xs[i], y: 1.95, w: ws[i], h: 0.24, fontSize: 10.5, bold: true, color: C.white, margin: 0, fit: "shrink" });
  });
  lodoLearned.forEach((row, r) => {
    const y = 2.45 + r * 0.62;
    slide.addShape(SHAPE.rect, { x: 0.62, y: y - 0.13, w: 9.95, h: 0.48, fill: { color: r % 2 ? "F8FAFC" : "EEF6FF" }, line: { color: C.line } });
    [row[0], row[1], row[2], row[3].toFixed(3), row[4].toFixed(3)].forEach((cell, c) => {
      slide.addText(cell, { x: xs[c], y, w: ws[c], h: 0.22, fontSize: 10.5, color: C.ink, margin: 0, fit: "shrink" });
    });
  });
  slide.addText("Average LODO F1: 0.066", { x: 8.0, y: 5.85, w: 2.2, h: 0.28, fontSize: 13, bold: true, color: C.red, margin: 0 });
  slide.addText("Signal: the learned selector currently overfits the five-dataset validation set.", { x: 0.65, y: 6.25, w: 8.6, h: 0.3, fontSize: 11.5, color: C.muted, margin: 0 });
}

function renderSlide(pptx, spec, idx) {
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  slide.addShape(SHAPE.rect, { x: 0, y: 0, w: 0.16, h: 7.5, fill: { color: C.blue }, line: { color: C.blue } });

  if (spec.kind === "cover") {
    slide.background = { color: "F7FAFF" };
    slide.addText("AutoAD+", { x: 0.78, y: 1.45, w: 5.4, h: 0.9, fontFace: "Aptos Display", fontSize: 56, bold: true, color: C.ink, margin: 0 });
    slide.addText(spec.kicker, { x: 0.83, y: 2.55, w: 6.9, h: 0.45, fontSize: 19, color: C.blue, margin: 0 });
    slide.addText(spec.body, { x: 0.85, y: 3.25, w: 5.8, h: 0.42, fontSize: 15, color: C.muted, margin: 0 });
    addBars(slide, [["Input", .95], ["Analyze", .82], ["Ensemble", .76], ["Evaluate", .67]], 8.05, 1.75, 4.3, 2.5, C.cyan);
  } else if (spec.kind === "section") {
    slide.background = { color: C.ink };
    slide.addText(spec.title, { x: 0.8, y: 2.3, w: 9.5, h: 0.65, fontSize: 38, bold: true, color: C.white, margin: 0 });
    slide.addText(spec.lead, { x: 0.84, y: 3.15, w: 8.3, h: 0.38, fontSize: 16, color: "D8E4F8", margin: 0 });
  } else if (spec.kind === "bars") {
    addTitle(slide, spec.title, spec.lead);
    addBars(slide, spec.data, 1.05, 2.0, 8.7, 3.3, spec.title.includes("Real") ? C.amber : C.blue);
    slide.addText(spec.axis, { x: 1.05, y: 5.7, w: 5.4, h: 0.25, fontSize: 10, color: C.muted, margin: 0 });
  } else if (spec.kind === "scenario") {
    addTitle(slide, spec.title, spec.lead);
    const data = scenario.map((r) => [r[0], r[1]]);
    addBars(slide, data, 0.9, 2.0, 5.6, 3.6, C.green);
    addBars(slide, scenario.map((r) => [r[0], r[2]]), 7.0, 2.0, 5.2, 3.6, C.purple);
    slide.addText("AutoAD F1", { x: 2.8, y: 1.72, w: 1.1, h: 0.22, fontSize: 10, color: C.green, bold: true, margin: 0 });
    slide.addText("Isolation Forest F1", { x: 8.65, y: 1.72, w: 1.8, h: 0.22, fontSize: 10, color: C.purple, bold: true, margin: 0 });
  } else if (spec.kind === "table") {
    addTitle(slide, spec.title, spec.lead);
    const rows = [
      ["Test", "Winner", "AutoAD rank", "Meaning"],
      ["Synthetic validation", "AutoAD", "1st", "Promising on controlled anomalies"],
      ["Real default", "OCSVM", "4th", "Generalization gap remains"],
      ["Real calibrated", "OCSVM", "2nd", "Calibration helps, not enough"],
    ];
    const xs = [0.9, 3.6, 5.6, 7.3];
    const ws = [2.4, 1.6, 1.3, 4.5];
    rows.forEach((row, r) => {
      const y = 2.15 + r * 0.72;
      slide.addShape(SHAPE.rect, { x: 0.78, y: y - 0.12, w: 11.65, h: 0.56, fill: { color: r === 0 ? C.ink : (r === 1 ? "EEF6FF" : "FFF7ED") }, line: { color: C.line } });
      row.forEach((cell, c) => slide.addText(cell, { x: xs[c], y, w: ws[c], h: 0.25, fontSize: r === 0 ? 10 : 11.5, bold: r === 0, color: r === 0 ? C.white : C.ink, margin: 0, fit: "shrink" }));
    });
  } else if (spec.kind === "lodoTable") {
    addTitle(slide, spec.title, spec.lead);
    addLodoTable(slide);
  } else if (spec.diagram) {
    addTitle(slide, spec.title, spec.lead);
    addDiagram(slide, spec.diagram);
  } else if (spec.code) {
    addTitle(slide, spec.title, spec.lead);
    spec.code.forEach((line, i) => {
      slide.addShape(SHAPE.rect, { x: 0.85, y: 2.05 + i * 1.15, w: 11.5, h: 0.74, fill: { color: "111827" }, line: { color: "111827" } });
      slide.addText(line, { x: 1.05, y: 2.28 + i * 1.15, w: 11.1, h: 0.22, fontFace: "Consolas", fontSize: 9.5, color: C.white, margin: 0, fit: "shrink" });
    });
  } else if (spec.kind === "close") {
    slide.background = { color: "F8FAFC" };
    slide.addText(spec.title, { x: 0.82, y: 2.35, w: 4.6, h: 0.8, fontSize: 48, bold: true, color: C.ink, margin: 0 });
    slide.addText(spec.lead, { x: 0.86, y: 3.35, w: 7.8, h: 0.35, fontSize: 17, color: C.blue, margin: 0 });
    addBars(slide, [["Synthetic", .552], ["Real calibrated", .119]], 8.0, 2.35, 4.1, 1.5, C.cyan);
  } else {
    addTitle(slide, spec.title, spec.lead);
    addBullets(slide, spec.bullets || []);
  }

  if (spec.kind !== "cover" && spec.kind !== "section") addFooter(slide, idx + 1);
  slide.addNotes(spec.note || "");
}

function escapeXml(s) {
  return String(s).replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[m]));
}

async function makePreview(spec, idx) {
  const title = escapeXml(spec.title);
  const lead = escapeXml(spec.lead || spec.kicker || "");
  const bg = spec.kind === "section" ? `#${C.ink}` : "#FFFFFF";
  const fg = spec.kind === "section" ? "#FFFFFF" : `#${C.ink}`;
  const sub = spec.kind === "section" ? "#D8E4F8" : `#${C.muted}`;
  let body = "";
  if (spec.bullets) {
    body = spec.bullets.slice(0, 5).map((b, i) => `<text x="86" y="${210 + i * 52}" font-size="22" fill="#${C.ink}">- ${escapeXml(b)}</text>`).join("");
  } else if (spec.kind === "bars") {
    body = spec.data.map((d, i) => {
      const bw = Math.round(480 * d[1]);
      return `<text x="90" y="${225 + i * 58}" font-size="18" fill="#${C.ink}">${escapeXml(d[0])}</text><rect x="310" y="${204 + i * 58}" width="${bw}" height="28" fill="#${spec.title.includes("Real") ? C.amber : C.blue}"/><text x="${330 + bw}" y="${225 + i * 58}" font-size="17" fill="#${C.ink}">${d[1].toFixed(3)}</text>`;
    }).join("");
  } else if (spec.kind === "lodoTable") {
    body = lodoLearned.map((r, i) => `<text x="86" y="${215 + i * 48}" font-size="18" fill="#${C.ink}">${escapeXml(r[0])}: ${escapeXml(r[1])} -> F1 ${r[3].toFixed(3)}</text>`).join("");
  } else if (spec.code) {
    body = spec.code.map((line, i) => `<rect x="86" y="${195 + i * 70}" width="1000" height="46" rx="6" fill="#111827"/><text x="105" y="${225 + i * 70}" font-size="14" fill="#FFFFFF" font-family="Consolas">${escapeXml(line.slice(0, 115))}</text>`).join("");
  }
  const svg = `<svg width="1280" height="720" xmlns="http://www.w3.org/2000/svg"><rect width="1280" height="720" fill="${bg}"/><rect x="0" y="0" width="16" height="720" fill="#${C.blue}"/><text x="72" y="95" font-size="42" font-weight="700" fill="${fg}" font-family="Arial">${title}</text><text x="75" y="142" font-size="24" fill="${sub}" font-family="Arial">${lead}</text>${body}<text x="75" y="680" font-size="13" fill="#8A95A5" font-family="Arial">AutoAD+ project | ${String(idx + 1).padStart(2, "0")}</text></svg>`;
  const out = path.join(PREVIEW_DIR, `slide_${String(idx + 1).padStart(2, "0")}.png`);
  await sharp(Buffer.from(svg)).png().toFile(out);
  return out;
}

async function build() {
  ensureDirs();
  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "AutoAD+ project";
  pptx.subject = "AutoAD+ anomaly detection project presentation";
  pptx.title = "AutoAD+ Project Presentation";
  pptx.company = "CENG AI Project";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Aptos Display",
    bodyFontFace: "Aptos",
    lang: "en-US",
  };
  pptx.defineLayout({ name: "LAYOUT_WIDE", width: W, height: H });

  slides.forEach((spec, idx) => renderSlide(pptx, spec, idx));
  await pptx.writeFile({ fileName: PPTX_PATH });

  const previews = [];
  for (let i = 0; i < slides.length; i += 1) previews.push(await makePreview(slides[i], i));

  const contact = path.join(PREVIEW_DIR, "AutoAD_Project_Presentation_contact_sheet.png");
  const composites = await Promise.all(previews.map(async (p, i) => {
    const input = await sharp(p).resize(256, 144).png().toBuffer();
    return { input, left: (i % 5) * 256, top: Math.floor(i / 5) * 144 };
  }));
  await sharp({
    create: {
      width: 1280,
      height: Math.ceil(previews.length / 5) * 144,
      channels: 4,
      background: "#F3F6FA",
    },
  }).composite(composites).png().toFile(contact);

  console.log(`Wrote ${PPTX_PATH}`);
  console.log(`Wrote ${previews.length} previews to ${PREVIEW_DIR}`);
  console.log(`Wrote ${contact}`);
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});

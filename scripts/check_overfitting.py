"""
Check whether label-tuned anomaly detection results look overfit.

This script runs repeated stratified train/test splits on a labeled CSV. For each
split it lets AdvancedAnomalySystem choose the best score source and threshold on
the train split, then applies that same decision rule to a held-out test split.

Important: AdvancedAnomalySystem is unsupervised and currently fits on the frame
passed to run(). This is therefore a practical leakage/threshold-stability check,
not a full supervised "fit once on train, predict on test" evaluation API.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import warnings
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from advanced_system import AdvancedAnomalySystem  # noqa: E402


optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")


LABEL_COLUMN_CANDIDATES = ("ground_truth", "label", "target", "y_true", "is_anomaly", "anomaly")


def _auto_label_column(df: pd.DataFrame) -> Optional[str]:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in LABEL_COLUMN_CANDIDATES:
        if name in by_lower:
            return str(by_lower[name])
    return None


def _normalize_labels(series: pd.Series) -> np.ndarray:
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(
        {
            "1": 1,
            "true": 1,
            "yes": 1,
            "anomaly": 1,
            "abnormal": 1,
            "attack": 1,
            "0": 0,
            "false": 0,
            "no": 0,
            "normal": 0,
            "benign": 0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    out = mapped.where(mapped.notna(), numeric).fillna(0).astype(float)
    return (out > 0).astype(int).to_numpy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, len(yt))
    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "accuracy": round(float(accuracy), 6),
        "specificity": round(float(specificity), 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _best_percentile(scores: np.ndarray, y_true: np.ndarray, low: int = 70, high: int = 99) -> float:
    arr = np.asarray(scores, dtype=float)
    best_percentile = 95.0
    best_f1 = -1.0
    for percentile in range(low, high + 1):
        threshold = float(np.percentile(arr, percentile))
        f1 = float(_metrics(y_true, arr > threshold)["f1"])
        if f1 > best_f1:
            best_f1 = f1
            best_percentile = float(percentile)
    return best_percentile


def _run_system(
    df: pd.DataFrame,
    *,
    y_true: Optional[np.ndarray],
    label_column: str,
    threshold_percentile: float | str,
    quiet: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        return AdvancedAnomalySystem().run(
            df,
            threshold_percentile=threshold_percentile,
            y_true=y_true,
            exclude_columns=[label_column],
        )


def _score_source_scores(details: Dict[str, Any], ensemble_scores: np.ndarray, source: str) -> np.ndarray:
    if source == "ensemble":
        return np.asarray(ensemble_scores, dtype=float)
    normalized = details.get("normalized_model_scores", {})
    if source in normalized:
        return np.asarray(normalized[source], dtype=float)
    return np.asarray(ensemble_scores, dtype=float)


def _sample_if_requested(df: pd.DataFrame, y: np.ndarray, max_rows: Optional[int], random_state: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    train_idx, _ = train_test_split(
        np.arange(len(df)),
        train_size=max_rows,
        random_state=random_state,
        stratify=y,
    )
    return df.iloc[np.sort(train_idx)].reset_index(drop=True)


def _interpret_gap(gap: float) -> str:
    if gap >= 0.20:
        return "strong_overfitting_signal"
    if gap >= 0.10:
        return "possible_overfitting_signal"
    if gap <= -0.10:
        return "test_higher_than_train_check_split_variance"
    return "no_strong_overfitting_signal"


def run_check(args: argparse.Namespace) -> Dict[str, Any]:
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    df = pd.read_csv(csv_path)
    label_column = args.label_column or _auto_label_column(df)
    if not label_column or label_column not in df.columns:
        raise ValueError(
            f"No label column found. Use --label-column. Searched: {', '.join(LABEL_COLUMN_CANDIDATES)}"
        )

    y_all = _normalize_labels(df[label_column])
    positives = int(np.sum(y_all))
    negatives = int(len(y_all) - positives)
    if positives < 2 or negatives < 2:
        raise ValueError("Need at least two positive and two negative labels for stratified train/test splits.")

    df = _sample_if_requested(df, y_all, args.max_rows, args.random_state)
    y_all = _normalize_labels(df[label_column])

    splitter = StratifiedShuffleSplit(
        n_splits=args.splits,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    rows: List[Dict[str, Any]] = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(df, y_all), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        train_anom, train_scores, train_details = _run_system(
            train_df,
            y_true=y_train,
            label_column=label_column,
            threshold_percentile="auto",
            quiet=args.quiet,
        )
        selection = train_details.get("threshold_selection", {})
        score_source = args.score_source or str(
            selection.get("score_source", train_details.get("selected_score_source", "ensemble"))
        )
        train_source_scores = _score_source_scores(train_details, train_scores, score_source)
        selected_percentile = (
            _best_percentile(train_source_scores, y_train)
            if args.score_source
            else float(selection.get("percentile", train_details.get("threshold_percentile", 95.0)))
        )
        train_threshold = float(np.percentile(train_source_scores, selected_percentile))
        train_pred = train_source_scores > train_threshold
        train_metrics = _metrics(y_train, train_pred)

        _test_anom, test_scores, test_details = _run_system(
            test_df,
            y_true=None,
            label_column=label_column,
            threshold_percentile=95.0,
            quiet=args.quiet,
        )
        test_source_scores = _score_source_scores(test_details, test_scores, score_source)
        test_threshold = float(np.percentile(test_source_scores, selected_percentile))
        test_pred = test_source_scores > test_threshold
        test_metrics = _metrics(y_test, test_pred)

        row = {
            "split": split_id,
            "dataset": csv_path.name,
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "positive_train": int(np.sum(y_train)),
            "positive_test": int(np.sum(y_test)),
            "score_source": score_source,
            "threshold_transfer": "percentile",
            "train_selected_percentile": selected_percentile,
            "train_threshold": round(train_threshold, 6),
            "test_threshold": round(test_threshold, 6),
            "train_f1": train_metrics["f1"],
            "test_f1": test_metrics["f1"],
            "f1_gap_train_minus_test": round(float(train_metrics["f1"] - test_metrics["f1"]), 6),
            "interpretation": _interpret_gap(float(train_metrics["f1"] - test_metrics["f1"])),
            "train_precision": train_metrics["precision"],
            "test_precision": test_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "test_recall": test_metrics["recall"],
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "train_tp": train_metrics["tp"],
            "train_fp": train_metrics["fp"],
            "train_tn": train_metrics["tn"],
            "train_fn": train_metrics["fn"],
            "test_tp": test_metrics["tp"],
            "test_fp": test_metrics["fp"],
            "test_tn": test_metrics["tn"],
            "test_fn": test_metrics["fn"],
            "models_train": ",".join(train_details.get("models", {}).keys()),
            "models_test": ",".join(test_details.get("models", {}).keys()),
        }
        rows.append(row)
        print(
            f"split {split_id}: train_f1={row['train_f1']:.3f} "
            f"test_f1={row['test_f1']:.3f} gap={row['f1_gap_train_minus_test']:.3f} "
            f"source={score_source}"
        )

    output_csv = Path(args.output)
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gaps = [float(r["f1_gap_train_minus_test"]) for r in rows]
    train_f1s = [float(r["train_f1"]) for r in rows]
    test_f1s = [float(r["test_f1"]) for r in rows]
    summary = {
        "dataset": str(csv_path),
        "label_column": label_column,
        "rows_used": int(len(df)),
        "positive_count": int(np.sum(y_all)),
        "negative_count": int(len(y_all) - np.sum(y_all)),
        "splits": int(args.splits),
        "test_size": float(args.test_size),
        "train_f1_mean": round(mean(train_f1s), 6),
        "test_f1_mean": round(mean(test_f1s), 6),
        "f1_gap_mean": round(mean(gaps), 6),
        "f1_gap_std": round(pstdev(gaps) if len(gaps) > 1 else 0.0, 6),
        "overall_interpretation": _interpret_gap(mean(gaps)),
        "csv_output": str(output_csv),
        "note": (
            "This checks whether label-optimized threshold/source selection on train carries to a held-out split. "
            "Because AdvancedAnomalySystem refits unsupervised models per dataframe, it is a practical stability "
            "and leakage check rather than a full train-once/test-once generalization API."
        ),
    }

    output_json = output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a train/test overfitting diagnostic on a labeled anomaly CSV.")
    parser.add_argument("--csv", default="data/external/adb_annthyroid_21feat_normalised.csv", help="Input labeled CSV.")
    parser.add_argument("--label-column", default=None, help="Binary label column. Auto-detected when omitted.")
    parser.add_argument("--splits", type=int, default=3, help="Repeated stratified splits.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Held-out test fraction.")
    parser.add_argument("--max-rows", type=int, default=500, help="Optional stratified row cap for quicker checks.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--score-source", default=None, help="Optional fixed score source: ensemble, iforest, ocsvm, lof, knn_distance, autoencoder, lstm.")
    parser.add_argument("--output", default="results/overfitting_check.csv", help="CSV output path.")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True, help="Suppress model report prints.")
    return parser.parse_args()


if __name__ == "__main__":
    run_check(parse_args())

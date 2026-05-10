"""
Threshold / stability hints for the UI and an optional subsampled train–test check.

The subsampled check mirrors scripts/check_overfitting.py on a smaller stratified sample
to keep API latency tolerable when the user explicitly requests it.
"""

from __future__ import annotations

import contextlib
import io
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from advanced_system import AdvancedAnomalySystem


def build_overfit_hint(
    *,
    evaluation_available: bool,
    label_info: Optional[Dict[str, Any]],
    threshold_selection: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """No extra model runs — interprets existing threshold_selection + label provenance."""
    label_info = label_info or {}
    threshold_selection = threshold_selection or {}
    method = str(threshold_selection.get("method", "") or "")

    if not evaluation_available:
        return {
            "available": False,
            "level": "unknown",
            "title": "Overfit check needs labels",
            "summary": "Add a binary label column (e.g. ground_truth, label) to see threshold-risk guidance.",
            "threshold_method": method or None,
        }

    if label_info.get("strategy") == "synthetic_injection_for_unlabeled_upload":
        return {
            "available": True,
            "level": "synthetic",
            "title": "Synthetic benchmark labels only",
            "summary": "Metrics use injected synthetic ground truth, not real labels.",
            "detail": "Threshold tuning may still use auto mode if the pipeline sees labels; interpret F1 as a stress test, not real-world performance.",
            "threshold_method": method or None,
        }

    if method == "holdout_validated_best_f1_on_labels":
        return {
            "available": True,
            "level": "low",
            "title": "Holdout-validated threshold (lower optimistic bias)",
            "summary": "Percentile and score source were chosen using an internal calibration vs validation split on your labels.",
            "detail": "This reduces in-sample threshold overfitting compared with tuning on all rows. A subsampled train/test check below is still useful for stability across random splits.",
            "threshold_method": method,
            "score_source": threshold_selection.get("score_source"),
        }

    if method == "best_f1_on_labels":
        return {
            "available": True,
            "level": "medium",
            "title": "In-sample label tuning (metrics can be optimistic)",
            "summary": "Best F1 percentile was searched on all labeled rows (dataset too small for internal holdout split).",
            "detail": "Reported F1 may be optimistic. Use the optional subsampled train/test check for a rough stability signal.",
            "threshold_method": method,
            "score_source": threshold_selection.get("score_source"),
        }

    if method in ("fixed_percentile", "fallback_percentile", ""):
        return {
            "available": True,
            "level": "unknown",
            "title": "Fixed or fallback percentile",
            "summary": "Threshold is not driven by a label-based F1 search on this run.",
            "detail": "Overfitting of the percentile rule to labels is less relevant; ranking metrics (ROC-AUC / PR-AUC) still describe separation quality.",
            "threshold_method": method or "fixed_percentile",
        }

    return {
        "available": True,
        "level": "unknown",
        "title": "Threshold selection",
        "summary": f"Method: {method or 'default'}",
        "detail": None,
        "threshold_method": method or None,
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, len(yt))
    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "accuracy": round(float(accuracy), 6),
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


def _score_source_scores(
    details: Dict[str, Any], ensemble_scores: np.ndarray, source: str
) -> np.ndarray:
    if source == "ensemble":
        return np.asarray(ensemble_scores, dtype=float)
    normalized = details.get("normalized_model_scores", {})
    if source in normalized:
        return np.asarray(normalized[source], dtype=float)
    return np.asarray(ensemble_scores, dtype=float)


def _run_system(
    runner: AdvancedAnomalySystem,
    df: pd.DataFrame,
    *,
    y_true: Optional[np.ndarray],
    label_column: str,
    threshold_percentile: float | str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        return runner.run(
            df,
            threshold_percentile=threshold_percentile,
            y_true=y_true,
            exclude_columns=[label_column],
        )


def _sample_stratified(
    df: pd.DataFrame, y: np.ndarray, max_rows: int, random_state: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    if len(df) <= max_rows:
        return df.reset_index(drop=True), y
    train_idx, _ = train_test_split(
        np.arange(len(df)),
        train_size=max_rows,
        random_state=random_state,
        stratify=y,
    )
    idx = np.sort(train_idx)
    return df.iloc[idx].reset_index(drop=True), y[idx]


def interpret_gap(gap: float) -> str:
    if gap >= 0.20:
        return "strong_overfitting_signal"
    if gap >= 0.10:
        return "possible_overfitting_signal"
    if gap <= -0.10:
        return "test_higher_than_train_check_split_variance"
    return "no_strong_overfitting_signal"


def run_subsampled_overfit_diagnostic(
    df: pd.DataFrame,
    label_column: str,
    y_all: np.ndarray,
    *,
    runner: Optional[AdvancedAnomalySystem] = None,
    n_splits: int = 2,
    test_size: float = 0.3,
    max_rows: int = 450,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Repeated stratified splits on a capped sample. Each split refits on train and test
    frames (same semantics as scripts/check_overfitting.py).
    """
    positives = int(np.sum(y_all))
    negatives = int(len(y_all) - positives)
    if positives < 2 or negatives < 2:
        return {
            "available": False,
            "skipped_reason": "Need at least two positive and two negative labels for stratified splits.",
        }

    runner = runner or AdvancedAnomalySystem()
    work_df, y_work = _sample_stratified(df, y_all, max_rows, random_state)

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    rows: List[Dict[str, Any]] = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(work_df, y_work), start=1):
        train_df = work_df.iloc[train_idx].reset_index(drop=True)
        test_df = work_df.iloc[test_idx].reset_index(drop=True)
        y_train = y_work[train_idx]
        y_test = y_work[test_idx]

        _train_anom, train_scores, train_details = _run_system(
            runner,
            train_df,
            y_true=y_train,
            label_column=label_column,
            threshold_percentile="auto",
        )
        selection = train_details.get("threshold_selection", {})
        score_source = str(
            selection.get("score_source", train_details.get("selected_score_source", "ensemble"))
        )
        train_source_scores = _score_source_scores(train_details, train_scores, score_source)
        selected_percentile = float(
            selection.get("percentile", train_details.get("threshold_percentile", 95.0))
        )
        train_threshold = float(np.percentile(train_source_scores, selected_percentile))
        train_pred = train_source_scores > train_threshold
        train_metrics = _metrics(y_train, train_pred)

        _test_anom, test_scores, test_details = _run_system(
            runner,
            test_df,
            y_true=None,
            label_column=label_column,
            threshold_percentile=95.0,
        )
        test_source_scores = _score_source_scores(test_details, test_scores, score_source)
        test_threshold = float(np.percentile(test_source_scores, selected_percentile))
        test_pred = test_source_scores > test_threshold
        test_metrics = _metrics(y_test, test_pred)

        gap = float(train_metrics["f1"] - test_metrics["f1"])
        rows.append(
            {
                "split": split_id,
                "rows_train": int(len(train_df)),
                "rows_test": int(len(test_df)),
                "score_source": score_source,
                "train_selected_percentile": selected_percentile,
                "train_f1": train_metrics["f1"],
                "test_f1": test_metrics["f1"],
                "f1_gap_train_minus_test": round(gap, 6),
                "interpretation": interpret_gap(gap),
            }
        )

    gaps = [float(r["f1_gap_train_minus_test"]) for r in rows]
    train_f1s = [float(r["train_f1"]) for r in rows]
    test_f1s = [float(r["test_f1"]) for r in rows]

    return {
        "available": True,
        "label_column": label_column,
        "rows_in_upload": int(len(df)),
        "rows_used": int(len(work_df)),
        "max_rows_cap": max_rows,
        "n_splits": n_splits,
        "test_size": test_size,
        "splits": rows,
        "summary": {
            "train_f1_mean": round(mean(train_f1s), 6),
            "test_f1_mean": round(mean(test_f1s), 6),
            "f1_gap_mean": round(mean(gaps), 6),
            "f1_gap_std": round(pstdev(gaps) if len(gaps) > 1 else 0.0, 6),
            "overall_interpretation": interpret_gap(mean(gaps)),
        },
        "note": (
            "Subsampled train/test stability check: each split refits unsupervised models on train "
            "and test frames; the same percentile rule is transferred. This is a practical leakage "
            "signal, not a full train-once supervised evaluation. See scripts/check_overfitting.py for "
            "CLI defaults."
        ),
    }

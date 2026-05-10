from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from advanced_system import AdvancedAnomalySystem  # noqa: E402
from synthetic_injection import binary_classification_metrics, binary_score_metrics  # noqa: E402


LABEL_CANDIDATES = ("ground_truth", "label", "target", "y_true", "is_anomaly", "anomaly")


def _label_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit and explicit in df.columns:
        return explicit
    by_lower = {str(c).lower(): c for c in df.columns}
    for cand in LABEL_CANDIDATES:
        if cand in by_lower:
            return by_lower[cand]
    raise ValueError(f"No label column found. Tried: {LABEL_CANDIDATES}")


def _labels(series: pd.Series) -> np.ndarray:
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


def _run_scores(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        _, scores, details = AdvancedAnomalySystem().run(df)
    out = {"ensemble": np.asarray(scores, dtype=float)}
    out.update({k: np.asarray(v, dtype=float) for k, v in details.get("normalized_model_scores", {}).items()})
    return out


def _metric(y: np.ndarray, scores: np.ndarray, metric: str) -> float:
    if len(np.unique(y)) < 2 or float(np.max(scores) - np.min(scores)) < 1e-12:
        return 0.0
    if metric == "pr_auc":
        return float(average_precision_score(y, scores))
    return float(roc_auc_score(y, scores))


def _combine(scores_by_name: Dict[str, np.ndarray], names: List[str], weights: np.ndarray) -> np.ndarray:
    combined = np.zeros_like(scores_by_name[names[0]], dtype=float)
    for name, weight in zip(names, weights):
        combined += float(weight) * scores_by_name[name]
    return combined


def _search_weights(
    y: np.ndarray,
    scores_by_name: Dict[str, np.ndarray],
    *,
    metric: str,
    iterations: int,
    seed: int,
) -> Tuple[Dict[str, float], float]:
    names = [n for n in scores_by_name if n != "ensemble"]
    rng = np.random.default_rng(seed)
    candidates = [np.ones(len(names), dtype=float) / max(len(names), 1)]
    for i, name in enumerate(names):
        one = np.zeros(len(names), dtype=float)
        one[i] = 1.0
        candidates.append(one)

    best_weights = candidates[0]
    best_score = -1.0
    for weights in candidates:
        score = _metric(y, _combine(scores_by_name, names, weights), metric)
        if score > best_score:
            best_score = score
            best_weights = weights

    for _ in range(iterations):
        weights = rng.dirichlet(np.ones(len(names), dtype=float))
        score = _metric(y, _combine(scores_by_name, names, weights), metric)
        if score > best_score:
            best_score = score
            best_weights = weights

    return {name: round(float(weight), 6) for name, weight in zip(names, best_weights)}, float(best_score)


def _default_f1(y: np.ndarray, scores: np.ndarray, contamination: float) -> float:
    threshold = float(np.percentile(scores, 100.0 * (1.0 - contamination)))
    return float(f1_score(y, scores > threshold, zero_division=0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate AutoAD ensemble weights on a labeled dataset split.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--out", default=str(ROOT / "configs" / "model_weights.yaml"))
    parser.add_argument("--metric", choices=["roc_auc", "pr_auc"], default="roc_auc")
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    df = pd.read_csv(dataset)
    label_col = _label_column(df, args.label_column)
    y = _labels(df[label_col])
    X = df.drop(columns=[label_col])

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    split = int(round(len(indices) * (1.0 - args.test_size)))
    cal_idx = np.sort(indices[:split])
    test_idx = np.sort(indices[split:])

    cal_scores = _run_scores(X.iloc[cal_idx].reset_index(drop=True))
    test_scores = _run_scores(X.iloc[test_idx].reset_index(drop=True))
    weights, cal_score = _search_weights(
        y[cal_idx],
        cal_scores,
        metric=args.metric,
        iterations=args.iterations,
        seed=args.seed,
    )

    names = list(weights)
    weight_arr = np.asarray([weights[n] for n in names], dtype=float)
    weight_arr = weight_arr / max(float(np.sum(weight_arr)), 1e-12)
    test_combined = _combine(test_scores, names, weight_arr)
    contamination = max(float(np.mean(y[cal_idx])), 1.0 / max(len(cal_idx), 1))
    test_pred = test_combined > float(np.percentile(test_combined, 100.0 * (1.0 - contamination)))
    test_class = binary_classification_metrics(y[test_idx], test_pred)
    test_rank = binary_score_metrics(y[test_idx], test_combined)

    payload = {
        "weights": weights,
        "calibration": {
            "dataset": str(dataset),
            "label_column": label_col,
            "metric": args.metric,
            "calibration_score": round(cal_score, 6),
            "calibration_rows": int(len(cal_idx)),
            "test_rows": int(len(test_idx)),
            "expected_contamination": round(contamination, 6),
            "heldout_f1": round(float(test_class["f1"]), 6),
            "heldout_roc_auc": round(float(test_rank["roc_auc"]), 6),
            "heldout_pr_auc": round(float(test_rank["pr_auc"]), 6),
        },
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except Exception:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

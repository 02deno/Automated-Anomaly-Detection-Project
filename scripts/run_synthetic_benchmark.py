from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from advanced_system import AdvancedAnomalySystem  # noqa: E402
from synthetic_injection import (  # noqa: E402
    binary_classification_metrics,
    binary_score_metrics,
    inject,
    list_scenarios,
)


DEFAULT_SCENARIOS = [
    "spike_single",
    "joint_shift",
    "scale_burst",
    "temporal_block",
    "dead_sensor",
    "sign_flip",
]


def _round_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: round(float(v), 6) for k, v in metrics.items()}


def _labels_at_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float) > float(threshold)).astype(int)


def _best_percentile(scores: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    best = {
        "percentile": 95.0,
        "threshold": float(np.percentile(scores, 95)),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    for percentile in range(50, 100):
        threshold = float(np.percentile(scores, percentile))
        pred = _labels_at_threshold(scores, threshold)
        metrics = binary_classification_metrics(y_true, pred)
        if metrics["f1"] > best["f1"]:
            best = {
                "percentile": float(percentile),
                "threshold": threshold,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            }
    return best


def _evaluate_score_source(
    *,
    scenario: str,
    seed: int,
    rows: int,
    source_name: str,
    scores: np.ndarray,
    default_threshold: float,
    y_true: np.ndarray,
    model_names: List[str],
) -> Dict[str, Any]:
    default_pred = _labels_at_threshold(scores, default_threshold)
    injected = np.where(y_true == 1)[0].astype(int).tolist()
    detected = np.where(default_pred == 1)[0].astype(int).tolist()
    overlap = sorted(set(injected) & set(detected))
    top_scores = np.argsort(scores)[-10:][::-1].astype(int).tolist()
    default_metrics = {
        **binary_classification_metrics(y_true, default_pred),
        **binary_score_metrics(y_true, scores),
    }
    best = _best_percentile(scores, y_true)

    return {
        "scenario": scenario,
        "seed": seed,
        "rows": rows,
        "score_source": source_name,
        "injected_count": int(np.sum(y_true)),
        "detected_count": int(np.sum(default_pred)),
        "injected_rows": injected,
        "detected_rows": detected,
        "overlap_rows": overlap,
        "top_score_rows": top_scores,
        "threshold": float(default_threshold),
        "best_percentile": best["percentile"],
        "best_threshold": best["threshold"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_f1": best["f1"],
        "models_used": model_names,
        "metrics": _round_metrics(default_metrics),
    }


def run_one(df: pd.DataFrame, scenario: str, seed: int) -> Dict[str, Any]:
    corrupted, y_true = inject(df, scenario, random_seed=seed)
    anomalies, scores, details = AdvancedAnomalySystem().run(corrupted)
    model_names = list(details["models"].keys())
    per_source = [
        _evaluate_score_source(
            scenario=scenario,
            seed=seed,
            rows=int(len(df)),
            source_name="ensemble",
            scores=scores,
            default_threshold=float(details["threshold"]),
            y_true=y_true,
            model_names=model_names,
        )
    ]
    for model_name, model_scores in details.get("normalized_model_scores", {}).items():
        per_source.append(
            _evaluate_score_source(
                scenario=scenario,
                seed=seed,
                rows=int(len(df)),
                source_name=model_name,
                scores=np.asarray(model_scores, dtype=float),
                default_threshold=float(np.percentile(model_scores, 95)),
                y_true=y_true,
                model_names=model_names,
            )
        )
    return {"scenario": scenario, "rows": per_source}


def write_summary_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "score_source",
        "seed",
        "rows",
        "injected_count",
        "detected_count",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "best_percentile",
        "best_precision",
        "best_recall",
        "best_f1",
        "injected_rows",
        "detected_rows",
        "overlap_rows",
        "top_score_rows",
        "threshold",
        "models_used",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scenario_result in rows:
            for row in scenario_result["rows"]:
                metrics = row["metrics"]
                writer.writerow(
                    {
                        "scenario": row["scenario"],
                        "score_source": row["score_source"],
                        "seed": row["seed"],
                        "rows": row["rows"],
                        "injected_count": row["injected_count"],
                        "detected_count": row["detected_count"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"],
                        "best_percentile": row["best_percentile"],
                        "best_precision": round(float(row["best_precision"]), 6),
                        "best_recall": round(float(row["best_recall"]), 6),
                        "best_f1": round(float(row["best_f1"]), 6),
                        "injected_rows": row["injected_rows"],
                        "detected_rows": row["detected_rows"],
                        "overlap_rows": row["overlap_rows"],
                        "top_score_rows": row["top_score_rows"],
                        "threshold": row["threshold"],
                        "models_used": row["models_used"],
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject synthetic anomalies into a CSV and evaluate the anomaly detector."
    )
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "data" / "synthetic_examples" / "baseline_server_metrics.csv"),
        help="CSV file to corrupt with synthetic anomalies.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=list_scenarios(),
        help="Scenario to run. Repeat for multiple scenarios. Defaults to common numeric scenarios.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "synthetic_benchmark_summary.csv"),
        help="Where to write the summary CSV.",
    )
    args = parser.parse_args()

    logging.getLogger("optuna").setLevel(logging.WARNING)

    dataset = Path(args.dataset)
    df = pd.read_csv(dataset)
    scenarios = args.scenario or DEFAULT_SCENARIOS
    rows = [run_one(df, scenario, args.seed) for scenario in scenarios]
    write_summary_csv(rows, Path(args.out))

    print(f"Dataset: {dataset}")
    print(f"Rows: {len(df)}")
    print(f"Summary CSV: {args.out}")
    print()
    for row in rows:
        print(row["scenario"])
        for source_row in row["rows"]:
            metrics = source_row["metrics"]
            print(
                f"  {source_row['score_source']}: "
                f"injected={source_row['injected_rows']} detected={source_row['detected_rows']} "
                f"overlap={source_row['overlap_rows']} "
                f"f1@95={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f} "
                f"pr_auc={metrics['pr_auc']:.3f} "
                f"best_p={source_row['best_percentile']:.0f} "
                f"best_f1={source_row['best_f1']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

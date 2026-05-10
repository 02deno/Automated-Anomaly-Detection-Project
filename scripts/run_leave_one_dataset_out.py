from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from advanced_system import AdvancedAnomalySystem, MetaSelectionLayer  # noqa: E402
from synthetic_injection import binary_classification_metrics, binary_score_metrics  # noqa: E402


LABEL_CANDIDATES = ("ground_truth", "label", "target", "y_true", "is_anomaly", "anomaly")
DEFAULT_ALLOWED = ["ensemble", "iforest", "ocsvm", "lof", "temporal_change"]


def _dataset_entries(values: List[str]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for value in values:
        parts = value.split("::")
        path = parts[0]
        label = parts[1] if len(parts) > 1 else Path(path).stem
        label_col = parts[2] if len(parts) > 2 else "ground_truth"
        entries.append({"path": path, "label": label, "label_column": label_col})
    return entries


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


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


def _load_dataset(entry: Dict[str, str]) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(_resolve(entry["path"]))
    label_col = entry.get("label_column")
    if not label_col or label_col not in df.columns:
        by_lower = {str(c).lower(): c for c in df.columns}
        label_col = next((by_lower[c] for c in LABEL_CANDIDATES if c in by_lower), None)
    if not label_col or label_col not in df.columns:
        raise ValueError(f"No label column found for {entry['label']}")
    y = _normalize_labels(df[label_col])
    return df.drop(columns=[label_col]), y


def _limit_rows(df: pd.DataFrame, y: np.ndarray, *, max_rows: Optional[int], seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    if not max_rows or len(df) <= max_rows:
        return df.reset_index(drop=True), y
    rng = np.random.default_rng(seed)
    chosen: List[np.ndarray] = []
    for value in sorted(np.unique(y)):
        cls = np.flatnonzero(y == value)
        take = max(1, int(round(max_rows * len(cls) / len(y))))
        chosen.append(rng.choice(cls, size=min(take, len(cls)), replace=False))
    selected = np.concatenate(chosen)
    if len(selected) > max_rows:
        selected = rng.choice(selected, size=max_rows, replace=False)
    elif len(selected) < max_rows:
        remaining = np.setdiff1d(np.arange(len(y)), selected, assume_unique=False)
        extra = rng.choice(remaining, size=min(max_rows - len(selected), len(remaining)), replace=False)
        selected = np.concatenate([selected, extra])
    selected = np.sort(selected)
    return df.iloc[selected].reset_index(drop=True), y[selected]


def _best_source(summary: pd.DataFrame, dataset: str, metric: str, allowed: List[str]) -> Dict[str, Any]:
    rows = summary[(summary["dataset"] == dataset) & summary["score_source"].isin(allowed)].copy()
    if rows.empty:
        raise ValueError(f"No summary rows for dataset {dataset}")
    for col in ("f1", "roc_auc", "pr_auc", "rows", "positive_count"):
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0.0)
    rows = rows.sort_values([metric, "roc_auc", "pr_auc"], ascending=[False, False, False])
    winner = rows.iloc[0]
    return {
        "selected_source": str(winner["score_source"]),
        "selected_metric": metric,
        "selected_score": round(float(winner[metric]), 6),
        "source_roc_auc": round(float(winner.get("roc_auc", 0.0)), 6),
        "source_pr_auc": round(float(winner.get("pr_auc", 0.0)), 6),
        "evaluated_rows": int(float(winner.get("rows", 0.0))),
        "expected_contamination": round(
            float(winner.get("positive_count", 0.0)) / max(float(winner.get("rows", 1.0)), 1.0),
            6,
        ),
    }


def _profile(
    entry: Dict[str, str],
    details: Dict[str, Any],
    winner: Dict[str, Any],
    allowed: List[str],
) -> Dict[str, Any]:
    meta = details["meta"]
    selector = MetaSelectionLayer([{**winner, "_allowed_sources": allowed}])
    return {
        "dataset": entry["label"],
        "sample_count": int(winner.get("evaluated_rows") or meta["samples"]),
        "feature_count": int(meta["features"]),
        "missing_rate": round(float(meta["missing_rate"]), 6),
        "sparsity_zero_rate": round(float(meta["sparsity_zero_rate"]), 6),
        "correlation_abs_mean": round(float(meta["correlation_abs_mean"]), 6),
        "high_corr_pair_count": int(meta["high_corr_pair_count"]),
        "threshold_strategy": "expected_contamination",
        "feature_vector": selector.feature_vector(meta, details.get("normalized_model_scores", {})),
        **winner,
    }


def _write_meta_config(profiles: List[Dict[str, Any]], allowed: List[str], metric: str, path: Path) -> None:
    payload = {
        "description": "Temporary leave-one-dataset-out learned meta-selector config.",
        "selector_mode": "learned_score_diagnostics",
        "metric": metric,
        "allowed_sources": allowed,
        "profiles": profiles,
    }
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_model(
    df: pd.DataFrame,
    *,
    weights_config: Optional[str],
    meta_config: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        return AdvancedAnomalySystem(
            weights_config_path=weights_config,
            meta_config_path=meta_config,
        ).run(df)


def _evaluate_scores(dataset: str, y: np.ndarray, scores: np.ndarray, pred: np.ndarray, details: Dict[str, Any]) -> Dict[str, Any]:
    cls = binary_classification_metrics(y, pred.astype(int))
    rank = binary_score_metrics(y, scores)
    return {
        "dataset": dataset,
        "score_source": "autoad_lodo",
        "selected_source": details.get("meta_selection", {}).get("selected_source"),
        "matched_dataset": details.get("meta_selection", {}).get("matched_dataset"),
        "rows": int(len(y)),
        "positive_count": int(np.sum(y)),
        "predicted_positive_count": int(np.sum(pred)),
        **{k: round(float(v), 6) for k, v in cls.items()},
        **{k: round(float(v), 6) for k, v in rank.items()},
    }


def _write_rows(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-dataset-out validation for AutoAD meta-selection.")
    parser.add_argument("--summary", required=True, help="Real-data summary used to choose winners for training folds.")
    parser.add_argument("--dataset", action="append", required=True, help="PATH::LABEL::LABEL_COLUMN")
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights-config", default=None)
    parser.add_argument("--metric", choices=["f1", "roc_auc", "pr_auc"], default="f1")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed-source", action="append", default=None)
    args = parser.parse_args()

    allowed = args.allowed_source or DEFAULT_ALLOWED
    summary = pd.read_csv(_resolve(args.summary))
    entries = _dataset_entries(args.dataset)
    loaded = {}
    for entry in entries:
        X, y = _load_dataset(entry)
        X, y = _limit_rows(X, y, max_rows=args.max_rows, seed=args.seed)
        loaded[entry["label"]] = (entry, X, y)

    base_details: Dict[str, Dict[str, Any]] = {}
    for label, (_, X, _) in loaded.items():
        _, _, details = _run_model(X, weights_config=args.weights_config, meta_config=None)
        base_details[label] = details

    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for heldout_label, (_, X_test, y_test) in loaded.items():
            profiles = []
            for train_label, (entry, X_train, _) in loaded.items():
                if train_label == heldout_label:
                    continue
                winner = _best_source(summary, train_label, args.metric, allowed)
                profiles.append(_profile(entry, base_details[train_label], winner, allowed))
            cfg_path = tmp_path / f"meta_{heldout_label}.yaml"
            _write_meta_config(profiles, allowed, args.metric, cfg_path)
            pred, scores, details = _run_model(X_test, weights_config=args.weights_config, meta_config=cfg_path)
            rows.append(_evaluate_scores(heldout_label, y_test, scores, pred, details))
            print(
                f"{heldout_label}: selected={rows[-1]['selected_source']} "
                f"matched={rows[-1]['matched_dataset']} f1={rows[-1]['f1']:.3f} "
                f"roc_auc={rows[-1]['roc_auc']:.3f}"
            )

    out = _resolve(args.out)
    _write_rows(rows, out)
    avg_f1 = float(np.mean([r["f1"] for r in rows]))
    avg_roc = float(np.mean([r["roc_auc"] for r in rows]))
    avg_pr = float(np.mean([r["pr_auc"] for r in rows]))
    print(f"Average: f1={avg_f1:.6f} roc_auc={avg_roc:.6f} pr_auc={avg_pr:.6f}")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

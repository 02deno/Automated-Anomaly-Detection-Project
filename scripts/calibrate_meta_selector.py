from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from advanced_system import AdvancedAnomalySystem, MetaSelectionLayer  # noqa: E402


def _dataset_entries(values: Optional[List[str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for value in values or []:
        parts = value.split("::")
        path = parts[0]
        label = parts[1] if len(parts) > 1 else Path(path).stem
        label_col = parts[2] if len(parts) > 2 else "ground_truth"
        out.append({"path": path, "label": label, "label_column": label_col})
    return out


def _load_features(entry: Dict[str, str]) -> pd.DataFrame:
    path = Path(entry["path"])
    if not path.is_absolute():
        path = ROOT / path
    df = pd.read_csv(path)
    label_col = entry.get("label_column")
    if label_col and label_col in df.columns:
        df = df.drop(columns=[label_col])
    return df


def _run_for_profile(df: pd.DataFrame, weights_config: Optional[str]) -> Dict[str, Any]:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        _, _, details = AdvancedAnomalySystem(weights_config_path=weights_config).run(df)
    return details


def _best_source(summary: pd.DataFrame, dataset: str, metric: str, allowed: List[str]) -> Dict[str, Any]:
    rows = summary[(summary["dataset"] == dataset) & summary["score_source"].isin(allowed)].copy()
    if rows.empty:
        raise ValueError(f"No summary rows found for dataset {dataset!r}")
    rows[metric] = pd.to_numeric(rows[metric], errors="coerce").fillna(0.0)
    rows = rows.sort_values([metric, "roc_auc", "pr_auc"], ascending=[False, False, False])
    winner = rows.iloc[0].to_dict()
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Learn dataset-aware score-source selection profiles.")
    parser.add_argument("--summary", required=True, help="Real-data evaluation summary CSV.")
    parser.add_argument("--dataset", action="append", required=True, help="PATH::LABEL::LABEL_COLUMN")
    parser.add_argument("--out", default=str(ROOT / "configs" / "meta_selector.yaml"))
    parser.add_argument("--metric", default="f1", choices=["f1", "roc_auc", "pr_auc"])
    parser.add_argument("--weights-config", default=None, help="Optional calibrated weights used while building score diagnostics.")
    parser.add_argument(
        "--allowed-source",
        action="append",
        default=None,
        help="Allowed source name. Repeatable. Defaults to ensemble/iforest/ocsvm/lof/temporal_change.",
    )
    args = parser.parse_args()

    allowed = args.allowed_source or ["ensemble", "iforest", "ocsvm", "lof", "temporal_change"]
    summary = pd.read_csv(ROOT / args.summary if not Path(args.summary).is_absolute() else args.summary)
    profiles = []
    for entry in _dataset_entries(args.dataset):
        features = _load_features(entry)
        details = _run_for_profile(features, args.weights_config)
        meta = details["meta"]
        winner = _best_source(summary, entry["label"], args.metric, allowed)
        selector = MetaSelectionLayer([{**winner, "_allowed_sources": allowed}])
        profiles.append(
            {
                "dataset": entry["label"],
                "sample_count": int(winner_rows if (winner_rows := winner.get("evaluated_rows", 0)) else meta["samples"]),
                "feature_count": int(meta["features"]),
                "missing_rate": round(float(meta["missing_rate"]), 6),
                "sparsity_zero_rate": round(float(meta["sparsity_zero_rate"]), 6),
                "correlation_abs_mean": round(float(meta["correlation_abs_mean"]), 6),
                "high_corr_pair_count": int(meta["high_corr_pair_count"]),
                "threshold_strategy": "expected_contamination",
                "feature_vector": selector.feature_vector(meta, details.get("normalized_model_scores", {})),
                **winner,
            }
        )

    payload = {
        "description": "Learned dataset-aware source selector trained from real-data validation results and runtime score diagnostics.",
        "selector_mode": "learned_score_diagnostics",
        "metric": args.metric,
        "allowed_sources": allowed,
        "profiles": profiles,
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

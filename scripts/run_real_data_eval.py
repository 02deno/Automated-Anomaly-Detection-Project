"""
Evaluate AdvancedAnomalySystem on **real** datasets that ship with binary anomaly labels.

For each labeled CSV (e.g. Annthyroid, KDD Cup 1999 SMTP/HTTP) the script:
  1. Drops the label column from model features and keeps it as y_true.
  2. Runs the full pipeline once (Optuna + ensemble) and records ROC-AUC / PR-AUC for the
     ensemble plus every component model that the analysis layer selected.
  3. Optionally re-runs after applying a synthetic injector on top of the real frame so
     we can compare "real anomalies" vs "real + synthetic" and check whether the
     model ranking is consistent (the headline contribution claim).

Outputs:
  results/real_data_summary.csv               # one row per (dataset, score_source)
  results/real_vs_synthetic_consistency.csv   # only when --inject is provided

The detector is unsupervised; the labels are used only to score continuous outputs
(ROC-AUC / PR-AUC are threshold-independent so they are comparable across datasets).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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


DEFAULT_DATASETS: List[Dict[str, Any]] = [
    {
        "label": "adb_annthyroid",
        "path": "data/external/adb_annthyroid_21feat_normalised.csv",
        "label_column": "ground_truth",
    },
    {
        "label": "kddcup99_http",
        "path": "data/external/kddcup99_http_percent10.csv",
        "label_column": "ground_truth",
    },
    {
        "label": "kddcup99_smtp",
        "path": "data/external/kddcup99_smtp_percent10.csv",
        "label_column": "ground_truth",
    },
]

LABEL_COLUMN_CANDIDATES = ("ground_truth", "label", "target", "y_true", "is_anomaly", "anomaly")


def _auto_label_column(df: pd.DataFrame) -> Optional[str]:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in LABEL_COLUMN_CANDIDATES:
        if cand in by_lower:
            return by_lower[cand]
    return None


def _normalize_labels(series: pd.Series) -> np.ndarray:
    """Coerce common label representations to {0,1}; non-binary → 0."""
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(
        {
            "1": 1, "true": 1, "yes": 1, "anomaly": 1, "abnormal": 1, "attack": 1,
            "0": 0, "false": 0, "no": 0, "normal": 0, "benign": 0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    out = mapped.where(mapped.notna(), numeric).fillna(0).astype(float)
    return (out > 0).astype(int).to_numpy()


def _run_once(df: pd.DataFrame, *, quiet: bool = True):
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        anomalies, scores, details = AdvancedAnomalySystem().run(df)
    return anomalies, scores, details


def _evaluate(*, dataset_label: str, df: pd.DataFrame, y_true: np.ndarray,
              tag: str, quiet: bool) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    if y_true.size != len(df):
        raise ValueError(
            f"y_true length {y_true.size} != dataset rows {len(df)} for {dataset_label!r}"
        )
    anomalies, scores, details = _run_once(df, quiet=quiet)
    rows: List[Dict[str, Any]] = []
    ensemble_classification = binary_classification_metrics(y_true, anomalies.astype(int))
    rows.append(
        {
            "dataset": dataset_label,
            "scenario": tag,
            "score_source": "ensemble",
            "rows": int(len(df)),
            "positive_count": int(np.sum(y_true)),
            "predicted_positive_count": int(np.sum(anomalies)),
            **{k: round(float(v), 6) for k, v in ensemble_classification.items()},
            **{k: round(float(v), 6) for k, v in binary_score_metrics(y_true, np.asarray(scores, dtype=float)).items()},
            "models_used": ",".join(details["models"].keys()),
        }
    )
    for model_name, model_scores in details.get("normalized_model_scores", {}).items():
        arr = np.asarray(model_scores, dtype=float)
        pred = arr > float(np.percentile(arr, 95))
        rows.append(
            {
                "dataset": dataset_label,
                "scenario": tag,
                "score_source": str(model_name),
                "rows": int(len(df)),
                "positive_count": int(np.sum(y_true)),
                "predicted_positive_count": int(np.sum(pred)),
                **{k: round(float(v), 6) for k, v in binary_classification_metrics(y_true, pred).items()},
                **{k: round(float(v), 6) for k, v in binary_score_metrics(y_true, arr).items()},
                "models_used": ",".join(details["models"].keys()),
            }
        )
    return rows


def _consistency_rows(real_rows: List[Dict[str, Any]],
                      synth_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For each (dataset, score_source) compare ROC-AUC ranking real vs real+synthetic."""

    def _by_source(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {(r["dataset"], r["score_source"]): r for r in rows}

    real_by = _by_source(real_rows)
    synth_by = _by_source(synth_rows)
    out: List[Dict[str, Any]] = []
    for key, real in real_by.items():
        synth = synth_by.get(key)
        if synth is None:
            continue
        out.append(
            {
                "dataset": key[0],
                "score_source": key[1],
                "real_roc_auc": real.get("roc_auc", 0.0),
                "synthetic_roc_auc": synth.get("roc_auc", 0.0),
                "delta_roc_auc": round(float(synth.get("roc_auc", 0.0)) - float(real.get("roc_auc", 0.0)), 6),
                "real_pr_auc": real.get("pr_auc", 0.0),
                "synthetic_pr_auc": synth.get("pr_auc", 0.0),
                "delta_pr_auc": round(float(synth.get("pr_auc", 0.0)) - float(real.get("pr_auc", 0.0)), 6),
            }
        )
    return out


def _ranking_agreement(real_rows: List[Dict[str, Any]],
                       synth_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per dataset, compare model ranking by ROC-AUC between real and real+synthetic."""

    def _ordering(rows: Iterable[Dict[str, Any]], dataset: str) -> List[str]:
        slice_rows = [r for r in rows if r["dataset"] == dataset]
        slice_rows.sort(key=lambda r: float(r.get("roc_auc", 0.0)), reverse=True)
        return [r["score_source"] for r in slice_rows]

    datasets = sorted({r["dataset"] for r in real_rows} & {r["dataset"] for r in synth_rows})
    summary: Dict[str, Dict[str, Any]] = {}
    for ds in datasets:
        real_order = _ordering(real_rows, ds)
        synth_order = _ordering(synth_rows, ds)
        summary[ds] = {
            "real_ranking": real_order,
            "synthetic_ranking": synth_order,
            "top_match": (bool(real_order and synth_order and real_order[0] == synth_order[0])),
        }
    return summary


def _resolve_dataset_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset:
        specs: List[Dict[str, Any]] = []
        for entry in args.dataset:
            parts = entry.split("::")
            path = parts[0]
            label = parts[1] if len(parts) > 1 else Path(path).stem
            label_column = parts[2] if len(parts) > 2 else None
            specs.append({"label": label, "path": path, "label_column": label_column})
        return specs
    return DEFAULT_DATASETS


def _write_rows(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the unsupervised pipeline against ground-truth labels on real datasets.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset entry as 'PATH[::LABEL[::LABEL_COLUMN]]' (repeatable). "
        "Defaults to Annthyroid + KDD'99 HTTP/SMTP if omitted.",
    )
    parser.add_argument(
        "--inject",
        choices=list_scenarios(),
        help="Optional synthetic scenario to apply on top of the real frame; "
        "produces real_vs_synthetic_consistency.csv.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the synthetic injector.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "real_data_summary.csv"),
        help="Where to write the per-(dataset, score_source) summary CSV.",
    )
    parser.add_argument(
        "--consistency-out",
        default=str(ROOT / "results" / "real_vs_synthetic_consistency.csv"),
        help="Where to write the real vs synthetic consistency CSV (used only with --inject).",
    )
    parser.add_argument("--verbose", action="store_true", help="Do not silence detector stdout.")
    args = parser.parse_args(argv)

    logging.getLogger("optuna").setLevel(logging.WARNING)

    specs = _resolve_dataset_specs(args)
    real_rows: List[Dict[str, Any]] = []
    synth_rows: List[Dict[str, Any]] = []

    for spec in specs:
        path = Path(spec["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            print(f"SKIP {spec['label']}: {path} not found "
                  f"(run scripts/fetch_public_datasets.py to populate)")
            continue
        df = pd.read_csv(path)
        label_col = spec.get("label_column") or _auto_label_column(df)
        if not label_col or label_col not in df.columns:
            print(f"SKIP {spec['label']}: no label column found "
                  f"(searched: {LABEL_COLUMN_CANDIDATES})")
            continue
        y_true = _normalize_labels(df[label_col])
        feat_df = df.drop(columns=[label_col])
        if feat_df.empty or feat_df.shape[0] == 0:
            print(f"SKIP {spec['label']}: no feature columns after dropping label.")
            continue
        print(f"[real] {spec['label']} rows={len(df)} positives={int(y_true.sum())} label_col={label_col}")
        real_rows.extend(_evaluate(
            dataset_label=str(spec["label"]),
            df=feat_df,
            y_true=y_true,
            tag="real",
            quiet=not args.verbose,
        ))

        if args.inject:
            corrupted, y_synth = inject(feat_df, args.inject, random_seed=int(args.seed))
            y_combined = np.maximum(y_true, y_synth)
            print(f"[real+synthetic={args.inject}] {spec['label']} extra_positives={int(y_synth.sum())}")
            synth_rows.extend(_evaluate(
                dataset_label=str(spec["label"]),
                df=corrupted,
                y_true=y_combined,
                tag=f"real+{args.inject}",
                quiet=not args.verbose,
            ))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    _write_rows(real_rows, out_path)
    print(f"Wrote {len(real_rows)} rows -> {out_path}")

    if args.inject:
        consistency = _consistency_rows(real_rows, synth_rows)
        consistency_path = Path(args.consistency_out)
        if not consistency_path.is_absolute():
            consistency_path = ROOT / consistency_path
        _write_rows(consistency, consistency_path)
        ranking = _ranking_agreement(real_rows, synth_rows)
        print(f"Wrote {len(consistency)} rows -> {consistency_path}")
        for ds, info in ranking.items():
            print(
                f"  {ds}: real_top={info['real_ranking'][:1]} "
                f"synth_top={info['synthetic_ranking'][:1]} "
                f"top_match={info['top_match']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

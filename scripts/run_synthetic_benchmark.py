"""
Reproducible synthetic-anomaly benchmark across datasets, seeds, scenarios, and grids.

Two modes:

* Legacy single-run CLI (kept for backward compatibility): ``--dataset PATH --scenario X``.
* YAML-driven study (preferred for the report): ``--config configs/experiments/quick.yaml``.

Outputs (paths configurable via the YAML ``output:`` block):

* ``per_run`` CSV: one row per (dataset × scenario × seed × noise × grid × score_source).
* ``aggregated`` CSV: groupby (dataset, scenario, params_signature, noise_std, score_source)
  reporting ``f1_mean/std``, ``roc_auc_mean/std``, ``pr_auc_mean/std``, ``best_f1_mean``.
* ``worst_case`` CSV: per ``score_source`` the (dataset, scenario, params, noise_std)
  combination with the lowest ``f1_mean`` — the natural "robustness floor" per model.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from advanced_system import AdvancedAnomalySystem  # noqa: E402
from synthetic_injection import (  # noqa: E402
    add_feature_noise,
    binary_classification_metrics,
    binary_score_metrics,
    inject,
    list_scenarios,
    merged_params,
)


DEFAULT_SCENARIOS = [
    "spike_single",
    "joint_shift",
    "scale_burst",
    "temporal_block",
    "dead_sensor",
    "sign_flip",
]


# -----------------------------
# Data classes
# -----------------------------
@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path
    drop_columns: Tuple[str, ...] = ()


@dataclass
class RunResult:
    dataset_label: str
    scenario: str
    score_source: str
    seed: int
    noise_std: float
    rows: int
    injected_count: int
    detected_count: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    best_percentile: float
    best_precision: float
    best_recall: float
    best_f1: float
    threshold: float
    models_used: Tuple[str, ...]
    params_signature: str
    params_effective: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------
def _round(x: float, ndigits: int = 6) -> float:
    return round(float(x), ndigits)


def _labels_at_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float) > float(threshold)).astype(int)


def _best_percentile(scores: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    best = {"percentile": 95.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for percentile in range(50, 100):
        threshold = float(np.percentile(scores, percentile))
        pred = _labels_at_threshold(scores, threshold)
        metrics = binary_classification_metrics(y_true, pred)
        if metrics["f1"] > best["f1"]:
            best = {
                "percentile": float(percentile),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            }
    return best


def _params_signature(scenario: str, params: Dict[str, Any], noise_std: float) -> str:
    """Stable hashable string used to group runs that vary only by seed."""
    payload = {
        "scenario": scenario,
        "noise_std": _round(float(noise_std)),
        "params": {k: params.get(k) for k in sorted(params)},
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _scenario_grid_combos(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product over grid; empty grid yields a single empty combo."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _evaluate_score_source(
    *,
    dataset_label: str,
    scenario: str,
    seed: int,
    noise_std: float,
    rows: int,
    source_name: str,
    scores: np.ndarray,
    default_threshold: float,
    y_true: np.ndarray,
    models_used: Tuple[str, ...],
    params_signature: str,
    params_effective: Dict[str, Any],
) -> RunResult:
    default_pred = _labels_at_threshold(scores, default_threshold)
    default_metrics = {
        **binary_classification_metrics(y_true, default_pred),
        **binary_score_metrics(y_true, scores),
    }
    best = _best_percentile(scores, y_true)
    return RunResult(
        dataset_label=dataset_label,
        scenario=scenario,
        score_source=source_name,
        seed=int(seed),
        noise_std=float(noise_std),
        rows=int(rows),
        injected_count=int(np.sum(y_true)),
        detected_count=int(np.sum(default_pred)),
        precision=_round(default_metrics["precision"]),
        recall=_round(default_metrics["recall"]),
        f1=_round(default_metrics["f1"]),
        roc_auc=_round(default_metrics["roc_auc"]),
        pr_auc=_round(default_metrics["pr_auc"]),
        best_percentile=_round(best["percentile"]),
        best_precision=_round(best["precision"]),
        best_recall=_round(best["recall"]),
        best_f1=_round(best["f1"]),
        threshold=_round(default_threshold),
        models_used=tuple(models_used),
        params_signature=params_signature,
        params_effective=params_effective,
    )


# -----------------------------
# One unit of work: dataset × scenario × seed × noise × grid combo
# -----------------------------
def run_unit(
    *,
    dataset_label: str,
    df: pd.DataFrame,
    scenario: str,
    seed: int,
    noise_std: float,
    overrides: Dict[str, Any],
    quiet: bool = True,
) -> List[RunResult]:
    cfg_full = merged_params(scenario, overrides)
    base_df = add_feature_noise(df, noise_std=float(noise_std), random_seed=int(seed))
    corrupted, y_true = inject(base_df, scenario, random_seed=int(seed), params=overrides)

    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        anomalies, scores, details = AdvancedAnomalySystem().run(corrupted)

    models_used = tuple(details["models"].keys())
    sig = _params_signature(scenario, cfg_full, float(noise_std))
    params_effective = {k: v for k, v in cfg_full.items() if v is not None}

    results: List[RunResult] = [
        _evaluate_score_source(
            dataset_label=dataset_label,
            scenario=scenario,
            seed=int(seed),
            noise_std=float(noise_std),
            rows=int(len(df)),
            source_name="ensemble",
            scores=np.asarray(scores, dtype=float),
            default_threshold=float(details["threshold"]),
            y_true=y_true,
            models_used=models_used,
            params_signature=sig,
            params_effective=params_effective,
        )
    ]
    for model_name, model_scores in details.get("normalized_model_scores", {}).items():
        arr = np.asarray(model_scores, dtype=float)
        results.append(
            _evaluate_score_source(
                dataset_label=dataset_label,
                scenario=scenario,
                seed=int(seed),
                noise_std=float(noise_std),
                rows=int(len(df)),
                source_name=str(model_name),
                scores=arr,
                default_threshold=float(np.percentile(arr, 95)),
                y_true=y_true,
                models_used=models_used,
                params_signature=sig,
                params_effective=params_effective,
            )
        )
    return results


# -----------------------------
# Config-driven iteration
# -----------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required for --config. Install it with `pip install pyyaml`."
        ) from e
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_dataset_specs(raw: Iterable[Any]) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    for item in raw:
        if isinstance(item, str):
            p = Path(item)
            specs.append(DatasetSpec(label=p.stem, path=(ROOT / p) if not p.is_absolute() else p))
            continue
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError(f"Invalid dataset entry: {item!r}")
        path = Path(str(item["path"]))
        full = (ROOT / path) if not path.is_absolute() else path
        label = str(item.get("label") or full.stem)
        drop = tuple(str(c) for c in item.get("drop_columns") or [])
        specs.append(DatasetSpec(label=label, path=full, drop_columns=drop))
    return specs


def _load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(
            f"Dataset {spec.label!r} not found at {spec.path}. "
            "Run `python scripts/fetch_public_datasets.py --dataset adb_annthyroid` (etc.) first."
        )
    df = pd.read_csv(spec.path)
    drop = [c for c in spec.drop_columns if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    return df


def iter_units(cfg: Dict[str, Any]) -> Iterator[Tuple[DatasetSpec, pd.DataFrame, str, int, float, Dict[str, Any]]]:
    datasets = _parse_dataset_specs(cfg.get("datasets") or [])
    seeds = [int(s) for s in (cfg.get("seeds") or [42])]
    noise_levels = [float(n) for n in (cfg.get("noise_std") or [0.0])]
    scenarios = cfg.get("scenarios") or []

    cache: Dict[str, pd.DataFrame] = {}
    for spec in datasets:
        if spec.label not in cache:
            cache[spec.label] = _load_dataset(spec)

    for spec in datasets:
        df = cache[spec.label]
        for entry in scenarios:
            scenario_id = entry["id"] if isinstance(entry, dict) else str(entry)
            if scenario_id not in list_scenarios():
                raise ValueError(
                    f"Unknown scenario {scenario_id!r}; valid: {list_scenarios()}"
                )
            grid = entry.get("grid") if isinstance(entry, dict) else None
            combos = _scenario_grid_combos(grid or {})
            for combo in combos:
                for noise_std in noise_levels:
                    for seed in seeds:
                        yield spec, df, scenario_id, int(seed), float(noise_std), dict(combo)


# -----------------------------
# Aggregation + worst-case
# -----------------------------
def _aggregate(rows: List[RunResult]) -> List[Dict[str, Any]]:
    keyed: Dict[Tuple[str, str, str, float, str], List[RunResult]] = {}
    for r in rows:
        key = (r.dataset_label, r.scenario, r.score_source, _round(r.noise_std), r.params_signature)
        keyed.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for (dataset_label, scenario, source, noise_std, sig), bucket in keyed.items():
        f1s = [b.f1 for b in bucket]
        roc = [b.roc_auc for b in bucket]
        pr = [b.pr_auc for b in bucket]
        best = [b.best_f1 for b in bucket]
        pcts = [b.best_percentile for b in bucket]
        reference = bucket[0]
        out.append(
            {
                "dataset": dataset_label,
                "scenario": scenario,
                "score_source": source,
                "noise_std": _round(noise_std),
                "n_runs": len(bucket),
                "f1_mean": _round(mean(f1s)),
                "f1_std": _round(pstdev(f1s) if len(f1s) > 1 else 0.0),
                "roc_auc_mean": _round(mean(roc)),
                "roc_auc_std": _round(pstdev(roc) if len(roc) > 1 else 0.0),
                "pr_auc_mean": _round(mean(pr)),
                "pr_auc_std": _round(pstdev(pr) if len(pr) > 1 else 0.0),
                "best_f1_mean": _round(mean(best)),
                "best_f1_std": _round(pstdev(best) if len(best) > 1 else 0.0),
                "best_percentile_median": _round(float(np.median(pcts))),
                "rows": reference.rows,
                "params_effective": json.dumps(reference.params_effective, sort_keys=True, default=str),
                "models_used": ",".join(reference.models_used),
                "params_signature": sig,
            }
        )
    return out


def _worst_case_per_model(aggregated: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_source: Dict[str, Dict[str, Any]] = {}
    for row in aggregated:
        src = row["score_source"]
        prev = by_source.get(src)
        if prev is None or row["f1_mean"] < prev["f1_mean"]:
            by_source[src] = row
    out: List[Dict[str, Any]] = []
    for src, row in sorted(by_source.items(), key=lambda kv: kv[1]["f1_mean"]):
        out.append(
            {
                "score_source": src,
                "worst_f1_mean": row["f1_mean"],
                "worst_f1_std": row["f1_std"],
                "worst_dataset": row["dataset"],
                "worst_scenario": row["scenario"],
                "worst_noise_std": row["noise_std"],
                "worst_params_effective": row["params_effective"],
                "n_runs": row["n_runs"],
            }
        )
    return out


# -----------------------------
# CSV writers
# -----------------------------
def _write_per_run(rows: List[RunResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "scenario",
        "score_source",
        "seed",
        "noise_std",
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
        "threshold",
        "models_used",
        "params_effective",
        "params_signature",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "dataset": r.dataset_label,
                    "scenario": r.scenario,
                    "score_source": r.score_source,
                    "seed": r.seed,
                    "noise_std": _round(r.noise_std),
                    "rows": r.rows,
                    "injected_count": r.injected_count,
                    "detected_count": r.detected_count,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "roc_auc": r.roc_auc,
                    "pr_auc": r.pr_auc,
                    "best_percentile": r.best_percentile,
                    "best_precision": r.best_precision,
                    "best_recall": r.best_recall,
                    "best_f1": r.best_f1,
                    "threshold": r.threshold,
                    "models_used": ",".join(r.models_used),
                    "params_effective": json.dumps(r.params_effective, sort_keys=True, default=str),
                    "params_signature": r.params_signature,
                }
            )


def _write_dict_rows(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Legacy single-run summary CSV (kept for backward compatibility)
# -----------------------------
def _write_legacy_summary(rows: List[RunResult], out_path: Path) -> None:
    """Same column layout the previous version emitted, for users who consumed it."""
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
        "threshold",
        "models_used",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "scenario": r.scenario,
                    "score_source": r.score_source,
                    "seed": r.seed,
                    "rows": r.rows,
                    "injected_count": r.injected_count,
                    "detected_count": r.detected_count,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "roc_auc": r.roc_auc,
                    "pr_auc": r.pr_auc,
                    "best_percentile": r.best_percentile,
                    "best_precision": r.best_precision,
                    "best_recall": r.best_recall,
                    "best_f1": r.best_f1,
                    "threshold": r.threshold,
                    "models_used": ",".join(r.models_used),
                }
            )


# -----------------------------
# Orchestration
# -----------------------------
def run_from_config(cfg: Dict[str, Any], *, quiet: bool = True, fail_fast: bool = False) -> Dict[str, Any]:
    output_cfg = cfg.get("output") or {}
    per_run_path = ROOT / Path(output_cfg.get("per_run") or "results/synthetic_benchmark_runs.csv")
    aggregated_path = ROOT / Path(output_cfg.get("aggregated") or "results/synthetic_benchmark_aggregated.csv")
    worst_path = ROOT / Path(output_cfg.get("worst_case") or "results/synthetic_benchmark_worst_case.csv")

    units = list(iter_units(cfg))
    total = len(units)
    print(f"Planned runs: {total}")
    if total == 0:
        return {"per_run": per_run_path, "aggregated": aggregated_path, "worst_case": worst_path, "rows": []}

    all_rows: List[RunResult] = []
    failures: List[Dict[str, Any]] = []
    for idx, (spec, df, scenario, seed, noise_std, overrides) in enumerate(units, start=1):
        tag = f"[{idx}/{total}] {spec.label} | {scenario} | seed={seed} | noise={noise_std} | {overrides}"
        start = time.perf_counter()
        try:
            unit_rows = run_unit(
                dataset_label=spec.label,
                df=df,
                scenario=scenario,
                seed=seed,
                noise_std=noise_std,
                overrides=overrides,
                quiet=quiet,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"tag": tag, "error": repr(exc)})
            print(f"{tag} FAILED: {exc}")
            if fail_fast:
                raise
            continue
        elapsed = time.perf_counter() - start
        ensemble = next((r for r in unit_rows if r.score_source == "ensemble"), unit_rows[0])
        print(
            f"{tag} | ensemble f1={ensemble.f1:.3f} roc_auc={ensemble.roc_auc:.3f} "
            f"best_f1={ensemble.best_f1:.3f} ({elapsed:.1f}s)"
        )
        all_rows.extend(unit_rows)

    aggregated = _aggregate(all_rows)
    worst = _worst_case_per_model(aggregated)
    _write_per_run(all_rows, per_run_path)
    _write_dict_rows(aggregated, aggregated_path)
    _write_dict_rows(worst, worst_path)
    print()
    print(f"Per-run rows : {len(all_rows)} -> {per_run_path}")
    print(f"Aggregated   : {len(aggregated)} -> {aggregated_path}")
    print(f"Worst-case   : {len(worst)} -> {worst_path}")
    if failures:
        print(f"Failures     : {len(failures)}")
    return {
        "per_run": per_run_path,
        "aggregated": aggregated_path,
        "worst_case": worst_path,
        "rows": all_rows,
        "failures": failures,
    }


def run_legacy_single(args: argparse.Namespace) -> int:
    """Backward-compatible single-dataset, fixed-seed mode for older docs/CI."""
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    df = pd.read_csv(dataset)
    scenarios = args.scenario or DEFAULT_SCENARIOS
    rows: List[RunResult] = []
    for scenario in scenarios:
        rows.extend(
            run_unit(
                dataset_label=dataset.stem,
                df=df,
                scenario=scenario,
                seed=int(args.seed),
                noise_std=0.0,
                overrides={},
                quiet=True,
            )
        )
    out_path = Path(args.out) if args.out else (ROOT / "results" / "synthetic_benchmark_summary.csv")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    _write_legacy_summary(rows, out_path)
    print(f"Dataset: {dataset}")
    print(f"Rows: {len(df)}")
    print(f"Summary CSV: {out_path}")
    print()
    last_scenario: Optional[str] = None
    for r in rows:
        if r.scenario != last_scenario:
            print(r.scenario)
            last_scenario = r.scenario
        print(
            f"  {r.score_source}: f1@95={r.f1:.3f} roc_auc={r.roc_auc:.3f} "
            f"pr_auc={r.pr_auc:.3f} best_p={r.best_percentile:.0f} best_f1={r.best_f1:.3f}"
        )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inject synthetic anomalies into one or more CSVs and benchmark detector robustness.",
    )
    parser.add_argument("--config", type=str, help="YAML experiment config (multi-dataset, multi-seed).")
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "data" / "synthetic_examples" / "baseline_server_metrics.csv"),
        help="Legacy single-dataset CSV (used only when --config is omitted).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=list_scenarios(),
        help="Legacy: scenario id (repeatable). Defaults to common numeric scenarios.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Legacy single seed (default 42).")
    parser.add_argument(
        "--out",
        default=None,
        help="Legacy summary CSV path. Defaults to results/synthetic_benchmark_summary.csv.",
    )
    parser.add_argument("--verbose", action="store_true", help="Do not silence detector stdout.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on the first run error.")
    args = parser.parse_args(argv)

    logging.getLogger("optuna").setLevel(logging.WARNING)

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = ROOT / cfg_path
        cfg = _load_yaml(cfg_path)
        run_from_config(cfg, quiet=not args.verbose, fail_fast=args.fail_fast)
        return 0

    return run_legacy_single(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""
Deterministic synthetic anomaly injection for controlled evaluation.

Scenarios are documented in SCENARIO_DEFAULTS. Numeric scenarios perturb numeric columns
only; ``categorical_flip`` targets non-numeric (string/object) columns; ``missing_value``
operates on any dtype by setting cells to NaN.

Human-oriented explanation, analogies, and sample CSVs: docs/SYNTHETIC_SCENARIOS.md
HTTP preview (no full model run): POST /synthetic-preview in api/main.py
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Catalog: scenario id -> default parameters (override via inject(..., params=...)).
SCENARIO_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "spike_single": {
        "contamination": 0.1,
        "column": None,  # first numeric column if None
        "magnitude_in_std": 5.0,
    },
    "joint_shift": {
        "contamination": 0.1,
        "columns": None,  # all numeric columns if None
        "magnitude_in_std": 4.0,
    },
    "scale_burst": {
        "contamination": 0.08,
        "columns": None,  # all numeric if None
        "scale_factor": 3.0,
    },
    "dead_sensor": {
        "contamination": 0.08,
        "columns": None,  # all numeric if None
        "constant": None,  # explicit constant override
        "mode": "median",  # median, zero, previous, random_constant
    },
    "sign_flip": {
        "contamination": 0.05,
        "columns": None,  # all numeric if None
    },
    "temporal_block": {
        "contamination": 0.1,
        "columns": None,  # all numeric if None
        "magnitude_in_std": 4.0,
        "block_count": 1,
    },
    "categorical_flip": {
        "contamination": 0.1,
        "column": None,  # first non-numeric column if None
        "mode": "swap",  # "swap" -> existing other category; "sentinel" -> sentinel value
        "sentinel": "__UNKNOWN__",
    },
    "missing_value": {
        "contamination": 0.08,
        "columns": None,  # all columns if None
    },
}


def list_scenarios() -> List[str]:
    return list(SCENARIO_DEFAULTS.keys())


def merged_params(scenario: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Defaults for ``scenario`` merged with ``overrides`` (caller filters unknown keys)."""
    if scenario not in SCENARIO_DEFAULTS:
        raise KeyError(f"Unknown scenario {scenario!r}; known: {list_scenarios()}")
    return {**SCENARIO_DEFAULTS[scenario], **(overrides or {})}


def _numeric_column_names(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _non_numeric_column_names(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def _injected_row_count(n: int, contamination: float) -> int:
    if n < 2 or contamination <= 0:
        return 0
    k = int(math.ceil(n * float(contamination)))
    k = max(1, k)
    return min(k, n - 1)


def _pick_random_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n, size=k, replace=False)


def _pick_block_indices(n: int, k: int, block_count: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``k`` indices spread across ``block_count`` non-overlapping contiguous blocks."""
    block_count = max(1, int(block_count))
    block_count = min(block_count, k)  # at least 1 row per block
    base = k // block_count
    extra = k % block_count
    sizes = [base + (1 if i < extra else 0) for i in range(block_count)]
    # Place blocks left-to-right with random gaps; total span <= n
    total = sum(sizes)
    free = n - total
    # Distribute free space into block_count + 1 gaps
    if free <= 0:
        starts = []
        cursor = 0
        for s in sizes:
            starts.append(cursor)
            cursor += s
    else:
        cuts = sorted(rng.integers(0, free + 1, size=block_count).tolist())
        starts = []
        cursor = 0
        for i, s in enumerate(sizes):
            cursor += cuts[i] - (cuts[i - 1] if i > 0 else 0)
            starts.append(cursor)
            cursor += s
    out: List[int] = []
    for start, s in zip(starts, sizes):
        out.extend(range(start, start + s))
    return np.array(out, dtype=int)


def _resolve_columns(df: pd.DataFrame, requested: Optional[List[str]], numeric_only: bool) -> List[str]:
    pool = _numeric_column_names(df) if numeric_only else list(df.columns)
    cols = requested or pool
    return [c for c in cols if c in df.columns and (not numeric_only or pd.api.types.is_numeric_dtype(df[c]))]


def _promote_int_to_float(df: pd.DataFrame, cols: List[str]) -> None:
    """Promote integer-typed columns in place so float assignments do not raise (pandas >=2.2)."""
    for col in cols:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)


def inject(
    df: pd.DataFrame,
    scenario: str,
    *,
    random_seed: int = 42,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Return a copy of df with synthetic anomalies injected and binary y_true (1 = injected row).

    Does not mutate the input DataFrame.
    """
    cfg = merged_params(scenario, params)
    rng = np.random.default_rng(int(random_seed))
    out = df.copy()
    n = len(out)
    y_true = np.zeros(n, dtype=np.int8)

    k = _injected_row_count(n, float(cfg["contamination"]))
    if k == 0:
        return out, y_true

    if scenario == "temporal_block":
        indices = _pick_block_indices(n, k, int(cfg.get("block_count", 1) or 1), rng)
    else:
        indices = _pick_random_indices(n, k, rng)
    y_true[indices] = 1

    if scenario == "spike_single":
        numeric_cols = _numeric_column_names(out)
        if not numeric_cols:
            raise ValueError("spike_single requires at least one numeric column.")
        col = cfg["column"] or numeric_cols[0]
        if col not in out.columns:
            raise KeyError(f"Column {col!r} not in DataFrame.")
        if not pd.api.types.is_numeric_dtype(out[col]):
            raise ValueError(f"Column {col!r} is not numeric.")
        mag = float(cfg["magnitude_in_std"])
        std = float(out[col].std(ddof=0)) or 1.0
        _promote_int_to_float(out, [col])
        out.loc[indices, col] = out.loc[indices, col].astype(float) + mag * std

    elif scenario in ("joint_shift", "temporal_block"):
        cols = _resolve_columns(out, cfg.get("columns"), numeric_only=True)
        if not cols:
            raise ValueError(f"{scenario} requires numeric columns.")
        mag = float(cfg["magnitude_in_std"])
        _promote_int_to_float(out, cols)
        for col in cols:
            std = float(out[col].std(ddof=0)) or 1.0
            out.loc[indices, col] = out.loc[indices, col].astype(float) + mag * std

    elif scenario == "scale_burst":
        cols = _resolve_columns(out, cfg.get("columns"), numeric_only=True)
        if not cols:
            raise ValueError("scale_burst requires numeric columns.")
        factor = float(cfg["scale_factor"])
        _promote_int_to_float(out, cols)
        out.loc[indices, cols] = out.loc[indices, cols].astype(float) * factor

    elif scenario == "dead_sensor":
        cols = _resolve_columns(out, cfg.get("columns"), numeric_only=True)
        if not cols:
            raise ValueError("dead_sensor requires numeric columns.")
        constant = cfg.get("constant")
        mode = str(cfg.get("mode") or "median").lower()
        # Promote when the constant or any median is non-integral, to avoid int-cast surprises.
        _promote_int_to_float(out, cols)
        for col in cols:
            if constant is not None:
                value = float(constant)
                out.loc[indices, col] = value
            elif mode == "median":
                out.loc[indices, col] = float(out[col].median())
            elif mode == "zero":
                out.loc[indices, col] = 0.0
            elif mode == "previous":
                for idx in indices:
                    prev_idx = max(0, int(idx) - 1)
                    out.at[out.index[idx], col] = float(out.iloc[prev_idx][col])
            elif mode == "random_constant":
                values = out[col].dropna().astype(float)
                if values.empty:
                    value = 0.0
                else:
                    low = float(values.quantile(0.10))
                    high = float(values.quantile(0.90))
                    value = float(rng.uniform(low, high)) if high > low else float(values.median())
                out.loc[indices, col] = value
            else:
                raise ValueError(
                    f"Unknown dead_sensor mode {mode!r}; use median, zero, previous, or random_constant."
                )

    elif scenario == "sign_flip":
        cols = _resolve_columns(out, cfg.get("columns"), numeric_only=True)
        if not cols:
            raise ValueError("sign_flip requires numeric columns.")
        # -1 * int is still int, but keep dtype consistent for downstream consumers.
        out.loc[indices, cols] = out.loc[indices, cols] * -1

    elif scenario == "categorical_flip":
        non_num = _non_numeric_column_names(out)
        if not non_num:
            raise ValueError("categorical_flip requires at least one non-numeric column.")
        col = cfg.get("column") or non_num[0]
        if col not in out.columns:
            raise KeyError(f"Column {col!r} not in DataFrame.")
        if pd.api.types.is_numeric_dtype(out[col]):
            raise ValueError(f"categorical_flip target {col!r} must be non-numeric.")
        mode = str(cfg.get("mode") or "swap").lower()
        if mode == "sentinel":
            sentinel = str(cfg.get("sentinel") or "__UNKNOWN__")
            # Coerce dtype to object so sentinel string is accepted on categorical columns too
            out[col] = out[col].astype(object)
            out.loc[indices, col] = sentinel
        elif mode == "swap":
            categories = pd.Series(out[col]).dropna().astype(object).unique().tolist()
            if len(categories) < 2:
                raise ValueError(
                    f"categorical_flip mode='swap' needs >=2 distinct categories in {col!r}."
                )
            out[col] = out[col].astype(object)
            for idx in indices:
                current = out.at[idx, col]
                alternatives = [v for v in categories if v != current]
                if not alternatives:
                    alternatives = categories
                out.at[idx, col] = alternatives[int(rng.integers(0, len(alternatives)))]
        else:
            raise ValueError(f"Unknown categorical_flip mode {mode!r}; use 'swap' or 'sentinel'.")

    elif scenario == "missing_value":
        requested = cfg.get("columns")
        cols = [c for c in (requested or list(out.columns)) if c in out.columns]
        if not cols:
            raise ValueError("missing_value requires at least one valid column.")
        for col in cols:
            if pd.api.types.is_integer_dtype(out[col]):
                # NaN cannot live in an integer column; promote to float
                out[col] = out[col].astype(float)
            elif not pd.api.types.is_float_dtype(out[col]) and not pd.api.types.is_object_dtype(out[col]):
                out[col] = out[col].astype(object)
            out.loc[indices, col] = np.nan

    else:
        raise AssertionError(f"Unhandled scenario {scenario}")

    return out, y_true


def add_feature_noise(
    df: pd.DataFrame,
    *,
    noise_std: float = 0.1,
    columns: Optional[List[str]] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Return a copy of ``df`` with independent Gaussian noise added to each numeric column.

    Noise scale is ``noise_std * column_std(ddof=0)`` per column (zero-std columns get
    plain ``N(0, noise_std)``). This is a **dataset-wide** background perturbation
    composed with ``inject()`` for robustness sweeps; it does **not** produce ``y_true``
    because every row is touched. Pair it with ``inject(...)`` so that injected rows
    remain the ground-truth positives and noise stays additive contamination.
    """
    if noise_std <= 0:
        return df.copy()
    rng = np.random.default_rng(int(random_seed))
    out = df.copy()
    pool = _numeric_column_names(out)
    cols = [c for c in (columns or pool) if c in pool]
    if not cols:
        return out
    _promote_int_to_float(out, cols)
    for col in cols:
        std = float(out[col].std(ddof=0)) or 1.0
        eps = rng.normal(loc=0.0, scale=float(noise_std) * std, size=len(out))
        out[col] = out[col].astype(float) + eps
    return out


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Precision, recall, F1 for binary labels; zero_division safe."""
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    return {
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
    }


def binary_score_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Threshold-independent ranking metrics for continuous anomaly scores.

    Returns ROC-AUC and PR-AUC (average precision). Falls back to NaN-safe 0.0 when
    only one class is present in ``y_true`` or when scores are constant.
    """
    yt = np.asarray(y_true).astype(int).ravel()
    sc = np.asarray(scores).astype(float).ravel()
    out = {"roc_auc": 0.0, "pr_auc": 0.0}
    if yt.size == 0 or sc.size == 0 or yt.size != sc.size:
        return out
    if len(np.unique(yt)) < 2:
        return out
    try:
        out["roc_auc"] = float(roc_auc_score(yt, sc))
    except ValueError:
        out["roc_auc"] = 0.0
    try:
        out["pr_auc"] = float(average_precision_score(yt, sc))
    except ValueError:
        out["pr_auc"] = 0.0
    return out

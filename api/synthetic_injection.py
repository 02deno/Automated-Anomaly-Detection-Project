"""
Deterministic synthetic anomaly injection for controlled evaluation.

Scenarios are documented in SCENARIO_DEFAULTS; only numeric columns are perturbed,
matching AdvancedAnomalySystem preprocessing (numeric-only).

Human-oriented explanation, analogies, and sample CSVs: docs/SYNTHETIC_SCENARIOS.md
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

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


def _injected_row_count(n: int, contamination: float) -> int:
    if n < 2 or contamination <= 0:
        return 0
    k = int(math.ceil(n * float(contamination)))
    k = max(1, k)
    return min(k, n - 1)


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

    numeric_cols = _numeric_column_names(out)
    if not numeric_cols:
        raise ValueError("DataFrame has no numeric columns to inject into.")

    k = _injected_row_count(n, float(cfg["contamination"]))
    if k == 0:
        return out, y_true

    indices = rng.choice(n, size=k, replace=False)
    y_true[indices] = 1

    if scenario == "spike_single":
        col = cfg["column"] or numeric_cols[0]
        if col not in out.columns:
            raise KeyError(f"Column {col!r} not in DataFrame.")
        mag = float(cfg["magnitude_in_std"])
        std = float(out[col].std(ddof=0)) or 1.0
        out.loc[indices, col] = out.loc[indices, col].astype(float) + mag * std

    elif scenario == "joint_shift":
        cols: List[str] = cfg["columns"] or numeric_cols
        cols = [c for c in cols if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
        if not cols:
            raise ValueError("No valid numeric columns for joint_shift.")
        mag = float(cfg["magnitude_in_std"])
        for col in cols:
            std = float(out[col].std(ddof=0)) or 1.0
            out.loc[indices, col] = out.loc[indices, col].astype(float) + mag * std

    elif scenario == "scale_burst":
        cols = cfg["columns"] or numeric_cols
        cols = [c for c in cols if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
        if not cols:
            raise ValueError("No valid numeric columns for scale_burst.")
        factor = float(cfg["scale_factor"])
        out.loc[indices, cols] = out.loc[indices, cols].astype(float) * factor

    else:
        raise AssertionError(f"Unhandled scenario {scenario}")

    return out, y_true


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Precision, recall, F1 for binary labels; zero_division safe."""
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    return {
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
    }

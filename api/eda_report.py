"""
Lightweight EDA payload for the dashboard: dtypes, missingness, numeric summaries,
boxplot statistics (Tukey fences), scatter sample for the strongest linear pair among numerics,
histograms (bounded), and a correlation matrix slice — suitable for JSON + Chart.js.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EDA_MAX_ROWS = 50_000
MAX_HISTOGRAM_COLUMNS = 6
MAX_BOXPLOT_COLUMNS = 6
MAX_CORR_COLUMNS = 14
MAX_SCATTER_POINTS = 2800
HIST_BINS = 22


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _histogram_series(s: pd.Series, bins: int = HIST_BINS) -> Tuple[List[float], List[int]]:
    clean = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if clean.size == 0:
        return [], []
    arr = clean.to_numpy()
    counts, edges = np.histogram(arr, bins=bins)
    # bin left edges + last right edge for Chart.js category alignment
    edge_list = [float(edges[i]) for i in range(len(edges))]
    return edge_list, [int(c) for c in counts.tolist()]


def build_eda_report(df: pd.DataFrame) -> Dict[str, Any]:
    n_raw = len(df)
    sampled = n_raw > EDA_MAX_ROWS
    work = df.iloc[:EDA_MAX_ROWS].copy() if sampled else df.copy()

    warnings: List[str] = []
    if sampled:
        warnings.append(f"EDA uses the first {EDA_MAX_ROWS:,} rows only (file has {n_raw:,} rows).")

    columns_meta: List[Dict[str, Any]] = []
    for col in work.columns:
        ser = work[col]
        missing = int(ser.isna().sum())
        miss_pct = float(missing / len(work)) if len(work) else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(ser)
        nunique = int(ser.nunique(dropna=True))
        columns_meta.append(
            {
                "name": col,
                "dtype": str(ser.dtype),
                "missing_count": missing,
                "missing_pct": round(miss_pct * 100, 3),
                "is_numeric": bool(is_numeric),
                "nunique": nunique,
            }
        )
        if miss_pct > 0.4:
            warnings.append(f"Column {col!r} has {miss_pct*100:.1f}% missing values.")

    numeric = work.select_dtypes(include=[np.number])
    numeric_cols = numeric.columns.tolist()
    if not numeric_cols:
        warnings.append("No numeric columns detected; correlation and distribution plots are skipped.")

    numeric_summary: Dict[str, Dict[str, Any]] = {}
    for col in numeric_cols:
        s = pd.to_numeric(work[col], errors="coerce").dropna()
        if s.size == 0:
            numeric_summary[col] = {"note": "all null or non-finite"}
            continue
        if float(s.std(ddof=0) or 0) < 1e-12:
            warnings.append(f"Numeric column {col!r} is nearly constant; little distributional signal.")
        numeric_summary[col] = {
            "count": int(s.size),
            "mean": _safe_float(s.mean()),
            "std": _safe_float(s.std(ddof=0)),
            "min": _safe_float(s.min()),
            "p25": _safe_float(s.quantile(0.25)),
            "median": _safe_float(s.median()),
            "p75": _safe_float(s.quantile(0.75)),
            "max": _safe_float(s.max()),
            "skew": _safe_float(s.skew()),
        }

    # Histograms: highest-variance numeric columns first (typical EDA priority)
    histograms: List[Dict[str, Any]] = []
    if numeric_cols:
        variances = numeric.var(ddof=0).replace(np.nan, 0.0).sort_values(ascending=False)
        hist_cols = variances.index[: min(MAX_HISTOGRAM_COLUMNS, len(variances))].tolist()
        for col in hist_cols:
            edges, counts = _histogram_series(work[col])
            if not edges:
                continue
            labels = [f"{edges[i]:.4g}" for i in range(len(counts))]
            histograms.append(
                {
                    "column": col,
                    "bin_edges": edges,
                    "labels": labels,
                    "counts": counts,
                }
            )

    correlation: Optional[Dict[str, Any]] = None
    scatter: Optional[Dict[str, Any]] = None
    if len(numeric_cols) >= 2:
        variances = numeric.var(ddof=0).replace(np.nan, 0.0).sort_values(ascending=False)
        corr_cols = variances.index[: min(MAX_CORR_COLUMNS, len(variances))].tolist()
        sub = numeric[corr_cols].dropna()
        if len(sub) >= 2:
            cmat = sub.corr()
            cmat = cmat.fillna(0.0)
            correlation = {
                "columns": corr_cols,
                "matrix": [[_safe_float(cmat.iloc[i, j]) or 0.0 for j in range(len(corr_cols))] for i in range(len(corr_cols))],
            }
            best_i, best_j, best_r = 0, 1, -1.0
            for i in range(len(corr_cols)):
                for j in range(len(corr_cols)):
                    if i == j:
                        continue
                    r = abs(float(cmat.iloc[i, j]))
                    if r > best_r:
                        best_r, best_i, best_j = r, i, j
            xc, yc = corr_cols[best_i], corr_cols[best_j]
            pair_df = numeric[[xc, yc]].dropna()
            if len(pair_df) > MAX_SCATTER_POINTS:
                pair_df = pair_df.sample(n=MAX_SCATTER_POINTS, random_state=0)
            scatter = {
                "x_column": xc,
                "y_column": yc,
                "pearson_r": _safe_float(cmat.iloc[best_i, best_j]) or 0.0,
                "x": pair_df[xc].astype(float).tolist(),
                "y": pair_df[yc].astype(float).tolist(),
            }

    boxplots: List[Dict[str, Any]] = []
    if numeric_cols:
        variances_bp = numeric.var(ddof=0).replace(np.nan, 0.0).sort_values(ascending=False)
        bp_cols = variances_bp.index[: min(MAX_BOXPLOT_COLUMNS, len(variances_bp))].tolist()
        for col in bp_cols:
            s = pd.to_numeric(work[col], errors="coerce").dropna().astype(float)
            if len(s) < 4:
                continue
            q1 = float(s.quantile(0.25))
            med = float(s.quantile(0.5))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1 or 1e-12
            lo_f = max(float(s.min()), q1 - 1.5 * iqr)
            hi_f = min(float(s.max()), q3 + 1.5 * iqr)
            boxplots.append(
                {
                    "column": col,
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "whisker_low": lo_f,
                    "q1": q1,
                    "median": med,
                    "q3": q3,
                    "whisker_high": hi_f,
                }
            )

    return {
        "row_count_raw": int(n_raw),
        "row_count_used": int(len(work)),
        "column_count": int(work.shape[1]),
        "sampled": sampled,
        "columns": columns_meta,
        "numeric_column_names": numeric_cols,
        "numeric_summary": numeric_summary,
        "histograms": histograms,
        "correlation": correlation,
        "scatter": scatter,
        "boxplots": boxplots,
        "warnings": warnings,
    }

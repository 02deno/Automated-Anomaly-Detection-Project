"""
Lightweight EDA payload for the dashboard: dtypes, missingness, numeric summaries,
boxplot statistics (Tukey fences), scatter sample for the strongest linear pair among numerics,
histograms (bounded), Pearson + Spearman correlation slices, top correlated pairs, categorical
top-k frequencies, datetime-like detection, and basic data quality metrics
(duplicate rows, per-column outlier counts, kurtosis, skew warnings) — JSON + Chart.js friendly.
"""

from __future__ import annotations

import warnings as _py_warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EDA_MAX_ROWS = 50_000
MAX_HISTOGRAM_COLUMNS = 6
MAX_BOXPLOT_COLUMNS = 6
MAX_CORR_COLUMNS = 14
MAX_SCATTER_POINTS = 2800
HIST_BINS = 22
MAX_CATEGORICAL_COLUMNS = 8
MAX_CATEGORICAL_TOP = 10
MAX_TOP_CORR = 10
DATETIME_PARSE_THRESHOLD = 0.8
DATETIME_SAMPLE_LIMIT = 5_000
RARE_CATEGORY_PCT = 0.01  # 1%
SKEW_WARN_THRESHOLD = 2.0


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
    edge_list = [float(edges[i]) for i in range(len(edges))]
    return edge_list, [int(c) for c in counts.tolist()]


def _tukey_outlier_count(s: pd.Series) -> int:
    if s.size < 4:
        return 0
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return 0
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum())


def _z_outlier_count(s: pd.Series) -> int:
    std = float(s.std(ddof=0))
    if std <= 0:
        return 0
    z = (s - float(s.mean())) / std
    return int((z.abs() > 3.0).sum())


def _build_numeric_summary(
    work: pd.DataFrame, numeric_cols: List[str], warnings: List[str]
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for col in numeric_cols:
        s = pd.to_numeric(work[col], errors="coerce").dropna()
        if s.size == 0:
            summary[col] = {"note": "all null or non-finite"}
            continue
        if float(s.std(ddof=0) or 0) < 1e-12:
            warnings.append(f"Numeric column {col!r} is nearly constant; little distributional signal.")
        skew_val = _safe_float(s.skew())
        if skew_val is not None and abs(skew_val) > SKEW_WARN_THRESHOLD:
            warnings.append(
                f"Numeric column {col!r} has skew {skew_val:.2f}; consider a log transform."
            )
        summary[col] = {
            "count": int(s.size),
            "mean": _safe_float(s.mean()),
            "std": _safe_float(s.std(ddof=0)),
            "min": _safe_float(s.min()),
            "p25": _safe_float(s.quantile(0.25)),
            "median": _safe_float(s.median()),
            "p75": _safe_float(s.quantile(0.75)),
            "max": _safe_float(s.max()),
            "skew": skew_val,
            "kurtosis": _safe_float(s.kurt()),
            "tukey_outlier_count": _tukey_outlier_count(s),
            "z_outlier_count": _z_outlier_count(s),
        }
    return summary


def _build_categorical_summary(
    work: pd.DataFrame, columns_meta: List[Dict[str, Any]], warnings: List[str]
) -> List[Dict[str, Any]]:
    """Top-k frequencies for non-numeric (object/string/category) columns."""
    candidates = [
        c["name"] for c in columns_meta
        if not c["is_numeric"] and c["nunique"] > 0
    ]
    candidates = candidates[:MAX_CATEGORICAL_COLUMNS]
    out: List[Dict[str, Any]] = []
    n = len(work)
    if n == 0:
        return out
    for col in candidates:
        s = work[col].dropna().astype(object)
        if s.size == 0:
            continue
        counts = s.value_counts()
        nunique = int(counts.size)
        top_items = counts.head(MAX_CATEGORICAL_TOP)
        rare_count = int((counts / max(1, n) < RARE_CATEGORY_PCT).sum())
        if rare_count > 0 and nunique > MAX_CATEGORICAL_TOP:
            warnings.append(
                f"Column {col!r} has {rare_count} rare categor{'y' if rare_count == 1 else 'ies'} "
                f"(<1% of rows each); consider grouping."
            )
        out.append({
            "column": col,
            "nunique": nunique,
            "top": [
                {"value": str(v), "count": int(c), "pct": round(float(c) / float(n) * 100.0, 3)}
                for v, c in top_items.items()
            ],
            "rare_count": rare_count,
        })
    return out


def _detect_datetime_columns(work: pd.DataFrame, columns_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Object-dtype columns where >=80% of values parse as datetimes."""
    out: List[Dict[str, Any]] = []
    for meta in columns_meta:
        if meta["is_numeric"]:
            continue
        col = meta["name"]
        if "datetime" in meta["dtype"].lower():
            ser = work[col].dropna()
            if ser.empty:
                continue
            out.append({
                "column": col,
                "parsed_pct": 100.0,
                "min": str(ser.min()),
                "max": str(ser.max()),
                "range_days": _safe_float((ser.max() - ser.min()).total_seconds() / 86400.0),
            })
            continue
        ser = work[col].dropna()
        if ser.empty:
            continue
        sample = ser.iloc[:DATETIME_SAMPLE_LIMIT].astype(str)
        with _py_warnings.catch_warnings():
            _py_warnings.simplefilter("ignore", category=UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        ratio = float(parsed.notna().sum()) / float(len(sample))
        if ratio < DATETIME_PARSE_THRESHOLD:
            continue
        valid = parsed.dropna()
        if valid.empty:
            continue
        out.append({
            "column": col,
            "parsed_pct": round(ratio * 100.0, 2),
            "min": str(valid.min()),
            "max": str(valid.max()),
            "range_days": _safe_float((valid.max() - valid.min()).total_seconds() / 86400.0),
        })
    return out


def _build_correlations(
    numeric: pd.DataFrame,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pearson + Spearman matrices, scatter for strongest |Pearson| pair, and top-N pair list."""
    if numeric.shape[1] < 2:
        return None, None, None, []
    variances = numeric.var(ddof=0).replace(np.nan, 0.0).sort_values(ascending=False)
    corr_cols = variances.index[: min(MAX_CORR_COLUMNS, len(variances))].tolist()
    sub = numeric[corr_cols].dropna()
    if len(sub) < 2:
        return None, None, None, []

    pearson_mat = sub.corr(method="pearson").fillna(0.0)
    spearman_mat = sub.corr(method="spearman").fillna(0.0)

    pearson = {
        "columns": corr_cols,
        "matrix": [[_safe_float(pearson_mat.iloc[i, j]) or 0.0 for j in range(len(corr_cols))] for i in range(len(corr_cols))],
    }
    spearman = {
        "columns": corr_cols,
        "matrix": [[_safe_float(spearman_mat.iloc[i, j]) or 0.0 for j in range(len(corr_cols))] for i in range(len(corr_cols))],
    }

    pairs: List[Dict[str, Any]] = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            pr = float(pearson_mat.iloc[i, j])
            sp = float(spearman_mat.iloc[i, j])
            pairs.append({
                "a": corr_cols[i],
                "b": corr_cols[j],
                "pearson": _safe_float(pr) or 0.0,
                "spearman": _safe_float(sp) or 0.0,
                "abs_pearson": abs(pr),
            })
    pairs.sort(key=lambda d: d["abs_pearson"], reverse=True)
    top_pairs = pairs[:MAX_TOP_CORR]

    scatter: Optional[Dict[str, Any]] = None
    best = max(pairs, key=lambda d: d["abs_pearson"], default=None)
    if best is not None:
        xc, yc = best["a"], best["b"]
        pair_df = numeric[[xc, yc]].dropna()
        if len(pair_df) > MAX_SCATTER_POINTS:
            pair_df = pair_df.sample(n=MAX_SCATTER_POINTS, random_state=0)
        scatter = {
            "x_column": xc,
            "y_column": yc,
            "pearson_r": best["pearson"],
            "x": pair_df[xc].astype(float).tolist(),
            "y": pair_df[yc].astype(float).tolist(),
        }
    return pearson, spearman, scatter, top_pairs


def _build_boxplots(work: pd.DataFrame, numeric_cols: List[str], numeric: pd.DataFrame) -> List[Dict[str, Any]]:
    if not numeric_cols:
        return []
    variances_bp = numeric.var(ddof=0).replace(np.nan, 0.0).sort_values(ascending=False)
    bp_cols = variances_bp.index[: min(MAX_BOXPLOT_COLUMNS, len(variances_bp))].tolist()
    boxplots: List[Dict[str, Any]] = []
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
        boxplots.append({
            "column": col,
            "min": float(s.min()),
            "max": float(s.max()),
            "whisker_low": lo_f,
            "q1": q1,
            "median": med,
            "q3": q3,
            "whisker_high": hi_f,
        })
    return boxplots


def _build_histograms(work: pd.DataFrame, numeric_cols: List[str], numeric: pd.DataFrame) -> List[Dict[str, Any]]:
    if not numeric_cols:
        return []
    variances = numeric.var(ddof=0).replace(np.nan, 0.0).sort_values(ascending=False)
    hist_cols = variances.index[: min(MAX_HISTOGRAM_COLUMNS, len(variances))].tolist()
    histograms: List[Dict[str, Any]] = []
    for col in hist_cols:
        edges, counts = _histogram_series(work[col])
        if not edges:
            continue
        labels = [f"{edges[i]:.4g}" for i in range(len(counts))]
        histograms.append({
            "column": col,
            "bin_edges": edges,
            "labels": labels,
            "counts": counts,
        })
    return histograms


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
        columns_meta.append({
            "name": col,
            "dtype": str(ser.dtype),
            "missing_count": missing,
            "missing_pct": round(miss_pct * 100, 3),
            "is_numeric": bool(is_numeric),
            "nunique": nunique,
        })
        if miss_pct > 0.4:
            warnings.append(f"Column {col!r} has {miss_pct*100:.1f}% missing values.")

    numeric = work.select_dtypes(include=[np.number])
    numeric_cols = numeric.columns.tolist()
    if not numeric_cols:
        warnings.append("No numeric columns detected; correlation and distribution plots are skipped.")

    duplicate_count = int(work.duplicated().sum())
    duplicate_pct = round(float(duplicate_count) / float(len(work)) * 100.0, 3) if len(work) else 0.0
    if duplicate_count > 0:
        warnings.append(f"{duplicate_count} duplicate row(s) detected ({duplicate_pct}%).")

    numeric_summary = _build_numeric_summary(work, numeric_cols, warnings)
    histograms = _build_histograms(work, numeric_cols, numeric)
    pearson, spearman, scatter, top_pairs = _build_correlations(numeric)
    boxplots = _build_boxplots(work, numeric_cols, numeric)
    categorical_summary = _build_categorical_summary(work, columns_meta, warnings)
    datetime_columns = _detect_datetime_columns(work, columns_meta)

    return {
        "row_count_raw": int(n_raw),
        "row_count_used": int(len(work)),
        "column_count": int(work.shape[1]),
        "sampled": sampled,
        "duplicate_row_count": duplicate_count,
        "duplicate_row_pct": duplicate_pct,
        "columns": columns_meta,
        "numeric_column_names": numeric_cols,
        "numeric_summary": numeric_summary,
        "histograms": histograms,
        "correlation": pearson,
        "correlation_spearman": spearman,
        "top_correlations": top_pairs,
        "scatter": scatter,
        "boxplots": boxplots,
        "categorical_summary": categorical_summary,
        "datetime_columns": datetime_columns,
        "warnings": warnings,
    }

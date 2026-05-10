from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse, Response
import io
import math
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_system import AdvancedAnomalySystem
from eda_report import build_eda_report
from synthetic_injection import binary_score_metrics, inject, list_scenarios, merged_params

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_UI_DIR = _REPO_ROOT / "ui"

system = AdvancedAnomalySystem()

LABEL_COLUMN_CANDIDATES = [
    "label",
    "target",
    "ground_truth",
    "y_true",
    "is_anomaly",
    "anomaly",
]


def _optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    t = str(value).strip()
    if not t:
        return None
    return float(t)


def _cells_differ(bv: Any, av: Any) -> bool:
    """NaN-aware equality across numeric and string cells."""
    b_na = bv is None or (isinstance(bv, float) and math.isnan(bv)) or (pd.isna(bv) if not isinstance(bv, str) else False)
    a_na = av is None or (isinstance(av, float) and math.isnan(av)) or (pd.isna(av) if not isinstance(av, str) else False)
    if b_na and a_na:
        return False
    if b_na != a_na:
        return True
    try:
        bf = float(bv)
        af = float(av)
        return abs(bf - af) > 1e-9
    except (TypeError, ValueError):
        return str(bv) != str(av)


def _coerce_cell(v: Any) -> Any:
    """JSON-safe representation: NaN -> None, numpy scalars -> python scalars."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _find_binary_label_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first label-like binary column usable for evaluation, if present."""
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in LABEL_COLUMN_CANDIDATES:
        col = by_lower.get(name)
        if col is None:
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        coerced = pd.to_numeric(values, errors="coerce")
        if coerced.isna().any():
            lowered = values.astype(str).str.strip().str.lower()
            if not lowered.isin(["0", "1", "true", "false", "normal", "anomaly"]).all():
                continue
        normalized = _binary_label_array(df[col])
        uniq = set(np.unique(normalized).astype(int).tolist())
        if uniq.issubset({0, 1}) and len(uniq) >= 1:
            return str(col)
    return None


def _binary_label_array(series: pd.Series) -> np.ndarray:
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(
        {
            "1": 1,
            "true": 1,
            "yes": 1,
            "anomaly": 1,
            "abnormal": 1,
            "0": 0,
            "false": 0,
            "no": 0,
            "normal": 0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    out = mapped.where(mapped.notna(), numeric).fillna(0).astype(float)
    return (out > 0).astype(int).to_numpy()


def _evaluation_report(
    df: pd.DataFrame,
    anomalies: np.ndarray,
    scores: np.ndarray,
    label_column: Optional[str],
) -> Dict[str, Any]:
    if not label_column:
        return {
            "available": False,
            "label_column": None,
            "searched_columns": LABEL_COLUMN_CANDIDATES,
            "note": "No binary label column was found. Add a column like label, ground_truth, or y_true with 0/1 values to see precision/recall/F1.",
        }

    y_true = _binary_label_array(df[label_column])
    y_pred = np.asarray(anomalies).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))
    ranking = binary_score_metrics(y_true, scores)
    return {
        "available": True,
        "label_column": label_column,
        "positive_label_count": int(np.sum(y_true)),
        "negative_label_count": int(len(y_true) - np.sum(y_true)),
        "predicted_positive_count": int(np.sum(y_pred)),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "specificity": float(specificity),
            "roc_auc": float(ranking.get("roc_auc", 0.0)),
            "pr_auc": float(ranking.get("pr_auc", 0.0)),
        },
        "note": f"Column {label_column!r} was used only for evaluation and excluded from model features.",
    }


def _cell_diffs(before: pd.DataFrame, after: pd.DataFrame, limit: int = 48) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = len(before)
    cols = list(before.columns)
    for i in range(n):
        for col in cols:
            bv = before.iloc[i][col]
            av = after.iloc[i][col]
            if _cells_differ(bv, av):
                out.append({"row": i, "column": col, "before": _coerce_cell(bv), "after": _coerce_cell(av)})
                if len(out) >= limit:
                    return out
    return out


def _injection_explanation(
    scenario: str,
    cfg: Dict[str, Any],
    k: int,
    numeric_cols: List[str],
    diffs: List[Dict[str, Any]],
    *,
    resolved_spike_column: Optional[str] = None,
) -> str:
    if k == 0:
        return "No rows were injected (contamination produced zero affected rows, or dataset too small)."
    cols_touched = sorted({d["column"] for d in diffs})
    if scenario == "spike_single":
        col = resolved_spike_column or cfg.get("column") or (numeric_cols[0] if numeric_cols else "?")
        mag = cfg.get("magnitude_in_std", 5.0)
        return (
            f"Scenario spike_single: chose {k} row(s) at random; added {mag}× "
            f"(column std of {col!r}) to that column only. Numeric columns in file: {numeric_cols}."
        )
    if scenario == "joint_shift":
        mag = cfg.get("magnitude_in_std", 4.0)
        return (
            f"Scenario joint_shift: chose {k} row(s); added {mag}× (per-column std) "
            f"to the same rows on columns {cols_touched or 'all numeric'}."
        )
    if scenario == "scale_burst":
        fac = cfg.get("scale_factor", 3.0)
        return (
            f"Scenario scale_burst: chose {k} row(s); multiplied selected numeric values by {fac} "
            f"(columns touched: {cols_touched or 'all numeric'})."
        )
    if scenario == "dead_sensor":
        const = cfg.get("constant")
        const_str = f"constant {const}" if const is not None else "per-column median"
        return (
            f"Scenario dead_sensor: chose {k} row(s); replaced selected numeric cells with "
            f"{const_str} (columns touched: {cols_touched or 'all numeric'})."
        )
    if scenario == "sign_flip":
        return (
            f"Scenario sign_flip: chose {k} row(s); multiplied selected numeric cells by -1 "
            f"(columns touched: {cols_touched or 'all numeric'})."
        )
    if scenario == "temporal_block":
        mag = cfg.get("magnitude_in_std", 4.0)
        bc = int(cfg.get("block_count", 1) or 1)
        return (
            f"Scenario temporal_block: picked {bc} contiguous block(s) totalling {k} row(s); "
            f"added {mag}× (per-column std) on those rows for columns {cols_touched or 'all numeric'}."
        )
    if scenario == "categorical_flip":
        mode = cfg.get("mode", "swap")
        col = cfg.get("column") or (cols_touched[0] if cols_touched else "?")
        if mode == "sentinel":
            sent = cfg.get("sentinel", "__UNKNOWN__")
            return (
                f"Scenario categorical_flip (sentinel): chose {k} row(s); set column {col!r} "
                f"to {sent!r} on those rows."
            )
        return (
            f"Scenario categorical_flip (swap): chose {k} row(s); replaced column {col!r} "
            f"with a different existing category on each row."
        )
    if scenario == "missing_value":
        return (
            f"Scenario missing_value: chose {k} row(s); set cells to NaN on columns "
            f"{cols_touched or 'all'} for those rows."
        )
    return f"Scenario {scenario!r}: {k} row(s) flagged as injected."


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    t = str(value).strip()
    if not t:
        return None
    return int(float(t))


def _synthetic_overrides_from_form(
    contamination: Optional[str],
    magnitude_in_std: Optional[str],
    scale_factor: Optional[str],
    column: Optional[str],
    columns: Optional[str],
    constant: Optional[str] = None,
    mode: Optional[str] = None,
    sentinel: Optional[str] = None,
    block_count: Optional[str] = None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    c = _optional_float(contamination)
    if c is not None:
        overrides["contamination"] = c
    m = _optional_float(magnitude_in_std)
    if m is not None:
        overrides["magnitude_in_std"] = m
    s = _optional_float(scale_factor)
    if s is not None:
        overrides["scale_factor"] = s
    if column and str(column).strip():
        overrides["column"] = str(column).strip()
    if columns and str(columns).strip():
        overrides["columns"] = [x.strip() for x in str(columns).split(",") if x.strip()]
    k = _optional_float(constant)
    if k is not None:
        overrides["constant"] = k
    if mode and str(mode).strip():
        overrides["mode"] = str(mode).strip().lower()
    if sentinel is not None and str(sentinel).strip():
        overrides["sentinel"] = str(sentinel).strip()
    bc = _optional_int(block_count)
    if bc is not None:
        overrides["block_count"] = bc
    return overrides


def _synthetic_inject_from_upload(
    content: bytes,
    scenario: str,
    random_seed: int,
    overrides: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, Any]]:
    if scenario not in list_scenarios():
        raise HTTPException(status_code=400, detail=f"Unknown scenario; use one of: {list_scenarios()}")
    before = pd.read_csv(io.BytesIO(content))
    cfg = merged_params(scenario, overrides)
    after, y_true = inject(before, scenario, random_seed=int(random_seed), params=overrides)
    return before, after, y_true, cfg


@app.get("/")
async def root():
    """Dashboard lives under /ui/ (static `ui/index.html`)."""
    return RedirectResponse(url="/ui/")


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    threshold_percentile: float = Form(95.0),
    threshold_strategy: str = Form("adaptive_gap"),
    expected_contamination: Optional[float] = Form(None),
):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    label_column = _find_binary_label_column(df)
    label_columns_ignored = [label_column] if label_column else []
    model_df = df.drop(columns=label_columns_ignored) if label_columns_ignored else df

    pct = max(50.0, min(99.9, float(threshold_percentile)))
    allowed_strategies = {"adaptive_gap", "percentile", "expected_contamination", "mean_std"}
    strategy = threshold_strategy if threshold_strategy in allowed_strategies else "adaptive_gap"
    contamination = None
    if expected_contamination is not None:
        contamination = max(0.001, min(0.5, float(expected_contamination)))
    anomalies, scores, details = system.run(
        model_df,
        threshold_strategy=strategy,
        threshold_percentile=pct,
        expected_contamination=contamination,
    )
    result_df = df.copy()
    result_df["anomaly_score"] = scores
    prediction_column = "predicted_is_anomaly" if "is_anomaly" in result_df.columns else "is_anomaly"
    result_df[prediction_column] = anomalies.astype(int)
    evaluation = _evaluation_report(df, anomalies, scores, label_column)

    return jsonable_encoder(
        {
            "anomaly_count": int(anomalies.sum()),
            "sample_scores": scores[:20].tolist(),
            "sample_anomalies": anomalies[:20].tolist(),
            "summary": details["report"],
            "columns": result_df.columns.tolist(),
            "full_data": result_df.values.tolist(),
            "full_anomalies": anomalies.tolist(),
            "full_scores": scores.tolist(),
            "prediction_column": prediction_column,
            "threshold": float(details["threshold"]),
            "models_used": list(details["models"].keys()),
            "model_weights": details.get("model_weights", {}),
            "meta": {**details["meta"], "label_columns_ignored": label_columns_ignored},
            "evaluation": evaluation,
            "threshold_rule": details.get("threshold_strategy", "adaptive_gap"),
            "threshold_percentile": float(details.get("threshold_percentile", pct)),
            "expected_contamination": details.get("expected_contamination"),
            "threshold_note": (
                "Rows with combined ensemble score above the adaptive score-gap threshold are flagged."
                if details.get("threshold_strategy") == "adaptive_gap"
                else "Rows above the expected-contamination threshold are flagged."
                if details.get("threshold_strategy") == "expected_contamination"
                else f"Rows with combined ensemble score above the {pct:g}th percentile are flagged."
            ),
        }
    )


@app.post("/eda")
async def eda_profile(file: UploadFile = File(...)):
    """
    Exploratory profile of an uploaded table: dtypes, missingness, numeric summaries,
    histograms (top variance numeric columns), and a correlation matrix slice — no ML models.
    """
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    return jsonable_encoder(build_eda_report(df))


@app.post("/synthetic-preview")
async def synthetic_preview(
    file: UploadFile = File(...),
    scenario: str = Form("spike_single"),
    random_seed: int = Form(42),
    preview_rows: int = Form(20),
    contamination: Optional[str] = Form(None),
    magnitude_in_std: Optional[str] = Form(None),
    scale_factor: Optional[str] = Form(None),
    column: Optional[str] = Form(None),
    columns: Optional[str] = Form(None),
    constant: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    sentinel: Optional[str] = Form(None),
    block_count: Optional[str] = Form(None),
):
    """
    Apply ``inject`` to the uploaded CSV and return before/after previews plus a short explanation.

    Optional form fields override defaults from ``SCENARIO_DEFAULTS``; ``columns`` is a
    comma-separated list for joint_shift / scale_burst / dead_sensor / sign_flip /
    temporal_block / missing_value when you want a subset. ``constant`` applies to
    ``dead_sensor``; ``mode``/``sentinel`` apply to ``categorical_flip``; ``block_count``
    applies to ``temporal_block``.
    """
    content = await file.read()
    overrides = _synthetic_overrides_from_form(
        contamination,
        magnitude_in_std,
        scale_factor,
        column,
        columns,
        constant=constant,
        mode=mode,
        sentinel=sentinel,
        block_count=block_count,
    )
    before, after, y_true, cfg = _synthetic_inject_from_upload(
        content, scenario, int(random_seed), overrides
    )

    numeric_cols = before.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in before.columns if c not in numeric_cols]
    resolved_spike_column = None
    if scenario == "spike_single" and numeric_cols:
        resolved_spike_column = cfg.get("column") or numeric_cols[0]
    k = int(np.sum(y_true))
    diffs = _cell_diffs(before, after)
    explanation = _injection_explanation(
        scenario, cfg, k, numeric_cols, diffs, resolved_spike_column=resolved_spike_column
    )

    pr = max(1, min(int(preview_rows), 80, len(before)))
    cols = list(before.columns)
    before_head = before.iloc[:pr].replace({np.nan: None}).to_dict(orient="records")
    after_head = after.iloc[:pr].replace({np.nan: None}).to_dict(orient="records")
    y_head = [int(x) for x in y_true[:pr].tolist()]

    injected_idx = np.where(y_true == 1)[0].astype(int).tolist()

    return {
        "scenario": scenario,
        "random_seed": int(random_seed),
        "params_effective": {k: v for k, v in cfg.items() if v is not None},
        "resolved_spike_column": resolved_spike_column,
        "columns": cols,
        "numeric_columns": numeric_cols,
        "non_numeric_columns": non_numeric_cols,
        "row_count": len(before),
        "injected_row_count": k,
        "injected_row_indices": injected_idx,
        "explanation": explanation,
        "cell_changes_sample": diffs,
        "preview_row_count": pr,
        "before_preview": before_head,
        "after_preview": after_head,
        "y_true_preview": y_head,
    }


@app.post("/synthetic-export")
async def synthetic_export(
    file: UploadFile = File(...),
    scenario: str = Form("spike_single"),
    random_seed: int = Form(42),
    contamination: Optional[str] = Form(None),
    magnitude_in_std: Optional[str] = Form(None),
    scale_factor: Optional[str] = Form(None),
    column: Optional[str] = Form(None),
    columns: Optional[str] = Form(None),
    constant: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    sentinel: Optional[str] = Form(None),
    block_count: Optional[str] = Form(None),
):
    """
    Same injection parameters as ``/synthetic-preview``, but returns the **full** perturbed
    table as a CSV attachment (all rows). Use this file as input to **Run Analysis** for
    pipeline evaluation on corrupted data.
    """
    content = await file.read()
    overrides = _synthetic_overrides_from_form(
        contamination,
        magnitude_in_std,
        scale_factor,
        column,
        columns,
        constant=constant,
        mode=mode,
        sentinel=sentinel,
        block_count=block_count,
    )
    _before, after, _y_true, _cfg = _synthetic_inject_from_upload(
        content, scenario, int(random_seed), overrides
    )
    buf = io.StringIO()
    after.to_csv(buf, index=False)
    body = buf.getvalue().encode("utf-8")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scenario)[:40]
    fname = f"synthetic_after_{safe}_seed{int(random_seed)}.csv"
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

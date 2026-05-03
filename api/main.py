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
from synthetic_injection import inject, list_scenarios, merged_params

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
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    anomalies, scores, details = system.run(df)

    return jsonable_encoder(
        {
            "anomaly_count": int(anomalies.sum()),
            "sample_scores": scores[:20].tolist(),
            "sample_anomalies": anomalies[:20].tolist(),
            "summary": details["report"],
            "full_data": details["results"].values.tolist(),
            "full_anomalies": anomalies.tolist(),
            "full_scores": scores.tolist(),
            "threshold": float(details["threshold"]),
            "models_used": list(details["models"].keys()),
            "meta": details["meta"],
            "threshold_rule": "percentile_95",
            "threshold_note": "Rows with combined ensemble score above the 95th percentile are flagged (~5% of rows in expectation for continuous scores).",
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
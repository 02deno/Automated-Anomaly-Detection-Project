from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
import io
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_system import AdvancedAnomalySystem
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


def _numeric_cell_diffs(before: pd.DataFrame, after: pd.DataFrame, limit: int = 48) -> List[Dict[str, Any]]:
    num_cols = before.select_dtypes(include=[np.number]).columns.tolist()
    out: List[Dict[str, Any]] = []
    n = len(before)
    for i in range(n):
        for col in num_cols:
            bv = before.iloc[i][col]
            av = after.iloc[i][col]
            try:
                bf = float(bv)
                af = float(av)
            except (TypeError, ValueError):
                continue
            if abs(bf - af) > 1e-9:
                out.append({"row": i, "column": col, "before": bf, "after": af})
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
    return f"Scenario {scenario!r}: {k} row(s) flagged as injected."


@app.get("/")
async def root():
    """Dashboard lives under /ui/ (static `ui/index.html`)."""
    return RedirectResponse(url="/ui/")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    anomalies, scores, details = system.run(df)

    return {
        "anomaly_count": int(anomalies.sum()),
        "sample_scores": scores[:20].tolist(),
        "sample_anomalies": anomalies[:20].tolist(),
        "summary": details["report"],
        "full_data": details["results"].values.tolist(),
        "full_anomalies": anomalies.tolist(),
        "full_scores": scores.tolist(),
    }


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
):
    """
    Apply ``inject`` to the uploaded CSV and return before/after previews plus a short explanation.

    Optional form fields override defaults from ``SCENARIO_DEFAULTS``; ``columns`` is a
    comma-separated list for joint_shift / scale_burst when you want a subset.
    """
    if scenario not in list_scenarios():
        raise HTTPException(status_code=400, detail=f"Unknown scenario; use one of: {list_scenarios()}")

    content = await file.read()
    before = pd.read_csv(io.BytesIO(content))

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

    cfg = merged_params(scenario, overrides)
    after, y_true = inject(before, scenario, random_seed=int(random_seed), params=overrides)

    numeric_cols = before.select_dtypes(include=[np.number]).columns.tolist()
    resolved_spike_column = None
    if scenario == "spike_single" and numeric_cols:
        resolved_spike_column = cfg.get("column") or numeric_cols[0]
    k = int(np.sum(y_true))
    diffs = _numeric_cell_diffs(before, after)
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


if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")
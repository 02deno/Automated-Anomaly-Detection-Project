"""
Render robustness figures from an aggregated benchmark CSV.

Inputs:
  --aggregated  results/<...>_aggregated.csv (default: robustness_aggregated.csv)

Outputs (under --out, default: results/figures/):
  - <dataset>_<noise>_heatmap_f1.png       # rows=scenario, cols=score_source
  - <dataset>_<noise>_heatmap_roc_auc.png  # same layout, ROC-AUC mean
  - <dataset>_sweep_<scenario>_<x>.png     # e.g. spike_single magnitude/contamination sweep

For "which model is more robust" the heatmaps + worst-case CSV are the headline; the
sweep curves explain *where* a model degrades faster than its peers (subtle vs extreme).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _import_matplotlib():
    try:
        import matplotlib  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    return plt


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))[:60]


def _decode_params(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _heatmap_grid(df: pd.DataFrame, value_col: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Build the (rows=scenario, cols=score_source) matrix used by the heatmaps.

    For repeats inside one dataset/noise slice we average; this only happens when the
    aggregated rows already collapsed across the param grid (we still pivot defensively).
    """
    pivot = df.pivot_table(index="scenario", columns="score_source", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return list(pivot.index), list(pivot.columns), pivot.to_numpy()


def _draw_heatmap(plt, *, matrix: np.ndarray, rows: List[str], cols: List[str],
                  title: str, value_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(cols) + 3), max(4, 0.55 * len(rows) + 2)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Score source (model / ensemble)")
    ax.set_ylabel("Scenario")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="white", fontsize=9)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label=value_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_sweep(plt, *, df: pd.DataFrame, x_key: str, y_col: str,
                title: str, out_path: Path) -> None:
    """One sweep figure per (dataset, scenario, x_key) — line per score_source."""
    grouped: Dict[str, Dict[float, List[float]]] = {}
    for _, row in df.iterrows():
        params = _decode_params(row.get("params_effective"))
        if x_key not in params:
            continue
        x_val = float(params[x_key])
        src = str(row["score_source"])
        bucket = grouped.setdefault(src, {})
        bucket.setdefault(x_val, []).append(float(row[y_col]))
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for src in sorted(grouped):
        items = sorted(grouped[src].items())
        xs = [item[0] for item in items]
        ys = [float(np.mean(item[1])) for item in items]
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=src)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_col)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_figures(aggregated_path: Path, out_dir: Path, *, sweep_keys: Optional[List[str]] = None) -> List[Path]:
    if not aggregated_path.exists():
        raise FileNotFoundError(f"Aggregated CSV not found: {aggregated_path}")
    df = pd.read_csv(aggregated_path)
    if df.empty:
        print(f"WARNING: {aggregated_path} is empty; nothing to plot.")
        return []
    plt = _import_matplotlib()
    written: List[Path] = []

    sweep_keys = sweep_keys or ["magnitude_in_std", "contamination", "scale_factor"]

    for (dataset, noise), slice_df in df.groupby(["dataset", "noise_std"], sort=True):
        slug = f"{_safe_filename(str(dataset))}_noise{noise:g}"
        for value_col, label in (("f1_mean", "F1 (mean)"), ("roc_auc_mean", "ROC-AUC (mean)")):
            rows, cols, matrix = _heatmap_grid(slice_df, value_col)
            if matrix.size == 0:
                continue
            out_path = out_dir / f"{slug}_heatmap_{value_col}.png"
            _draw_heatmap(
                plt,
                matrix=matrix,
                rows=rows,
                cols=cols,
                title=f"{dataset} (noise_std={noise:g}) — {label}",
                value_label=label,
                out_path=out_path,
            )
            written.append(out_path)

        for scenario, scen_df in slice_df.groupby("scenario", sort=True):
            for x_key in sweep_keys:
                if scen_df.empty:
                    continue
                values = [_decode_params(p).get(x_key) for p in scen_df["params_effective"].tolist()]
                if sum(1 for v in values if v is not None) < 2:
                    continue
                if len({float(v) for v in values if v is not None}) < 2:
                    continue
                title = f"{dataset} | {scenario} | sweep over {x_key} (noise_std={noise:g})"
                out_path = out_dir / f"{slug}_sweep_{_safe_filename(scenario)}_{_safe_filename(x_key)}.png"
                _draw_sweep(plt, df=scen_df, x_key=x_key, y_col="f1_mean",
                            title=title, out_path=out_path)
                written.append(out_path)

    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render robustness heatmaps + sweep curves.")
    parser.add_argument(
        "--aggregated",
        default=str(ROOT / "results" / "robustness_aggregated.csv"),
        help="Aggregated benchmark CSV produced by run_synthetic_benchmark.py.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "figures"),
        help="Output directory for PNG figures.",
    )
    parser.add_argument(
        "--sweep-key",
        action="append",
        help="Param key to use as sweep x-axis (repeatable). Defaults: magnitude_in_std, contamination, scale_factor.",
    )
    args = parser.parse_args(argv)

    aggregated = Path(args.aggregated)
    if not aggregated.is_absolute():
        aggregated = ROOT / aggregated
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    figures = render_figures(aggregated, out_dir, sweep_keys=args.sweep_key)
    print(f"Wrote {len(figures)} figure(s) to {out_dir}")
    for path in figures:
        print(f"  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

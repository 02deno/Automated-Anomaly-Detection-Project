from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from synthetic_injection import inject, list_scenarios  # noqa: E402

CLASS_LIKE_COLUMNS = ["label", "target", "class", "category", "type", "digit", "glass_type"]


def _parse_values(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _coerce_like_column(values: List[str], series: pd.Series) -> List[Any]:
    if pd.api.types.is_numeric_dtype(series):
        return [float(v) if "." in v else int(v) for v in values]
    return values


def _auto_label_column(df: pd.DataFrame) -> str:
    by_lower = {str(c).strip().lower(): str(c) for c in df.columns}
    for name in CLASS_LIKE_COLUMNS:
        if name in by_lower:
            return by_lower[name]
    raise ValueError(
        "Could not infer a label column. Pass --label-column with a class-like column name."
    )


def _rare_values(series: pd.Series) -> List[Any]:
    counts = series.dropna().value_counts()
    if counts.empty or counts.shape[0] < 2:
        raise ValueError("Label column must contain at least two classes.")
    cutoff = float(counts.quantile(0.25))
    rare = counts[counts <= cutoff].index.tolist()
    if not rare or len(rare) == counts.shape[0]:
        rare = [counts.idxmin()]
    return rare


def prepare_labeled_csv(
    input_path: Path,
    output_path: Path,
    *,
    label_column: str | None,
    anomaly_values: List[str] | None,
    strategy: str,
    scenario: str,
    random_seed: int,
    contamination: float | None,
    magnitude_in_std: float | None,
    column: str | None,
    columns: List[str] | None,
) -> None:
    df = pd.read_csv(input_path)
    if strategy == "synthetic":
        overrides: dict[str, Any] = {}
        if contamination is not None:
            overrides["contamination"] = contamination
        if magnitude_in_std is not None:
            overrides["magnitude_in_std"] = magnitude_in_std
        if column:
            overrides["column"] = column
        if columns:
            overrides["columns"] = columns
        out, y_true = inject(df, scenario, random_seed=random_seed, params=overrides)
        out["ground_truth"] = y_true.astype(int)
        positive_count = int(out["ground_truth"].sum())
        if positive_count == 0:
            raise ValueError("Synthetic injection produced zero anomalies. Increase --contamination.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)

        print(f"Wrote {output_path}")
        print(f"Rows: {len(out)}")
        print(f"Strategy: synthetic")
        print(f"Scenario: {scenario}")
        print(f"Random seed: {random_seed}")
        print(f"ground_truth=1 rows: {positive_count}")
        print(f"ground_truth=0 rows: {len(out) - positive_count}")
        return

    source_col = label_column or _auto_label_column(df)
    if source_col not in df.columns:
        raise ValueError(f"Column {source_col!r} was not found. Available columns: {list(df.columns)}")

    if anomaly_values:
        positives = _coerce_like_column(anomaly_values, df[source_col])
    elif strategy == "rare":
        positives = _rare_values(df[source_col])
    else:
        raise ValueError("Use --anomaly-values or --strategy rare.")

    out = df.copy()
    out["ground_truth"] = out[source_col].isin(positives).astype(int)
    positive_count = int(out["ground_truth"].sum())
    if positive_count == 0 or positive_count == len(out):
        raise ValueError(
            "The generated ground_truth column is not useful; all rows have the same label. "
            "Check --label-column and --anomaly-values."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Wrote {output_path}")
    print(f"Rows: {len(out)}")
    print(f"Source label column: {source_col}")
    print(f"Strategy: {strategy}")
    print(f"Anomaly values: {[str(v) for v in positives]}")
    print(f"ground_truth=1 rows: {positive_count}")
    print(f"ground_truth=0 rows: {len(out) - positive_count}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a CSV with binary ground_truth labels for UI evaluation."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "external" / "uci_glass.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Default: input filename with _labeled suffix.",
    )
    parser.add_argument(
        "--label-column",
        default="",
        help="Class/label column to convert. If omitted, common names are inferred.",
    )
    parser.add_argument(
        "--anomaly-values",
        default="",
        help="Comma-separated values to mark as anomalies, e.g. 5,6 or attack,fraud.",
    )
    parser.add_argument(
        "--strategy",
        choices=("rare", "synthetic"),
        default="rare",
        help="rare: convert rare classes to labels. synthetic: inject anomalies and create ground_truth.",
    )
    parser.add_argument(
        "--scenario",
        choices=list_scenarios(),
        default="spike_single",
        help="Synthetic scenario to use when --strategy synthetic.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for synthetic labeling.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=None,
        help="Optional synthetic anomaly fraction, e.g. 0.1.",
    )
    parser.add_argument(
        "--magnitude-in-std",
        type=float,
        default=None,
        help="Optional magnitude override for spike/joint/temporal scenarios.",
    )
    parser.add_argument(
        "--column",
        default="",
        help="Optional single column for synthetic scenarios such as spike_single.",
    )
    parser.add_argument(
        "--columns",
        default="",
        help="Optional comma-separated columns for multi-column synthetic scenarios.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_labeled.csv")
    prepare_labeled_csv(
        input_path,
        output_path,
        label_column=args.label_column.strip() or None,
        anomaly_values=_parse_values(args.anomaly_values) if args.anomaly_values.strip() else None,
        strategy=args.strategy,
        scenario=args.scenario,
        random_seed=args.random_seed,
        contamination=args.contamination,
        magnitude_in_std=args.magnitude_in_std,
        column=args.column.strip() or None,
        columns=_parse_values(args.columns) if args.columns.strip() else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

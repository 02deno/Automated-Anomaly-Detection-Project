from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark CSV by score source.")
    parser.add_argument("input", help="Per-run benchmark CSV.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV. Defaults to <input stem>_model_comparison.csv.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}_model_comparison.csv")

    df = pd.read_csv(in_path)
    keep = df[df["score_source"].isin(["ensemble", "iforest", "lof", "ocsvm"])].copy()
    if "best_f1" not in keep.columns:
        keep["best_f1"] = keep["f1"]
    grouped = (
        keep.groupby("score_source", as_index=False)
        .agg(
            runs=("f1", "count"),
            avg_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            avg_roc_auc=("roc_auc", "mean"),
            avg_pr_auc=("pr_auc", "mean"),
            avg_best_f1=("best_f1", "mean"),
            worst_f1=("f1", "min"),
        )
        .fillna(0.0)
    )
    grouped["rank_by_f1"] = grouped["avg_f1"].rank(ascending=False, method="min").astype(int)
    grouped = grouped.sort_values(["rank_by_f1", "score_source"])
    for col in ["avg_f1", "std_f1", "avg_roc_auc", "avg_pr_auc", "avg_best_f1", "worst_f1"]:
        grouped[col] = grouped[col].round(6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)
    print(grouped.to_string(index=False))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

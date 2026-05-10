from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external"


def prepare_glass() -> Path:
    src = DATA_DIR / "uci_glass.csv"
    df = pd.read_csv(src)
    out = df.drop(columns=["id"], errors="ignore").copy()
    # UCI Glass has no native anomaly label. Use the rarest class as a derived
    # anomaly class so this can be scored as a real tabular benchmark.
    rare_class = int(out["glass_type"].value_counts().idxmin())
    out["ground_truth"] = (out["glass_type"] == rare_class).astype(int)
    out = out.drop(columns=["glass_type"])
    path = DATA_DIR / "uci_glass_rare_class.csv"
    out.to_csv(path, index=False)
    return path


def prepare_pendigits() -> Path:
    src = DATA_DIR / "uci_pendigits_train.csv"
    df = pd.read_csv(src)
    out = df.copy()
    # Pendigits is multiclass, not a native anomaly dataset. Treat digit 0 as
    # the held-out anomaly class, a common one-vs-rest style benchmark setup.
    out["ground_truth"] = (out["digit"] == 0).astype(int)
    out = out.drop(columns=["digit"])
    path = DATA_DIR / "uci_pendigits_digit0_anomaly.csv"
    out.to_csv(path, index=False)
    return path


def prepare_ionosphere() -> Path:
    src = DATA_DIR / "uci_ionosphere.csv"
    df = pd.read_csv(src)
    out = df.copy()
    # Ionosphere is a labeled radar-return dataset. Treat bad returns as anomalies.
    out["ground_truth"] = out["label"].astype(str).str.lower().eq("b").astype(int)
    out = out.drop(columns=["label"])
    path = DATA_DIR / "uci_ionosphere_bad.csv"
    out.to_csv(path, index=False)
    return path


def prepare_wdbc() -> Path:
    src = DATA_DIR / "uci_wdbc.csv"
    df = pd.read_csv(src)
    out = df.drop(columns=["id"], errors="ignore").copy()
    # WDBC is not a native anomaly corpus, but malignant cases are the rarer
    # labeled class and provide another real tabular validation task.
    out["ground_truth"] = out["diagnosis"].astype(str).str.upper().eq("M").astype(int)
    out = out.drop(columns=["diagnosis"])
    path = DATA_DIR / "uci_wdbc_malignant.csv"
    out.to_csv(path, index=False)
    return path


def prepare_wine() -> Path:
    src = DATA_DIR / "uci_wine.csv"
    df = pd.read_csv(src)
    out = df.copy()
    rare_class = int(out["wine_class"].value_counts().idxmin())
    out["ground_truth"] = (out["wine_class"] == rare_class).astype(int)
    out = out.drop(columns=["wine_class"])
    path = DATA_DIR / "uci_wine_rare_class.csv"
    out.to_csv(path, index=False)
    return path


def prepare_ecoli() -> Path:
    src = DATA_DIR / "uci_ecoli.csv"
    df = pd.read_csv(src)
    out = df.drop(columns=["sequence_name"], errors="ignore").copy()
    rare_class = str(out["site"].value_counts().idxmin())
    out["ground_truth"] = out["site"].astype(str).eq(rare_class).astype(int)
    out = out.drop(columns=["site"])
    path = DATA_DIR / "uci_ecoli_rare_class.csv"
    out.to_csv(path, index=False)
    return path


def prepare_yeast() -> Path:
    src = DATA_DIR / "uci_yeast.csv"
    df = pd.read_csv(src)
    out = df.drop(columns=["sequence_name"], errors="ignore").copy()
    rare_class = str(out["site"].value_counts().idxmin())
    out["ground_truth"] = out["site"].astype(str).eq(rare_class).astype(int)
    out = out.drop(columns=["site"])
    path = DATA_DIR / "uci_yeast_rare_class.csv"
    out.to_csv(path, index=False)
    return path


def main() -> int:
    written = [
        prepare_glass(),
        prepare_pendigits(),
        prepare_ionosphere(),
        prepare_wdbc(),
        prepare_wine(),
        prepare_ecoli(),
        prepare_yeast(),
    ]
    for path in written:
        df = pd.read_csv(path)
        print(
            f"Wrote {path.relative_to(ROOT)} rows={len(df)} "
            f"positives={int(df['ground_truth'].sum())}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

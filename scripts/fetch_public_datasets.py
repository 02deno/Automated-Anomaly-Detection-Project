"""
Download public tabular datasets used in anomaly-detection benchmarks.

Sources:
  - UCI Machine Learning Repository (HTTP): Glass, Pen Digits, Ionosphere,
    WDBC Breast Cancer, Wine, Ecoli, Yeast.
  - scikit-learn fetcher (downloads from network on first use): KDD Cup 1999
    subsets SMTP / HTTP (10% samples) with attack vs normal ground truth.
  - mala-lab/ADBenchmarks-anomaly-detection-datasets (raw GitHub): DevNet
    ``annthyroid_21feat_normalised.csv`` (tabular, ~7k rows; suitable default).

Run from repository root:
  python scripts/fetch_public_datasets.py --dataset glass
  python scripts/fetch_public_datasets.py --dataset pendigits
  python scripts/fetch_public_datasets.py --dataset ionosphere
  python scripts/fetch_public_datasets.py --dataset wdbc
  python scripts/fetch_public_datasets.py --dataset wine
  python scripts/fetch_public_datasets.py --dataset ecoli
  python scripts/fetch_public_datasets.py --dataset yeast
  python scripts/fetch_public_datasets.py --dataset kddcup99_smtp
  python scripts/fetch_public_datasets.py --dataset kddcup99_http
  python scripts/fetch_public_datasets.py --dataset adb_annthyroid
  python scripts/fetch_public_datasets.py --dataset all

Outputs go to data/external/ (see data/external/README.md for citations).
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "external"

UCI_GLASS = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
UCI_PENDIGITS_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"
UCI_IONOSPHERE = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
UCI_WDBC = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
UCI_WINE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
UCI_ECOLI = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
UCI_YEAST = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"

ADB_RAW_BASE = (
    "https://raw.githubusercontent.com/mala-lab/ADBenchmarks-anomaly-detection-datasets/main"
)
ADB_ANNTHYROID_CSV = (
    f"{ADB_RAW_BASE}/numerical%20data/DevNet%20datasets/annthyroid_21feat_normalised.csv"
)


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _download_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Automated-Anomaly-Detection-Project/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8")


def _download_bytes(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Automated-Anomaly-Detection-Project/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_glass() -> Path:
    """UCI Glass Identification; multiclass (types 1–7). Last column is class."""
    _ensure_out_dir()
    raw = _download_text(UCI_GLASS)
    cols = [
        "id",
        "ri",
        "na",
        "mg",
        "al",
        "si",
        "k",
        "ca",
        "ba",
        "fe",
        "glass_type",
    ]
    df = pd.read_csv(io.StringIO(raw), header=None, names=cols)
    path = OUT_DIR / "uci_glass.csv"
    df.to_csv(path, index=False)
    return path


def fetch_pendigits() -> Path:
    """UCI Pen Digits training; 16 input attributes + digit label (0–9)."""
    _ensure_out_dir()
    raw = _download_text(UCI_PENDIGITS_TRAIN)
    feat = [f"f{i}" for i in range(16)]
    cols = feat + ["digit"]
    df = pd.read_csv(io.StringIO(raw), header=None, skipinitialspace=True)
    if df.shape[1] != 17:
        raise ValueError(f"Unexpected pendigits shape: {df.shape}")
    df.columns = cols
    path = OUT_DIR / "uci_pendigits_train.csv"
    df.to_csv(path, index=False)
    return path


def fetch_ionosphere() -> Path:
    """UCI Ionosphere; label is g=good radar return, b=bad return."""
    _ensure_out_dir()
    raw = _download_text(UCI_IONOSPHERE)
    cols = [f"f{i}" for i in range(34)] + ["label"]
    df = pd.read_csv(io.StringIO(raw), header=None, names=cols)
    path = OUT_DIR / "uci_ionosphere.csv"
    df.to_csv(path, index=False)
    return path


def fetch_wdbc() -> Path:
    """Wisconsin Diagnostic Breast Cancer; diagnosis M/B is used as a binary label."""
    _ensure_out_dir()
    raw = _download_text(UCI_WDBC)
    cols = ["id", "diagnosis"] + [f"f{i}" for i in range(30)]
    df = pd.read_csv(io.StringIO(raw), header=None, names=cols)
    path = OUT_DIR / "uci_wdbc.csv"
    df.to_csv(path, index=False)
    return path


def fetch_wine() -> Path:
    """UCI Wine; multiclass label is later converted to a rare-class anomaly task."""
    _ensure_out_dir()
    raw = _download_text(UCI_WINE)
    cols = ["wine_class"] + [f"f{i}" for i in range(13)]
    df = pd.read_csv(io.StringIO(raw), header=None, names=cols)
    path = OUT_DIR / "uci_wine.csv"
    df.to_csv(path, index=False)
    return path


def fetch_ecoli() -> Path:
    """UCI Ecoli localization; multiclass label is later converted to rare-class anomaly."""
    _ensure_out_dir()
    raw = _download_text(UCI_ECOLI)
    cols = ["sequence_name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "site"]
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=cols, engine="python")
    path = OUT_DIR / "uci_ecoli.csv"
    df.to_csv(path, index=False)
    return path


def fetch_yeast() -> Path:
    """UCI Yeast localization; multiclass label is later converted to rare-class anomaly."""
    _ensure_out_dir()
    raw = _download_text(UCI_YEAST)
    cols = ["sequence_name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "site"]
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=cols, engine="python")
    path = OUT_DIR / "uci_yeast.csv"
    df.to_csv(path, index=False)
    return path


def fetch_adb_annthyroid() -> Path:
    """
    DevNet-format Annthyroid (tabular, normalised) from ADBenchmarks repo.

    Original column ``class`` (0/1) is saved as ``ground_truth`` for consistency
    with other exports in this project.
    """
    _ensure_out_dir()
    raw = _download_bytes(ADB_ANNTHYROID_CSV)
    df = pd.read_csv(io.BytesIO(raw))
    if "class" in df.columns:
        df = df.rename(columns={"class": "ground_truth"})
    path = OUT_DIR / "adb_annthyroid_21feat_normalised.csv"
    df.to_csv(path, index=False)
    return path


def fetch_kddcup99(subset: str) -> Path:
    """KDD Cup 1999 intrusion subset (SMTP or HTTP); ground_truth=1 for attack."""
    try:
        from sklearn.datasets import fetch_kddcup99
    except ImportError as e:
        raise SystemExit("scikit-learn is required for KDD datasets. pip install -r requirements.txt") from e

    _ensure_out_dir()
    subset_lower = subset.lower()
    bunch = fetch_kddcup99(subset=subset_lower, percent10=True, as_frame=True)
    df = bunch.frame.copy()
    if "labels" not in df.columns:
        raise ValueError("Expected 'labels' column in KDD frame from sklearn.")
    labels = df.pop("labels")
    # Benign traffic is labeled b'normal.' in this corpus; everything else is attack.
    df["ground_truth"] = (~labels.eq(b"normal.")).astype(int)
    name = f"kddcup99_{subset_lower}_percent10.csv"
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    return path


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download public benchmark CSVs into data/external/")
    p.add_argument(
        "--dataset",
        choices=(
            "glass",
            "pendigits",
            "ionosphere",
            "wdbc",
            "wine",
            "ecoli",
            "yeast",
            "kddcup99_smtp",
            "kddcup99_http",
            "adb_annthyroid",
            "all",
        ),
        default="glass",
        help="Which dataset to fetch (default: glass)",
    )
    args = p.parse_args(argv)

    targets: Iterable[str]
    if args.dataset == "all":
        targets = (
            "glass",
            "pendigits",
            "ionosphere",
            "wdbc",
            "wine",
            "ecoli",
            "yeast",
            "kddcup99_smtp",
            "kddcup99_http",
            "adb_annthyroid",
        )
    else:
        targets = (args.dataset,)

    written: List[Path] = []
    for name in targets:
        if name == "glass":
            written.append(fetch_glass())
        elif name == "pendigits":
            written.append(fetch_pendigits())
        elif name == "ionosphere":
            written.append(fetch_ionosphere())
        elif name == "wdbc":
            written.append(fetch_wdbc())
        elif name == "wine":
            written.append(fetch_wine())
        elif name == "ecoli":
            written.append(fetch_ecoli())
        elif name == "yeast":
            written.append(fetch_yeast())
        elif name == "kddcup99_smtp":
            written.append(fetch_kddcup99("smtp"))
        elif name == "kddcup99_http":
            written.append(fetch_kddcup99("http"))
        elif name == "adb_annthyroid":
            written.append(fetch_adb_annthyroid())
        else:
            raise AssertionError(name)

    for path in written:
        print(f"Wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1:])

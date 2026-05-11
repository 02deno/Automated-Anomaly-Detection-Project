# External (downloaded) datasets

CSV files in this folder are **not** committed by default (see root `.gitignore`). Populate them with:

```bash
python scripts/fetch_public_datasets.py --dataset all
```

Or one at a time: `glass`, `pendigits`, `ionosphere`, `wdbc`, `wine`, `ecoli`, `yeast`, `kddcup99_smtp`, `kddcup99_http`, `adb_annthyroid`.

## What you get

| Output file | Source | Notes |
|-------------|--------|--------|
| `uci_glass.csv` | [UCI Glass Identification](https://archive.ics.uci.edu/ml/datasets/Glass+Identification) | Multiclass chemistry data; last column `glass_type`. Often used by treating **rare classes as anomalies** in research settings (not a native “outlier” column). |
| `uci_pendigits_train.csv` | [UCI Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) | 16 features + `digit` (0–9). Common setup: pick one digit as “normal” and others as anomalies for controlled experiments. |
| `uci_ionosphere.csv` | [UCI Ionosphere](https://archive.ics.uci.edu/ml/datasets/Ionosphere) | 34 radar features + `label`; prepared as `uci_ionosphere_bad.csv` with bad returns as anomalies. |
| `uci_wdbc.csv` | [UCI Wisconsin Diagnostic Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) | 30 numeric features + diagnosis; prepared as `uci_wdbc_malignant.csv` with malignant cases as the positive class. |
| `uci_wine.csv` | [UCI Wine](https://archive.ics.uci.edu/ml/datasets/Wine) | Multiclass chemistry data; prepared as `uci_wine_rare_class.csv` using the rarest class as positive. |
| `uci_ecoli.csv` | [UCI Ecoli](https://archive.ics.uci.edu/ml/datasets/Ecoli) | Protein localization data; prepared as `uci_ecoli_rare_class.csv` using the rarest class as positive. |
| `uci_yeast.csv` | [UCI Yeast](https://archive.ics.uci.edu/ml/datasets/Yeast) | Protein localization data; prepared as `uci_yeast_rare_class.csv` using the rarest class as positive. |
| `kddcup99_smtp_percent10.csv` | [KDD Cup 1999](https://kdd.ig.ucsd.edu/databases/kddcup99/kddcup99.html) via `sklearn.datasets.fetch_kddcup99` | **SMTP** subset, 10% sampling. Column `ground_truth`: `1` = attack, `0` = normal (benign). |
| `kddcup99_http_percent10.csv` | same | **HTTP** subset, 10% sampling; same `ground_truth` convention. |
| `adb_annthyroid_21feat_normalised.csv` | [mala-lab/ADBenchmarks-anomaly-detection-datasets](https://github.com/mala-lab/ADBenchmarks-anomaly-detection-datasets) (DevNet numerical folder) | **Annthyroid**, normalised 21 features; `ground_truth` is the DevNet label (renamed from `class`). Fits this pipeline (numeric features + binary label for evaluation). |

### ADBenchmarks repository (larger / non-CSV)

The same GitHub repo also ships **very large** CSVs (e.g. donors ~24 MB) and **archives** (`creditcardfraud_normalised.tar.xz`, `census-income-…tar.xz`). This project’s fetch script only automates **Annthyroid** as a practical default. For other DevNet files, download manually from:

`https://github.com/mala-lab/ADBenchmarks-anomaly-detection-datasets/tree/main/numerical%20data/DevNet%20datasets`

and cite the repository license (GPL-3.0) and the DevNet / survey papers listed there.

## Using with this project

- The detection pipeline (`AdvancedAnomalySystem`) **does not use label columns**; it only reads numeric features. You can **drop** `ground_truth`, `digit`, `glass_type`, or `id` before `run()`, or keep them and let preprocessing ignore non-numeric columns.
- For **supervised-style evaluation** against real labels, compare `ground_truth` (or your derived anomaly label) to `is_anomaly` in the result `DataFrame`—similar to synthetic `y_true`, but labels come from the dataset, not from `inject()`.

## Licenses and attribution

- UCI datasets are typically for academic use; cite the UCI repository and original donors as required by your course.
- KDD Cup 1999 data has its own usage terms; see the official KDD page linked above.
- **ADBenchmarks** datasets: follow [their LICENSE](https://github.com/mala-lab/ADBenchmarks-anomaly-detection-datasets/blob/main/LICENSE) and cite [Pang et al., KDD 2019 (DevNet)](https://doi.org/10.1145/3292500.3330871) and/or the repository README as appropriate.

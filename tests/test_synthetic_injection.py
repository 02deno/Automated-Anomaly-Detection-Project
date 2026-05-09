"""Synthetic injection contracts: determinism, no-mutation, label semantics, edge cases."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import pytest

from synthetic_injection import (  # type: ignore  # added to sys.path via conftest
    SCENARIO_DEFAULTS,
    add_feature_noise,
    binary_classification_metrics,
    binary_score_metrics,
    inject,
    list_scenarios,
    merged_params,
)


SCENARIO_IDS: List[str] = list_scenarios()
NUMERIC_SCENARIOS: List[str] = [
    "spike_single",
    "joint_shift",
    "scale_burst",
    "dead_sensor",
    "sign_flip",
    "temporal_block",
]


def _numeric_frame(rows: int = 60, cols: int = 4, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(cols)])


def _mixed_frame(rows: int = 50) -> pd.DataFrame:
    df = _numeric_frame(rows=rows, cols=3, seed=42).copy()
    df["device_id"] = ["pump-A", "pump-B", "pump-C"] * (rows // 3) + ["pump-A"] * (rows - 3 * (rows // 3))
    return df


# ---------------------------------------------------------------------------
# Catalog / params
# ---------------------------------------------------------------------------
def test_list_scenarios_matches_defaults():
    assert sorted(SCENARIO_IDS) == sorted(SCENARIO_DEFAULTS.keys())
    assert len(SCENARIO_IDS) == 8


def test_merged_params_overrides():
    base = SCENARIO_DEFAULTS["spike_single"]
    merged = merged_params("spike_single", {"contamination": 0.25, "magnitude_in_std": 9.0})
    assert merged["contamination"] == 0.25
    assert merged["magnitude_in_std"] == 9.0
    assert "column" in merged  # preserved from defaults
    assert SCENARIO_DEFAULTS["spike_single"] == base, "defaults must not mutate"


def test_merged_params_unknown_scenario_raises():
    with pytest.raises(KeyError):
        merged_params("does_not_exist", {})


# ---------------------------------------------------------------------------
# inject() — determinism, no mutation, ground-truth shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", NUMERIC_SCENARIOS)
def test_inject_determinism(scenario: str):
    df = _numeric_frame()
    a_df, a_y = inject(df, scenario, random_seed=42)
    b_df, b_y = inject(df, scenario, random_seed=42)
    pd.testing.assert_frame_equal(a_df, b_df)
    np.testing.assert_array_equal(a_y, b_y)


@pytest.mark.parametrize("scenario", NUMERIC_SCENARIOS)
def test_inject_does_not_mutate_input(scenario: str):
    df = _numeric_frame()
    snapshot = df.copy(deep=True)
    inject(df, scenario, random_seed=42)
    pd.testing.assert_frame_equal(df, snapshot)


@pytest.mark.parametrize("scenario", NUMERIC_SCENARIOS)
def test_inject_y_true_shape_and_label_count(scenario: str):
    df = _numeric_frame(rows=80)
    out, y_true = inject(df, scenario, random_seed=7)
    assert out.shape == df.shape
    assert y_true.shape == (len(df),)
    assert int(y_true.sum()) >= 1
    assert set(np.unique(y_true)).issubset({0, 1})


def test_inject_indices_are_subset_of_rows():
    df = _numeric_frame(rows=40)
    _, y_true = inject(df, "spike_single", random_seed=42)
    indices = np.where(y_true == 1)[0]
    assert indices.tolist() == sorted(indices.tolist())
    assert (indices >= 0).all() and (indices < len(df)).all()


def test_inject_different_seeds_produce_different_indices():
    df = _numeric_frame(rows=200)
    _, y_a = inject(df, "spike_single", random_seed=1)
    _, y_b = inject(df, "spike_single", random_seed=2)
    assert not np.array_equal(y_a, y_b), "different seeds must yield different injection sets"


def test_inject_temporal_block_is_contiguous():
    df = _numeric_frame(rows=100)
    _, y_true = inject(df, "temporal_block", random_seed=42, params={"block_count": 1})
    indices = np.where(y_true == 1)[0]
    assert indices.size >= 2
    assert (np.diff(indices) == 1).all(), "single block_count=1 must produce contiguous indices"


def test_inject_categorical_flip_swap():
    df = _mixed_frame(rows=60)
    out, y_true = inject(df, "categorical_flip", random_seed=42)
    assert int(y_true.sum()) >= 1
    affected = np.where(y_true == 1)[0]
    diffs = (df.loc[affected, "device_id"].astype(str).to_numpy()
             != out.loc[affected, "device_id"].astype(str).to_numpy())
    assert diffs.all(), "every injected row must change the categorical value"


def test_inject_missing_value_creates_nan():
    df = _numeric_frame(rows=40)
    out, y_true = inject(df, "missing_value", random_seed=42)
    affected = np.where(y_true == 1)[0]
    assert out.loc[affected].isna().any(axis=None)


def test_inject_spike_single_changes_only_target_column():
    df = _numeric_frame(rows=50)
    target = "f0"
    out, y_true = inject(df, "spike_single", random_seed=42, params={"column": target})
    affected = np.where(y_true == 1)[0]
    diffs = (df.loc[affected, target] != out.loc[affected, target]).to_numpy()
    assert diffs.all()
    other_cols = [c for c in df.columns if c != target]
    pd.testing.assert_frame_equal(df.loc[:, other_cols], out.loc[:, other_cols])


# ---------------------------------------------------------------------------
# Gaussian noise helper
# ---------------------------------------------------------------------------
def test_add_feature_noise_determinism():
    df = _numeric_frame(rows=50)
    a = add_feature_noise(df, noise_std=0.2, random_seed=42)
    b = add_feature_noise(df, noise_std=0.2, random_seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_add_feature_noise_zero_std_returns_copy():
    df = _numeric_frame(rows=50)
    out = add_feature_noise(df, noise_std=0.0, random_seed=42)
    pd.testing.assert_frame_equal(df, out)
    assert out is not df


def test_add_feature_noise_different_seeds_differ():
    df = _numeric_frame(rows=200)
    a = add_feature_noise(df, noise_std=0.5, random_seed=1)
    b = add_feature_noise(df, noise_std=0.5, random_seed=2)
    assert not np.array_equal(a.to_numpy(), b.to_numpy())


def test_add_feature_noise_does_not_mutate_input():
    df = _numeric_frame(rows=50)
    snapshot = df.copy(deep=True)
    add_feature_noise(df, noise_std=0.3, random_seed=42)
    pd.testing.assert_frame_equal(df, snapshot)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def test_binary_classification_metrics_zero_division_safe():
    metrics = binary_classification_metrics(np.array([0, 0, 0]), np.array([0, 0, 0]))
    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_binary_score_metrics_single_class_returns_zero():
    out = binary_score_metrics(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9]))
    assert out == {"roc_auc": 0.0, "pr_auc": 0.0}


def test_binary_score_metrics_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    out = binary_score_metrics(y, s)
    assert math.isclose(out["roc_auc"], 1.0, abs_tol=1e-9)
    assert math.isclose(out["pr_auc"], 1.0, abs_tol=1e-9)

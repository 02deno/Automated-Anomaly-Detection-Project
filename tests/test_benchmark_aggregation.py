"""Aggregation contracts for the benchmark script — fast, no full pipeline run."""

from __future__ import annotations

from typing import List

from run_synthetic_benchmark import (  # type: ignore  # added to sys.path via conftest
    RunResult,
    _aggregate,
    _params_signature,
    _scenario_grid_combos,
    _worst_case_per_model,
)


def _make(
    *,
    dataset: str = "ds",
    scenario: str = "spike_single",
    score_source: str = "ensemble",
    seed: int = 42,
    f1: float = 0.8,
    roc_auc: float = 0.9,
    pr_auc: float = 0.7,
    best_f1: float = 0.85,
    best_percentile: float = 92.0,
    noise_std: float = 0.0,
) -> RunResult:
    return RunResult(
        dataset_label=dataset,
        scenario=scenario,
        score_source=score_source,
        seed=seed,
        noise_std=noise_std,
        rows=100,
        injected_count=10,
        detected_count=9,
        precision=0.8,
        recall=0.75,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        best_percentile=best_percentile,
        best_precision=0.9,
        best_recall=0.8,
        best_f1=best_f1,
        threshold=0.5,
        models_used=("iforest", "ocsvm", "lof"),
        params_signature=_params_signature(scenario, {"contamination": 0.1}, noise_std),
        params_effective={"contamination": 0.1},
    )


def test_scenario_grid_combos_empty_yields_one_combo():
    assert _scenario_grid_combos({}) == [{}]


def test_scenario_grid_combos_cartesian():
    out = _scenario_grid_combos({"a": [1, 2], "b": [10, 20]})
    assert len(out) == 4
    assert {"a": 1, "b": 10} in out
    assert {"a": 2, "b": 20} in out


def test_aggregate_collapses_seeds():
    rows: List[RunResult] = [_make(seed=1, f1=0.6), _make(seed=2, f1=0.8)]
    aggregated = _aggregate(rows)
    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["n_runs"] == 2
    assert row["f1_mean"] == 0.7
    assert row["f1_std"] > 0


def test_aggregate_keeps_distinct_groups():
    rows: List[RunResult] = [
        _make(score_source="ensemble", f1=0.8),
        _make(score_source="iforest", f1=0.6),
        _make(score_source="ensemble", f1=0.7),
    ]
    aggregated = _aggregate(rows)
    assert len(aggregated) == 2
    by_source = {r["score_source"]: r for r in aggregated}
    assert by_source["ensemble"]["n_runs"] == 2
    assert by_source["iforest"]["n_runs"] == 1


def test_worst_case_picks_lowest_f1_per_source():
    aggregated = [
        {"dataset": "ds", "scenario": "spike_single", "score_source": "ensemble",
         "f1_mean": 0.9, "f1_std": 0.0, "noise_std": 0.0,
         "params_effective": "{}", "n_runs": 2},
        {"dataset": "ds", "scenario": "dead_sensor", "score_source": "ensemble",
         "f1_mean": 0.1, "f1_std": 0.0, "noise_std": 0.0,
         "params_effective": "{}", "n_runs": 2},
        {"dataset": "ds", "scenario": "spike_single", "score_source": "iforest",
         "f1_mean": 0.5, "f1_std": 0.0, "noise_std": 0.0,
         "params_effective": "{}", "n_runs": 2},
    ]
    worst = _worst_case_per_model(aggregated)
    assert {row["score_source"] for row in worst} == {"ensemble", "iforest"}
    by_source = {row["score_source"]: row for row in worst}
    assert by_source["ensemble"]["worst_scenario"] == "dead_sensor"
    assert by_source["ensemble"]["worst_f1_mean"] == 0.1
    assert by_source["iforest"]["worst_scenario"] == "spike_single"

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.RAG_cmapss.lhi_case_adapter import _dominant_top_sensors
from src.zero_shot_cmapss.lhi_indicator import add_top_drift_sensors


WINDOW_KEY = {
    "covariate_mode": "cluster_covariate",
    "fd": "FD001",
    "unit_id": 1,
    "cutoff_cycle": 50,
    "forecast_start_cycle": 51,
}


def test_window_top_sensors_use_one_rmse_over_all_cycles() -> None:
    scores = pd.DataFrame(
        [
            {**WINDOW_KEY, "cycle": 51},
            {**WINDOW_KEY, "cycle": 52},
        ]
    )
    values = {
        "S1": [1.0, 3.0],
        "S2": [2.0, 2.0],
        "S3": [1.5, 1.5],
        "S4": [0.5, 0.5],
    }
    sensor_rows = []
    for cycle_index, cycle in enumerate([51, 52]):
        for sensor, rmses in values.items():
            sensor_rows.append(
                {
                    **WINDOW_KEY,
                    "cycle": cycle,
                    "sensor": sensor,
                    "sensor_d_rmse": rmses[cycle_index],
                    "sensor_d_mae": rmses[cycle_index],
                    "n": 1,
                    "healthy_condition_n": 50,
                    "past_range": 1.0,
                }
            )

    ranked_scores, top_rows = add_top_drift_sensors(
        scores,
        pd.DataFrame(sensor_rows),
        top_k=2,
    )

    assert ranked_scores["window_top_drift_sensors"].unique().tolist() == ["S1,S2,S3"]
    values_by_sensor = _parse_value_map(ranked_scores.iloc[0]["window_top_sensor_rmse_values"])
    assert values_by_sensor["S1"] == pytest.approx(math.sqrt(5.0), rel=1e-5)
    assert values_by_sensor["S2"] == pytest.approx(2.0)
    assert values_by_sensor["S3"] == pytest.approx(1.5)
    assert "_sensor_d_squared_sum" not in top_rows.columns


def test_window_rmse_combines_preaggregated_rows_using_n() -> None:
    scores = pd.DataFrame(
        [
            {**WINDOW_KEY, "cycle": 51},
            {**WINDOW_KEY, "cycle": 52},
        ]
    )
    sensor_scores = pd.DataFrame(
        [
            {
                **WINDOW_KEY,
                "cycle": 51,
                "sensor": "S1",
                "sensor_d_rmse": 1.0,
                "sensor_d_mae": 1.0,
                "n": 3,
                "healthy_condition_n": 50,
                "past_range": 1.0,
            },
            {
                **WINDOW_KEY,
                "cycle": 52,
                "sensor": "S1",
                "sensor_d_rmse": 3.0,
                "sensor_d_mae": 3.0,
                "n": 1,
                "healthy_condition_n": 50,
                "past_range": 1.0,
            },
            {
                **WINDOW_KEY,
                "cycle": 51,
                "sensor": "S2",
                "sensor_d_rmse": 1.8,
                "sensor_d_mae": 1.8,
                "n": 1,
                "healthy_condition_n": 50,
                "past_range": 1.0,
            },
            {
                **WINDOW_KEY,
                "cycle": 52,
                "sensor": "S2",
                "sensor_d_rmse": 1.8,
                "sensor_d_mae": 1.8,
                "n": 1,
                "healthy_condition_n": 50,
                "past_range": 1.0,
            },
        ]
    )

    ranked_scores, _ = add_top_drift_sensors(scores, sensor_scores, top_k=1, window_top_k=2)

    # S1: sqrt((1^2 * 3 + 3^2 * 1) / 4) = sqrt(3), so S2 ranks first.
    assert ranked_scores.iloc[0]["window_top_drift_sensors"] == "S2,S1"


def test_case_builder_prefers_persisted_window_ranking() -> None:
    window = pd.DataFrame(
        {
            "top_drift_sensors": ["S1,S2,S3", "S1,S2,S3"],
            "window_top_drift_sensors": ["S3,S2,S1", "S3,S2,S1"],
        }
    )

    assert _dominant_top_sensors(pd.DataFrame(), window) == ["S3", "S2", "S1"]


def test_case_builder_rejects_inconsistent_window_rankings() -> None:
    window = pd.DataFrame(
        {
            "window_top_drift_sensors": ["S1,S2,S3", "S2,S1,S3"],
        }
    )

    with pytest.raises(ValueError, match="inconsistent window_top_drift_sensors"):
        _dominant_top_sensors(pd.DataFrame(), window)


def _parse_value_map(value: str) -> dict[str, float]:
    return {
        sensor: float(raw)
        for item in value.split(";")
        for sensor, raw in [item.split(":", 1)]
    }

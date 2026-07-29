from __future__ import annotations

import numpy as np
import pandas as pd

from src.zero_shot_cmapss.chronos2_cmapss_forecast import (
    SENSOR_COLUMNS,
    iter_windows,
    select_metric_windows,
    summarize_metrics,
)
from src.zero_shot_cmapss.lhi_indicator import add_top_drift_sensors, compute_lhi_scores


def _unit_frame() -> pd.DataFrame:
    rows = []
    for idx in range(8):
        historical_condition = idx % 2
        settings = (
            (0.0, 0.0, 100.0)
            if historical_condition == 0
            else (10.0, 0.2, 100.0)
        )
        if idx >= 4:
            # These future settings must never enter a forecasting task.
            settings = (999.0, 9.99, 999.0)
        row = {
            "unit_id": 1,
            "cycle": idx + 1,
            "setting1": settings[0],
            "setting2": settings[1],
            "setting3": settings[2],
        }
        row.update({sensor: float(idx + sensor_idx) for sensor_idx, sensor in enumerate(SENSOR_COLUMNS)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_each_historical_condition_gets_full_horizon_without_future_settings() -> None:
    frame = _unit_frame()
    tasks = list(
        iter_windows(
            fd_name="FD002",
            eval_df=frame,
            model_input_df=frame,
            sensors=["s2", "s3"],
            covariate_mode="cluster_covariate",
            target_transform="none",
            context_length=0,
            prediction_length=3,
            stride=1,
            forecast_start_cycle=5,
            forecast_end_cycle=5,
        )
    )
    assert len(tasks) == 2
    assert {task[0]["op_condition_key"] for task in tasks} == {"0|0|100", "10|20|100"}
    for meta, chronos_input, truth, transform in tasks:
        assert meta["condition_source"] == "past_context_only"
        assert meta["n_condition_forecast_tasks"] == 2
        assert meta["condition_prediction_points"] == 6
        assert meta["group_prediction_length"] == 3
        assert truth.shape == (2, 3)
        assert transform["future_horizons"].tolist() == [1, 2, 3]
        assert transform["metric_op_condition_keys"].tolist() == [
            "999|999|999",
            "999|999|999",
            "999|999|999",
        ]
        for values in chronos_input["future_covariates"].values():
            assert len(values) == 3
            assert np.unique(values).size == 1
            assert 999.0 not in values


def test_metrics_keep_only_realized_condition_prediction() -> None:
    rows = []
    realized_keys = ["0|0|100", "10|20|100", "0|0|100"]
    for condition_idx, condition_key in enumerate(["0|0|100", "10|20|100"]):
        for horizon, realized_key in enumerate(realized_keys, start=1):
            rows.append(
                {
                    "covariate_mode": "cluster_covariate",
                    "fd": "FD002",
                    "unit_id": 1,
                    "forecast_start_cycle": 5,
                    "sensor": "s2",
                    "horizon": horizon,
                    "has_ground_truth": 1,
                    "op_condition_key": condition_key,
                    "metric_op_condition_key": realized_key,
                    "is_metric_condition_match": int(condition_key == realized_key),
                    "y_true": 10.0 * horizon,
                    "y_pred": 10.0 * horizon + condition_idx + 1.0,
                }
            )

    selected = select_metric_windows(pd.DataFrame(rows), prediction_length=3)
    assert len(selected) == 3
    assert selected["horizon"].nunique() == 3
    assert (
        selected["op_condition_key"] == selected["metric_op_condition_key"]
    ).all()

    metrics = summarize_metrics(selected)
    sensor_metrics = metrics[metrics["sensor"] == "s2"].iloc[0]
    # Realized-condition errors are [1, 2, 1], not all six scenario errors.
    assert sensor_metrics["n"] == 3
    assert np.isclose(sensor_metrics["mae"], 4.0 / 3.0)
    assert np.isclose(sensor_metrics["mse"], 2.0)


def test_metrics_drop_window_when_realized_condition_has_no_forecast() -> None:
    rows = pd.DataFrame(
        [
            {
                "covariate_mode": "cluster_covariate",
                "fd": "FD002",
                "unit_id": 1,
                "forecast_start_cycle": 5,
                "sensor": "s2",
                "horizon": horizon,
                "has_ground_truth": 1,
                "is_metric_condition_match": int(horizon < 3),
                "y_true": float(horizon),
                "y_pred": float(horizon),
            }
            for horizon in range(1, 4)
        ]
    )
    selected = select_metric_windows(rows, prediction_length=3)
    assert selected.empty


def test_lhi_uses_all_condition_predictions_at_each_horizon() -> None:
    forecasts = []
    for condition_idx, condition_key in enumerate(["0|0|100", "10|20|100"]):
        for horizon in range(1, 4):
            for sensor_idx, sensor in enumerate(["s2", "s3"]):
                forecasts.append(
                    {
                        "covariate_mode": "cluster_covariate",
                        "fd": "FD002",
                        "unit_id": 1,
                        "context_start_cycle": 1,
                        "cutoff_cycle": 4,
                        "forecast_start_cycle": 5,
                        "cycle": 4 + horizon,
                        "sensor": sensor,
                        "op_condition_key": condition_key,
                        "y_pred": float(condition_idx + sensor_idx + horizon),
                        "prediction_length": 3,
                        "n_condition_forecast_tasks": 2,
                        "condition_prediction_points": 6,
                    }
                )
    forecasts = pd.DataFrame(forecasts)
    ranges = pd.DataFrame(
        [
            {
                "covariate_mode": "cluster_covariate",
                "fd": "FD002",
                "unit_id": 1,
                "context_start_cycle": 1,
                "cutoff_cycle": 4,
                "sensor": sensor,
                "past_min": 0.0,
                "past_range": 10.0,
                "range_usable": True,
            }
            for sensor in ["s2", "s3"]
        ]
    )
    condition_means = pd.DataFrame(
        [
            {
                "fd": "FD002",
                "unit_id": 1,
                "op_condition_key": condition_key,
                "sensor": sensor,
                "healthy_condition_mean_raw": 0.0,
                "healthy_condition_n": 10,
            }
            for condition_key in ["0|0|100", "10|20|100"]
            for sensor in ["s2", "s3"]
        ]
    )
    scores, sensor_scores = compute_lhi_scores(forecasts, ranges, condition_means)
    assert len(scores) == 3
    assert set(scores["row_count"]) == {4}
    assert set(scores["condition_count"]) == {2}
    assert set(scores["condition_prediction_points"]) == {6}
    assert set(sensor_scores["n"]) == {2}

    _, top_rows = add_top_drift_sensors(scores, sensor_scores, top_k=2, window_top_k=2)
    # Each sensor RMSE combines 2 conditions x 3 horizon positions.
    assert set(
        top_rows.groupby("sensor")["n"].sum()
    ) == {6}

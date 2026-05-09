from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .lhi_indicator import (
        add_lhi_columns,
        add_rolling_scores,
        add_top_drift_sensors,
        build_condition_means,
        compute_lhi_scores,
        compute_initial_forecast_baselines,
        compute_past_sensor_ranges,
        load_eval_frames,
        summarize_fd,
    )
    from .plot_operating_condition_clusters import DEFAULT_SENSORS, FD_NAMES
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.zero_shot_cmapss.lhi_indicator import (
        add_lhi_columns,
        add_rolling_scores,
        add_top_drift_sensors,
        build_condition_means,
        compute_lhi_scores,
        compute_initial_forecast_baselines,
        compute_past_sensor_ranges,
        load_eval_frames,
        summarize_fd,
    )
    from src.zero_shot_cmapss.plot_operating_condition_clusters import DEFAULT_SENSORS, FD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the same condition-matched log-ratio LHI as lhi_indicator.py, "
            "but directly on observed C-MAPSS sensor values instead of forecast outputs."
        )
    )
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/CMAPSS/raw_lhi"))
    parser.add_argument("--eval_split", choices=["train", "test"], default="train")
    parser.add_argument("--fds", nargs="+", default=list(FD_NAMES), choices=list(FD_NAMES))
    parser.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS)
    parser.add_argument("--healthy_cycles", type=int, default=50)
    parser.add_argument(
        "--min_context_cycles",
        type=int,
        default=None,
        help=(
            "Optional minimum observed history length required before scoring a raw window. "
            "Default is no extra filter, matching lhi_indicator.py on window_forecasts.csv."
        ),
    )
    parser.add_argument("--range_epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--minmax_scope",
        choices=["past_context", "past_and_forecast"],
        default="past_and_forecast",
        help=(
            "Values used to compute each raw window's min-max normalization. past_context uses only "
            "observed history up to cutoff_cycle. past_and_forecast uses observed history plus the "
            "current raw target horizon values."
        ),
    )
    parser.add_argument("--lhi_epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--baseline_cycles",
        type=int,
        default=0,
        help=(
            "Number of initial post-healthy observed cycles used to calibrate B. "
            "0 means use the first observed monitor block when --baseline_source post_healthy_block."
        ),
    )
    parser.add_argument(
        "--baseline_source",
        choices=["post_healthy_block", "healthy_target_cycles"],
        default="healthy_target_cycles",
        help=(
            "healthy_target_cycles uses raw observed cycles 1..--healthy_cycles as B, "
            "independent of the scored target window start. post_healthy_block keeps the older "
            "initial observed monitor block baseline."
        ),
    )
    parser.add_argument("--rolling_window", type=int, default=5)
    parser.add_argument("--top_k_sensors", type=int, default=5)
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=20,
        help="Raw-observed target horizon length. Use 20 to match the forecast LHI experiment.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=20,
        help=(
            "Cutoff stride between raw-observed windows. Default 20 gives non-overlapping raw "
            "horizons; forecast runs may use stride=1 for denser experimental evaluation."
        ),
    )
    parser.add_argument(
        "--forecast_start_cycle",
        type=int,
        default=20,
        help="First absolute target cycle to score. Default 20 matches outputs/CMAPSS/cluster_20.",
    )
    return parser.parse_args()


def build_raw_observation_rows(
    frames: dict[str, pd.DataFrame],
    sensors: Sequence[str],
    prediction_length: int,
    stride: int,
    min_context_cycles: int,
    forecast_start_cycle: int,
) -> pd.DataFrame:
    if prediction_length < 1:
        raise ValueError("--prediction_length must be >= 1.")
    if stride < 1:
        raise ValueError("--stride must be >= 1.")
    rows = []
    for fd_name, frame in frames.items():
        for unit_id, unit_df in frame.groupby("unit_id", sort=True):
            unit_df = unit_df.sort_values("cycle").copy()
            context_start = int(unit_df["cycle"].min())
            max_cycle = int(unit_df["cycle"].max())
            first_cutoff = max(context_start, int(forecast_start_cycle) - 1)
            if min_context_cycles > 0:
                first_cutoff = max(first_cutoff, context_start + int(min_context_cycles) - 1)
            for cutoff_cycle in range(first_cutoff, max_cycle, stride):
                forecast_start = cutoff_cycle + 1
                forecast_end = cutoff_cycle + prediction_length
                horizon = unit_df[
                    (unit_df["cycle"] >= forecast_start) & (unit_df["cycle"] <= forecast_end)
                ].copy()
                if horizon.empty:
                    continue
                horizon["fd"] = fd_name
                horizon["covariate_mode"] = "raw_observed"
                horizon["context_start_cycle"] = context_start
                horizon["cutoff_cycle"] = cutoff_cycle
                horizon["forecast_start_cycle"] = forecast_start
                horizon["prediction_length"] = prediction_length
                melted = horizon.melt(
                    id_vars=[
                        "covariate_mode",
                        "fd",
                        "unit_id",
                        "context_start_cycle",
                        "cutoff_cycle",
                        "forecast_start_cycle",
                        "cycle",
                        "op_condition_key",
                        "prediction_length",
                    ],
                    value_vars=list(sensors),
                    var_name="sensor",
                    value_name="y_pred",
                )
                rows.append(melted)
    if not rows:
        raise ValueError("No raw observation rows were built.")
    return pd.concat(rows, ignore_index=True)


def compute_healthy_target_cycle_baselines(
    frames: dict[str, pd.DataFrame],
    condition_means: pd.DataFrame,
    sensors: Sequence[str],
    healthy_cycles: int,
    range_epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fd_name, frame in frames.items():
        healthy_frame = frame[frame["cycle"] <= int(healthy_cycles)].copy()
        if healthy_frame.empty:
            continue
        for unit_id, unit_df in healthy_frame.groupby("unit_id", sort=True):
            unit_df = unit_df.sort_values("cycle").copy()
            sensor_values = unit_df.loc[:, sensors]
            sensor_min = sensor_values.min()
            sensor_range = sensor_values.max() - sensor_min
            melted = unit_df.melt(
                id_vars=["unit_id", "cycle", "op_condition_key"],
                value_vars=list(sensors),
                var_name="sensor",
                value_name="y_pred",
            )
            melted["fd"] = fd_name
            keyed = melted.merge(
                condition_means,
                on=["fd", "unit_id", "op_condition_key", "sensor"],
                how="left",
                validate="many_to_one",
            )
            keyed["past_min"] = keyed["sensor"].map(sensor_min.to_dict())
            keyed["past_range"] = keyed["sensor"].map(sensor_range.to_dict())
            keyed = keyed[
                keyed["healthy_condition_mean_raw"].notna()
                & keyed["past_range"].notna()
                & (keyed["past_range"] > float(range_epsilon))
            ].copy()
            if keyed.empty:
                continue
            keyed["y_pred_minmax"] = (keyed["y_pred"] - keyed["past_min"]) / keyed["past_range"]
            keyed["healthy_condition_mean_minmax"] = (
                keyed["healthy_condition_mean_raw"] - keyed["past_min"]
            ) / keyed["past_range"]
            keyed["drift"] = (keyed["y_pred_minmax"] - keyed["healthy_condition_mean_minmax"]).abs()
            keyed["drift_sq"] = keyed["drift"] ** 2
            scores = (
                keyed.groupby(["fd", "unit_id", "cycle"], sort=True)
                .agg(
                    d_mae=("drift", "mean"),
                    d_rmse=("drift_sq", lambda s: float(np.sqrt(np.mean(s)))),
                    sensor_count=("sensor", "nunique"),
                    row_count=("drift", "size"),
                )
                .reset_index()
            )
            scores["covariate_mode"] = "raw_observed"
            scores["cutoff_cycle"] = int(healthy_cycles)
            scores["forecast_start_cycle"] = 1
            rows.append(scores)
    if not rows:
        raise ValueError(f"No healthy baseline scores were computed from cycles 1..{healthy_cycles}.")
    healthy = pd.concat(rows, ignore_index=True)
    point_cols = [
        "covariate_mode",
        "fd",
        "unit_id",
        "cutoff_cycle",
        "forecast_start_cycle",
        "cycle",
        "d_mae",
        "d_rmse",
        "sensor_count",
        "row_count",
    ]
    baseline_points = healthy[point_cols].assign(
        baseline_source=f"raw_observed_cycles_1_to_{int(healthy_cycles)}"
    )
    baselines = (
        healthy.groupby(["covariate_mode", "fd", "unit_id"], sort=True)
        .agg(
            b_mae=("d_mae", "mean"),
            b_rmse=("d_rmse", "mean"),
            baseline_start_cycle=("cycle", "min"),
            baseline_end_cycle=("cycle", "max"),
            baseline_rows=("cycle", "size"),
        )
        .reset_index()
    )
    baselines["baseline_source"] = f"raw_observed_cycles_1_to_{int(healthy_cycles)}"
    return baseline_points, baselines


def main() -> None:
    args = parse_args()
    if args.range_epsilon <= 0:
        raise ValueError("--range_epsilon must be positive.")
    if args.lhi_epsilon <= 0:
        raise ValueError("--lhi_epsilon must be positive.")
    if args.top_k_sensors < 1:
        raise ValueError("--top_k_sensors must be >= 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_eval_frames(args.data_dir, args.eval_split, args.fds)
    condition_means = build_condition_means(frames, args.sensors, args.healthy_cycles)
    min_context_cycles = int(args.min_context_cycles or 0)
    raw_observations = build_raw_observation_rows(
        frames,
        args.sensors,
        prediction_length=args.prediction_length,
        stride=args.stride,
        min_context_cycles=min_context_cycles,
        forecast_start_cycle=args.forecast_start_cycle,
    )
    if min_context_cycles > 0:
        raw_observations = raw_observations[
            (raw_observations["cutoff_cycle"] - raw_observations["context_start_cycle"] + 1) >= min_context_cycles
        ].copy()
    if raw_observations.empty:
        raise ValueError(f"No raw observation rows remain after --min_context_cycles {min_context_cycles}.")
    past_ranges = compute_past_sensor_ranges(
        frames=frames,
        forecasts=raw_observations,
        sensors=args.sensors,
        range_epsilon=args.range_epsilon,
        minmax_scope=args.minmax_scope,
    )
    all_scores, sensor_scores = compute_lhi_scores(
        forecasts=raw_observations,
        past_ranges=past_ranges,
        condition_means=condition_means,
    )
    if all_scores.empty:
        raise ValueError("No raw LHI score rows were computed.")

    if args.baseline_source == "healthy_target_cycles":
        baseline_points, baselines = compute_healthy_target_cycle_baselines(
            frames=frames,
            condition_means=condition_means,
            sensors=args.sensors,
            healthy_cycles=args.healthy_cycles,
            range_epsilon=args.range_epsilon,
        )
    else:
        baseline_points, baselines = compute_initial_forecast_baselines(
            all_scores,
            healthy_cycles=args.healthy_cycles,
            baseline_cycles=args.baseline_cycles,
        )
    scores = add_lhi_columns(all_scores, baselines, args.lhi_epsilon)
    scores, top_drift_sensors = add_top_drift_sensors(scores, sensor_scores, args.top_k_sensors)
    scores = add_rolling_scores(scores, args.rolling_window)
    fd_summary = summarize_fd(scores)

    past_ranges.to_csv(args.output_dir / "past_minmax_ranges.csv", index=False)
    condition_means.to_csv(args.output_dir / "healthy_condition_means.csv", index=False)
    raw_observations.to_csv(args.output_dir / "raw_observation_rows.csv", index=False)
    baseline_points.to_csv(args.output_dir / "baseline_forecast_points.csv", index=False)
    baselines.to_csv(args.output_dir / "lhi_baselines.csv", index=False)
    scores.to_csv(args.output_dir / "lhi_scores.csv", index=False)
    top_drift_sensors.to_csv(args.output_dir / "top_drift_sensors.csv", index=False)
    sensor_scores.to_csv(args.output_dir / "sensor_lhi_components.csv", index=False)
    fd_summary.to_csv(args.output_dir / "fd_lhi_summary.csv", index=False)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    usable_ranges = past_ranges["range_usable"].mean()
    print(f"Usable raw min-max ranges: {usable_ranges:.2%}", flush=True)
    print(
        "Raw LHI summary by FD:\n"
        + scores.groupby(["covariate_mode", "fd"], sort=True)
        .agg(
            units=("unit_id", "nunique"),
            cycles=("cycle", "nunique"),
            median_b_rmse=("b_rmse", "median"),
            median_lhi_rmse=("lhi_rmse", "median"),
            q90_lhi_rmse=("lhi_rmse", lambda s: float(s.quantile(0.90))),
        )
        .reset_index()
        .to_string(index=False),
        flush=True,
    )
    print(f"Saved raw LHI outputs to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

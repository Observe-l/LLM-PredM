from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .plot_operating_condition_clusters import DEFAULT_SENSORS, FD_NAMES, load_cmapss_file, make_condition_keys
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.zero_shot_cmapss.plot_operating_condition_clusters import (
        DEFAULT_SENSORS,
        FD_NAMES,
        load_cmapss_file,
        make_condition_keys,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a leakage-free log-ratio health indicator from condition-matched "
            "min-max drift."
        )
    )
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--forecast_dir", type=Path, default=Path("outputs/cluster_20"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--fds", nargs="+", default=list(FD_NAMES), choices=list(FD_NAMES))
    parser.add_argument(
        "--covariate_modes",
        nargs="+",
        default=None,
        help="Covariate modes to evaluate. Defaults to all modes present in window_forecasts.csv.",
    )
    parser.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS)
    parser.add_argument(
        "--healthy_cycles",
        type=int,
        default=50,
        help="Cycles <= this value define the unit-specific healthy reference.",
    )
    parser.add_argument(
        "--range_epsilon",
        type=float,
        default=1e-6,
        help="Minimum min-max range required for a sensor to be used.",
    )
    parser.add_argument(
        "--minmax_scope",
        choices=["past_context", "past_and_forecast"],
        default="past_and_forecast",
        help=(
            "Values used to compute each window's min-max normalization. past_context uses only "
            "observed history up to cutoff_cycle. past_and_forecast uses observed history plus "
            "the current forecast/raw target horizon values."
        ),
    )
    parser.add_argument(
        "--lhi_epsilon",
        type=float,
        default=1e-6,
        help="Small positive epsilon in LHI = log((D + eps) / (B + eps)).",
    )
    parser.add_argument(
        "--baseline_cycles",
        type=int,
        default=0,
        help=(
            "Number of initial monitor target cycles used to calibrate B. "
            "0 means use the first forecast block after --healthy_cycles."
        ),
    )
    parser.add_argument(
        "--baseline_source",
        choices=["raw_healthy_cycles", "initial_forecast_block"],
        default="raw_healthy_cycles",
        help=(
            "Source used to calibrate B. raw_healthy_cycles uses observed raw cycles "
            "1..--healthy_cycles and is independent of forecast outputs. "
            "initial_forecast_block uses the original forecast-derived monitor baseline."
        ),
    )
    parser.add_argument("--rolling_window", type=int, default=5)
    parser.add_argument(
        "--top_k_sensors",
        type=int,
        default=5,
        help="Number of highest-drift sensors to report for each LHI point.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per CSV chunk when streaming window_forecasts.csv. Use 0 to run the legacy in-memory path.",
    )
    return parser.parse_args()


def load_window_forecasts(args: argparse.Namespace) -> pd.DataFrame:
    path = args.forecast_dir / "window_forecasts.csv"
    required = {
        "covariate_mode",
        "fd",
        "unit_id",
        "cutoff_cycle",
        "forecast_start_cycle",
        "context_start_cycle",
        "cycle",
        "sensor",
        "y_true",
        "y_pred",
        "op_condition_key",
        "prediction_length",
    }
    forecasts = pd.read_csv(path, usecols=sorted(required))
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    mode_filter = forecasts["covariate_mode"].isin(args.covariate_modes) if args.covariate_modes else True
    forecasts = forecasts[
        forecasts["fd"].isin(args.fds)
        & mode_filter
        & forecasts["sensor"].isin(args.sensors)
    ].copy()
    if forecasts.empty:
        raise ValueError(f"No forecast rows left after filtering {path}.")
    return forecasts


def iter_window_forecast_chunks(args: argparse.Namespace):
    path = args.forecast_dir / "window_forecasts.csv"
    required = {
        "covariate_mode",
        "fd",
        "unit_id",
        "cutoff_cycle",
        "forecast_start_cycle",
        "context_start_cycle",
        "cycle",
        "sensor",
        "y_pred",
        "op_condition_key",
        "prediction_length",
    }
    reader = pd.read_csv(path, usecols=sorted(required), chunksize=args.chunksize)
    for chunk in reader:
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        mode_filter = chunk["covariate_mode"].isin(args.covariate_modes) if args.covariate_modes else True
        chunk = chunk[
            chunk["fd"].isin(args.fds)
            & mode_filter
            & chunk["sensor"].isin(args.sensors)
        ].copy()
        if not chunk.empty:
            yield chunk


def load_eval_frames(data_dir: Path, eval_split: str, fds: Sequence[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for fd_name in fds:
        frame = load_cmapss_file(data_dir / fd_name / f"{eval_split}_{fd_name}.txt")
        frame = frame.copy()
        frame["op_condition_key"] = make_condition_keys(frame)
        frames[fd_name] = frame
    return frames


def build_condition_means(
    frames: dict[str, pd.DataFrame],
    sensors: Sequence[str],
    healthy_cycles: int,
) -> pd.DataFrame:
    rows = []
    for fd_name, frame in frames.items():
        healthy = frame[frame["cycle"] <= healthy_cycles].copy()
        if healthy.empty:
            continue
        melted = healthy.melt(
            id_vars=["unit_id", "cycle", "op_condition_key"],
            value_vars=list(sensors),
            var_name="sensor",
            value_name="y_true",
        )
        melted["fd"] = fd_name
        rows.append(melted)
    if not rows:
        raise ValueError(f"No healthy rows found for cycle <= {healthy_cycles}.")
    healthy_actuals = pd.concat(rows, ignore_index=True)
    condition_means = (
        healthy_actuals.groupby(["fd", "unit_id", "op_condition_key", "sensor"], sort=True)
        .agg(
            healthy_condition_mean_raw=("y_true", "mean"),
            healthy_condition_n=("y_true", "size"),
        )
        .reset_index()
    )
    return condition_means


def compute_past_sensor_ranges(
    frames: dict[str, pd.DataFrame],
    forecasts: pd.DataFrame,
    sensors: Sequence[str],
    range_epsilon: float,
    minmax_scope: str = "past_and_forecast",
) -> pd.DataFrame:
    if minmax_scope not in {"past_context", "past_and_forecast"}:
        raise ValueError("--minmax_scope must be one of: past_context, past_and_forecast.")
    rows = []
    mode_cols = ["covariate_mode"] if "covariate_mode" in forecasts.columns else []
    key_cols = [*mode_cols, "fd", "unit_id", "context_start_cycle", "cutoff_cycle"]
    window_keys = (
        forecasts[key_cols]
        .drop_duplicates()
        .sort_values(key_cols)
    )
    forecast_unit_groups = {
        (str(fd_name), int(unit_id)): group
        for (fd_name, unit_id), group in forecasts.groupby(["fd", "unit_id"], sort=False)
    }
    for (fd_name, unit_id), key_group in window_keys.groupby(["fd", "unit_id"], sort=True):
        unit_frame = (
            frames[str(fd_name)][frames[str(fd_name)]["unit_id"] == int(unit_id)]
            .sort_values("cycle")
            .set_index("cycle", drop=False)
        )
        if unit_frame.empty:
            raise ValueError(f"No rows for {fd_name} unit {unit_id}.")

        sensor_values = unit_frame.loc[:, sensors]
        uses_expanding_context = minmax_scope == "past_context" and (
            key_group["context_start_cycle"].nunique() == 1
            and int(key_group["context_start_cycle"].iloc[0]) == int(unit_frame["cycle"].min())
        )
        if uses_expanding_context:
            expanding_min = sensor_values.expanding().min()
            expanding_max = sensor_values.expanding().max()

        unit_forecasts = forecast_unit_groups[(str(fd_name), int(unit_id))]
        for key_values, forecast_group in unit_forecasts.groupby(key_cols, sort=True):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            key = dict(zip(key_cols, key_values))
            context_start_cycle = int(key["context_start_cycle"])
            cutoff_cycle = int(key["cutoff_cycle"])
            if cutoff_cycle not in unit_frame.index:
                raise ValueError(f"Missing cutoff cycle {cutoff_cycle} for {fd_name} unit {unit_id}.")
            if uses_expanding_context:
                minimums = expanding_min.loc[cutoff_cycle]
                maximums = expanding_max.loc[cutoff_cycle]
                past_n = int(cutoff_cycle - context_start_cycle + 1)
            else:
                past = sensor_values.loc[context_start_cycle:cutoff_cycle]
                if past.empty:
                    raise ValueError(
                        f"No past rows for {fd_name} unit {unit_id}, "
                        f"context_start_cycle={context_start_cycle}, cutoff_cycle={cutoff_cycle}."
                    )
                minimums = past.min()
                maximums = past.max()
                past_n = int(len(past))

            for sensor in sensors:
                minimum = float(minimums[sensor])
                maximum = float(maximums[sensor])
                forecast_n = 0
                if minmax_scope == "past_and_forecast":
                    sensor_forecast = forecast_group[forecast_group["sensor"] == sensor]["y_pred"]
                    forecast_n = int(sensor_forecast.notna().sum())
                    if forecast_n > 0:
                        minimum = min(minimum, float(sensor_forecast.min()))
                        maximum = max(maximum, float(sensor_forecast.max()))
                value_range = maximum - minimum
                row = {
                    "fd": fd_name,
                    "unit_id": int(unit_id),
                    "context_start_cycle": context_start_cycle,
                    "cutoff_cycle": cutoff_cycle,
                    "sensor": sensor,
                    "past_min": minimum,
                    "past_max": maximum,
                    "past_range": value_range,
                    "range_usable": bool(np.isfinite(value_range) and value_range > range_epsilon),
                    "past_n": past_n,
                    "forecast_n": forecast_n,
                    "minmax_scope": minmax_scope,
                }
                for col in mode_cols:
                    row[col] = key[col]
                rows.append(row)
    return pd.DataFrame(rows)


def compute_lhi_scores(
    forecasts: pd.DataFrame,
    past_ranges: pd.DataFrame,
    condition_means: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    range_key_cols = ["fd", "unit_id", "context_start_cycle", "cutoff_cycle", "sensor"]
    if "covariate_mode" in past_ranges.columns:
        range_key_cols = ["covariate_mode", *range_key_cols]
    keyed = forecasts.merge(
        past_ranges,
        on=range_key_cols,
        how="left",
        validate="many_to_one",
    )
    if keyed["past_min"].isna().any() or keyed["past_range"].isna().any():
        raise ValueError("Missing min-max ranges for some forecast rows.")
    keyed = keyed[keyed["range_usable"]].copy()
    keyed["y_pred_minmax"] = (keyed["y_pred"] - keyed["past_min"]) / keyed["past_range"]
    keyed = keyed.merge(
        condition_means,
        on=["fd", "unit_id", "op_condition_key", "sensor"],
        how="left",
        validate="many_to_one",
    )
    keyed["condition_reference_available"] = keyed["healthy_condition_mean_raw"].notna()
    available = keyed[keyed["condition_reference_available"]].copy()
    available["healthy_condition_mean_minmax"] = (
        available["healthy_condition_mean_raw"] - available["past_min"]
    ) / available["past_range"]
    available["drift"] = (available["y_pred_minmax"] - available["healthy_condition_mean_minmax"]).abs()
    available["drift_sq"] = available["drift"] ** 2

    score_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"]
    sensor_scores = (
        available.groupby(score_cols + ["sensor"], sort=True)
        .agg(
            sensor_d_mae=("drift", "mean"),
            sensor_d_rmse=("drift_sq", lambda s: float(np.sqrt(np.mean(s)))),
            n=("drift", "size"),
            healthy_condition_n=("healthy_condition_n", "min"),
            past_range=("past_range", "first"),
        )
        .reset_index()
    )
    scores = (
        available.groupby(score_cols, sort=True)
        .agg(
            d_mae=("drift", "mean"),
            d_rmse=("drift_sq", lambda s: float(np.sqrt(np.mean(s)))),
            sensor_count=("sensor", "nunique"),
            row_count=("drift", "size"),
            prediction_length=("prediction_length", "max"),
        )
        .reset_index()
    )
    return scores, sensor_scores


def aggregate_lhi_chunk(
    forecasts: pd.DataFrame,
    past_ranges: pd.DataFrame,
    condition_means: pd.DataFrame,
) -> pd.DataFrame:
    range_key_cols = ["fd", "unit_id", "context_start_cycle", "cutoff_cycle", "sensor"]
    if "covariate_mode" in past_ranges.columns:
        range_key_cols = ["covariate_mode", *range_key_cols]
    keyed = forecasts.merge(
        past_ranges,
        on=range_key_cols,
        how="left",
        validate="many_to_one",
    )
    if keyed["past_min"].isna().any() or keyed["past_range"].isna().any():
        raise ValueError("Missing min-max ranges for some forecast rows.")
    keyed = keyed[keyed["range_usable"]].copy()
    keyed["y_pred_minmax"] = (keyed["y_pred"] - keyed["past_min"]) / keyed["past_range"]
    keyed = keyed.merge(
        condition_means,
        on=["fd", "unit_id", "op_condition_key", "sensor"],
        how="left",
        validate="many_to_one",
    )
    keyed = keyed[keyed["healthy_condition_mean_raw"].notna()].copy()
    keyed["healthy_condition_mean_minmax"] = (
        keyed["healthy_condition_mean_raw"] - keyed["past_min"]
    ) / keyed["past_range"]
    keyed["drift"] = (keyed["y_pred_minmax"] - keyed["healthy_condition_mean_minmax"]).abs()
    keyed["drift_sq"] = keyed["drift"] ** 2

    score_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"]
    return (
        keyed.groupby(score_cols + ["sensor"], sort=True)
        .agg(
            drift_sum=("drift", "sum"),
            drift_sq_sum=("drift_sq", "sum"),
            n=("drift", "size"),
            healthy_condition_n=("healthy_condition_n", "min"),
            past_range=("past_range", "first"),
            prediction_length=("prediction_length", "max"),
        )
        .reset_index()
    )


def finalize_lhi_partials(partials: Sequence[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not partials:
        raise ValueError("No LHI partial aggregates were produced.")
    partial = pd.concat(partials, ignore_index=True)
    score_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"]
    sensor_scores = (
        partial.groupby(score_cols + ["sensor"], sort=True)
        .agg(
            drift_sum=("drift_sum", "sum"),
            drift_sq_sum=("drift_sq_sum", "sum"),
            n=("n", "sum"),
            healthy_condition_n=("healthy_condition_n", "min"),
            past_range=("past_range", "first"),
            prediction_length=("prediction_length", "max"),
        )
        .reset_index()
    )
    sensor_scores["sensor_d_mae"] = sensor_scores["drift_sum"] / sensor_scores["n"]
    sensor_scores["sensor_d_rmse"] = np.sqrt(sensor_scores["drift_sq_sum"] / sensor_scores["n"])

    score_partials = (
        sensor_scores.groupby(score_cols, sort=True)
        .agg(
            drift_sum=("drift_sum", "sum"),
            drift_sq_sum=("drift_sq_sum", "sum"),
            row_count=("n", "sum"),
            sensor_count=("sensor", "nunique"),
            prediction_length=("prediction_length", "max"),
        )
        .reset_index()
    )
    score_partials["d_mae"] = score_partials["drift_sum"] / score_partials["row_count"]
    score_partials["d_rmse"] = np.sqrt(score_partials["drift_sq_sum"] / score_partials["row_count"])
    scores = score_partials[
        [*score_cols, "d_mae", "d_rmse", "sensor_count", "row_count", "prediction_length"]
    ].copy()
    sensor_scores = sensor_scores[
        [
            *score_cols,
            "sensor",
            "sensor_d_mae",
            "sensor_d_rmse",
            "n",
            "healthy_condition_n",
            "past_range",
        ]
    ].copy()
    return scores, sensor_scores


def compute_lhi_scores_streaming(
    args: argparse.Namespace,
    frames: dict[str, pd.DataFrame],
    condition_means: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    range_partials = []
    lhi_partials = []
    range_cache: dict[tuple, pd.DataFrame] = {}
    total_rows = 0
    kept_rows = 0
    for chunk_idx, chunk in enumerate(iter_window_forecast_chunks(args), start=1):
        total_rows += len(chunk)
        window_keys = chunk[["fd", "unit_id", "context_start_cycle", "cutoff_cycle"]].drop_duplicates()
        missing_keys = []
        for key in window_keys.itertuples(index=False):
            cache_key = (str(key.fd), int(key.unit_id), int(key.context_start_cycle), int(key.cutoff_cycle))
            if cache_key not in range_cache:
                missing_keys.append(
                    {
                        "fd": cache_key[0],
                        "unit_id": cache_key[1],
                        "context_start_cycle": cache_key[2],
                        "cutoff_cycle": cache_key[3],
                    }
                )
        if missing_keys:
            missing_frame = pd.DataFrame(missing_keys).drop_duplicates()
            ranges = compute_past_sensor_ranges(
                frames,
                missing_frame,
                args.sensors,
                args.range_epsilon,
                minmax_scope=args.minmax_scope,
            )
            for cache_key, range_group in ranges.groupby(
                ["fd", "unit_id", "context_start_cycle", "cutoff_cycle"], sort=False
            ):
                range_cache[(str(cache_key[0]), int(cache_key[1]), int(cache_key[2]), int(cache_key[3]))] = range_group
            range_partials.append(ranges)

        chunk_ranges = pd.concat(
            [
                range_cache[(str(key.fd), int(key.unit_id), int(key.context_start_cycle), int(key.cutoff_cycle))]
                for key in window_keys.itertuples(index=False)
            ],
            ignore_index=True,
        )
        partial = aggregate_lhi_chunk(chunk, chunk_ranges, condition_means)
        kept_rows += int(partial["n"].sum()) if not partial.empty else 0
        lhi_partials.append(partial)
        print(
            f"  chunk {chunk_idx}: input rows={len(chunk):,}, partial sensor groups={len(partial):,}",
            flush=True,
        )

    if total_rows == 0:
        raise ValueError("No forecast rows left after filtering window_forecasts.csv.")
    print(f"Streamed LHI rows: forecast rows={total_rows:,}, usable drift rows={kept_rows:,}", flush=True)
    if range_partials:
        past_ranges = pd.concat(range_partials, ignore_index=True)
        range_key_cols = [
            col
            for col in ["covariate_mode", "fd", "unit_id", "context_start_cycle", "cutoff_cycle", "sensor"]
            if col in past_ranges.columns
        ]
        past_ranges = past_ranges.drop_duplicates(range_key_cols).reset_index(drop=True)
    else:
        past_ranges = pd.DataFrame()
    scores, sensor_scores = finalize_lhi_partials(lhi_partials)
    return scores, sensor_scores, past_ranges


def compute_initial_forecast_baselines(
    scores: pd.DataFrame,
    healthy_cycles: int,
    baseline_cycles: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if baseline_cycles < 0:
        raise ValueError("--baseline_cycles must be >= 0.")
    rows = []
    baseline_point_rows = []
    for (mode, fd_name, unit_id), group in scores.groupby(["covariate_mode", "fd", "unit_id"], sort=True):
        group = group.sort_values(["cycle", "forecast_start_cycle"]).copy()
        monitor = group[group["cycle"] > healthy_cycles].copy()
        if monitor.empty:
            monitor = group.copy()
        if baseline_cycles == 0:
            full_monitor = group[group["forecast_start_cycle"] > healthy_cycles].copy()
            if full_monitor.empty:
                full_monitor = monitor
            first_start = int(full_monitor["forecast_start_cycle"].min())
            baseline_group = full_monitor[full_monitor["forecast_start_cycle"] == first_start].copy()
            baseline_source = f"first_full_monitor_forecast_block_start_{first_start}"
        else:
            first_cycle = int(monitor["cycle"].min())
            last_cycle = first_cycle + int(baseline_cycles) - 1
            baseline_group = monitor[(monitor["cycle"] >= first_cycle) & (monitor["cycle"] <= last_cycle)].copy()
            baseline_source = f"first_{baseline_cycles}_monitor_cycles_{first_cycle}_{last_cycle}"
        if baseline_group.empty:
            baseline_group = monitor.head(1).copy()
            baseline_source = "first_monitor_row_fallback"
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
        baseline_point_rows.append(baseline_group[point_cols].assign(baseline_source=baseline_source))
        rows.append(
            {
                "covariate_mode": mode,
                "fd": fd_name,
                "unit_id": int(unit_id),
                "b_mae": float(baseline_group["d_mae"].mean()),
                "b_rmse": float(baseline_group["d_rmse"].mean()),
                "baseline_start_cycle": int(baseline_group["cycle"].min()),
                "baseline_end_cycle": int(baseline_group["cycle"].max()),
                "baseline_rows": int(len(baseline_group)),
                "baseline_source": baseline_source,
            }
        )
    baseline_points = pd.concat(baseline_point_rows, ignore_index=True) if baseline_point_rows else pd.DataFrame()
    return baseline_points, pd.DataFrame(rows)


def compute_raw_healthy_cycle_baselines(
    frames: dict[str, pd.DataFrame],
    condition_means: pd.DataFrame,
    sensors: Sequence[str],
    healthy_cycles: int,
    range_epsilon: float,
    score_index: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_rows = []
    baseline_point_rows = []
    required_units = score_index[["covariate_mode", "fd", "unit_id"]].drop_duplicates()
    for (fd_name, unit_id), mode_group in required_units.groupby(["fd", "unit_id"], sort=True):
        frame = frames[str(fd_name)]
        unit_df = frame[
            (frame["unit_id"] == int(unit_id)) & (frame["cycle"] <= int(healthy_cycles))
        ].sort_values("cycle")
        if unit_df.empty:
            continue

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
        healthy_scores = (
            keyed.groupby(["fd", "unit_id", "cycle"], sort=True)
            .agg(
                d_mae=("drift", "mean"),
                d_rmse=("drift_sq", lambda s: float(np.sqrt(np.mean(s)))),
                sensor_count=("sensor", "nunique"),
                row_count=("drift", "size"),
            )
            .reset_index()
        )

        for mode in sorted(mode_group["covariate_mode"].unique()):
            mode_scores = healthy_scores.copy()
            mode_scores["covariate_mode"] = mode
            mode_scores["cutoff_cycle"] = int(healthy_cycles)
            mode_scores["forecast_start_cycle"] = 1
            baseline_source = f"raw_observed_cycles_1_to_{int(healthy_cycles)}"
            baseline_point_rows.append(
                mode_scores[
                    [
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
                ].assign(baseline_source=baseline_source)
            )
            baseline_rows.append(
                {
                    "covariate_mode": mode,
                    "fd": fd_name,
                    "unit_id": int(unit_id),
                    "b_mae": float(mode_scores["d_mae"].mean()),
                    "b_rmse": float(mode_scores["d_rmse"].mean()),
                    "baseline_start_cycle": int(mode_scores["cycle"].min()),
                    "baseline_end_cycle": int(mode_scores["cycle"].max()),
                    "baseline_rows": int(len(mode_scores)),
                    "baseline_source": baseline_source,
                }
            )

    if not baseline_rows:
        raise ValueError(f"No raw healthy-cycle baselines were computed from cycles 1..{healthy_cycles}.")
    baseline_points = pd.concat(baseline_point_rows, ignore_index=True)
    baselines = pd.DataFrame(baseline_rows)
    expected = required_units.set_index(["covariate_mode", "fd", "unit_id"]).index
    actual = baselines.set_index(["covariate_mode", "fd", "unit_id"]).index
    if len(expected.difference(actual)) > 0:
        missing = list(expected.difference(actual))[:5]
        raise ValueError(f"Missing raw healthy-cycle baseline for forecast units, examples: {missing}")
    return baseline_points, baselines


def add_lhi_columns(scores: pd.DataFrame, baselines: pd.DataFrame, lhi_epsilon: float) -> pd.DataFrame:
    scores = scores.merge(
        baselines,
        on=["covariate_mode", "fd", "unit_id"],
        how="left",
        validate="many_to_one",
    )
    if scores["b_mae"].isna().any() or scores["b_rmse"].isna().any():
        raise ValueError("Missing LHI baseline for some unit rows.")
    scores["lhi_mae"] = np.log((scores["d_mae"] + lhi_epsilon) / (scores["b_mae"] + lhi_epsilon))
    scores["lhi_rmse"] = np.log((scores["d_rmse"] + lhi_epsilon) / (scores["b_rmse"] + lhi_epsilon))
    return scores


def add_top_drift_sensors(
    scores: pd.DataFrame,
    sensor_scores: pd.DataFrame,
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if top_k < 1:
        raise ValueError("--top_k_sensors must be >= 1.")

    key_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"]
    ranked = sensor_scores.merge(scores[key_cols], on=key_cols, how="inner").copy()
    if ranked.empty:
        empty_cols = key_cols + [
            "top_drift_rank",
            "sensor",
            "sensor_d_mae",
            "sensor_d_rmse",
            "n",
            "healthy_condition_n",
            "past_range",
        ]
        return scores.assign(
            top_drift_sensors="",
            top_drift_sensor_rmse_values="",
            top_drift_sensor_mae_values="",
        ), pd.DataFrame(columns=empty_cols)

    ranked = ranked.sort_values(
        key_cols + ["sensor_d_rmse", "sensor_d_mae", "sensor"],
        ascending=[True, True, True, True, True, True, False, False, True],
    )
    ranked["top_drift_rank"] = ranked.groupby(key_cols, sort=False).cumcount() + 1
    top_rows = ranked[ranked["top_drift_rank"] <= top_k].copy()

    summary_rows = []
    for key, group in top_rows.groupby(key_cols, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(key_cols, key_values))
        group = group.sort_values("top_drift_rank")
        row["top_drift_sensors"] = ",".join(group["sensor"].astype(str))
        row["top_drift_sensor_rmse_values"] = ";".join(
            f"{r.sensor}:{float(r.sensor_d_rmse):.6g}" for r in group.itertuples(index=False)
        )
        row["top_drift_sensor_mae_values"] = ";".join(
            f"{r.sensor}:{float(r.sensor_d_mae):.6g}" for r in group.itertuples(index=False)
        )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    scores = scores.merge(summary, on=key_cols, how="left", validate="one_to_one")
    for col in ["top_drift_sensors", "top_drift_sensor_rmse_values", "top_drift_sensor_mae_values"]:
        scores[col] = scores[col].fillna("")

    ordered_cols = key_cols + [
        "top_drift_rank",
        "sensor",
        "sensor_d_rmse",
        "sensor_d_mae",
        "n",
        "healthy_condition_n",
        "past_range",
    ]
    remaining_cols = [col for col in top_rows.columns if col not in ordered_cols]
    return scores, top_rows[ordered_cols + remaining_cols]


def add_rolling_scores(scores: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    if rolling_window < 1:
        raise ValueError("--rolling_window must be >= 1.")
    rows = []
    for _, group in scores.groupby(["covariate_mode", "fd", "unit_id"], sort=True):
        group = group.sort_values(["cycle", "forecast_start_cycle"]).copy()
        for col in ["d_mae", "d_rmse", "lhi_mae", "lhi_rmse"]:
            group[f"{col}_roll_mean"] = group[col].rolling(rolling_window, min_periods=1).mean()
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def summarize_fd(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, fd_name, cycle), group in scores.groupby(["covariate_mode", "fd", "cycle"], sort=True):
        rows.append(
            {
                "covariate_mode": mode,
                "fd": fd_name,
                "cycle": int(cycle),
                "unit_count": int(group["unit_id"].nunique()),
                "median_d_rmse": float(group["d_rmse"].median()),
                "q90_d_rmse": float(group["d_rmse"].quantile(0.90)),
                "median_lhi_rmse": float(group["lhi_rmse"].median()),
                "q90_lhi_rmse": float(group["lhi_rmse"].quantile(0.90)),
                "median_lhi_rmse_roll_mean": float(group["lhi_rmse_roll_mean"].median()),
                "q90_lhi_rmse_roll_mean": float(group["lhi_rmse_roll_mean"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.range_epsilon <= 0:
        raise ValueError("--range_epsilon must be positive.")
    if args.lhi_epsilon <= 0:
        raise ValueError("--lhi_epsilon must be positive.")
    if args.top_k_sensors < 1:
        raise ValueError("--top_k_sensors must be >= 1.")
    output_dir = args.output_dir or (args.forecast_dir / "lhi")
    output_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in (
        "unit_healthy_minmax_ranges.csv",
        "healthy_baseline_points.csv",
        "healthy_reference_baselines.csv",
    ):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    print(f"Loading {args.forecast_dir / 'window_forecasts.csv'}...", flush=True)
    run_config_path = args.forecast_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    eval_split = str(run_config.get("eval_split", "train"))
    frames = load_eval_frames(args.data_dir, eval_split, args.fds)
    condition_means = build_condition_means(frames, args.sensors, args.healthy_cycles)
    use_streaming = bool(args.chunksize and args.chunksize > 0 and args.minmax_scope == "past_context")
    if args.chunksize and args.chunksize > 0 and args.minmax_scope == "past_and_forecast":
        print(
            "Using in-memory LHI path because --minmax_scope past_and_forecast requires forecast values "
            "from complete windows when computing min-max ranges.",
            flush=True,
        )
    if use_streaming:
        all_scores, sensor_scores, past_ranges = compute_lhi_scores_streaming(
            args=args,
            frames=frames,
            condition_means=condition_means,
        )
    else:
        forecasts = load_window_forecasts(args)
        past_ranges = compute_past_sensor_ranges(
            frames,
            forecasts,
            args.sensors,
            args.range_epsilon,
            minmax_scope=args.minmax_scope,
        )
        all_scores, sensor_scores = compute_lhi_scores(
            forecasts=forecasts,
            past_ranges=past_ranges,
            condition_means=condition_means,
        )
    scores = all_scores.copy()
    if scores.empty:
        raise ValueError("No LHI score rows were computed.")
    if args.baseline_source == "raw_healthy_cycles":
        baseline_points, baselines = compute_raw_healthy_cycle_baselines(
            frames=frames,
            condition_means=condition_means,
            sensors=args.sensors,
            healthy_cycles=args.healthy_cycles,
            range_epsilon=args.range_epsilon,
            score_index=scores,
        )
    else:
        baseline_points, baselines = compute_initial_forecast_baselines(
            scores,
            healthy_cycles=args.healthy_cycles,
            baseline_cycles=args.baseline_cycles,
        )
    scores = add_lhi_columns(scores, baselines, args.lhi_epsilon)
    scores, top_drift_sensors = add_top_drift_sensors(scores, sensor_scores, args.top_k_sensors)
    scores = add_rolling_scores(scores, args.rolling_window)
    fd_summary = summarize_fd(scores)

    past_ranges.to_csv(output_dir / "past_minmax_ranges.csv", index=False)
    condition_means.to_csv(output_dir / "healthy_condition_means.csv", index=False)
    baseline_points.to_csv(output_dir / "baseline_forecast_points.csv", index=False)
    baselines.to_csv(output_dir / "lhi_baselines.csv", index=False)
    scores.to_csv(output_dir / "lhi_scores.csv", index=False)
    top_drift_sensors.to_csv(output_dir / "top_drift_sensors.csv", index=False)
    sensor_scores.to_csv(output_dir / "sensor_lhi_components.csv", index=False)
    fd_summary.to_csv(output_dir / "fd_lhi_summary.csv", index=False)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    usable_ranges = past_ranges["range_usable"].mean()
    print(f"Usable past-context min-max ranges: {usable_ranges:.2%}", flush=True)
    print(
        "LHI summary by FD/mode:\n"
        + scores.groupby(["covariate_mode", "fd"], sort=True)
        .agg(
            units=("unit_id", "nunique"),
            median_b_rmse=("b_rmse", "median"),
            median_lhi_rmse=("lhi_rmse", "median"),
            q90_lhi_rmse=("lhi_rmse", lambda s: float(s.quantile(0.90))),
        )
        .reset_index()
        .to_string(index=False),
        flush=True,
    )
    print(f"Saved LHI outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

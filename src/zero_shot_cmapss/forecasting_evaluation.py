from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
            "Evaluate Chronos-2 C-MAPSS forecasts with min-max forecast error "
            "and condition-matched forecast state drift."
        )
    )
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--forecast_dir", type=Path, default=Path("outputs/roll_5"))
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
        help="First N cycles per unit used as healthy reference for condition-matched drift.",
    )
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=5,
        help="Rolling window used to save smoothed metric columns.",
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def load_eval_frames(data_dir: Path, eval_split: str) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for fd_name in FD_NAMES:
        frames[fd_name] = load_cmapss_file(data_dir / fd_name / f"{eval_split}_{fd_name}.txt")
    return frames


def build_healthy_reference_from_frames(
    frames: Dict[str, pd.DataFrame],
    sensors: Sequence[str],
    healthy_cycles: int,
) -> pd.DataFrame:
    rows = []
    for fd_name, frame in frames.items():
        healthy = frame[frame["cycle"] <= healthy_cycles].copy()
        if healthy.empty:
            continue
        healthy["op_condition_key"] = make_condition_keys(healthy)
        melted = healthy.melt(
            id_vars=["unit_id", "cycle", "op_condition_key"],
            value_vars=list(sensors),
            var_name="sensor",
            value_name="y_true",
        )
        melted["fd"] = fd_name
        rows.append(melted)
    if not rows:
        raise ValueError(f"No healthy rows found for cycle <= --healthy_cycles ({healthy_cycles}).")
    healthy_actuals = pd.concat(rows, ignore_index=True)
    return (
        healthy_actuals.groupby(["fd", "unit_id", "op_condition_key", "sensor"], sort=True)
        .agg(
            ref_mean_raw=("y_true", "mean"),
            ref_std_raw=("y_true", lambda s: float(np.std(s.to_numpy(dtype=np.float32), ddof=0))),
            ref_n=("y_true", "size"),
        )
        .reset_index()
    )


def compute_past_sensor_ranges(
    frames: Dict[str, pd.DataFrame],
    forecasts: pd.DataFrame,
    sensors: Sequence[str],
) -> pd.DataFrame:
    rows = []
    window_keys = (
        forecasts[["fd", "unit_id", "context_start_cycle", "cutoff_cycle"]]
        .drop_duplicates()
        .sort_values(["fd", "unit_id", "cutoff_cycle"])
    )
    for key in window_keys.itertuples(index=False):
        fd_name = str(key.fd)
        unit_id = int(key.unit_id)
        context_start_cycle = int(key.context_start_cycle)
        cutoff_cycle = int(key.cutoff_cycle)
        frame = frames[fd_name]
        past = frame[
            (frame["unit_id"] == unit_id)
            & (frame["cycle"] >= context_start_cycle)
            & (frame["cycle"] <= cutoff_cycle)
        ]
        if past.empty:
            raise ValueError(
                f"No past rows for {fd_name} unit {unit_id}, "
                f"context_start_cycle={context_start_cycle}, cutoff_cycle={cutoff_cycle}."
            )
        for sensor in sensors:
            minimum = float(past[sensor].min())
            maximum = float(past[sensor].max())
            value_range = maximum - minimum
            if not np.isfinite(value_range) or value_range <= 1e-8:
                value_range = 1.0
            rows.append(
                {
                    "fd": fd_name,
                    "unit_id": unit_id,
                    "context_start_cycle": context_start_cycle,
                    "cutoff_cycle": cutoff_cycle,
                    "sensor": sensor,
                    "past_min": minimum,
                    "past_max": maximum,
                    "past_range": value_range,
                    "past_n": int(len(past)),
                }
            )
    return pd.DataFrame(rows)


def add_minmax_columns(forecasts: pd.DataFrame, past_ranges: pd.DataFrame) -> pd.DataFrame:
    keyed = forecasts.merge(
        past_ranges,
        on=["fd", "unit_id", "context_start_cycle", "cutoff_cycle", "sensor"],
        how="left",
        validate="many_to_one",
    )
    if keyed["past_min"].isna().any() or keyed["past_range"].isna().any():
        raise ValueError("Missing leakage-free past min-max stats for some forecast rows.")
    keyed["y_true_minmax"] = (keyed["y_true"] - keyed["past_min"]) / keyed["past_range"]
    keyed["y_pred_minmax"] = (keyed["y_pred"] - keyed["past_min"]) / keyed["past_range"]
    keyed["minmax_error"] = keyed["y_pred_minmax"] - keyed["y_true_minmax"]
    keyed["minmax_abs_error"] = keyed["minmax_error"].abs()
    keyed["minmax_sq_error"] = keyed["minmax_error"] ** 2
    return keyed


def select_metric_windows(forecasts: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    if "has_ground_truth" not in forecasts.columns:
        forecasts = forecasts.assign(has_ground_truth=forecasts["y_true"].notna().astype(int))
    key_cols = ["covariate_mode", "fd", "unit_id", "forecast_start_cycle"]
    horizon_counts = (
        forecasts[forecasts["has_ground_truth"].astype(bool)]
        .groupby(key_cols, sort=True)["horizon"]
        .nunique()
        .reset_index(name="ground_truth_horizon_count")
    )
    full_keys = horizon_counts[horizon_counts["ground_truth_horizon_count"] >= prediction_length].copy()
    if full_keys.empty:
        return forecasts.iloc[0:0].copy()

    selected = []
    for _, group in full_keys.groupby(["covariate_mode", "fd", "unit_id"], sort=True):
        first_start = int(group["forecast_start_cycle"].min())
        selected.append(group[(group["forecast_start_cycle"] - first_start) % prediction_length == 0][key_cols])
    selected_keys = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=key_cols)
    metric_forecasts = forecasts.merge(selected_keys, on=key_cols, how="inner")
    return metric_forecasts[metric_forecasts["has_ground_truth"].astype(bool)].copy()


def summarize_forecast_error(forecasts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    window_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"]
    sensor_cols = [*window_cols, "sensor"]
    sensor_round = (
        forecasts.groupby(sensor_cols, sort=True)
        .agg(
            mae=("minmax_abs_error", "mean"),
            rmse=("minmax_sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            n=("minmax_error", "size"),
        )
        .reset_index()
    )
    overall_round = (
        forecasts.groupby(window_cols, sort=True)
        .agg(
            mae=("minmax_abs_error", "mean"),
            rmse=("minmax_sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            n=("minmax_error", "size"),
        )
        .reset_index()
    )
    overall_round["sensor"] = "ALL"
    return sensor_round, overall_round[[*sensor_cols, "mae", "rmse", "n"]]


def summarize_error_partials(partials: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not partials:
        return pd.DataFrame()
    partial = pd.concat(partials, ignore_index=True)
    group_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle", "sensor"]
    out = (
        partial.groupby(group_cols, sort=True)
        .agg(
            abs_sum=("abs_sum", "sum"),
            sq_sum=("sq_sum", "sum"),
            n=("n", "sum"),
        )
        .reset_index()
    )
    out["mae"] = out["abs_sum"] / out["n"]
    out["rmse"] = np.sqrt(out["sq_sum"] / out["n"])
    return out[[*group_cols, "mae", "rmse", "n"]]


def make_overall_from_sensor_round(sensor_round: pd.DataFrame) -> pd.DataFrame:
    window_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"]
    partial = sensor_round.copy()
    partial["abs_sum"] = partial["mae"] * partial["n"]
    partial["sq_sum"] = (partial["rmse"] ** 2) * partial["n"]
    overall = (
        partial.groupby(window_cols, sort=True)
        .agg(abs_sum=("abs_sum", "sum"), sq_sum=("sq_sum", "sum"), n=("n", "sum"))
        .reset_index()
    )
    overall["mae"] = overall["abs_sum"] / overall["n"]
    overall["rmse"] = np.sqrt(overall["sq_sum"] / overall["n"])
    overall["sensor"] = "ALL"
    return overall[[*window_cols, "sensor", "mae", "rmse", "n"]]


def iter_forecast_chunks(path: Path, args: argparse.Namespace, usecols: Sequence[str]):
    for chunk in pd.read_csv(path, usecols=list(usecols), chunksize=args.chunksize):
        mode_filter = chunk["covariate_mode"].isin(args.covariate_modes) if args.covariate_modes else True
        chunk = chunk[
            chunk["fd"].isin(args.fds)
            & mode_filter
            & chunk["sensor"].isin(args.sensors)
        ].copy()
        if not chunk.empty:
            yield chunk


def summarize_forecast_error_streaming(
    args: argparse.Namespace,
    frames: Dict[str, pd.DataFrame],
    prediction_length: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_path = args.forecast_dir / "metric_window_forecasts.csv"
    if not metric_path.exists():
        metric_path = args.forecast_dir / "window_forecasts.csv"
    usecols = [
        "covariate_mode",
        "fd",
        "unit_id",
        "cutoff_cycle",
        "forecast_start_cycle",
        "context_start_cycle",
        "cycle",
        "sensor",
        "horizon",
        "has_ground_truth",
        "y_true",
        "y_pred",
    ]
    partials = []
    metric_rows = []
    for chunk_idx, chunk in enumerate(iter_forecast_chunks(metric_path, args, usecols), start=1):
        if metric_path.name == "window_forecasts.csv":
            chunk = select_metric_windows(chunk, prediction_length)
        chunk = chunk[chunk["y_true"].notna()].copy()
        if chunk.empty:
            continue
        past_ranges = compute_past_sensor_ranges(frames, chunk, args.sensors)
        chunk = add_minmax_columns(chunk, past_ranges)
        chunk["abs_sum"] = chunk["minmax_abs_error"]
        chunk["sq_sum"] = chunk["minmax_sq_error"]
        group_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle", "sensor"]
        partials.append(
            chunk.groupby(group_cols, sort=True)
            .agg(abs_sum=("abs_sum", "sum"), sq_sum=("sq_sum", "sum"), n=("minmax_error", "size"))
            .reset_index()
        )
        metric_rows.append(chunk)
        print(f"  forecast-error chunk {chunk_idx}: rows={len(chunk):,}", flush=True)
    sensor_round = summarize_error_partials(partials)
    overall_round = make_overall_from_sensor_round(sensor_round) if not sensor_round.empty else pd.DataFrame()
    metric_forecasts = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    metric_forecasts.to_csv(output_dir / "forecast_error_metric_rows.csv", index=False)
    return sensor_round, overall_round, metric_forecasts


def build_healthy_reference(forecasts: pd.DataFrame, healthy_cycles: int) -> pd.DataFrame:
    if "op_condition_key" not in forecasts.columns:
        raise ValueError("window_forecasts.csv must contain op_condition_key; rerun clustered forecasting first.")
    healthy = forecasts[(forecasts["cycle"] <= healthy_cycles) & forecasts["y_true"].notna()].copy()
    if healthy.empty:
        raise ValueError(f"No forecast rows found for cycle <= --healthy_cycles ({healthy_cycles}).")
    healthy = healthy.drop_duplicates(["fd", "unit_id", "cycle", "sensor", "op_condition_key"])
    reference = (
        healthy.groupby(["fd", "unit_id", "op_condition_key", "sensor"], sort=True)
        .agg(
            ref_mean_raw=("y_true", "mean"),
            ref_std_raw=("y_true", lambda s: float(np.std(s.to_numpy(dtype=np.float32), ddof=0))),
            ref_n=("y_true", "size"),
        )
        .reset_index()
    )
    return reference


def summarize_condition_matched_drift(
    forecasts: pd.DataFrame,
    healthy_reference: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "op_condition_key" not in forecasts.columns:
        raise ValueError("window_forecasts.csv must contain op_condition_key; evaluation no longer reclassifies conditions.")
    keyed = forecasts.copy()
    keyed = keyed.merge(
        healthy_reference[["fd", "unit_id", "op_condition_key", "sensor", "ref_mean_raw", "ref_std_raw", "ref_n"]],
        on=["fd", "unit_id", "op_condition_key", "sensor"],
        how="left",
        validate="many_to_one",
    )
    keyed["ref_mean_minmax"] = (keyed["ref_mean_raw"] - keyed["past_min"]) / keyed["past_range"]
    keyed["ref_std_minmax"] = keyed["ref_std_raw"] / keyed["past_range"]
    keyed["drift_error"] = keyed["y_pred_minmax"] - keyed["ref_mean_minmax"]
    keyed["drift_abs_error"] = keyed["drift_error"].abs()
    keyed["drift_sq_error"] = keyed["drift_error"] ** 2

    available = keyed[keyed["ref_mean_minmax"].notna()].copy()
    window_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"]
    sensor_cols = [*window_cols, "sensor"]
    sensor_round = (
        available.groupby(sensor_cols, sort=True)
        .agg(
            mae=("drift_abs_error", "mean"),
            rmse=("drift_sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            n=("drift_error", "size"),
            ref_n_min=("ref_n", "min"),
            ref_std_minmax_median=("ref_std_minmax", "median"),
        )
        .reset_index()
    )
    overall_round = (
        available.groupby(window_cols, sort=True)
        .agg(
            mae=("drift_abs_error", "mean"),
            rmse=("drift_sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            n=("drift_error", "size"),
            ref_n_min=("ref_n", "min"),
            ref_std_minmax_median=("ref_std_minmax", "median"),
        )
        .reset_index()
    )
    overall_round["sensor"] = "ALL"
    return keyed, sensor_round, overall_round[[*sensor_cols, "mae", "rmse", "n", "ref_n_min", "ref_std_minmax_median"]]


def summarize_condition_matched_drift_streaming(
    args: argparse.Namespace,
    frames: Dict[str, pd.DataFrame],
    healthy_reference: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = args.forecast_dir / "window_forecasts.csv"
    usecols = [
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
    ]
    drift_path = output_dir / "condition_matched_drift_rows.csv"
    if drift_path.exists():
        drift_path.unlink()
    partials = []
    total_rows = 0
    missing_rows = 0
    for chunk_idx, chunk in enumerate(iter_forecast_chunks(path, args, usecols), start=1):
        past_ranges = compute_past_sensor_ranges(frames, chunk, args.sensors)
        chunk = chunk.merge(
            past_ranges,
            on=["fd", "unit_id", "context_start_cycle", "cutoff_cycle", "sensor"],
            how="left",
            validate="many_to_one",
        )
        chunk["y_pred_minmax"] = (chunk["y_pred"] - chunk["past_min"]) / chunk["past_range"]
        chunk = chunk.merge(
            healthy_reference[["fd", "unit_id", "op_condition_key", "sensor", "ref_mean_raw", "ref_std_raw", "ref_n"]],
            on=["fd", "unit_id", "op_condition_key", "sensor"],
            how="left",
            validate="many_to_one",
        )
        chunk["ref_mean_minmax"] = (chunk["ref_mean_raw"] - chunk["past_min"]) / chunk["past_range"]
        chunk["ref_std_minmax"] = chunk["ref_std_raw"] / chunk["past_range"]
        chunk["drift_error"] = chunk["y_pred_minmax"] - chunk["ref_mean_minmax"]
        chunk["drift_abs_error"] = chunk["drift_error"].abs()
        chunk["drift_sq_error"] = chunk["drift_error"] ** 2
        total_rows += len(chunk)
        missing_rows += int(chunk["ref_mean_minmax"].isna().sum())
        chunk.to_csv(drift_path, index=False, mode="a", header=not drift_path.exists())

        available = chunk[chunk["ref_mean_minmax"].notna()].copy()
        if not available.empty:
            group_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle", "sensor"]
            partials.append(
                available.groupby(group_cols, sort=True)
                .agg(
                    abs_sum=("drift_abs_error", "sum"),
                    sq_sum=("drift_sq_error", "sum"),
                    n=("drift_error", "size"),
                    ref_n_min=("ref_n", "min"),
                    ref_std_minmax_median=("ref_std_minmax", "median"),
                )
                .reset_index()
            )
        print(f"  condition-drift chunk {chunk_idx}: rows={len(chunk):,}", flush=True)
    sensor_round = summarize_error_partials(partials)
    if not sensor_round.empty and "ref_n_min" not in sensor_round.columns:
        # Recompute with extra columns when present in partials.
        partial = pd.concat(partials, ignore_index=True) if partials else pd.DataFrame()
        group_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle", "sensor"]
        sensor_round = (
            partial.groupby(group_cols, sort=True)
            .agg(
                abs_sum=("abs_sum", "sum"),
                sq_sum=("sq_sum", "sum"),
                n=("n", "sum"),
                ref_n_min=("ref_n_min", "min"),
                ref_std_minmax_median=("ref_std_minmax_median", "median"),
            )
            .reset_index()
        )
        sensor_round["mae"] = sensor_round["abs_sum"] / sensor_round["n"]
        sensor_round["rmse"] = np.sqrt(sensor_round["sq_sum"] / sensor_round["n"])
        sensor_round = sensor_round[[*group_cols, "mae", "rmse", "n", "ref_n_min", "ref_std_minmax_median"]]
    overall_round = make_overall_from_sensor_round(sensor_round) if not sensor_round.empty else pd.DataFrame()
    if not overall_round.empty:
        overall_round["ref_n_min"] = np.nan
        overall_round["ref_std_minmax_median"] = np.nan
    reference_coverage = pd.DataFrame(
        [
            {
                "fd": "ALL",
                "covariate_mode": "ALL",
                "sensor": "ALL",
                "coverage": float(1.0 - (missing_rows / total_rows)) if total_rows else np.nan,
                "rows": int(total_rows),
                "missing_rows": int(missing_rows),
            }
        ]
    )
    return sensor_round, overall_round, reference_coverage


def summarize_reference_coverage(keyed: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "fd": "ALL",
            "covariate_mode": "ALL",
            "sensor": "ALL",
            "coverage": float(keyed["ref_mean_minmax"].notna().mean()),
            "rows": int(len(keyed)),
            "missing_rows": int(keyed["ref_mean_minmax"].isna().sum()),
        }
    ]
    for (fd_name, mode), group in keyed.groupby(["fd", "covariate_mode"], sort=True):
        rows.append(
            {
                "fd": fd_name,
                "covariate_mode": mode,
                "sensor": "ALL",
                "coverage": float(group["ref_mean_minmax"].notna().mean()),
                "rows": int(len(group)),
                "missing_rows": int(group["ref_mean_minmax"].isna().sum()),
            }
        )
    for (fd_name, mode, sensor), group in keyed.groupby(["fd", "covariate_mode", "sensor"], sort=True):
        rows.append(
            {
                "fd": fd_name,
                "covariate_mode": mode,
                "sensor": sensor,
                "coverage": float(group["ref_mean_minmax"].notna().mean()),
                "rows": int(len(group)),
                "missing_rows": int(group["ref_mean_minmax"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def smooth_round_metrics(frame: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    smoothed = []
    for _, group in frame.groupby(["covariate_mode", "fd", "unit_id", "sensor"], sort=True):
        group = group.sort_values(["cycle", "forecast_start_cycle"]).copy()
        group["mae_roll_mean"] = group["mae"].rolling(rolling_window, min_periods=1).mean()
        group["rmse_roll_mean"] = group["rmse"].rolling(rolling_window, min_periods=1).mean()
        smoothed.append(group)
    return pd.concat(smoothed, ignore_index=True) if smoothed else frame


def aggregate_fd_level(round_metrics: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    rows = []
    for (mode, fd_name, cycle, sensor), group in round_metrics.groupby(
        ["covariate_mode", "fd", "cycle", "sensor"],
        sort=True,
    ):
        rows.append(
            {
                "covariate_mode": mode,
                "fd": fd_name,
                "cycle": int(cycle),
                "forecast_start_cycle": int(group["forecast_start_cycle"].min()),
                "sensor": sensor,
                "mae": float(group["mae"].median()),
                "rmse": float(group["rmse"].median()),
                "unit_count": int(group["unit_id"].nunique()),
                "round_count": int(len(group)),
            }
        )
    fd_level = pd.DataFrame(rows)
    return smooth_round_metrics(fd_level.assign(unit_id=0), rolling_window).drop(columns=["unit_id"])


def plot_fd_level(metric_frame: pd.DataFrame, metric_name: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_data = metric_frame[metric_frame["sensor"] == "ALL"].copy()
    if plot_data.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    axes_flat = axes.ravel()
    colors = {
        "cluster_covariate": "#ff7f0e",
        "future_covariate": "#1f77b4",
        "no_covariate": "#2ca02c",
        "known_future": "#ff7f0e",
        "none": "#2ca02c",
    }
    for ax, fd_name in zip(axes_flat, FD_NAMES):
        fd_df = plot_data[plot_data["fd"] == fd_name]
        for mode, mode_df in fd_df.groupby("covariate_mode", sort=True):
            mode_df = mode_df.sort_values(["cycle", "forecast_start_cycle"])
            color = colors.get(str(mode))
            ax.plot(mode_df["cycle"], mode_df["mae_roll_mean"], color=color, label=f"{mode} MAE rolling")
            ax.plot(mode_df["cycle"], mode_df["rmse"], color=color, alpha=0.25, linewidth=0.9, label=f"{mode} RMSE raw")
            ax.plot(mode_df["cycle"], mode_df["rmse_roll_mean"], color=color, linestyle="--", label=f"{mode} RMSE rolling")
        ax.set_title(fd_name)
        ax.set_xlabel("target cycle")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
    fig.suptitle(f"{metric_name}: rolling MAE/RMSE over forecast rounds")
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric_name.lower().replace(' ', '_')}_fd_level.png", dpi=160)
    plt.close(fig)


def plot_unit_level(metric_frame: pd.DataFrame, metric_name: str, output_dir: Path, units: Sequence[Tuple[str, int]]) -> None:
    import matplotlib.pyplot as plt

    unit_dir = output_dir / "unit_plots"
    unit_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "cluster_covariate": "#ff7f0e",
        "future_covariate": "#1f77b4",
        "no_covariate": "#2ca02c",
        "known_future": "#ff7f0e",
        "none": "#2ca02c",
    }
    for fd_name, unit_id in units:
        unit_df = metric_frame[
            (metric_frame["fd"] == fd_name)
            & (metric_frame["unit_id"] == unit_id)
            & (metric_frame["sensor"] == "ALL")
        ]
        if unit_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        for mode, mode_df in unit_df.groupby("covariate_mode", sort=True):
            mode_df = mode_df.sort_values(["cycle", "forecast_start_cycle"])
            color = colors.get(str(mode))
            ax.plot(mode_df["cycle"], mode_df["mae_roll_mean"], color=color, label=f"{mode} MAE rolling")
            ax.plot(mode_df["cycle"], mode_df["rmse"], color=color, alpha=0.25, linewidth=0.9, label=f"{mode} RMSE raw")
            ax.plot(mode_df["cycle"], mode_df["rmse_roll_mean"], color=color, linestyle="--", label=f"{mode} RMSE rolling")
        ax.set_title(f"{metric_name}: {fd_name} unit {unit_id}")
        ax.set_xlabel("target cycle")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(unit_dir / f"{metric_name.lower().replace(' ', '_')}_{fd_name}_unit{unit_id}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.forecast_dir / "forecasting_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = args.forecast_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    eval_split = str(run_config.get("eval_split", "train"))
    prediction_length = int(run_config.get("prediction_length", 1))
    eval_frames = load_eval_frames(args.data_dir, eval_split)

    print("Computing min-max forecasting error metrics...", flush=True)
    fe_sensor, fe_overall, metric_forecasts = summarize_forecast_error_streaming(
        args=args,
        frames=eval_frames,
        prediction_length=prediction_length,
        output_dir=output_dir,
    )
    print(
        f"Forecast Error uses full-ground-truth, non-overlapping starts: "
        f"{len(metric_forecasts):,} rows.",
        flush=True,
    )
    fe_round = pd.concat([fe_overall, fe_sensor], ignore_index=True)
    fe_round = smooth_round_metrics(fe_round, args.rolling_window)
    fe_fd_level = aggregate_fd_level(fe_round, args.rolling_window)
    fe_round.to_csv(output_dir / "forecast_error_round_metrics.csv", index=False)
    fe_fd_level.to_csv(output_dir / "forecast_error_fd_level.csv", index=False)

    print("Computing condition-matched forecast state drift metrics...", flush=True)
    healthy_reference = build_healthy_reference_from_frames(eval_frames, args.sensors, args.healthy_cycles)
    healthy_reference.to_csv(output_dir / "healthy_condition_reference.csv", index=False)
    drift_sensor, drift_overall, reference_coverage = summarize_condition_matched_drift_streaming(
        args=args,
        frames=eval_frames,
        healthy_reference=healthy_reference,
        output_dir=output_dir,
    )
    reference_coverage.to_csv(output_dir / "reference_coverage.csv", index=False)
    print(
        "Reference coverage by FD/mode:\n"
        + reference_coverage[reference_coverage["sensor"] == "ALL"].to_string(index=False),
        flush=True,
    )
    drift_round = pd.concat([drift_overall, drift_sensor], ignore_index=True)
    drift_round = smooth_round_metrics(drift_round, args.rolling_window)
    drift_fd_level = aggregate_fd_level(drift_round, args.rolling_window)
    drift_round.to_csv(output_dir / "condition_matched_drift_round_metrics.csv", index=False)
    drift_fd_level.to_csv(output_dir / "condition_matched_drift_fd_level.csv", index=False)

    print(f"Saved forecasting evaluation outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

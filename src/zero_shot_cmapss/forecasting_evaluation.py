from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .plot_operating_condition_clusters import DEFAULT_SENSORS, FD_NAMES, load_cmapss_file, make_condition_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Chronos-2 C-MAPSS forecasts with z-score forecast error "
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
        default=30,
        help="First N cycles per unit used as healthy reference for condition-matched drift.",
    )
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=5,
        help="Optional rolling window over forecast rounds for smoother plots.",
    )
    parser.add_argument(
        "--plot_units",
        nargs="*",
        default=None,
        help="Optional unit-level plots, e.g. FD001:4 FD004:3. If omitted, plots first units per FD.",
    )
    parser.add_argument("--plot_examples", type=int, default=3)
    return parser.parse_args()


def parse_plot_units(items: Sequence[str] | None, forecasts: pd.DataFrame, fds: Sequence[str]) -> List[Tuple[str, int]]:
    if items:
        parsed = []
        for item in items:
            if ":" not in item:
                raise ValueError(f"--plot_units entries must look like FD004:3, got {item!r}")
            fd_name, unit_text = item.split(":", 1)
            if fd_name not in FD_NAMES:
                raise ValueError(f"Unknown FD in --plot_units: {fd_name!r}")
            parsed.append((fd_name, int(unit_text)))
        return parsed

    keys = (
        forecasts[forecasts["fd"].isin(fds)][["fd", "unit_id"]]
        .drop_duplicates()
        .sort_values(["fd", "unit_id"])
        .groupby("fd", as_index=False)
        .head(1)
    )
    return [(str(row.fd), int(row.unit_id)) for row in keys.itertuples(index=False)]


def load_eval_frames(data_dir: Path, eval_split: str) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for fd_name in FD_NAMES:
        frames[fd_name] = load_cmapss_file(data_dir / fd_name / f"{eval_split}_{fd_name}.txt")
    return frames


def compute_sensor_stats(frames: Dict[str, pd.DataFrame], sensors: Sequence[str]) -> pd.DataFrame:
    rows = []
    for fd_name, frame in frames.items():
        for sensor in sensors:
            mean = float(frame[sensor].mean())
            std = float(frame[sensor].std(ddof=0))
            if not np.isfinite(std) or std <= 1e-8:
                std = 1.0
            rows.append({"fd": fd_name, "sensor": sensor, "mean": mean, "std": std})
    return pd.DataFrame(rows)


def add_zscore_columns(forecasts: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    keyed = forecasts.merge(stats, on=["fd", "sensor"], how="left", validate="many_to_one")
    if keyed["mean"].isna().any() or keyed["std"].isna().any():
        raise ValueError("Missing z-score stats for some forecast rows.")
    keyed["y_true_z"] = (keyed["y_true"] - keyed["mean"]) / keyed["std"]
    keyed["y_pred_z"] = (keyed["y_pred"] - keyed["mean"]) / keyed["std"]
    keyed["z_error"] = keyed["y_pred_z"] - keyed["y_true_z"]
    keyed["z_abs_error"] = keyed["z_error"].abs()
    keyed["z_sq_error"] = keyed["z_error"] ** 2
    return keyed


def summarize_forecast_error(forecasts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    window_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle"]
    sensor_cols = [*window_cols, "sensor"]
    sensor_round = (
        forecasts.groupby(sensor_cols, sort=True)
        .agg(
            mae=("z_abs_error", "mean"),
            rmse=("z_sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            n=("z_error", "size"),
        )
        .reset_index()
    )
    overall_round = (
        forecasts.groupby(window_cols, sort=True)
        .agg(
            mae=("z_abs_error", "mean"),
            rmse=("z_sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            n=("z_error", "size"),
        )
        .reset_index()
    )
    overall_round["sensor"] = "ALL"
    return sensor_round, overall_round[[*sensor_cols, "mae", "rmse", "n"]]


def label_eval_conditions(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    labeled: Dict[str, pd.DataFrame] = {}
    for fd_name, frame in frames.items():
        frame = frame.copy()
        frame["op_condition_key"] = make_condition_keys(frame)
        labeled[fd_name] = frame
    return labeled


def build_cycle_condition_lookup(labeled_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fd_name, frame in labeled_frames.items():
        rows.append(frame[["unit_id", "cycle", "op_condition_key"]].assign(fd=fd_name))
    return pd.concat(rows, ignore_index=True)


def build_healthy_reference(
    labeled_frames: Dict[str, pd.DataFrame],
    stats: pd.DataFrame,
    sensors: Sequence[str],
    healthy_cycles: int,
) -> pd.DataFrame:
    stat_lookup = stats.set_index(["fd", "sensor"])
    rows = []
    for fd_name, frame in labeled_frames.items():
        healthy = frame[frame["cycle"] <= healthy_cycles].copy()
        for sensor in sensors:
            mean = float(stat_lookup.loc[(fd_name, sensor), "mean"])
            std = float(stat_lookup.loc[(fd_name, sensor), "std"])
            healthy[f"{sensor}_z"] = (healthy[sensor] - mean) / std
            grouped = (
                healthy.groupby(["unit_id", "op_condition_key"], sort=True)[f"{sensor}_z"]
                .agg(ref_mean_z="mean", ref_std_z="std", ref_n="size")
                .reset_index()
            )
            grouped["fd"] = fd_name
            grouped["sensor"] = sensor
            rows.append(grouped)
    reference = pd.concat(rows, ignore_index=True)
    reference["ref_std_z"] = reference["ref_std_z"].replace(0.0, np.nan).fillna(1.0)
    return reference


def summarize_condition_matched_drift(
    forecasts: pd.DataFrame,
    condition_lookup: pd.DataFrame,
    healthy_reference: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keyed = forecasts.merge(
        condition_lookup,
        on=["fd", "unit_id", "cycle"],
        how="left",
        validate="many_to_one",
    )
    keyed = keyed.merge(
        healthy_reference[["fd", "unit_id", "op_condition_key", "sensor", "ref_mean_z", "ref_n"]],
        on=["fd", "unit_id", "op_condition_key", "sensor"],
        how="left",
        validate="many_to_one",
    )
    keyed["drift_error"] = keyed["y_pred_z"] - keyed["ref_mean_z"]
    keyed["drift_abs_error"] = keyed["drift_error"].abs()
    keyed["drift_sq_error"] = keyed["drift_error"] ** 2

    available = keyed[keyed["ref_mean_z"].notna()].copy()
    window_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle"]
    sensor_cols = [*window_cols, "sensor"]
    sensor_round = (
        available.groupby(sensor_cols, sort=True)
        .agg(
            mae=("drift_abs_error", "mean"),
            rmse=("drift_sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            n=("drift_error", "size"),
            ref_n_min=("ref_n", "min"),
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
        )
        .reset_index()
    )
    overall_round["sensor"] = "ALL"
    return keyed, sensor_round, overall_round[[*sensor_cols, "mae", "rmse", "n", "ref_n_min"]]


def smooth_round_metrics(frame: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    smoothed = []
    for _, group in frame.groupby(["covariate_mode", "fd", "unit_id", "sensor"], sort=True):
        group = group.sort_values("forecast_start_cycle").copy()
        group["mae_roll_mean"] = group["mae"].rolling(rolling_window, min_periods=1).mean()
        group["rmse_roll_mean"] = group["rmse"].rolling(rolling_window, min_periods=1).mean()
        smoothed.append(group)
    return pd.concat(smoothed, ignore_index=True) if smoothed else frame


def aggregate_fd_level(round_metrics: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    rows = []
    for (mode, fd_name, cycle, sensor), group in round_metrics.groupby(
        ["covariate_mode", "fd", "forecast_start_cycle", "sensor"],
        sort=True,
    ):
        rows.append(
            {
                "covariate_mode": mode,
                "fd": fd_name,
                "forecast_start_cycle": int(cycle),
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
    colors = {"past_only": "#1f77b4", "known_future": "#ff7f0e", "none": "#2ca02c"}
    for ax, fd_name in zip(axes_flat, FD_NAMES):
        fd_df = plot_data[plot_data["fd"] == fd_name]
        for mode, mode_df in fd_df.groupby("covariate_mode", sort=True):
            mode_df = mode_df.sort_values("forecast_start_cycle")
            color = colors.get(str(mode))
            ax.plot(mode_df["forecast_start_cycle"], mode_df["mae_roll_mean"], color=color, label=f"{mode} MAE")
            ax.plot(mode_df["forecast_start_cycle"], mode_df["rmse_roll_mean"], color=color, linestyle="--", label=f"{mode} RMSE")
        ax.set_title(fd_name)
        ax.set_xlabel("forecast start cycle")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{metric_name}: rolling MAE/RMSE over forecast rounds")
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric_name.lower().replace(' ', '_')}_fd_level.png", dpi=160)
    plt.close(fig)


def plot_unit_level(metric_frame: pd.DataFrame, metric_name: str, output_dir: Path, units: Sequence[Tuple[str, int]]) -> None:
    import matplotlib.pyplot as plt

    unit_dir = output_dir / "unit_plots"
    unit_dir.mkdir(parents=True, exist_ok=True)
    colors = {"past_only": "#1f77b4", "known_future": "#ff7f0e", "none": "#2ca02c"}
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
            mode_df = mode_df.sort_values("forecast_start_cycle")
            color = colors.get(str(mode))
            ax.plot(mode_df["forecast_start_cycle"], mode_df["mae_roll_mean"], color=color, label=f"{mode} MAE")
            ax.plot(mode_df["forecast_start_cycle"], mode_df["rmse_roll_mean"], color=color, linestyle="--", label=f"{mode} RMSE")
        ax.set_title(f"{metric_name}: {fd_name} unit {unit_id}")
        ax.set_xlabel("forecast start cycle")
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
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    run_config_path = args.forecast_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    eval_split = str(run_config.get("eval_split", "train"))
    print(f"Loading {args.forecast_dir / 'window_forecasts.csv'}...", flush=True)
    forecasts = pd.read_csv(args.forecast_dir / "window_forecasts.csv")
    required = {"covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle", "sensor", "y_true", "y_pred"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Missing required columns in window_forecasts.csv: {sorted(missing)}")
    mode_filter = forecasts["covariate_mode"].isin(args.covariate_modes) if args.covariate_modes else True
    forecasts = forecasts[
        forecasts["fd"].isin(args.fds)
        & mode_filter
        & forecasts["sensor"].isin(args.sensors)
    ].copy()

    eval_frames = load_eval_frames(args.data_dir, eval_split)
    stats = compute_sensor_stats(eval_frames, args.sensors)
    stats.to_csv(output_dir / "zscore_stats.csv", index=False)
    forecasts = add_zscore_columns(forecasts, stats)

    print("Computing z-score forecasting error metrics...", flush=True)
    fe_sensor, fe_overall = summarize_forecast_error(forecasts)
    fe_round = pd.concat([fe_overall, fe_sensor], ignore_index=True)
    fe_round = smooth_round_metrics(fe_round, args.rolling_window)
    fe_fd_level = aggregate_fd_level(fe_round, args.rolling_window)
    fe_round.to_csv(output_dir / "forecast_error_round_metrics.csv", index=False)
    fe_fd_level.to_csv(output_dir / "forecast_error_fd_level.csv", index=False)

    print("Computing condition-matched forecast state drift metrics...", flush=True)
    labeled_frames = label_eval_conditions(eval_frames)
    condition_lookup = build_cycle_condition_lookup(labeled_frames)
    healthy_reference = build_healthy_reference(labeled_frames, stats, args.sensors, args.healthy_cycles)
    healthy_reference.to_csv(output_dir / "healthy_condition_reference.csv", index=False)
    drift_rows, drift_sensor, drift_overall = summarize_condition_matched_drift(
        forecasts=forecasts,
        condition_lookup=condition_lookup,
        healthy_reference=healthy_reference,
    )
    drift_round = pd.concat([drift_overall, drift_sensor], ignore_index=True)
    drift_round = smooth_round_metrics(drift_round, args.rolling_window)
    drift_fd_level = aggregate_fd_level(drift_round, args.rolling_window)
    drift_rows.to_csv(output_dir / "condition_matched_drift_rows.csv", index=False)
    drift_round.to_csv(output_dir / "condition_matched_drift_round_metrics.csv", index=False)
    drift_fd_level.to_csv(output_dir / "condition_matched_drift_fd_level.csv", index=False)

    units = parse_plot_units(args.plot_units, forecasts, args.fds)
    plot_fd_level(fe_fd_level, "Forecast Error", output_dir)
    plot_fd_level(drift_fd_level, "Condition Matched Drift", output_dir)
    plot_unit_level(fe_round, "Forecast Error", output_dir, units)
    plot_unit_level(drift_round, "Condition Matched Drift", output_dir, units)
    print(f"Saved forecasting evaluation outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

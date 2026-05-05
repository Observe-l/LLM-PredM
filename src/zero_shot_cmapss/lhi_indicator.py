from __future__ import annotations

import argparse
import json
import os
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
            "min-max drift. Min-max ranges are computed from each forecast window's past context."
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
        help="Minimum past-context min-max range required for a sensor to be used.",
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
    parser.add_argument("--rolling_window", type=int, default=5)
    parser.add_argument("--plot_examples", type=int, default=4)
    parser.add_argument(
        "--plot_units",
        nargs="*",
        default=None,
        help="Optional unit-level plots, e.g. FD001:1 FD004:3. If omitted, plots first --plot_examples units per FD.",
    )
    return parser.parse_args()


def parse_plot_units(items: Sequence[str] | None, scores: pd.DataFrame, examples_per_fd: int) -> List[Tuple[str, int]]:
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
        scores[["fd", "unit_id"]]
        .drop_duplicates()
        .sort_values(["fd", "unit_id"])
        .groupby("fd", as_index=False)
        .head(examples_per_fd)
    )
    return [(str(row.fd), int(row.unit_id)) for row in keys.itertuples(index=False)]


def load_window_forecasts(args: argparse.Namespace) -> pd.DataFrame:
    path = args.forecast_dir / "window_forecasts.csv"
    forecasts = pd.read_csv(path)
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
                    "range_usable": bool(np.isfinite(value_range) and value_range > range_epsilon),
                    "past_n": int(len(past)),
                }
            )
    return pd.DataFrame(rows)


def compute_lhi_scores(
    forecasts: pd.DataFrame,
    past_ranges: pd.DataFrame,
    condition_means: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keyed = forecasts.merge(
        past_ranges,
        on=["fd", "unit_id", "context_start_cycle", "cutoff_cycle", "sensor"],
        how="left",
        validate="many_to_one",
    )
    if keyed["past_min"].isna().any() or keyed["past_range"].isna().any():
        raise ValueError("Missing past-context min-max ranges for some forecast rows.")
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
            first_start = int(monitor["forecast_start_cycle"].min())
            baseline_group = monitor[monitor["forecast_start_cycle"] == first_start].copy()
            baseline_source = f"first_monitor_forecast_block_start_{first_start}"
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


def plot_units(scores: pd.DataFrame, units: Sequence[Tuple[str, int]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    drift_dir = output_dir / "unit_drift"
    lhi_dir = output_dir / "unit_lhi"
    drift_dir.mkdir(parents=True, exist_ok=True)
    lhi_dir.mkdir(parents=True, exist_ok=True)
    colors = {"past_only": "#1f77b4", "known_future": "#ff7f0e", "none": "#2ca02c"}
    baseline_styles = {"past_only": "--", "known_future": "-", "none": ":"}
    for fd_name, unit_id in units:
        unit_df = scores[(scores["fd"] == fd_name) & (scores["unit_id"] == unit_id)]
        if unit_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        for mode, group in unit_df.groupby("covariate_mode", sort=True):
            group = group.sort_values(["cycle", "forecast_start_cycle"])
            color = colors.get(str(mode))
            ax.plot(group["cycle"], group["d_rmse"], color=color, alpha=0.25, linewidth=0.9, label=f"{mode} D_RMSE raw")
            ax.plot(group["cycle"], group["d_rmse_roll_mean"], color=color, linewidth=1.8, label=f"{mode} D_RMSE rolling")
            baseline = float(group["b_rmse"].iloc[0])
            ax.axhline(
                baseline,
                color="#ef4444",
                linestyle=baseline_styles.get(str(mode), "-"),
                linewidth=1.6,
                label=f"{mode} B_RMSE={baseline:.4g}",
            )
        ax.set_title(f"Condition-matched min-max drift: {fd_name} unit {unit_id}")
        ax.set_xlabel("target cycle")
        ax.set_ylabel("D_RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(drift_dir / f"{fd_name}_unit{unit_id}_drift.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhline(0.0, color="#6b7280", linewidth=1.0, label="healthy baseline")
        for mode, group in unit_df.groupby("covariate_mode", sort=True):
            group = group.sort_values(["cycle", "forecast_start_cycle"])
            color = colors.get(str(mode))
            ax.plot(group["cycle"], group["lhi_rmse"], color=color, alpha=0.25, linewidth=0.9, label=f"{mode} LHI raw")
            ax.plot(group["cycle"], group["lhi_rmse_roll_mean"], color=color, linewidth=1.8, label=f"{mode} LHI rolling")
        ax.set_title(f"Log-ratio LHI: {fd_name} unit {unit_id}")
        ax.set_xlabel("target cycle")
        ax.set_ylabel("log((D_RMSE + eps) / (B_RMSE + eps))")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(lhi_dir / f"{fd_name}_unit{unit_id}_lhi.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.range_epsilon <= 0:
        raise ValueError("--range_epsilon must be positive.")
    if args.lhi_epsilon <= 0:
        raise ValueError("--lhi_epsilon must be positive.")
    output_dir = args.output_dir or (args.forecast_dir / "lhi")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    for legacy_name in (
        "unit_healthy_minmax_ranges.csv",
        "healthy_baseline_points.csv",
        "healthy_reference_baselines.csv",
    ):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    print(f"Loading {args.forecast_dir / 'window_forecasts.csv'}...", flush=True)
    forecasts = load_window_forecasts(args)
    run_config_path = args.forecast_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    eval_split = str(run_config.get("eval_split", "train"))
    frames = load_eval_frames(args.data_dir, eval_split, args.fds)
    condition_means = build_condition_means(frames, args.sensors, args.healthy_cycles)
    past_ranges = compute_past_sensor_ranges(frames, forecasts, args.sensors, args.range_epsilon)
    all_scores, sensor_scores = compute_lhi_scores(
        forecasts=forecasts,
        past_ranges=past_ranges,
        condition_means=condition_means,
    )
    baseline_points, baselines = compute_initial_forecast_baselines(
        all_scores,
        healthy_cycles=args.healthy_cycles,
        baseline_cycles=args.baseline_cycles,
    )
    scores = all_scores[all_scores["cycle"] > args.healthy_cycles].copy()
    if scores.empty:
        raise ValueError(f"No monitor forecast rows found for cycle > {args.healthy_cycles}.")
    scores = add_lhi_columns(scores, baselines, args.lhi_epsilon)
    scores = add_rolling_scores(scores, args.rolling_window)
    fd_summary = summarize_fd(scores)

    past_ranges.to_csv(output_dir / "past_minmax_ranges.csv", index=False)
    condition_means.to_csv(output_dir / "healthy_condition_means.csv", index=False)
    baseline_points.to_csv(output_dir / "baseline_forecast_points.csv", index=False)
    baselines.to_csv(output_dir / "lhi_baselines.csv", index=False)
    scores.to_csv(output_dir / "lhi_scores.csv", index=False)
    sensor_scores.to_csv(output_dir / "sensor_lhi_components.csv", index=False)
    fd_summary.to_csv(output_dir / "fd_lhi_summary.csv", index=False)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    units = parse_plot_units(args.plot_units, scores, args.plot_examples)
    plot_units(scores, units, output_dir)

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

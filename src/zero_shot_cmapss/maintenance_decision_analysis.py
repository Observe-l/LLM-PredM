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
    from src.zero_shot_cmapss.plot_operating_condition_clusters import DEFAULT_SENSORS, FD_NAMES, load_cmapss_file, make_condition_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze condition-matched drift as a maintenance decision signal."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--forecast_dir", type=Path, default=Path("outputs/cluster_roll_10"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--fds", nargs="+", default=list(FD_NAMES), choices=list(FD_NAMES))
    parser.add_argument(
        "--covariate_mode",
        choices=["cluster_covariate", "future_covariate", "no_covariate", "known_future"],
        default="cluster_covariate",
        help="Forecast experiment mode used for the decision signal. known_future is accepted as a legacy alias.",
    )
    parser.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS)
    parser.add_argument(
        "--healthy_cycles",
        type=int,
        default=50,
        help="First N cycles per unit used to build the healthy condition reference.",
    )
    parser.add_argument(
        "--start_cycle",
        type=int,
        default=20,
        help="Earliest forecast target cycle included in the maintenance signal.",
    )
    parser.add_argument(
        "--history_windows",
        type=int,
        default=3,
        help="Minimum previous RMSE windows required before theta_0.95/theta_0.99 are available.",
    )
    parser.add_argument("--slope_window", type=int, default=3)
    parser.add_argument(
        "--lhi_epsilon",
        type=float,
        default=1e-6,
        help="Small positive epsilon used in log-ratio LHI = log((RMSE + eps) / (B + eps)).",
    )
    parser.add_argument(
        "--drift_epsilon",
        type=float,
        default=1e-6,
        help="Small positive epsilon used in D = abs(y_pred - healthy_mean) / (healthy_std + eps).",
    )
    parser.add_argument(
        "--min_healthy_reference_n",
        type=int,
        default=3,
        help="Minimum healthy samples required for a condition-matched mean/std reference.",
    )
    parser.add_argument(
        "--lhi_rolling_window",
        type=int,
        default=5,
        help="Rolling window over target cycles used to smooth log-ratio LHI plots and summaries.",
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def load_decision_forecasts(args: argparse.Namespace) -> Tuple[pd.DataFrame, str]:
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
    }
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    mode = "cluster_covariate" if args.covariate_mode == "known_future" else args.covariate_mode
    forecasts = forecasts[
        (forecasts["covariate_mode"] == mode)
        & forecasts["fd"].isin(args.fds)
        & forecasts["sensor"].isin(args.sensors)
    ].copy()
    if forecasts.empty:
        raise ValueError(f"No {mode} forecast rows found in {path}.")

    run_config_path = args.forecast_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    return forecasts, str(run_config.get("eval_split", "train"))


def build_condition_healthy_reference(forecasts: pd.DataFrame, healthy_cycles: int) -> pd.DataFrame:
    healthy = forecasts[forecasts["cycle"] <= healthy_cycles].copy()
    if healthy.empty:
        raise ValueError(f"No forecast rows found for cycle <= --healthy_cycles ({healthy_cycles}).")
    healthy = healthy.drop_duplicates(["fd", "unit_id", "cycle", "sensor", "op_condition_key"])
    reference = (
        healthy.groupby(["fd", "unit_id", "op_condition_key", "sensor"], sort=True)
        .agg(
            healthy_mean_raw=("y_true", "mean"),
            healthy_std_raw=("y_true", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
            healthy_n=("y_true", "size"),
        )
        .reset_index()
    )
    return reference


def load_eval_frames(data_dir: Path, eval_split: str, fds: Sequence[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for fd_name in fds:
        frame = load_cmapss_file(data_dir / fd_name / f"{eval_split}_{fd_name}.txt")
        frame = frame.copy()
        frame["op_condition_key"] = make_condition_keys(frame)
        frames[fd_name] = frame
    return frames


def build_condition_healthy_reference_from_frames(
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
        raise ValueError(f"No healthy rows found for cycle <= --healthy_cycles ({healthy_cycles}).")
    healthy_actuals = pd.concat(rows, ignore_index=True)
    return (
        healthy_actuals.groupby(["fd", "unit_id", "op_condition_key", "sensor"], sort=True)
        .agg(
            healthy_mean_raw=("y_true", "mean"),
            healthy_std_raw=("y_true", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
            healthy_n=("y_true", "size"),
        )
        .reset_index()
    )


def summarize_condition_matched_healthy_std_drift(
    forecasts: pd.DataFrame,
    healthy_reference: pd.DataFrame,
    epsilon: float,
    min_healthy_reference_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if epsilon <= 0:
        raise ValueError("--drift_epsilon must be positive.")
    if min_healthy_reference_n < 2:
        raise ValueError("--min_healthy_reference_n must be >= 2 for a meaningful healthy std.")
    keyed = forecasts.merge(
        healthy_reference,
        on=["fd", "unit_id", "op_condition_key", "sensor"],
        how="left",
        validate="many_to_one",
    )
    keyed["drift"] = (keyed["y_pred"] - keyed["healthy_mean_raw"]).abs() / (
        keyed["healthy_std_raw"] + epsilon
    )
    keyed["drift_sq"] = keyed["drift"] ** 2

    available = keyed[
        keyed["healthy_mean_raw"].notna()
        & (keyed["healthy_n"] >= min_healthy_reference_n)
        & (keyed["healthy_std_raw"] > epsilon)
    ].copy()
    window_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"]
    sensor_cols = [*window_cols, "sensor"]
    sensor_round = (
        available.groupby(sensor_cols, sort=True)
        .agg(
            mae=("drift", "mean"),
            rmse=("drift_sq", lambda s: float(np.sqrt(np.mean(s)))),
            n=("drift", "size"),
            healthy_n_min=("healthy_n", "min"),
            healthy_std_median=("healthy_std_raw", "median"),
        )
        .reset_index()
    )
    overall_round = (
        available.groupby(window_cols, sort=True)
        .agg(
            mae=("drift", "mean"),
            rmse=("drift_sq", lambda s: float(np.sqrt(np.mean(s)))),
            n=("drift", "size"),
            healthy_n_min=("healthy_n", "min"),
            healthy_std_median=("healthy_std_raw", "median"),
        )
        .reset_index()
    )
    overall_round["sensor"] = "ALL"
    return keyed, sensor_round, overall_round[[*sensor_cols, "mae", "rmse", "n", "healthy_n_min", "healthy_std_median"]]


def build_decision_rmse(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecasts, _ = load_decision_forecasts(args)
    reference_source = forecasts.copy()

    forecasts = forecasts[forecasts["cycle"] >= args.start_cycle].copy()
    if forecasts.empty:
        raise ValueError(f"No forecast rows remain after filtering to cycle >= --start_cycle ({args.start_cycle}).")

    healthy_reference = build_condition_healthy_reference(reference_source, args.healthy_cycles)
    drift_rows, drift_sensor, drift_overall = summarize_condition_matched_healthy_std_drift(
        forecasts=forecasts,
        healthy_reference=healthy_reference,
        epsilon=args.drift_epsilon,
        min_healthy_reference_n=args.min_healthy_reference_n,
    )
    metrics = pd.concat([drift_overall, drift_sensor], ignore_index=True)
    metrics = metrics[
        (metrics["covariate_mode"] == ("cluster_covariate" if args.covariate_mode == "known_future" else args.covariate_mode))
        & (metrics["sensor"] == "ALL")
    ].copy()
    metrics = metrics.sort_values(["fd", "unit_id", "cycle", "forecast_start_cycle"]).reset_index(drop=True)
    return metrics, drift_rows, healthy_reference


def iter_decision_forecast_chunks(args: argparse.Namespace):
    path = args.forecast_dir / "window_forecasts.csv"
    required = [
        "covariate_mode",
        "fd",
        "unit_id",
        "cutoff_cycle",
        "forecast_start_cycle",
        "cycle",
        "sensor",
        "y_pred",
        "op_condition_key",
    ]
    mode = "cluster_covariate" if args.covariate_mode == "known_future" else args.covariate_mode
    for chunk in pd.read_csv(path, usecols=required, chunksize=args.chunksize):
        chunk = chunk[
            (chunk["covariate_mode"] == mode)
            & chunk["fd"].isin(args.fds)
            & chunk["sensor"].isin(args.sensors)
            & (chunk["cycle"] >= args.start_cycle)
        ].copy()
        if not chunk.empty:
            yield chunk


def summarize_decision_drift_streaming(
    args: argparse.Namespace,
    healthy_reference: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    drift_path = output_dir / "decision_condition_matched_drift_rows.csv"
    if drift_path.exists():
        drift_path.unlink()
    partials = []
    for chunk_idx, chunk in enumerate(iter_decision_forecast_chunks(args), start=1):
        keyed = chunk.merge(
            healthy_reference,
            on=["fd", "unit_id", "op_condition_key", "sensor"],
            how="left",
            validate="many_to_one",
        )
        keyed["drift"] = (keyed["y_pred"] - keyed["healthy_mean_raw"]).abs() / (
            keyed["healthy_std_raw"] + args.drift_epsilon
        )
        keyed["drift_sq"] = keyed["drift"] ** 2
        available = keyed[
            keyed["healthy_mean_raw"].notna()
            & (keyed["healthy_n"] >= args.min_healthy_reference_n)
            & (keyed["healthy_std_raw"] > args.drift_epsilon)
        ].copy()
        keyed.to_csv(drift_path, index=False, mode="a", header=not drift_path.exists())
        if not available.empty:
            group_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle", "sensor"]
            partials.append(
                available.groupby(group_cols, sort=True)
                .agg(
                    drift_sum=("drift", "sum"),
                    drift_sq_sum=("drift_sq", "sum"),
                    n=("drift", "size"),
                    healthy_n_min=("healthy_n", "min"),
                    healthy_std_median=("healthy_std_raw", "median"),
                )
                .reset_index()
            )
        print(f"  decision chunk {chunk_idx}: rows={len(chunk):,}", flush=True)
    if not partials:
        raise ValueError("No decision drift rows were produced.")
    partial = pd.concat(partials, ignore_index=True)
    group_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle", "sensor"]
    sensor_round = (
        partial.groupby(group_cols, sort=True)
        .agg(
            drift_sum=("drift_sum", "sum"),
            drift_sq_sum=("drift_sq_sum", "sum"),
            n=("n", "sum"),
            healthy_n_min=("healthy_n_min", "min"),
            healthy_std_median=("healthy_std_median", "median"),
        )
        .reset_index()
    )
    sensor_round["mae"] = sensor_round["drift_sum"] / sensor_round["n"]
    sensor_round["rmse"] = np.sqrt(sensor_round["drift_sq_sum"] / sensor_round["n"])
    window_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"]
    overall = (
        sensor_round.groupby(window_cols, sort=True)
        .agg(
            drift_sum=("drift_sum", "sum"),
            drift_sq_sum=("drift_sq_sum", "sum"),
            n=("n", "sum"),
            healthy_n_min=("healthy_n_min", "min"),
            healthy_std_median=("healthy_std_median", "median"),
        )
        .reset_index()
    )
    overall["mae"] = overall["drift_sum"] / overall["n"]
    overall["rmse"] = np.sqrt(overall["drift_sq_sum"] / overall["n"])
    overall["sensor"] = "ALL"
    metrics = overall[[*window_cols, "sensor", "mae", "rmse", "n", "healthy_n_min", "healthy_std_median"]].copy()
    mode = "cluster_covariate" if args.covariate_mode == "known_future" else args.covariate_mode
    metrics = metrics[metrics["covariate_mode"] == mode].sort_values(
        ["fd", "unit_id", "cycle", "forecast_start_cycle"]
    ).reset_index(drop=True)
    drift_sensor = sensor_round[[*group_cols, "mae", "rmse", "n", "healthy_n_min", "healthy_std_median"]].copy()
    return metrics, drift_sensor


def add_dynamic_thresholds_and_slopes(
    metrics: pd.DataFrame,
    min_history_windows: int,
    slope_window: int,
    healthy_cycles: int,
) -> pd.DataFrame:
    if min_history_windows < 1:
        raise ValueError("--history_windows must be >= 1.")
    rows = []
    for _, group in metrics.groupby(["fd", "unit_id"], sort=True):
        group = group.sort_values(["cycle", "forecast_start_cycle"]).copy()
        cycles = group["cycle"].to_numpy(dtype=np.int64)
        cutoffs = group["cutoff_cycle"].to_numpy(dtype=np.int64)
        rmse = group["rmse"].to_numpy(dtype=float)
        history_pos = np.searchsorted(cycles, cutoffs, side="right")
        expanding_q95 = pd.Series(rmse).expanding(min_periods=min_history_windows).quantile(0.95).to_numpy()
        expanding_q99 = pd.Series(rmse).expanding(min_periods=min_history_windows).quantile(0.99).to_numpy()
        theta95 = np.full(len(group), np.nan, dtype=float)
        theta99 = np.full(len(group), np.nan, dtype=float)
        valid = history_pos >= min_history_windows
        theta95[valid] = expanding_q95[history_pos[valid] - 1]
        theta99[valid] = expanding_q99[history_pos[valid] - 1]

        group["history_count"] = history_pos
        group["healthy_reference_window"] = group["cycle"] <= healthy_cycles
        group["theta_0_95"] = theta95
        group["theta_0_99"] = theta99
        group["exceeds_theta_0_95"] = group["rmse"] > group["theta_0_95"]
        group["exceeds_theta_0_99"] = group["rmse"] > group["theta_0_99"]
        group.loc[group["theta_0_95"].isna(), "exceeds_theta_0_95"] = False
        group.loc[group["theta_0_99"].isna(), "exceeds_theta_0_99"] = False

        delta_t = group["cycle"].diff().replace(0, np.nan)
        group["rmse_slope"] = group["rmse"].diff() / delta_t
        group["rmse_slope_roll"] = group["rmse_slope"].rolling(slope_window, min_periods=1).mean()
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def add_log_ratio_lhi(metrics: pd.DataFrame, healthy_cycles: int, epsilon: float, rolling_window: int) -> pd.DataFrame:
    if epsilon <= 0:
        raise ValueError("--lhi_epsilon must be positive.")
    if rolling_window < 1:
        raise ValueError("--lhi_rolling_window must be >= 1.")
    rows = []
    for _, group in metrics.groupby(["fd", "unit_id"], sort=True):
        group = group.sort_values(["cycle", "forecast_start_cycle"]).copy()
        healthy = group[group["cycle"] <= healthy_cycles]
        if healthy.empty:
            baseline = float(group["rmse"].median())
            baseline_source = "all_median_fallback"
        else:
            latest_healthy_cutoff = healthy["cutoff_cycle"].max()
            latest_healthy = healthy[healthy["cutoff_cycle"] == latest_healthy_cutoff]
            baseline = float(latest_healthy["rmse"].median())
            baseline_source = f"latest_healthy_cutoff_{int(latest_healthy_cutoff)}"
        group["lhi_baseline_rmse"] = baseline
        group["lhi_baseline_source"] = baseline_source
        group["log_ratio_lhi"] = np.log((group["rmse"] + epsilon) / (baseline + epsilon))
        group["log_ratio_lhi_roll_mean"] = np.nan
        monitor_mask = group["cycle"] > healthy_cycles
        group.loc[monitor_mask, "log_ratio_lhi_roll_mean"] = (
            group.loc[monitor_mask, "log_ratio_lhi"].rolling(rolling_window, min_periods=1).mean()
        )
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def summarize_units(metrics: pd.DataFrame, healthy_cycles: int) -> pd.DataFrame:
    rows = []
    for (fd_name, unit_id), group in metrics.groupby(["fd", "unit_id"], sort=True):
        group = group.sort_values(["cycle", "forecast_start_cycle"])
        monitor = group[group["cycle"] > healthy_cycles]
        first95 = monitor.loc[monitor["exceeds_theta_0_95"], "cycle"]
        first99 = monitor.loc[monitor["exceeds_theta_0_99"], "cycle"]
        valid_theta95 = group["theta_0_95"].dropna()
        valid_theta99 = group["theta_0_99"].dropna()
        last_theta95 = float(valid_theta95.iloc[-1]) if not valid_theta95.empty else np.nan
        last_theta99 = float(valid_theta99.iloc[-1]) if not valid_theta99.empty else np.nan
        healthy = group[group["healthy_reference_window"]]
        monitor_lhi = monitor["log_ratio_lhi"].dropna() if "log_ratio_lhi" in monitor else pd.Series(dtype=float)
        summary_source = monitor if not monitor.empty else group.iloc[0:0]
        rows.append(
            {
                "fd": fd_name,
                "unit_id": int(unit_id),
                "last_theta_0_95": last_theta95,
                "last_theta_0_99": last_theta99,
                "healthy_window_rmse_median": float(healthy["rmse"].median()) if not healthy.empty else np.nan,
                "lhi_baseline_rmse": float(group["lhi_baseline_rmse"].iloc[0]) if "lhi_baseline_rmse" in group else np.nan,
                "lhi_baseline_source": str(group["lhi_baseline_source"].iloc[0]) if "lhi_baseline_source" in group else "",
                "last_log_ratio_lhi": float(summary_source["log_ratio_lhi"].iloc[-1]) if "log_ratio_lhi" in summary_source and not summary_source.empty else np.nan,
                "max_log_ratio_lhi": float(summary_source["log_ratio_lhi"].max()) if "log_ratio_lhi" in summary_source and not summary_source.empty else np.nan,
                "monitor_log_ratio_lhi_median": float(monitor_lhi.median()) if not monitor_lhi.empty else np.nan,
                "first_theta_0_95_cycle": int(first95.iloc[0]) if not first95.empty else np.nan,
                "first_theta_0_99_cycle": int(first99.iloc[0]) if not first99.empty else np.nan,
                "last_rmse": float(summary_source["rmse"].iloc[-1]) if not summary_source.empty else np.nan,
                "max_rmse": float(summary_source["rmse"].max()) if not summary_source.empty else np.nan,
                "max_rmse_over_last_theta_0_95": float(summary_source["rmse"].max() / last_theta95) if last_theta95 > 0 and not summary_source.empty else np.nan,
                "max_rmse_over_last_theta_0_99": float(summary_source["rmse"].max() / last_theta99) if last_theta99 > 0 and not summary_source.empty else np.nan,
                "exceed_theta_0_95_rate": float(summary_source["exceeds_theta_0_95"].mean()) if not summary_source.empty else np.nan,
                "exceed_theta_0_99_rate": float(summary_source["exceeds_theta_0_99"].mean()) if not summary_source.empty else np.nan,
                "max_positive_slope": float(summary_source["rmse_slope_roll"].max(skipna=True)) if not summary_source.empty else np.nan,
                "last_slope": float(summary_source["rmse_slope_roll"].iloc[-1]) if not summary_source.empty else np.nan,
                "num_rounds": int(len(summary_source)),
            }
        )
    return pd.DataFrame(rows)


def summarize_fd(metrics: pd.DataFrame, healthy_cycles: int) -> pd.DataFrame:
    rows = []
    metrics = metrics[metrics["cycle"] > healthy_cycles].copy()
    for (fd_name, cycle), group in metrics.groupby(["fd", "cycle"], sort=True):
        valid95 = group["theta_0_95"].notna()
        valid99 = group["theta_0_99"].notna()
        rows.append(
            {
                "fd": fd_name,
                "cycle": int(cycle),
                "forecast_start_cycle": int(group["forecast_start_cycle"].min()),
                "unit_count": int(group["unit_id"].nunique()),
                "median_rmse": float(group["rmse"].median()),
                "q90_rmse": float(group["rmse"].quantile(0.90)),
                "q95_rmse": float(group["rmse"].quantile(0.95)),
                "median_log_ratio_lhi": float(group["log_ratio_lhi"].median()) if "log_ratio_lhi" in group else np.nan,
                "q90_log_ratio_lhi": float(group["log_ratio_lhi"].quantile(0.90)) if "log_ratio_lhi" in group else np.nan,
                "q95_log_ratio_lhi": float(group["log_ratio_lhi"].quantile(0.95)) if "log_ratio_lhi" in group else np.nan,
                "median_log_ratio_lhi_roll_mean": float(group["log_ratio_lhi_roll_mean"].median()) if "log_ratio_lhi_roll_mean" in group else np.nan,
                "q90_log_ratio_lhi_roll_mean": float(group["log_ratio_lhi_roll_mean"].quantile(0.90)) if "log_ratio_lhi_roll_mean" in group else np.nan,
                "q95_log_ratio_lhi_roll_mean": float(group["log_ratio_lhi_roll_mean"].quantile(0.95)) if "log_ratio_lhi_roll_mean" in group else np.nan,
                "median_theta_0_95": float(group.loc[valid95, "theta_0_95"].median()) if valid95.any() else np.nan,
                "median_theta_0_99": float(group.loc[valid99, "theta_0_99"].median()) if valid99.any() else np.nan,
                "theta_0_95_exceed_unit_rate": float(group["exceeds_theta_0_95"].mean()),
                "theta_0_99_exceed_unit_rate": float(group["exceeds_theta_0_99"].mean()),
                "median_slope": float(group["rmse_slope_roll"].median(skipna=True)),
                "q90_slope": float(group["rmse_slope_roll"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def plot_threshold_examples(metrics: pd.DataFrame, units: Sequence[Tuple[str, int]], output_dir: Path, healthy_cycles: int) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "unit_thresholds"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for fd_name, unit_id in units:
        unit_df = metrics[
            (metrics["fd"] == fd_name)
            & (metrics["unit_id"] == unit_id)
            & (metrics["cycle"] > healthy_cycles)
        ].sort_values(["cycle", "forecast_start_cycle"])
        if unit_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        mode = str(unit_df["covariate_mode"].iloc[0]) if "covariate_mode" in unit_df else "decision"
        ax.plot(unit_df["cycle"], unit_df["rmse"], color="#1f2937", linewidth=1.8, label=f"{mode} RMSE")
        ax.scatter(unit_df["cycle"], unit_df["rmse"], color="#d97706", s=12, alpha=0.75, label="target RMSE")
        ax.plot(unit_df["cycle"], unit_df["theta_0_95"], color="#dc2626", linestyle="--", linewidth=1.3, label=r"$\theta_{0.95}$")
        ax.plot(unit_df["cycle"], unit_df["theta_0_99"], color="#7f1d1d", linestyle=":", linewidth=1.5, label=r"$\theta_{0.99}$")
        ax.set_title(f"Dynamic threshold decision signal: {fd_name} unit {unit_id}")
        ax.set_xlabel("target cycle")
        ax.set_ylabel("condition matched drift RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{fd_name}_unit{unit_id}_thresholds.png", dpi=160)
        plt.close(fig)


def plot_slope_examples(metrics: pd.DataFrame, units: Sequence[Tuple[str, int]], output_dir: Path, healthy_cycles: int) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "unit_slopes"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for fd_name, unit_id in units:
        unit_df = metrics[
            (metrics["fd"] == fd_name)
            & (metrics["unit_id"] == unit_id)
            & (metrics["cycle"] > healthy_cycles)
        ].sort_values(["cycle", "forecast_start_cycle"])
        if unit_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhline(0.0, color="#6b7280", linewidth=1.0)
        ax.plot(unit_df["cycle"], unit_df["rmse_slope"], color="#9ca3af", linewidth=1.0, label="raw d(RMSE)/dt")
        ax.plot(unit_df["cycle"], unit_df["rmse_slope_roll"], color="#7c3aed", linewidth=1.8, label="rolling slope")
        ax.set_title(f"RMSE slope: {fd_name} unit {unit_id}")
        ax.set_xlabel("target cycle")
        ax.set_ylabel("d(condition matched drift RMSE) / dt")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{fd_name}_unit{unit_id}_slope.png", dpi=160)
        plt.close(fig)


def plot_lhi_examples(metrics: pd.DataFrame, units: Sequence[Tuple[str, int]], output_dir: Path, healthy_cycles: int) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "unit_lhi"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for fd_name, unit_id in units:
        unit_df = metrics[
            (metrics["fd"] == fd_name)
            & (metrics["unit_id"] == unit_id)
            & (metrics["cycle"] > healthy_cycles)
        ].sort_values(["cycle", "forecast_start_cycle"])
        if unit_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhline(0.0, color="#6b7280", linewidth=1.0, label="healthy baseline")
        ax.plot(unit_df["cycle"], unit_df["log_ratio_lhi"], color="#94a3b8", linewidth=0.9, alpha=0.45, label="raw LHI")
        ax.plot(unit_df["cycle"], unit_df["log_ratio_lhi_roll_mean"], color="#0f172a", linewidth=2.0, label="rolling LHI")
        baseline = float(unit_df["lhi_baseline_rmse"].iloc[0])
        ax.set_title(f"Log-ratio Relative Health Index: {fd_name} unit {unit_id} (B={baseline:.4g})")
        ax.set_xlabel("target cycle")
        ax.set_ylabel(r"$\log((HI+\epsilon)/(B+\epsilon))$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{fd_name}_unit{unit_id}_log_ratio_lhi.png", dpi=160)
        plt.close(fig)


def plot_fd_overview(fd_summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    for ax, fd_name in zip(axes.ravel(), FD_NAMES):
        fd_df = fd_summary[fd_summary["fd"] == fd_name].sort_values(["cycle", "forecast_start_cycle"])
        if fd_df.empty:
            ax.set_title(fd_name)
            ax.axis("off")
            continue
        ax.plot(fd_df["cycle"], fd_df["median_rmse"], color="#111827", label="median RMSE")
        ax.plot(fd_df["cycle"], fd_df["q95_rmse"], color="#4b5563", linestyle="--", label="q95 RMSE")
        ax.plot(fd_df["cycle"], fd_df["median_theta_0_95"], color="#dc2626", linestyle="--", label="median theta0.95")
        ax.plot(fd_df["cycle"], fd_df["median_theta_0_99"], color="#7f1d1d", linestyle=":", label="median theta0.99")
        ax.plot(fd_df["cycle"], fd_df["theta_0_95_exceed_unit_rate"], color="#f97316", label="theta0.95 exceed rate")
        ax.set_title(fd_name)
        ax.set_xlabel("target cycle")
        ax.set_ylabel("RMSE / threshold / exceed rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Known-future condition matched drift: dynamic threshold overview")
    fig.tight_layout()
    fig.savefig(output_dir / "fd_threshold_overview.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    for ax, fd_name in zip(axes.ravel(), FD_NAMES):
        fd_df = fd_summary[fd_summary["fd"] == fd_name].sort_values(["cycle", "forecast_start_cycle"])
        if fd_df.empty:
            ax.set_title(fd_name)
            ax.axis("off")
            continue
        ax.axhline(0.0, color="#6b7280", linewidth=1.0)
        ax.plot(fd_df["cycle"], fd_df["median_slope"], color="#7c3aed", label="median slope")
        ax.plot(fd_df["cycle"], fd_df["q90_slope"], color="#a855f7", linestyle="--", label="q90 slope")
        ax.set_title(fd_name)
        ax.set_xlabel("target cycle")
        ax.set_ylabel("d(RMSE)/dt")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Known-future condition matched drift: slope overview")
    fig.tight_layout()
    fig.savefig(output_dir / "fd_slope_overview.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    for ax, fd_name in zip(axes.ravel(), FD_NAMES):
        fd_df = fd_summary[fd_summary["fd"] == fd_name].sort_values(["cycle", "forecast_start_cycle"])
        if fd_df.empty:
            ax.set_title(fd_name)
            ax.axis("off")
            continue
        ax.axhline(0.0, color="#6b7280", linewidth=1.0, label="healthy baseline")
        ax.plot(fd_df["cycle"], fd_df["median_log_ratio_lhi"], color="#94a3b8", alpha=0.35, linewidth=0.9, label="median raw LHI")
        ax.plot(fd_df["cycle"], fd_df["median_log_ratio_lhi_roll_mean"], color="#111827", label="median rolling LHI")
        ax.plot(fd_df["cycle"], fd_df["q90_log_ratio_lhi_roll_mean"], color="#4b5563", linestyle="--", label="q90 rolling LHI")
        ax.plot(fd_df["cycle"], fd_df["q95_log_ratio_lhi_roll_mean"], color="#991b1b", linestyle=":", label="q95 rolling LHI")
        ax.set_title(fd_name)
        ax.set_xlabel("target cycle")
        ax.set_ylabel(r"$\log((HI+\epsilon)/(B+\epsilon))$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Known-future condition matched drift: log-ratio Relative Health Index")
    fig.tight_layout()
    fig.savefig(output_dir / "fd_log_ratio_lhi_overview.png", dpi=160)
    plt.close(fig)


def plot_engine_scatter(unit_summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"FD001": "#2563eb", "FD002": "#dc2626", "FD003": "#16a34a", "FD004": "#d97706"}
    fig, ax = plt.subplots(figsize=(9, 7))
    for fd_name, group in unit_summary.groupby("fd", sort=True):
        ax.scatter(
            group["max_rmse_over_last_theta_0_95"],
            group["max_positive_slope"],
            s=28,
            alpha=0.72,
            color=colors.get(fd_name),
            label=fd_name,
        )
    ax.axvline(1.0, color="#991b1b", linestyle="--", linewidth=1.2, label="theta0.95 crossing")
    ax.axhline(0.0, color="#6b7280", linewidth=1.0)
    ax.set_xlabel("max RMSE / last theta_0.95")
    ax.set_ylabel("max rolling d(RMSE)/dt")
    ax.set_title("Engine-level maintenance signal scatter")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "engine_threshold_slope_scatter.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.forecast_dir / "maintenance_decision")
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "cluster_covariate" if args.covariate_mode == "known_future" else args.covariate_mode
    for legacy_name in ("known_future_rmse_with_thresholds.csv", "rmse_compare_with_evaluation.csv"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    run_config_path = args.forecast_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    eval_split = str(run_config.get("eval_split", "train"))
    frames = load_eval_frames(args.data_dir, eval_split, args.fds)
    healthy_reference = build_condition_healthy_reference_from_frames(frames, args.sensors, args.healthy_cycles)
    metrics, drift_sensor = summarize_decision_drift_streaming(
        args=args,
        healthy_reference=healthy_reference,
        output_dir=output_dir,
    )
    metrics = add_dynamic_thresholds_and_slopes(
        metrics=metrics,
        min_history_windows=args.history_windows,
        slope_window=args.slope_window,
        healthy_cycles=args.healthy_cycles,
    )
    metrics = add_log_ratio_lhi(metrics, args.healthy_cycles, args.lhi_epsilon, args.lhi_rolling_window)
    unit_summary = summarize_units(metrics, args.healthy_cycles)
    fd_summary = summarize_fd(metrics, args.healthy_cycles)

    metrics.to_csv(output_dir / f"{mode}_rmse_with_dynamic_thresholds.csv", index=False)
    metrics.to_csv(output_dir / f"{mode}_log_ratio_lhi.csv", index=False)
    metrics.to_csv(output_dir / "decision_log_ratio_lhi.csv", index=False)
    metrics[metrics["cycle"] > args.healthy_cycles].to_csv(
        output_dir / f"{mode}_log_ratio_lhi_monitoring.csv",
        index=False,
    )
    drift_sensor.to_csv(output_dir / f"{mode}_condition_matched_drift_sensor_metrics.csv", index=False)
    healthy_reference.to_csv(output_dir / "healthy_condition_reference.csv", index=False)
    unit_summary.to_csv(output_dir / "engine_maintenance_signal_summary.csv", index=False)
    fd_summary.to_csv(output_dir / "fd_maintenance_signal_summary.csv", index=False)

    print("Engine-level signal summary by FD:", flush=True)
    print(
        unit_summary.groupby("fd", sort=True)
        .agg(
            engines=("unit_id", "nunique"),
            theta95_cross_rate=("first_theta_0_95_cycle", lambda s: float(s.notna().mean())),
            theta99_cross_rate=("first_theta_0_99_cycle", lambda s: float(s.notna().mean())),
            median_max_rmse_over_last_theta95=("max_rmse_over_last_theta_0_95", "median"),
            median_max_positive_slope=("max_positive_slope", "median"),
        )
        .reset_index()
        .to_string(index=False),
        flush=True,
    )
    print(f"Saved maintenance decision analysis outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

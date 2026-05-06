from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

FD_NAMES = ("FD001", "FD002", "FD003", "FD004")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch plot unit-level LHI curves from lhi_indicator.py outputs."
    )
    parser.add_argument("--lhi_dir", type=Path, default=Path("outputs/cluster_20/lhi"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--fd", choices=list(FD_NAMES), required=True)
    parser.add_argument("--unit_start", type=int, required=True)
    parser.add_argument("--unit_end", type=int, required=True)
    parser.add_argument(
        "--covariate_modes",
        nargs="+",
        default=None,
        help="Covariate modes to plot. Defaults to all modes present in lhi_scores.csv.",
    )
    parser.add_argument(
        "--metric",
        choices=["rmse", "mae"],
        default="rmse",
        help="Plot D_RMSE/LHI_RMSE or D_MAE/LHI_MAE.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--fig_width", type=float, default=12.0)
    parser.add_argument("--fig_height", type=float, default=8.0)
    parser.add_argument(
        "--plot_stride",
        type=int,
        default=0,
        help="Forecast-start stride for plotting. 0 means read prediction_length from the source run_config.json.",
    )
    parser.add_argument(
        "--healthy_cycles",
        type=int,
        default=50,
        help="Only plot cycles after this healthy reference interval. Default: 50.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.unit_start <= 0 or args.unit_end <= 0:
        raise ValueError("--unit_start and --unit_end must be positive.")
    if args.unit_end < args.unit_start:
        raise ValueError("--unit_end must be >= --unit_start.")


def load_scores(lhi_dir: Path, fd_name: str, unit_start: int, unit_end: int, modes: Sequence[str] | None) -> pd.DataFrame:
    path = lhi_dir / "lhi_scores.csv"
    scores = pd.read_csv(path)
    required = {
        "covariate_mode",
        "fd",
        "unit_id",
        "cycle",
        "d_mae",
        "d_rmse",
        "b_mae",
        "b_rmse",
        "lhi_mae",
        "lhi_rmse",
        "d_mae_roll_mean",
        "d_rmse_roll_mean",
        "lhi_mae_roll_mean",
        "lhi_rmse_roll_mean",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    mode_filter = scores["covariate_mode"].isin(modes) if modes else True
    scores = scores[
        (scores["fd"] == fd_name)
        & (scores["unit_id"] >= unit_start)
        & (scores["unit_id"] <= unit_end)
        & mode_filter
    ].copy()
    if scores.empty:
        raise ValueError(
            f"No LHI rows found for {fd_name} units {unit_start}-{unit_end} in {path}."
        )
    return scores


def infer_plot_stride(lhi_dir: Path, requested_stride: int) -> int:
    if requested_stride > 0:
        return int(requested_stride)
    lhi_config_path = lhi_dir / "run_config.json"
    if lhi_config_path.exists():
        lhi_config = json.loads(lhi_config_path.read_text())
        forecast_dir = lhi_config.get("forecast_dir")
        if forecast_dir:
            forecast_config_path = Path(forecast_dir) / "run_config.json"
            if forecast_config_path.exists():
                forecast_config = json.loads(forecast_config_path.read_text())
                return int(forecast_config.get("prediction_length", 1))
        if "prediction_length" in lhi_config:
            return int(lhi_config["prediction_length"])
    parent_config_path = lhi_dir.parent / "run_config.json"
    if parent_config_path.exists():
        parent_config = json.loads(parent_config_path.read_text())
        return int(parent_config.get("prediction_length", 1))
    return 1


def filter_plot_windows(frame: pd.DataFrame, plot_stride: int) -> pd.DataFrame:
    if plot_stride <= 1 or "forecast_start_cycle" not in frame.columns:
        return frame
    rows = []
    for _, group in frame.groupby(["covariate_mode", "fd", "unit_id"], sort=True):
        first_start = int(group["forecast_start_cycle"].min())
        rows.append(group[(group["forecast_start_cycle"] - first_start) % plot_stride == 0])
    return pd.concat(rows, ignore_index=True) if rows else frame.iloc[0:0].copy()


def filter_monitor_cycles(frame: pd.DataFrame, healthy_cycles: int) -> pd.DataFrame:
    if "cycle" not in frame.columns:
        return frame
    return frame[frame["cycle"] > int(healthy_cycles)].copy()


def plot_unit(scores: pd.DataFrame, fd_name: str, unit_id: int, metric: str, output_dir: Path, dpi: int, fig_size: tuple[float, float]) -> bool:
    import matplotlib.pyplot as plt

    unit_df = scores[(scores["fd"] == fd_name) & (scores["unit_id"] == unit_id)].copy()
    if unit_df.empty:
        return False

    d_col = f"d_{metric}"
    b_col = f"b_{metric}"
    lhi_col = f"lhi_{metric}"
    d_roll_col = f"d_{metric}_roll_mean"
    lhi_roll_col = f"lhi_{metric}_roll_mean"
    metric_label = metric.upper()
    colors = {
        "cluster_covariate": "#ff7f0e",
        "future_covariate": "#1f77b4",
        "no_covariate": "#2ca02c",
        "known_future": "#ff7f0e",
        "none": "#2ca02c",
    }
    baseline_styles = {
        "cluster_covariate": "-",
        "future_covariate": "--",
        "no_covariate": ":",
        "known_future": "-",
        "none": ":",
    }

    fig, axes = plt.subplots(2, 1, figsize=fig_size, sharex=True)
    drift_ax, lhi_ax = axes

    for mode, group in unit_df.groupby("covariate_mode", sort=True):
        group = group.sort_values(["cycle", "forecast_start_cycle"] if "forecast_start_cycle" in group else ["cycle"])
        color = colors.get(str(mode))
        drift_ax.plot(group["cycle"], group[d_col], color=color, alpha=0.25, linewidth=0.9, label=f"{mode} D_{metric_label} raw")
        drift_ax.plot(group["cycle"], group[d_roll_col], color=color, linewidth=1.8, label=f"{mode} D_{metric_label} rolling")
        baseline = float(group[b_col].iloc[0])
        drift_ax.axhline(
            baseline,
            color="#ef4444",
            linestyle=baseline_styles.get(str(mode), "-"),
            linewidth=1.5,
            label=f"{mode} B_{metric_label}={baseline:.4g}",
        )

        lhi_ax.plot(group["cycle"], group[lhi_col], color=color, alpha=0.25, linewidth=0.9, label=f"{mode} LHI raw")
        lhi_ax.plot(group["cycle"], group[lhi_roll_col], color=color, linewidth=1.8, label=f"{mode} LHI rolling")

    drift_ax.set_title(f"Condition-matched drift: {fd_name} unit {unit_id}")
    drift_ax.set_ylabel(f"D_{metric_label}")
    drift_ax.grid(True, alpha=0.3)
    drift_ax.legend(fontsize=8)

    lhi_ax.axhline(0.0, color="#6b7280", linewidth=1.0, label="healthy baseline")
    lhi_ax.set_title(f"Log-ratio LHI: {fd_name} unit {unit_id}")
    lhi_ax.set_xlabel("target cycle")
    lhi_ax.set_ylabel(f"LHI_{metric_label}")
    lhi_ax.grid(True, alpha=0.3)
    lhi_ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / f"{fd_name}_unit{unit_id}_{metric}_lhi.png", dpi=dpi)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir or (args.lhi_dir / f"batch_{args.fd}_unit{args.unit_start}_{args.unit_end}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    scores = load_scores(args.lhi_dir, args.fd, args.unit_start, args.unit_end, args.covariate_modes)
    plot_stride = infer_plot_stride(args.lhi_dir, args.plot_stride)
    scores = filter_plot_windows(scores, plot_stride)
    scores = filter_monitor_cycles(scores, args.healthy_cycles)
    if scores.empty:
        raise ValueError(
            f"No LHI rows remain after applying plot stride {plot_stride} "
            f"and cycle > {args.healthy_cycles}."
        )
    plotted = 0
    skipped = []
    for unit_id in range(args.unit_start, args.unit_end + 1):
        did_plot = plot_unit(
            scores=scores,
            fd_name=args.fd,
            unit_id=unit_id,
            metric=args.metric,
            output_dir=output_dir,
            dpi=args.dpi,
            fig_size=(args.fig_width, args.fig_height),
        )
        if did_plot:
            plotted += 1
        else:
            skipped.append(unit_id)

    print(f"Saved {plotted} unit plots to: {output_dir}", flush=True)
    if skipped:
        print(f"Skipped units with no rows: {skipped}", flush=True)


if __name__ == "__main__":
    main()

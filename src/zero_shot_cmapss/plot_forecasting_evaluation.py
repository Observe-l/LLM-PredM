from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from .forecasting_evaluation import plot_fd_level, plot_unit_level
    from .plot_operating_condition_clusters import FD_NAMES
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.zero_shot_cmapss.forecasting_evaluation import plot_fd_level, plot_unit_level
    from src.zero_shot_cmapss.plot_operating_condition_clusters import FD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch plot forecasting evaluation metrics for a range of C-MAPSS engines."
    )
    parser.add_argument("--evaluation_dir", type=Path, default=Path("outputs/roll_5/forecasting_evaluation"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--fd", choices=list(FD_NAMES), required=True)
    parser.add_argument("--unit_start", type=int, required=True)
    parser.add_argument("--unit_end", type=int, required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["forecast_error", "condition_matched_drift"],
        default=["forecast_error", "condition_matched_drift"],
    )
    parser.add_argument("--plot_fd_level", action="store_true")
    parser.add_argument(
        "--plot_stride",
        type=int,
        default=0,
        help="Forecast-start stride for plotting. 0 means infer prediction_length from nearby run_config.json.",
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


def load_metric(evaluation_dir: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    round_path = evaluation_dir / f"{stem}_round_metrics.csv"
    fd_path = evaluation_dir / f"{stem}_fd_level.csv"
    round_metrics = pd.read_csv(round_path)
    fd_level = pd.read_csv(fd_path)
    return round_metrics, fd_level


def infer_plot_stride(evaluation_dir: Path, requested_stride: int) -> int:
    if requested_stride > 0:
        return int(requested_stride)
    for path in (
        evaluation_dir / "run_config.json",
        evaluation_dir.parent / "run_config.json",
        evaluation_dir.parent.parent / "run_config.json",
    ):
        if path.exists():
            config = json.loads(path.read_text())
            return int(config.get("prediction_length", 1))
    return 1


def filter_plot_windows(frame: pd.DataFrame, plot_stride: int) -> pd.DataFrame:
    if plot_stride <= 1 or "forecast_start_cycle" not in frame.columns:
        return frame
    if "unit_id" in frame.columns:
        group_cols = ["covariate_mode", "fd", "unit_id"]
    else:
        group_cols = ["covariate_mode", "fd"]
    rows = []
    for _, group in frame.groupby(group_cols, sort=True):
        first_start = int(group["forecast_start_cycle"].min())
        rows.append(group[(group["forecast_start_cycle"] - first_start) % plot_stride == 0])
    return pd.concat(rows, ignore_index=True) if rows else frame.iloc[0:0].copy()


def filter_monitor_cycles(frame: pd.DataFrame, healthy_cycles: int) -> pd.DataFrame:
    if "cycle" not in frame.columns:
        return frame
    return frame[frame["cycle"] > int(healthy_cycles)].copy()


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir or (
        args.evaluation_dir / f"plots_{args.fd}_unit{args.unit_start}_{args.unit_end}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    units = [(args.fd, unit_id) for unit_id in range(args.unit_start, args.unit_end + 1)]
    plot_stride = infer_plot_stride(args.evaluation_dir, args.plot_stride)
    plotted_metrics = 0
    for metric in args.metrics:
        stem = "forecast_error" if metric == "forecast_error" else "condition_matched_drift"
        label = "Forecast Error" if metric == "forecast_error" else "Condition Matched Drift"
        round_metrics, fd_level = load_metric(args.evaluation_dir, stem)
        round_metrics = filter_plot_windows(round_metrics, plot_stride)
        fd_level = filter_plot_windows(fd_level, plot_stride)
        round_metrics = filter_monitor_cycles(round_metrics, args.healthy_cycles)
        fd_level = filter_monitor_cycles(fd_level, args.healthy_cycles)
        plot_unit_level(round_metrics, label, output_dir, units)
        if args.plot_fd_level:
            plot_fd_level(fd_level, label, output_dir)
        plotted_metrics += 1

    print(f"Saved plots for {plotted_metrics} evaluation metric groups to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

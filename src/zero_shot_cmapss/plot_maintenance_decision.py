from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

try:
    from .maintenance_decision_analysis import (
        plot_engine_scatter,
        plot_fd_overview,
        plot_lhi_examples,
        plot_slope_examples,
        plot_threshold_examples,
    )
    from .plot_operating_condition_clusters import FD_NAMES
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.zero_shot_cmapss.maintenance_decision_analysis import (
        plot_engine_scatter,
        plot_fd_overview,
        plot_lhi_examples,
        plot_slope_examples,
        plot_threshold_examples,
    )
    from src.zero_shot_cmapss.plot_operating_condition_clusters import FD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch plot maintenance decision signals for a range of C-MAPSS engines."
    )
    parser.add_argument("--decision_dir", type=Path, default=Path("outputs/cluster_roll_10/maintenance_decision"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--fd", choices=list(FD_NAMES), required=True)
    parser.add_argument("--unit_start", type=int, required=True)
    parser.add_argument("--unit_end", type=int, required=True)
    parser.add_argument("--healthy_cycles", type=int, default=50)
    parser.add_argument("--plot_fd_level", action="store_true")
    parser.add_argument("--plot_engine_scatter", action="store_true")
    parser.add_argument(
        "--plot_stride",
        type=int,
        default=0,
        help="Forecast-start stride for plotting. 0 means infer prediction_length from nearby run_config.json.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.unit_start <= 0 or args.unit_end <= 0:
        raise ValueError("--unit_start and --unit_end must be positive.")
    if args.unit_end < args.unit_start:
        raise ValueError("--unit_end must be >= --unit_start.")


def infer_plot_stride(decision_dir: Path, requested_stride: int) -> int:
    if requested_stride > 0:
        return int(requested_stride)
    for path in (
        decision_dir / "run_config.json",
        decision_dir.parent / "run_config.json",
        decision_dir.parent.parent / "run_config.json",
    ):
        if path.exists():
            config = json.loads(path.read_text())
            return int(config.get("prediction_length", 1))
    return 1


def filter_plot_windows(frame: pd.DataFrame, plot_stride: int) -> pd.DataFrame:
    if plot_stride <= 1 or "forecast_start_cycle" not in frame.columns:
        return frame
    if "unit_id" in frame.columns:
        group_cols = ["covariate_mode", "fd", "unit_id"] if "covariate_mode" in frame.columns else ["fd", "unit_id"]
    else:
        group_cols = ["covariate_mode", "fd"] if "covariate_mode" in frame.columns else ["fd"]
    rows = []
    for _, group in frame.groupby(group_cols, sort=True):
        first_start = int(group["forecast_start_cycle"].min())
        rows.append(group[(group["forecast_start_cycle"] - first_start) % plot_stride == 0])
    return pd.concat(rows, ignore_index=True) if rows else frame.iloc[0:0].copy()


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir or (
        args.decision_dir / f"plots_{args.fd}_unit{args.unit_start}_{args.unit_end}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    metrics_path = args.decision_dir / "decision_log_ratio_lhi.csv"
    if not metrics_path.exists():
        metrics_path = args.decision_dir / "known_future_log_ratio_lhi.csv"
    metrics = pd.read_csv(metrics_path)
    unit_summary = pd.read_csv(args.decision_dir / "engine_maintenance_signal_summary.csv")
    fd_summary = pd.read_csv(args.decision_dir / "fd_maintenance_signal_summary.csv")
    plot_stride = infer_plot_stride(args.decision_dir, args.plot_stride)
    metrics = filter_plot_windows(metrics, plot_stride)
    fd_summary = filter_plot_windows(fd_summary, plot_stride)
    units = [(args.fd, unit_id) for unit_id in range(args.unit_start, args.unit_end + 1)]

    plot_threshold_examples(metrics, units, output_dir, args.healthy_cycles)
    plot_slope_examples(metrics, units, output_dir, args.healthy_cycles)
    plot_lhi_examples(metrics, units, output_dir, args.healthy_cycles)
    if args.plot_fd_level:
        plot_fd_overview(fd_summary, output_dir)
    if args.plot_engine_scatter:
        plot_engine_scatter(unit_summary, output_dir)

    print(f"Saved maintenance decision plots to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

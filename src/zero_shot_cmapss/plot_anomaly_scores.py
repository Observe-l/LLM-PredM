from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd


FD_NAMES = ("FD001", "FD002", "FD003", "FD004")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot rolling anomaly-score trends from anomaly_scores.csv.")
    parser.add_argument("--input_csv", type=Path, default=Path("outputs/chronos2_cmapss/anomaly_scores.csv"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=5,
        help="Rolling window over forecast windows, not raw cycles. With stride=5, window=5 smooths about 25 cycles.",
    )
    parser.add_argument(
        "--plot_units",
        nargs="*",
        default=None,
        help="Optional unit-level plots, e.g. FD001:4 FD004:3. If omitted, only FD-level plots are created.",
    )
    return parser.parse_args()


def parse_plot_units(items: Sequence[str] | None) -> List[Tuple[str, int]]:
    if not items:
        return []
    parsed: List[Tuple[str, int]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"--plot_units entries must look like FD004:3, got {item!r}")
        fd_name, unit_text = item.split(":", 1)
        if fd_name not in FD_NAMES:
            raise ValueError(f"Unknown FD in --plot_units: {fd_name!r}")
        parsed.append((fd_name, int(unit_text)))
    return parsed


def add_rolling_columns(frame: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    frame = frame.sort_values("forecast_start_cycle").copy()
    rolling = frame["anomaly_score"].rolling(rolling_window, min_periods=1)
    frame["rolling_mean"] = rolling.mean()
    frame["rolling_median"] = rolling.median()
    return frame


def make_fd_level_frame(scores: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    rows = []
    for (mode, fd_name, cycle), group in scores.groupby(["covariate_mode", "fd", "forecast_start_cycle"], sort=True):
        rows.append(
            {
                "covariate_mode": mode,
                "fd": fd_name,
                "forecast_start_cycle": cycle,
                "anomaly_score": float(group["anomaly_score"].median()),
                "unit_count": int(group["unit_id"].nunique()),
                "window_count": int(len(group)),
            }
        )
    fd_level = pd.DataFrame(rows)
    smoothed = []
    for _, group in fd_level.groupby(["covariate_mode", "fd"], sort=True):
        smoothed.append(add_rolling_columns(group, rolling_window))
    return pd.concat(smoothed, ignore_index=True)


def plot_fd_level(scores: pd.DataFrame, output_dir: Path, rolling_window: int) -> None:
    import matplotlib.pyplot as plt

    fd_level = make_fd_level_frame(scores, rolling_window)
    fd_level.to_csv(output_dir / "fd_level_anomaly_trends.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    colors = {"past_only": "#1f77b4", "known_future": "#ff7f0e", "none": "#2ca02c"}
    for ax, fd_name in zip(axes_flat, FD_NAMES):
        fd_df = fd_level[fd_level["fd"] == fd_name]
        for mode, mode_df in fd_df.groupby("covariate_mode", sort=True):
            color = colors.get(str(mode), None)
            ax.plot(
                mode_df["forecast_start_cycle"],
                mode_df["rolling_mean"],
                color=color,
                linewidth=1.8,
                label=f"{mode} rolling mean",
            )
            ax.plot(
                mode_df["forecast_start_cycle"],
                mode_df["rolling_median"],
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=f"{mode} rolling median",
            )
        ax.set_title(fd_name)
        ax.set_xlabel("forecast start cycle")
        ax.set_ylabel("anomaly score")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"FD-level anomaly-score trends, rolling window={rolling_window}")
    fig.tight_layout()
    fig.savefig(output_dir / "fd_level_anomaly_trends.png", dpi=160)
    plt.close(fig)


def plot_unit_level(scores: pd.DataFrame, output_dir: Path, rolling_window: int, units: Sequence[Tuple[str, int]]) -> None:
    if not units:
        return

    import matplotlib.pyplot as plt

    unit_dir = output_dir / "unit_trends"
    unit_dir.mkdir(parents=True, exist_ok=True)
    colors = {"past_only": "#1f77b4", "known_future": "#ff7f0e", "none": "#2ca02c"}

    for fd_name, unit_id in units:
        unit_df = scores[(scores["fd"] == fd_name) & (scores["unit_id"] == unit_id)]
        if unit_df.empty:
            print(f"Skipped {fd_name}:{unit_id}, no rows found.")
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        for mode, mode_df in unit_df.groupby("covariate_mode", sort=True):
            smooth = add_rolling_columns(mode_df, rolling_window)
            color = colors.get(str(mode), None)
            ax.plot(
                smooth["forecast_start_cycle"],
                smooth["rolling_mean"],
                color=color,
                linewidth=1.8,
                label=f"{mode} rolling mean",
            )
            ax.plot(
                smooth["forecast_start_cycle"],
                smooth["rolling_median"],
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=f"{mode} rolling median",
            )
        ax.set_title(f"{fd_name} unit {unit_id} anomaly-score trend, rolling window={rolling_window}")
        ax.set_xlabel("forecast start cycle")
        ax.set_ylabel("anomaly score")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(unit_dir / f"{fd_name}_unit{unit_id}_anomaly_trend.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_csv.parent / "anomaly_plots"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))

    scores = pd.read_csv(args.input_csv)
    required_cols = {"covariate_mode", "fd", "unit_id", "forecast_start_cycle", "anomaly_score"}
    missing = required_cols - set(scores.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.input_csv}: {sorted(missing)}")

    plot_fd_level(scores, args.output_dir, args.rolling_window)
    plot_unit_level(scores, args.output_dir, args.rolling_window, parse_plot_units(args.plot_units))
    print(f"Saved anomaly-score plots to: {args.output_dir}")


if __name__ == "__main__":
    main()

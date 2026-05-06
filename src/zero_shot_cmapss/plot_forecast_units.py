from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from .plot_operating_condition_clusters import DEFAULT_SENSORS, FD_NAMES
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.zero_shot_cmapss.plot_operating_condition_clusters import DEFAULT_SENSORS, FD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch plot Chronos-2 forecast curves for a range of C-MAPSS engines."
    )
    parser.add_argument("--forecast_dir", type=Path, default=Path("outputs/chronos2_cmapss"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--fd", choices=list(FD_NAMES), required=True)
    parser.add_argument("--unit_start", type=int, required=True)
    parser.add_argument("--unit_end", type=int, required=True)
    parser.add_argument(
        "--source",
        choices=["window_forecasts"],
        default="window_forecasts",
        help="Forecast CSV to plot.",
    )
    parser.add_argument(
        "--plot_stride",
        type=int,
        default=0,
        help=(
            "Forecast-start stride used for plotting. 0 means read prediction_length from run_config.json. "
            "Use prediction_length to avoid overlapping forecast horizons."
        ),
    )
    parser.add_argument(
        "--covariate_modes",
        nargs="+",
        default=None,
        help="Covariate modes to plot. Defaults to all modes present in the forecast file.",
    )
    parser.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--fig_width", type=float, default=18.0)
    parser.add_argument("--fig_height", type=float, default=12.0)
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


def load_forecasts(
    forecast_dir: Path,
    source: str,
    fd_name: str,
    unit_start: int,
    unit_end: int,
    modes: Sequence[str] | None,
    sensors: Sequence[str],
) -> pd.DataFrame:
    path = forecast_dir / f"{source}.csv"
    forecasts = pd.read_csv(path)
    required = {"covariate_mode", "fd", "unit_id", "sensor", "cycle", "y_true", "y_pred"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    mode_filter = forecasts["covariate_mode"].isin(modes) if modes else True
    forecasts = forecasts[
        (forecasts["fd"] == fd_name)
        & (forecasts["unit_id"] >= unit_start)
        & (forecasts["unit_id"] <= unit_end)
        & forecasts["sensor"].isin(sensors)
        & mode_filter
    ].copy()
    if forecasts.empty:
        raise ValueError(f"No forecast rows found for {fd_name} units {unit_start}-{unit_end} in {path}.")
    return forecasts


def infer_plot_stride(forecast_dir: Path, requested_stride: int) -> int:
    if requested_stride > 0:
        return int(requested_stride)
    config_path = forecast_dir / "run_config.json"
    if not config_path.exists():
        return 1
    import json

    config = json.loads(config_path.read_text())
    return int(config.get("prediction_length", 1))


def filter_plot_windows(forecasts: pd.DataFrame, plot_stride: int) -> pd.DataFrame:
    if "forecast_start_cycle" not in forecasts.columns:
        return forecasts
    rows = []
    for _, group in forecasts.groupby(["covariate_mode", "fd", "unit_id"], sort=True):
        first_start = int(group["forecast_start_cycle"].min())
        keep = group[(group["forecast_start_cycle"] - first_start) % plot_stride == 0]
        rows.append(keep)
    return pd.concat(rows, ignore_index=True) if rows else forecasts.iloc[0:0].copy()


def filter_monitor_cycles(frame: pd.DataFrame, healthy_cycles: int) -> pd.DataFrame:
    if "cycle" not in frame.columns:
        return frame
    return frame[frame["cycle"] > int(healthy_cycles)].copy()


def plot_unit(
    forecasts: pd.DataFrame,
    fd_name: str,
    unit_id: int,
    sensors: Sequence[str],
    output_dir: Path,
    dpi: int,
    fig_size: tuple[float, float],
) -> int:
    import matplotlib.pyplot as plt

    unit_df = forecasts[(forecasts["fd"] == fd_name) & (forecasts["unit_id"] == unit_id)].copy()
    if unit_df.empty:
        return 0

    plotted = 0
    for mode, mode_df in unit_df.groupby("covariate_mode", sort=True):
        fig, axes = plt.subplots(4, 4, figsize=fig_size, sharex=True)
        axes_flat = axes.ravel()
        for ax_idx, sensor in enumerate(sensors):
            ax = axes_flat[ax_idx]
            sensor_df = mode_df[mode_df["sensor"] == sensor].sort_values("cycle")
            if sensor_df.empty:
                ax.axis("off")
                continue
            ax.plot(sensor_df["cycle"], sensor_df["y_true"], label="ground truth", linewidth=1.4)
            ax.plot(sensor_df["cycle"], sensor_df["y_pred"], label="zero-shot forecast", linewidth=1.3)
            ax.set_title(sensor)
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[len(sensors) :]:
            ax.axis("off")
        axes_flat[0].legend(loc="best", fontsize=8)
        first_cycle = int(mode_df["cycle"].min())
        last_cycle = int(mode_df["cycle"].max())
        fig.suptitle(f"{mode} {fd_name} unit {unit_id} forecast curve, cycles {first_cycle}-{last_cycle}")
        fig.supxlabel("cycle")
        fig.supylabel("sensor reading")
        fig.tight_layout()
        fig.savefig(output_dir / f"{mode}_{fd_name}_unit{unit_id}_forecast.png", dpi=dpi)
        plt.close(fig)
        plotted += 1
    return plotted


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir or (
        args.forecast_dir / f"forecast_plots_{args.source}_{args.fd}_unit{args.unit_start}_{args.unit_end}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    forecasts = load_forecasts(
        forecast_dir=args.forecast_dir,
        source=args.source,
        fd_name=args.fd,
        unit_start=args.unit_start,
        unit_end=args.unit_end,
        modes=args.covariate_modes,
        sensors=args.sensors,
    )
    plot_stride = infer_plot_stride(args.forecast_dir, args.plot_stride)
    forecasts = filter_plot_windows(forecasts, plot_stride)
    forecasts = filter_monitor_cycles(forecasts, args.healthy_cycles)
    if forecasts.empty:
        raise ValueError(
            f"No forecast rows remain after applying plot stride {plot_stride} "
            f"and cycle > {args.healthy_cycles}."
        )
    plotted = 0
    skipped = []
    for unit_id in range(args.unit_start, args.unit_end + 1):
        n_plots = plot_unit(
            forecasts=forecasts,
            fd_name=args.fd,
            unit_id=unit_id,
            sensors=args.sensors,
            output_dir=output_dir,
            dpi=args.dpi,
            fig_size=(args.fig_width, args.fig_height),
        )
        if n_plots:
            plotted += n_plots
        else:
            skipped.append(unit_id)

    print(f"Saved {plotted} forecast plots to: {output_dir}", flush=True)
    if skipped:
        print(f"Skipped units with no rows: {skipped}", flush=True)


if __name__ == "__main__":
    main()

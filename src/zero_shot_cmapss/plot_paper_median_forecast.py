from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


DEFAULT_FDS = ("FD001", "FD004")
DEFAULT_SENSORS = ("s3", "s7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot median-error C-MAPSS zero-shot forecast examples for paper figures."
    )
    parser.add_argument("--forecast_dir", type=Path, default=Path("outputs/CMAPSS/cluster_20"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper_figure"))
    parser.add_argument("--fds", nargs="+", default=list(DEFAULT_FDS))
    parser.add_argument("--sensors", nargs="+", default=list(DEFAULT_SENSORS))
    parser.add_argument("--covariate_mode", default="cluster_covariate")
    parser.add_argument("--source", default="metric_window_forecasts.csv")
    parser.add_argument("--healthy_cycles", type=int, default=50)
    parser.add_argument("--plot_stride", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def infer_plot_stride(forecast_dir: Path, requested_stride: int) -> int:
    if requested_stride > 0:
        return requested_stride
    config_path = forecast_dir / "run_config.json"
    if not config_path.exists():
        return 1
    config = json.loads(config_path.read_text())
    return int(config.get("prediction_length", 1))


def iter_rows(path: Path, fieldnames: Iterable[str] | None = None):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(fieldnames or []) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        yield from reader


def find_first_starts(path: Path, fds: set[str], mode: str) -> dict[tuple[str, int], int]:
    first_starts: dict[tuple[str, int], int] = {}
    for row in iter_rows(path, ["covariate_mode", "fd", "unit_id", "forecast_start_cycle"]):
        if row["covariate_mode"] != mode or row["fd"] not in fds:
            continue
        key = (row["fd"], int(row["unit_id"]))
        start = int(row["forecast_start_cycle"])
        if key not in first_starts or start < first_starts[key]:
            first_starts[key] = start
    return first_starts


def keep_window(row: dict[str, str], first_starts: dict[tuple[str, int], int], plot_stride: int) -> bool:
    if plot_stride <= 1:
        return True
    key = (row["fd"], int(row["unit_id"]))
    first_start = first_starts[key]
    return (int(row["forecast_start_cycle"]) - first_start) % plot_stride == 0


def select_median_units(
    path: Path,
    fds: set[str],
    mode: str,
    first_starts: dict[tuple[str, int], int],
    plot_stride: int,
) -> tuple[dict[str, int], dict[str, list[tuple[int, float]]]]:
    sums = defaultdict(float)
    counts = defaultdict(int)
    required = [
        "covariate_mode",
        "fd",
        "unit_id",
        "forecast_start_cycle",
        "has_ground_truth",
        "y_true",
        "y_pred",
    ]
    for row in iter_rows(path, required):
        if row["covariate_mode"] != mode or row["fd"] not in fds:
            continue
        if row["has_ground_truth"] not in ("1", "True", "true"):
            continue
        if not keep_window(row, first_starts, plot_stride):
            continue
        err = float(row["y_pred"]) - float(row["y_true"])
        key = (row["fd"], int(row["unit_id"]))
        sums[key] += err * err
        counts[key] += 1

    ranked: dict[str, list[tuple[int, float]]] = {}
    for (fd_name, unit_id), sq_sum in sums.items():
        if counts[(fd_name, unit_id)]:
            ranked.setdefault(fd_name, []).append((unit_id, math.sqrt(sq_sum / counts[(fd_name, unit_id)])))
    for fd_name in ranked:
        ranked[fd_name].sort(key=lambda item: (item[1], item[0]))

    selected = {}
    for fd_name in sorted(fds):
        if fd_name not in ranked or not ranked[fd_name]:
            raise ValueError(f"No RMSE rows found for {fd_name}.")
        selected[fd_name] = ranked[fd_name][len(ranked[fd_name]) // 2][0]
    return selected, ranked


def load_prediction_series(
    path: Path,
    selected_units: dict[str, int],
    sensors: set[str],
    mode: str,
    first_starts: dict[tuple[str, int], int],
    plot_stride: int,
    healthy_cycles: int,
) -> dict[tuple[str, str], dict[int, tuple[float, float]]]:
    series: dict[tuple[str, str], dict[int, tuple[float, float]]] = defaultdict(dict)
    required = [
        "covariate_mode",
        "fd",
        "unit_id",
        "sensor",
        "cycle",
        "forecast_start_cycle",
        "has_ground_truth",
        "y_true",
        "y_pred",
    ]
    for row in iter_rows(path, required):
        fd_name = row["fd"]
        if row["covariate_mode"] != mode or fd_name not in selected_units:
            continue
        if int(row["unit_id"]) != selected_units[fd_name] or row["sensor"] not in sensors:
            continue
        if row["has_ground_truth"] not in ("1", "True", "true"):
            continue
        if int(row["cycle"]) <= healthy_cycles:
            continue
        if not keep_window(row, first_starts, plot_stride):
            continue
        series[(fd_name, row["sensor"])][int(row["cycle"])] = (float(row["y_true"]), float(row["y_pred"]))

    return series


def plot_on_axis(ax, rows: dict[int, tuple[float, float]], show_legend: bool = False) -> None:
    cycles = sorted(rows)
    y_true = [rows[cycle][0] for cycle in cycles]
    y_pred = [rows[cycle][1] for cycle in cycles]
    ax.plot(cycles, y_true, label="Ground Truth", linewidth=1.4)
    ax.plot(cycles, y_pred, label="OCC-ZSF", linewidth=1.3)
    ax.set_xlabel("cycle")
    ax.set_ylabel("sensor reading")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="best", fontsize=9)


def save_figures(
    output_dir: Path,
    series: dict[tuple[str, str], dict[int, tuple[float, float]]],
    selected_units: dict[str, int],
    ranked: dict[str, list[tuple[int, float]]],
    fds: list[str],
    sensors: list[str],
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=False)
    for row_idx, fd_name in enumerate(fds):
        for col_idx, sensor in enumerate(sensors):
            ax = axes[row_idx][col_idx]
            key = (fd_name, sensor)
            if not series.get(key):
                raise ValueError(f"No plot rows found for {fd_name} unit {selected_units[fd_name]} {sensor}.")
            plot_on_axis(ax, series[key], show_legend=(row_idx == 0 and col_idx == 0))
    fig.tight_layout()
    fig.savefig(output_dir / "median_error_forecast_2x2.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    for fd_name in fds:
        for sensor in sensors:
            key = (fd_name, sensor)
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            plot_on_axis(ax, series[key], show_legend=True)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"median_error_forecast_{fd_name}_unit{selected_units[fd_name]}_{sensor}.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))
    path = args.forecast_dir / args.source
    if not path.exists():
        raise FileNotFoundError(path)

    fds = list(args.fds)
    sensors = list(args.sensors)
    plot_stride = infer_plot_stride(args.forecast_dir, args.plot_stride)
    first_starts = find_first_starts(path, set(fds), args.covariate_mode)
    selected_units, ranked = select_median_units(path, set(fds), args.covariate_mode, first_starts, plot_stride)
    series = load_prediction_series(
        path=path,
        selected_units=selected_units,
        sensors=set(sensors),
        mode=args.covariate_mode,
        first_starts=first_starts,
        plot_stride=plot_stride,
        healthy_cycles=args.healthy_cycles,
    )
    save_figures(args.output_dir, series, selected_units, ranked, fds, sensors, args.dpi)

    print(f"Saved 5 figures to {args.output_dir}")
    for fd_name in fds:
        rmse = dict(ranked[fd_name])[selected_units[fd_name]]
        rank = [unit for unit, _ in ranked[fd_name]].index(selected_units[fd_name]) + 1
        print(f"{fd_name}: selected unit {selected_units[fd_name]} rank {rank}/{len(ranked[fd_name])}, RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()

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


FD_NAMES = ("FD001", "FD002", "FD003", "FD004")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper LHI comparisons for median-error C-MAPSS units.")
    parser.add_argument("--forecast_dir", type=Path, default=Path("outputs/CMAPSS/cluster_20"))
    parser.add_argument("--raw_lhi_dir", type=Path, default=Path("outputs/CMAPSS/raw_lhi"))
    parser.add_argument("--chronos_lhi_dir", type=Path, default=Path("outputs/CMAPSS/cov_20/lhi_fix"))
    parser.add_argument("--occ_lhi_dir", type=Path, default=Path("outputs/CMAPSS/cluster_20/lhi_fix"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper_figure"))
    parser.add_argument("--fds", nargs="+", default=list(FD_NAMES), choices=list(FD_NAMES))
    parser.add_argument("--forecast_source", default="metric_window_forecasts.csv")
    parser.add_argument("--forecast_mode", default="cluster_covariate")
    parser.add_argument("--raw_mode", default="raw_observed")
    parser.add_argument("--chronos_mode", default="future_covariate")
    parser.add_argument("--occ_mode", default="cluster_covariate")
    parser.add_argument("--lhi_column", default="lhi_rmse_roll_mean")
    parser.add_argument("--healthy_cycles", type=int, default=50)
    parser.add_argument("--plot_stride", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def iter_rows(path: Path, fieldnames: Iterable[str] | None = None):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(fieldnames or []) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        yield from reader


def infer_plot_stride(run_dir: Path, requested_stride: int) -> int:
    if requested_stride > 0:
        return requested_stride
    for path in (run_dir / "run_config.json", run_dir.parent / "run_config.json"):
        if path.exists():
            config = json.loads(path.read_text())
            return int(config.get("prediction_length", 1))
    return 1


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
    if key not in first_starts:
        return False
    return (int(row["forecast_start_cycle"]) - first_starts[key]) % plot_stride == 0


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
        if row["has_ground_truth"] not in {"1", "True", "true"}:
            continue
        if not keep_window(row, first_starts, plot_stride):
            continue
        err = float(row["y_pred"]) - float(row["y_true"])
        key = (row["fd"], int(row["unit_id"]))
        sums[key] += err * err
        counts[key] += 1

    ranked: dict[str, list[tuple[int, float]]] = {}
    for (fd_name, unit_id), sq_sum in sums.items():
        n = counts[(fd_name, unit_id)]
        if n:
            ranked.setdefault(fd_name, []).append((unit_id, math.sqrt(sq_sum / n)))
    for fd_name in ranked:
        ranked[fd_name].sort(key=lambda item: (item[1], item[0]))

    selected = {}
    for fd_name in sorted(fds):
        if fd_name not in ranked or not ranked[fd_name]:
            raise ValueError(f"No RMSE rows found for {fd_name}.")
        selected[fd_name] = ranked[fd_name][len(ranked[fd_name]) // 2][0]
    return selected, ranked


def load_lhi_series(
    lhi_dir: Path,
    mode: str,
    fd_name: str,
    unit_id: int,
    lhi_column: str,
    healthy_cycles: int,
    plot_stride: int,
) -> list[tuple[int, float]]:
    path = lhi_dir / "lhi_scores.csv"
    required = ["covariate_mode", "fd", "unit_id", "cycle", "forecast_start_cycle", lhi_column]
    first_start = None
    candidates = []
    for row in iter_rows(path, required):
        if row["covariate_mode"] != mode or row["fd"] != fd_name or int(row["unit_id"]) != unit_id:
            continue
        start = int(row["forecast_start_cycle"])
        first_start = start if first_start is None else min(first_start, start)
        candidates.append(row)
    if first_start is None:
        raise ValueError(f"No LHI rows found in {path} for {mode} {fd_name} unit {unit_id}.")

    by_cycle = {}
    for row in candidates:
        cycle = int(row["cycle"])
        if cycle <= healthy_cycles:
            continue
        if plot_stride > 1 and (int(row["forecast_start_cycle"]) - first_start) % plot_stride != 0:
            continue
        by_cycle[cycle] = float(row[lhi_column])
    return sorted(by_cycle.items())


def save_lhi_plot(
    output_dir: Path,
    fd_name: str,
    unit_id: int,
    raw_series: list[tuple[int, float]],
    chronos_series: list[tuple[int, float]],
    occ_series: list[tuple[int, float]],
    dpi: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(
        [x for x, _ in raw_series],
        [y for _, y in raw_series],
        label="Ground Truth",
        color="#1f77b4",
        linewidth=1.4,
    )
    ax.plot(
        [x for x, _ in chronos_series],
        [y for _, y in chronos_series],
        label="Chronos-2",
        color="#2ca02c",
        linewidth=1.3,
    )
    ax.plot(
        [x for x, _ in occ_series],
        [y for _, y in occ_series],
        label="OCC-ZSF",
        color="#ff7f0e",
        linewidth=1.3,
    )
    ax.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("cycle")
    ax.set_ylabel("HDI")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path = output_dir / f"log_ratio_lhi_{fd_name}_unit{unit_id}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))

    forecast_path = args.forecast_dir / args.forecast_source
    fds = set(args.fds)
    plot_stride = infer_plot_stride(args.forecast_dir, args.plot_stride)
    first_starts = find_first_starts(forecast_path, fds, args.forecast_mode)
    selected_units, ranked = select_median_units(
        forecast_path,
        fds,
        args.forecast_mode,
        first_starts,
        plot_stride,
    )

    saved = []
    for fd_name in args.fds:
        unit_id = selected_units[fd_name]
        raw_series = load_lhi_series(
            args.raw_lhi_dir,
            args.raw_mode,
            fd_name,
            unit_id,
            args.lhi_column,
            args.healthy_cycles,
            plot_stride,
        )
        chronos_series = load_lhi_series(
            args.chronos_lhi_dir,
            args.chronos_mode,
            fd_name,
            unit_id,
            args.lhi_column,
            args.healthy_cycles,
            plot_stride,
        )
        occ_series = load_lhi_series(
            args.occ_lhi_dir,
            args.occ_mode,
            fd_name,
            unit_id,
            args.lhi_column,
            args.healthy_cycles,
            plot_stride,
        )
        saved.append(save_lhi_plot(args.output_dir, fd_name, unit_id, raw_series, chronos_series, occ_series, args.dpi))

    print(f"Saved {len(saved)} LHI figures to {args.output_dir}")
    for fd_name in args.fds:
        unit_id = selected_units[fd_name]
        rmse = dict(ranked[fd_name])[unit_id]
        rank = [unit for unit, _ in ranked[fd_name]].index(unit_id) + 1
        print(f"{fd_name}: selected unit {unit_id} rank {rank}/{len(ranked[fd_name])}, RMSE={rmse:.6f}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()

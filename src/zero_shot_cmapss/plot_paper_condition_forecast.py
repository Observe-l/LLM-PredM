from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot C-MAPSS forecast comparisons grouped by operating condition."
    )
    parser.add_argument("--chronos_dir", type=Path, default=Path("outputs/CMAPSS/cov_20"))
    parser.add_argument("--occ_dir", type=Path, default=Path("outputs/CMAPSS/cluster_20"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper_figure"))
    parser.add_argument("--source", default="metric_window_forecasts.csv")
    parser.add_argument("--fd", default="FD004")
    parser.add_argument("--unit_id", type=int, default=93)
    parser.add_argument("--sensor", default="s3")
    parser.add_argument(
        "--forecast_start_cycle",
        type=int,
        default=0,
        help="Optional single forecast start to plot. 0 means plot the whole non-overlapping unit series.",
    )
    parser.add_argument("--chronos_mode", default="future_covariate")
    parser.add_argument("--occ_mode", default="cluster_covariate")
    parser.add_argument("--healthy_cycles", type=int, default=50)
    parser.add_argument("--plot_stride", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_rows(
    path: Path,
    mode: str,
    fd_name: str,
    unit_id: int,
    sensor: str,
    forecast_start_cycle: int,
    healthy_cycles: int,
    plot_stride: int,
):
    required = {
        "covariate_mode",
        "fd",
        "unit_id",
        "sensor",
        "forecast_start_cycle",
        "cycle",
        "has_ground_truth",
        "y_true",
        "y_pred",
    }
    rows = {}
    first_start = None
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        for row in reader:
            if (
                row["covariate_mode"] == mode
                and row["fd"] == fd_name
                and int(row["unit_id"]) == unit_id
                and row["sensor"] == sensor
                and row["has_ground_truth"] in {"1", "True", "true"}
            ):
                start = int(row["forecast_start_cycle"])
                first_start = start if first_start is None else min(first_start, start)
                if forecast_start_cycle > 0 and start != forecast_start_cycle:
                    continue
                if forecast_start_cycle == 0 and plot_stride > 1 and (start - first_start) % plot_stride != 0:
                    continue
                cycle = int(row["cycle"])
                if cycle <= healthy_cycles:
                    continue
                op_key = row["op_condition_key"]
                rows.setdefault(op_key, {})[cycle] = (float(row["y_true"]), float(row["y_pred"]))
    if not rows:
        raise ValueError(
            f"No rows found in {path} for {mode} {fd_name} unit {unit_id} "
            f"{sensor}."
        )
    return rows


def save_plot(
    chronos_rows: dict[str, dict[int, tuple[float, float]]],
    occ_rows: dict[str, dict[int, tuple[float, float]]],
    output_path: Path,
    dpi: int,
) -> None:
    op_keys = sorted(set(chronos_rows) & set(occ_rows))
    if not op_keys:
        raise ValueError("Chronos-2 and OCC-ZSF rows have no overlapping operating conditions.")

    fig_height = max(2.0 * len(op_keys), 6.0)
    fig, axes = plt.subplots(len(op_keys), 1, figsize=(8.0, fig_height), sharex=True)
    if len(op_keys) == 1:
        axes = [axes]

    for condition_idx, (ax, op_key) in enumerate(zip(axes, op_keys), start=1):
        cycles = sorted(set(chronos_rows[op_key]) & set(occ_rows[op_key]))
        if not cycles:
            ax.axis("off")
            continue
        ground_truth = [occ_rows[op_key][cycle][0] for cycle in cycles]
        chronos_pred = [chronos_rows[op_key][cycle][1] for cycle in cycles]
        occ_pred = [occ_rows[op_key][cycle][1] for cycle in cycles]

        ax.plot(
            cycles,
            ground_truth,
            label="Ground Truth",
            color="#1f77b4",
            linewidth=1.4,
            marker="o",
            markersize=3.2,
        )
        ax.plot(
            cycles,
            chronos_pred,
            label="Chronos-2",
            color="#2ca02c",
            linewidth=1.3,
            marker="x",
            markersize=3.6,
        )
        ax.plot(
            cycles,
            occ_pred,
            label="OCD-ZSF" if condition_idx == len(axes) else "_nolegend_",
            color="#ff7f0e",
            linewidth=1.3,
            marker=".",
            markersize=5.0,
        )
        ax.set_ylabel(f"condition {condition_idx}")
        ax.grid(True, alpha=0.3)
        if condition_idx == len(axes):
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("cycle")
    fig.supylabel("sensor reading")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))

    chronos_rows = load_rows(
        args.chronos_dir / args.source,
        args.chronos_mode,
        args.fd,
        args.unit_id,
        args.sensor,
        args.forecast_start_cycle,
        args.healthy_cycles,
        args.plot_stride,
    )
    occ_rows = load_rows(
        args.occ_dir / args.source,
        args.occ_mode,
        args.fd,
        args.unit_id,
        args.sensor,
        args.forecast_start_cycle,
        args.healthy_cycles,
        args.plot_stride,
    )
    suffix = f"start{args.forecast_start_cycle}" if args.forecast_start_cycle > 0 else "all_conditions"
    output_path = args.output_dir / f"condition_forecast_{args.fd}_unit{args.unit_id}_{args.sensor}_{suffix}.png"
    save_plot(chronos_rows, occ_rows, output_path, args.dpi)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()

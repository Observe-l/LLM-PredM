#!/usr/bin/env python3
"""Compute N-CMAPSS cycle-level LHI for one engine.

This is the raw-observation analogue of ``src.zero_shot_cmapss.lhi_indicator``.
The first cycle of the selected engine, including every time sample in that
cycle, is the health reference.  For each later cycle, sensor drift is the
absolute min-max-normalized deviation from the first-cycle sensor mean.  The
cycle LHI is the log ratio of its drift RMSE to the first-cycle drift RMSE.

N-CMAPSS has many samples per cycle, so the aggregation is performed over all
time samples in each cycle rather than treating one row as one cycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SENSORS = [
    "T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
    "Ps30", "P40", "P50", "Nf", "Nc", "Wf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/N-CMAPSS/N-CMAPSS_DS02-006.h5"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unit", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N-CMAPSS-figure/lhi_DS02-006_dev_unit2"))
    parser.add_argument("--lhi-epsilon", type=float, default=1e-6)
    parser.add_argument("--range-epsilon", type=float, default=1e-12)
    return parser.parse_args()


def read_engine(path: Path, split: str, unit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with h5py.File(path, "r") as hdf:
        a = np.asarray(hdf[f"A_{split}"][:, :2], dtype=np.int64)
        units = a[:, 0]
        if not np.any(units == unit):
            available = np.unique(units).tolist()
            raise ValueError(f"Unit {unit} not found in {path.name} {split}; available units: {available}")
        row_indices = np.flatnonzero(units == unit)
        x_s = np.asarray(hdf[f"X_s_{split}"][row_indices], dtype=np.float64)
        sensor_names = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in np.asarray(hdf["X_s_var"]).reshape(-1)
        ]
    return a[row_indices, 1], x_s, row_indices, sensor_names


def compute_scores(cycles: np.ndarray, values: np.ndarray, sensor_names: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if sensor_names != DEFAULT_SENSORS:
        # Keep the source's actual names in the result while making the intended
        # physical-sensor block explicit in the metadata.
        sensor_names = list(sensor_names)
    reference_cycle = int(cycles.min())
    target_cycles = np.sort(np.unique(cycles))
    if len(target_cycles) < 2:
        raise ValueError("The selected engine has fewer than two cycles.")
    reference = values[cycles == reference_cycle]
    reference_mean = reference.mean(axis=0)
    reference_min = reference.min(axis=0)
    reference_max = reference.max(axis=0)
    reference_range = reference_max - reference_min
    usable = np.isfinite(reference_range) & (reference_range > args.range_epsilon)
    if not np.any(usable):
        raise ValueError("No usable sensor range in the first-cycle health reference.")

    # This follows the existing C-MAPSS LHI convention: min-max normalize the
    # deviation from a healthy reference, then aggregate sensor/time errors.
    normalized = (values[:, usable] - reference_mean[usable]) / reference_range[usable]
    ref_normalized = normalized[cycles == reference_cycle]
    ref_abs = np.abs(ref_normalized)
    ref_sq = ref_normalized ** 2
    baseline_mae = float(ref_abs.mean())
    baseline_rmse = float(np.sqrt(ref_sq.mean()))
    eps = float(args.lhi_epsilon)

    cycle_rows: list[dict[str, float | int]] = []
    sensor_rows: list[dict[str, float | int | str]] = []
    for cycle in target_cycles:
        mask = cycles == cycle
        cycle_norm = normalized[mask]
        abs_error = np.abs(cycle_norm)
        squared_error = cycle_norm ** 2
        d_mae = float(abs_error.mean())
        d_rmse = float(np.sqrt(squared_error.mean()))
        cycle_rows.append(
            {
                "cycle": int(cycle),
                "sample_count": int(mask.sum()),
                "d_mae": d_mae,
                "d_rmse": d_rmse,
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "lhi_mae": float(np.log((d_mae + eps) / (baseline_mae + eps))),
                "lhi_rmse": float(np.log((d_rmse + eps) / (baseline_rmse + eps))),
            }
        )
        sensor_mae = abs_error.mean(axis=0)
        sensor_rmse = np.sqrt(squared_error.mean(axis=0))
        ref_sensor_rmse = np.sqrt(ref_sq.mean(axis=0))
        for idx, sensor in enumerate(np.asarray(sensor_names)[usable]):
            sensor_rows.append(
                {
                    "cycle": int(cycle),
                    "sensor": str(sensor),
                    "sample_count": int(mask.sum()),
                    "d_mae": float(sensor_mae[idx]),
                    "d_rmse": float(sensor_rmse[idx]),
                    "baseline_rmse": float(ref_sensor_rmse[idx]),
                    "lhi_rmse": float(np.log((sensor_rmse[idx] + eps) / (ref_sensor_rmse[idx] + eps))),
                }
            )

    cycle_scores = pd.DataFrame(cycle_rows)
    sensor_scores = pd.DataFrame(sensor_rows)
    metadata = {
        "method": "raw N-CMAPSS LHI adapted from src/zero_shot_cmapss/lhi_indicator.py",
        "reference": f"all X_s time samples from cycle {reference_cycle}",
        "sensor_block": "X_s (physical sensor readings)",
        "sensor_names": sensor_names,
        "usable_sensor_names": list(np.asarray(sensor_names)[usable]),
        "reference_cycle": reference_cycle,
        "first_cycle_sample_count": int(len(reference)),
        "total_cycle_count": int(len(target_cycles)),
        "lhi_formula": "log((cycle_drift + epsilon) / (first_cycle_drift + epsilon))",
        "normalization": "per-sensor min-max range from first-cycle samples",
        "lhi_epsilon": eps,
        "range_epsilon": float(args.range_epsilon),
    }
    return cycle_scores, sensor_scores, metadata


def plot_selected_cycle(cycle_scores: pd.DataFrame, sensor_scores: pd.DataFrame, cycle: int, output: Path, metadata: dict[str, object]) -> None:
    row = cycle_scores.loc[cycle_scores["cycle"] == cycle].iloc[0]
    sensors = sensor_scores.loc[sensor_scores["cycle"] == cycle].copy()
    sensors = sensors.sort_values("lhi_rmse", ascending=False)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1.15]})
    x = np.arange(len(sensors))
    width = 0.38
    axes[0].bar(x - width / 2, sensors["d_rmse"], width, color="#2f5d8c", label="cycle D_RMSE")
    axes[0].bar(x + width / 2, sensors["baseline_rmse"], width, color="#c9d5e3", edgecolor="#2f5d8c", label="cycle-1 reference D_RMSE")
    axes[0].set_ylabel("sensor drift RMSE")
    axes[0].set_title(f"N-CMAPSS LHI sensor drift: cycle {cycle} ({int(row['sample_count']):,} time samples)")
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, sensors["lhi_rmse"], color="#b45f06", alpha=0.88)
    axes[1].axhline(0.0, color="#4b5563", linewidth=1.0)
    axes[1].set_ylabel("sensor LHI_RMSE")
    axes[1].set_xlabel("physical sensor")
    axes[1].set_xticks(x, sensors["sensor"], rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Cycle-level LHI relative to first-cycle health reference | overall LHI_RMSE={float(row['lhi_rmse']):.4f}",
        fontsize=14,
    )
    fig.text(0.01, 0.01, "Reference: all samples in cycle 1; X_s physical sensors; positive LHI means larger normalized drift than cycle 1.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cycle_curve(cycle_scores: pd.DataFrame, selected: list[int], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)
    ax.plot(cycle_scores["cycle"], cycle_scores["lhi_rmse"], color="#2f5d8c", linewidth=1.8, label="LHI_RMSE")
    ax.plot(cycle_scores["cycle"], cycle_scores["lhi_mae"], color="#8f3b76", linewidth=1.3, alpha=0.8, label="LHI_MAE")
    colors = ["#b45f06", "#7f1d1d"]
    for cycle, color in zip(selected, colors):
        row = cycle_scores.loc[cycle_scores["cycle"] == cycle].iloc[0]
        ax.scatter([cycle], [row["lhi_rmse"]], color=color, s=52, zorder=5)
        ax.annotate(f"cycle {cycle}", (cycle, row["lhi_rmse"]), xytext=(6, 8), textcoords="offset points", fontsize=9)
    ax.axhline(0.0, color="#4b5563", linewidth=1.0)
    ax.set_title("N-CMAPSS cycle-level LHI across one engine")
    ax.set_xlabel("cycle")
    ax.set_ylabel("log-ratio LHI")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cycles, values, _, sensor_names = read_engine(args.data_file, args.split, args.unit)
    cycle_scores, sensor_scores, metadata = compute_scores(cycles, values, sensor_names, args)
    metadata.update({
        "data_file": str(args.data_file),
        "split": args.split,
        "unit": int(args.unit),
        "last_cycle": int(cycle_scores["cycle"].max()),
    })
    selected_cycles = [2, int(cycle_scores["cycle"].max())]
    cycle_scores.to_csv(args.output_dir / "cycle_lhi_scores.csv", index=False)
    sensor_scores.to_csv(args.output_dir / "sensor_lhi_scores.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for cycle in selected_cycles:
        plot_selected_cycle(
            cycle_scores,
            sensor_scores,
            cycle,
            args.output_dir / f"lhi_cycle_{cycle:03d}.png",
            metadata,
        )
    plot_cycle_curve(cycle_scores, selected_cycles, args.output_dir / "lhi_cycle_curve.png")
    print(cycle_scores[cycle_scores["cycle"].isin(selected_cycles)].to_string(index=False))
    print(f"Saved N-CMAPSS LHI outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

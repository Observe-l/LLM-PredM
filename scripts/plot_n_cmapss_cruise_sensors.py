#!/usr/bin/env python3
"""Plot all physical N-CMAPSS sensors inside selected cruise cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.plot_n_cmapss_cruise_lhi import cycle_local_samples, load_unit
except ModuleNotFoundError:  # direct execution: scripts/ is on sys.path
    from plot_n_cmapss_cruise_lhi import cycle_local_samples, load_unit


SENSOR_NAMES = [
    "T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
    "Ps30", "P40", "P50", "Nf", "Nc", "Wf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/N-CMAPSS/N-CMAPSS_DS02-006.h5"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unit", type=int, default=5)
    parser.add_argument("--cruise-statistics", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude/cruise_cycle_statistics.csv"))
    parser.add_argument("--cycles", type=int, nargs="+", default=[11, 89])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_sensor_readings_DS02-006_dev_unit5_cycle11_89"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervals = pd.read_csv(args.cruise_statistics)
    intervals = intervals[
        (intervals["dataset"] == "DS02-006")
        & (intervals["split"] == args.split)
        & (intervals["unit"] == args.unit)
        & (intervals["cruise_status"] == "accepted")
    ].set_index("cycle")
    missing = [cycle for cycle in args.cycles if cycle not in intervals.index]
    if missing:
        raise ValueError(f"No accepted cruise interval for cycles {missing}")

    cycles, _, sensors, _, sensor_names = load_unit(args.data_file, args.split, args.unit)
    if sensor_names != SENSOR_NAMES:
        raise ValueError(f"Unexpected X_s variables: {sensor_names}")
    local = cycle_local_samples(cycles)
    extracted: dict[int, pd.DataFrame] = {}
    for cycle in args.cycles:
        row = intervals.loc[cycle]
        mask = (
            (cycles == cycle)
            & (local >= int(row["cruise_start_sample"]))
            & (local <= int(row["cruise_end_sample"]))
        )
        values = sensors[mask]
        frame = pd.DataFrame(values, columns=SENSOR_NAMES)
        frame.insert(0, "time_sample", np.arange(1, len(frame) + 1))
        frame.insert(0, "cycle", cycle)
        extracted[cycle] = frame

        fig, axes = plt.subplots(7, 2, figsize=(15, 20), dpi=180)
        axes = axes.ravel()
        color = "#2f5d8c"
        for index, sensor in enumerate(SENSOR_NAMES):
            ax = axes[index]
            ax.plot(frame["time_sample"], frame[sensor], color=color, linewidth=0.75)
            ax.set_title(sensor)
            ax.set_xlabel("cruise time sample")
            ax.set_ylabel("sensor reading")
            ax.grid(alpha=0.3)
        fig.suptitle(
            f"N-CMAPSS DS02-006 dev unit {args.unit}: cycle {cycle} cruise sensor readings",
            fontsize=16,
            y=0.995,
        )
        fig.text(0.01, 0.005, "Raw X_s physical sensor readings; no operating-condition grouping or clustering.", fontsize=8.5)
        fig.tight_layout(rect=(0, 0.015, 1, 0.985))
        fig.savefig(args.output_dir / f"cruise_sensor_readings_cycle_{cycle:03d}.png", bbox_inches="tight")
        plt.close(fig)

    combined = pd.concat(extracted.values(), ignore_index=True)
    cycle_label = "_".join(f"cycle{cycle:03d}" for cycle in args.cycles)
    combined.to_csv(args.output_dir / f"cruise_sensor_readings_{cycle_label}.csv", index=False)
    metadata = {
        "data_file": str(args.data_file),
        "split": args.split,
        "unit": int(args.unit),
        "cycles": [int(cycle) for cycle in args.cycles],
        "cruise_only": True,
        "operating_condition_grouping": False,
        "sensor_block": "X_s physical sensor readings",
        "sensor_names": SENSOR_NAMES,
        "plot_units": "native sensor units; no normalization",
        "cruise_intervals": {
            str(cycle): {
                "start_sample": int(intervals.loc[cycle, "cruise_start_sample"]),
                "end_sample": int(intervals.loc[cycle, "cruise_end_sample"]),
                "cruise_samples": int(intervals.loc[cycle, "cruise_samples"]),
            }
            for cycle in args.cycles
        },
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(combined.groupby("cycle").size().to_string())
    print(f"Saved cruise sensor figures to: {args.output_dir}")


if __name__ == "__main__":
    main()

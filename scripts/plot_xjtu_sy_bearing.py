#!/usr/bin/env python3
"""Plot raw XJTU-SY bearing vibration signals from one CSV file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SENSOR_COLUMNS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/XJTU-SY-figure"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    missing = [column for column in SENSOR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")
    frame = frame[SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        raise ValueError("Input contains non-numeric or missing sensor values")

    sample_index = pd.Series(range(len(frame)), name="sample_index")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, dpi=180)
    colors = ["#2f5d8c", "#b45f06"]
    labels = ["Horizontal vibration", "Vertical vibration"]
    for ax, column, color, label in zip(axes, SENSOR_COLUMNS, colors, labels):
        ax.plot(sample_index, frame[column], color=color, linewidth=0.55)
        ax.set_title(label)
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("sample index")
    fig.suptitle("XJTU-SY Bearing1_1: raw vibration sensor signals (35 Hz, 12 kN)", fontsize=15)
    fig.text(0.01, 0.01, f"Source: {args.input}; samples={len(frame):,}; raw values, no filtering or normalization.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    output_path = args.output_dir / "35Hz12kN_Bearing1_1_file1_sensor_readings.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    summary = frame.describe().T.reset_index().rename(columns={"index": "sensor"})
    summary.to_csv(args.output_dir / "35Hz12kN_Bearing1_1_file1_sensor_summary.csv", index=False)
    metadata = {
        "input": str(args.input),
        "condition": "35Hz12kN",
        "bearing": "Bearing1_1",
        "samples": int(len(frame)),
        "sensors": SENSOR_COLUMNS,
        "plot": str(output_path),
        "processing": "raw CSV values; no filtering or normalization",
    }
    (args.output_dir / "35Hz12kN_Bearing1_1_file1_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()

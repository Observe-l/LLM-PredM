#!/usr/bin/env python3
"""Plot all one-minute-spaced XJTU-SY bearing vibration CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SENSORS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]
SENSOR_LABELS = {
    "Horizontal_vibration_signals": "Horizontal vibration",
    "Vertical_vibration_signals": "Vertical vibration",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/XJTU-SY-figure"))
    parser.add_argument("--heatmap-bins", type=int, default=512)
    return parser.parse_args()


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def binned_rms(values: np.ndarray, bins: int) -> np.ndarray:
    edges = np.linspace(0, len(values), bins + 1, dtype=int)
    output = np.empty(bins, dtype=float)
    for i, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        output[i] = rms(values[start:stop])
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.csv"), key=lambda path: int(path.stem))
    if not files:
        raise FileNotFoundError(f"No numeric CSV files found in {args.input_dir}")
    bearing_name = args.input_dir.name
    condition_name = args.input_dir.parent.name

    records: list[pd.DataFrame] = []
    arrays: list[np.ndarray] = []
    heatmaps = {sensor: [] for sensor in SENSORS}
    summary_rows: list[dict[str, float | int | str]] = []
    expected_length: int | None = None
    for measurement_index, path in enumerate(files, start=1):
        frame = pd.read_csv(path)
        missing = [sensor for sensor in SENSORS if sensor not in frame.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        values = frame[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite sensor values found")
        if expected_length is None:
            expected_length = len(values)
        if len(values) != expected_length:
            raise ValueError(f"{path}: length {len(values)} differs from {expected_length}")
        arrays.append(values)
        for sensor_index, sensor in enumerate(SENSORS):
            column = values[:, sensor_index].astype(float)
            heatmaps[sensor].append(binned_rms(column, args.heatmap_bins))
        summary_rows.append({
            "measurement": measurement_index,
            "file": path.name,
            "samples": len(values),
            "horizontal_mean": float(values[:, 0].mean()),
            "horizontal_std": float(values[:, 0].std()),
            "horizontal_rms": rms(values[:, 0]),
            "horizontal_abs_p95": float(np.percentile(np.abs(values[:, 0]), 95)),
            "horizontal_abs_max": float(np.abs(values[:, 0]).max()),
            "vertical_mean": float(values[:, 1].mean()),
            "vertical_std": float(values[:, 1].std()),
            "vertical_rms": rms(values[:, 1]),
            "vertical_abs_p95": float(np.percentile(np.abs(values[:, 1]), 95)),
            "vertical_abs_max": float(np.abs(values[:, 1]).max()),
        })

    n_files = len(files)
    file_tag = f"all{n_files}"
    summary = pd.DataFrame(summary_rows)
    output_prefix = f"{condition_name}_{bearing_name}_{file_tag}"
    summary.to_csv(args.output_dir / f"{output_prefix}_summary.csv", index=False)
    np.savez_compressed(
        args.output_dir / f"{output_prefix}_raw_arrays.npz",
        measurements=np.stack(arrays, axis=0),
        files=np.asarray([path.name for path in files]),
    )

    # All raw waveform segments. Each file is kept separate; the x coordinate
    # is the measurement number plus normalized position within that file.
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, dpi=180)
    colors = ["#2f5d8c", "#b45f06"]
    x_local = np.linspace(0, 1, expected_length, endpoint=False)
    for sensor_index, sensor in enumerate(SENSORS):
        ax = axes[sensor_index]
        for measurement_index, values in enumerate(arrays, start=1):
            ax.plot(measurement_index - 1 + x_local, values[:, sensor_index],
                    color=colors[sensor_index], linewidth=0.22, alpha=0.62)
        ax.set_ylabel("amplitude")
        ax.set_title(SENSOR_LABELS[sensor])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("measurement index; adjacent CSV files are sampled 1 minute apart")
    axes[-1].set_xlim(0, n_files)
    tick_step = max(1, n_files // 10)
    axes[-1].set_xticks(np.arange(0, n_files + 1, tick_step))
    fig.suptitle(f"XJTU-SY {bearing_name}: all {n_files} raw vibration recordings ({condition_name})", fontsize=15)
    fig.text(0.01, 0.01, f"{n_files} files × {expected_length:,} samples; each file is plotted as a separate 1-minute-spaced recording.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(args.output_dir / f"{output_prefix}_raw_waveforms.png", bbox_inches="tight")
    plt.close(fig)

    # Per-measurement feature trends.
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, dpi=180)
    for ax, prefix, label, color in [
        (axes[0], "horizontal", "Horizontal vibration", "#2f5d8c"),
        (axes[1], "vertical", "Vertical vibration", "#b45f06"),
    ]:
        ax.plot(summary["measurement"], summary[f"{prefix}_rms"], color=color, linewidth=1.3, label="RMS")
        ax.plot(summary["measurement"], summary[f"{prefix}_abs_p95"], color=color, linestyle="--", linewidth=1.0, label="95th percentile |amplitude|")
        ax.set_ylabel("amplitude")
        ax.set_title(label)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("measurement index (1-minute interval)")
    fig.suptitle(f"XJTU-SY {bearing_name}: vibration level across {n_files} measurements", fontsize=15)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(args.output_dir / f"{output_prefix}_feature_trends.png", bbox_inches="tight")
    plt.close(fig)

    # RMS heatmaps preserve the within-file pattern while showing all files.
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True, dpi=180)
    for ax, sensor, color_map in zip(axes, SENSORS, ["Blues", "Oranges"]):
        matrix = np.asarray(heatmaps[sensor])
        low, high = np.percentile(matrix, [1, 99])
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=color_map, vmin=low, vmax=high,
                          extent=[0, 1, 1, n_files])
        ax.set_ylabel("measurement / file")
        ax.set_title(f"{SENSOR_LABELS[sensor]}: local RMS heatmap")
        fig.colorbar(image, ax=ax, label="local RMS")
    axes[-1].set_xlabel("normalized position within each CSV recording")
    fig.suptitle(f"XJTU-SY {bearing_name}: local vibration energy across all measurements", fontsize=15)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(args.output_dir / f"{output_prefix}_rms_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "input_dir": str(args.input_dir),
        "condition": condition_name,
        "bearing": bearing_name,
        "file_count": n_files,
        "files": [path.name for path in files],
        "samples_per_file": int(expected_length),
        "sensors": SENSORS,
        "sampling_interval_between_files": "1 minute",
        "raw_waveform_plot": "each CSV plotted as a separate segment; within-file position normalized to [0, 1)",
        "heatmap": f"local RMS with {args.heatmap_bins} bins per file; color limits clipped to 1st-99th percentiles",
        "processing": "raw values for waveform; RMS only for overview heatmap and summary",
    }
    (args.output_dir / f"{output_prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"files={n_files}; samples_per_file={expected_length}; total_samples={n_files * expected_length:,}")
    print(summary[["measurement", "file", "horizontal_rms", "vertical_rms"]].head().to_string(index=False))
    print(f"Saved all-file plots to: {args.output_dir}")


if __name__ == "__main__":
    main()

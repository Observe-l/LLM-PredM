#!/usr/bin/env python3
"""Plot per-CSV RMS of sampled observed and Chronos-2 XJTU-SY sensor data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SENSOR_COLUMNS = ["horizontal_abs", "vertical_abs"]
PREDICTED_COLUMNS = ["predicted_horizontal_abs_q50", "predicted_vertical_abs_q50"]


def rms(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def plot_rms(comparison: pd.DataFrame, output: Path, prefix: str, reference_files: int) -> None:
    x = comparison["file_index"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, dpi=180)
    labels = [
        ("horizontal_rms", "Horizontal vibration RMS"),
        ("vertical_rms", "Vertical vibration RMS"),
    ]
    for ax, (sensor, title) in zip(axes, labels):
        ax.axvspan(1, reference_files, color="#dbeafe", alpha=0.8, label=f"health reference: first {reference_files} files")
        ax.plot(x, comparison[f"observed_{sensor}"], color="#1f4e79", linewidth=1.2, label="observed sampled data RMS")
        ax.plot(x, comparison[f"forecast_{sensor}_q50"], color="#b45f06", linewidth=1.35, linestyle="--", label="Chronos-2 forecast data RMS q50")
        ax.set_ylabel("RMS")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("measurement index; one CSV per minute")
    axes[-1].set_xlim(1, int(x.max()))
    tick_step = max(1, int(x.max()) // 10)
    axes[-1].set_xticks(np.arange(1, int(x.max()) + 1, tick_step))
    fig.suptitle(f"{prefix}: RMS of sampled sensor data and Chronos-2 forecast", fontsize=15)
    fig.text(
        0.01,
        0.01,
        "RMS is calculated independently for each CSV from its 256 sampled points; it is not used in the LHI calculation.",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-files", type=int, default=20)
    parser.add_argument("--center-points", type=int, default=256)
    args = parser.parse_args()
    if args.center_points != 256:
        raise ValueError("This experiment expects center-points=256")

    summary_rows: list[dict[str, object]] = []
    for bearing_dir in sorted(path for path in args.output_root.iterdir() if path.is_dir()):
        actual_paths = list(bearing_dir.glob("*_abs_middle256_sensor_readings.csv"))
        forecast_paths = list(bearing_dir.glob("*_chronos2_abs_middle256_sensor_forecast_q50.csv"))
        if not actual_paths or not forecast_paths:
            continue
        actual = pd.read_csv(actual_paths[0])
        forecast = pd.read_csv(forecast_paths[0])
        n_files = int(actual["file_index"].max())

        observed = actual.groupby("file_index", sort=True)[SENSOR_COLUMNS].agg(rms)
        predicted = forecast.groupby("target_file_index", sort=True)[PREDICTED_COLUMNS].agg(rms)
        predicted.columns = ["horizontal_rms", "vertical_rms"]
        predicted = predicted.reindex(range(1, n_files + 1))
        comparison = pd.DataFrame({
            "file_index": np.arange(1, n_files + 1),
            "time_minute": np.arange(n_files, dtype=float),
            "observed_horizontal_rms": observed["horizontal_abs"].to_numpy(dtype=float),
            "observed_vertical_rms": observed["vertical_abs"].to_numpy(dtype=float),
            "forecast_horizontal_rms_q50": predicted["horizontal_rms"].to_numpy(dtype=float),
            "forecast_vertical_rms_q50": predicted["vertical_rms"].to_numpy(dtype=float),
        })
        prefix = bearing_dir.name
        csv_path = bearing_dir / f"{prefix}_abs_middle256_rms_comparison.csv"
        plot_path = bearing_dir / f"{prefix}_abs_middle256_rms_comparison.png"
        comparison.to_csv(csv_path, index=False)
        plot_rms(comparison, plot_path, prefix, args.reference_files)

        valid = comparison["forecast_horizontal_rms_q50"].notna()
        summary_rows.append({
            "condition": prefix.split("_", 1)[0],
            "bearing": prefix.split("_", 1)[1],
            "measurement_count": n_files,
            "reference_files": args.reference_files,
            "forecast_measurements": int(valid.sum()),
            "horizontal_rms_mae": float(np.mean(np.abs(comparison.loc[valid, "forecast_horizontal_rms_q50"] - comparison.loc[valid, "observed_horizontal_rms"]))),
            "horizontal_rms_rmse": float(np.sqrt(np.mean((comparison.loc[valid, "forecast_horizontal_rms_q50"] - comparison.loc[valid, "observed_horizontal_rms"]) ** 2))),
            "vertical_rms_mae": float(np.mean(np.abs(comparison.loc[valid, "forecast_vertical_rms_q50"] - comparison.loc[valid, "observed_vertical_rms"]))),
            "vertical_rms_rmse": float(np.sqrt(np.mean((comparison.loc[valid, "forecast_vertical_rms_q50"] - comparison.loc[valid, "observed_vertical_rms"]) ** 2))),
            "plot": str(plot_path),
            "comparison": str(csv_path),
        })

    summary = pd.DataFrame(summary_rows).sort_values(["condition", "bearing"])
    summary_path = args.output_root / "all_bearings_abs_middle256_rms_summary.csv"
    summary.to_csv(summary_path, index=False)
    metadata = {
        "center_points": args.center_points,
        "reference_files": args.reference_files,
        "rms_definition": "sqrt(mean(sample_value^2)) over the 256 sampled points in each CSV",
        "observed_data": "abs(middle-256) sensor readings",
        "forecast_data": "Chronos-2 q50 predictions of abs(middle-256) sensor readings",
        "used_for_lhi": False,
        "summary_file": str(summary_path),
    }
    metadata_path = args.output_root / "all_bearings_abs_middle256_rms_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved RMS comparisons to: {args.output_root}")


if __name__ == "__main__":
    main()

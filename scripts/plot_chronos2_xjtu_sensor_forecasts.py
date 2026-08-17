#!/usr/bin/env python3
"""Plot saved Chronos-2 sensor-waveform forecasts for all XJTU-SY bearings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SENSORS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]
LABELS = ["Horizontal vibration", "Vertical vibration"]
COLORS = ["#2f855a", "#b45f06"]


def numeric_csvs(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("*.csv"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def load_center_waveforms(directory: Path, center_points: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for path in numeric_csvs(directory):
        frame = pd.read_csv(path)
        values = frame[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        start = (len(values) - center_points) // 2
        rows.append(values[start : start + center_points])
    return np.stack(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--center-points", type=int, default=128)
    parser.add_argument("--reference-minutes", type=int, default=20)
    args = parser.parse_args()

    for condition_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        for bearing_dir in sorted(path for path in condition_dir.iterdir() if path.is_dir()):
            prefix = f"{condition_dir.name}_{bearing_dir.name}"
            output_dir = args.output_root / prefix
            raw_path = output_dir / f"{prefix}_chronos2_forecast_raw_q50.npz"
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)

            actual = load_center_waveforms(bearing_dir, args.center_points)
            raw = np.load(raw_path)
            predicted_blocks = raw["forecasts"]
            predicted = np.concatenate(predicted_blocks, axis=1)
            n_minutes = len(actual)
            forecast_points = max(0, (n_minutes - args.reference_minutes) * args.center_points)
            predicted = predicted[:, :forecast_points]
            observed = actual[args.reference_minutes:].reshape(-1, 2).T[:, :forecast_points]
            x = args.reference_minutes + np.arange(forecast_points) / args.center_points

            fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, dpi=180)
            for sensor_index, (ax, label, color) in enumerate(zip(axes, LABELS, COLORS)):
                ax.plot(x, observed[sensor_index], color="#111827", linewidth=0.28, alpha=0.65, label="ground truth (middle 128)")
                ax.plot(x, predicted[sensor_index], color=color, linewidth=0.32, alpha=0.82, label="Chronos-2 q50 forecast")
                ax.set_ylabel("amplitude")
                ax.set_title(label)
                ax.grid(alpha=0.25)
                ax.legend(loc="upper left", fontsize=8)
            axes[-1].set_xlabel("measurement time (minutes; one CSV per minute)")
            axes[-1].set_xlim(args.reference_minutes, n_minutes)
            tick_step = max(1, n_minutes // 10)
            axes[-1].set_xticks(np.arange(args.reference_minutes, n_minutes + 1, tick_step))
            fig.suptitle(f"{prefix}: Chronos-2 sensor waveform forecast", fontsize=15)
            fig.text(
                0.01,
                0.01,
                f"Forecast starts after minute {args.reference_minutes}; each CSV contributes its middle {args.center_points} samples; q50 point forecast.",
                fontsize=8.5,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.96))
            output_path = output_dir / f"{prefix}_chronos2_sensor_forecast_q50.png"
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            print(f"[{prefix}] saved {output_path}", flush=True)


if __name__ == "__main__":
    main()

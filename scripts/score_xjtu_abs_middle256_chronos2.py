#!/usr/bin/env python3
"""Score all XJTU-SY absolute middle-256 Chronos-2 sensor forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    pooled_errors: list[np.ndarray] = []
    for bearing_dir in sorted(path for path in args.output_root.iterdir() if path.is_dir()):
        actual_paths = list(bearing_dir.glob("*_abs_middle256_sensor_readings.csv"))
        forecast_paths = list(bearing_dir.glob("*_chronos2_abs_middle256_sensor_forecast_q50.csv"))
        if not actual_paths or not forecast_paths:
            continue
        actual = pd.read_csv(actual_paths[0], usecols=["file_index", "sample_in_file", "horizontal_abs", "vertical_abs"])
        forecast = pd.read_csv(forecast_paths[0], usecols=["target_file_index", "sample_in_file", "predicted_horizontal_abs_q50", "predicted_vertical_abs_q50"])
        merged = forecast.merge(
            actual,
            left_on=["target_file_index", "sample_in_file"],
            right_on=["file_index", "sample_in_file"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(forecast):
            raise ValueError(f"{bearing_dir.name}: only {len(merged)}/{len(forecast)} forecast points matched ground truth")
        error = merged[["predicted_horizontal_abs_q50", "predicted_vertical_abs_q50"]].to_numpy() - merged[["horizontal_abs", "vertical_abs"]].to_numpy()
        pooled_errors.append(error)
        rows.append({
            "condition": bearing_dir.name.split("_", 1)[0],
            "bearing": bearing_dir.name.split("_", 1)[1],
            "forecast_points": int(len(merged)),
            "horizontal_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "vertical_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
            "combined_sensor_rmse": float(np.sqrt(np.mean(error ** 2))),
        })

    if not rows:
        raise FileNotFoundError(f"No forecast/ground-truth pairs found under {args.output_root}")
    summary = pd.DataFrame(rows).sort_values(["condition", "bearing"]).reset_index(drop=True)
    summary.to_csv(args.output_root / "all_bearings_abs_middle256_chronos2_rmse.csv", index=False)
    pooled = np.vstack(pooled_errors)
    pooled_row = {
        "forecast_points": int(len(pooled)),
        "horizontal_rmse": float(np.sqrt(np.mean(pooled[:, 0] ** 2))),
        "vertical_rmse": float(np.sqrt(np.mean(pooled[:, 1] ** 2))),
        "combined_sensor_rmse": float(np.sqrt(np.mean(pooled ** 2))),
    }
    print(summary.to_string(index=False))
    print("\nPooled overall RMSE:")
    print(pd.Series(pooled_row).to_string())
    print(f"\nSaved: {args.output_root / 'all_bearings_abs_middle256_chronos2_rmse.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot observed and Chronos-2 forecast LHI from abs middle-256 readings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_lhi(rms_values: np.ndarray, reference_count: int, epsilon: float = 1e-6) -> dict[str, np.ndarray]:
    reference = rms_values[:reference_count]
    reference_mean = reference.mean(axis=0)
    reference_range = reference.max(axis=0) - reference.min(axis=0)
    usable = np.isfinite(reference_range) & (reference_range > 1e-12)
    if not np.any(usable):
        raise ValueError("No usable sensor range in the health reference")
    normalized = np.abs((rms_values[:, usable] - reference_mean[usable]) / reference_range[usable])
    d_mae = normalized.mean(axis=1)
    d_rmse = np.sqrt(np.mean(normalized ** 2, axis=1))
    baseline_mae = float(d_mae[:reference_count].mean())
    baseline_rmse = float(np.sqrt(np.mean(d_rmse[:reference_count] ** 2)))
    return {
        "lhi_mae": np.log((d_mae + epsilon) / (baseline_mae + epsilon)),
        "lhi_rmse": np.log((d_rmse + epsilon) / (baseline_rmse + epsilon)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-files", type=int, default=20)
    parser.add_argument("--center-points", type=int, default=256)
    args = parser.parse_args()

    summary_rows: list[dict[str, object]] = []
    for bearing_dir in sorted(path for path in args.output_root.iterdir() if path.is_dir()):
        actual_paths = list(bearing_dir.glob("*_abs_middle256_sensor_readings.csv"))
        forecast_paths = list(bearing_dir.glob("*_chronos2_abs_middle256_sensor_forecast_q50.csv"))
        if not actual_paths or not forecast_paths:
            continue
        actual = pd.read_csv(actual_paths[0])
        forecast = pd.read_csv(forecast_paths[0])
        n_files = int(actual["file_index"].max())
        observed_rms = actual.groupby("file_index", sort=True).agg(
            horizontal_rms=("horizontal_abs", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            vertical_rms=("vertical_abs", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
        ).reindex(range(1, n_files + 1)).to_numpy(dtype=float)
        forecast_rms = forecast.groupby("target_file_index", sort=True).agg(
            horizontal_rms=("predicted_horizontal_abs_q50", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            vertical_rms=("predicted_vertical_abs_q50", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
        )
        predicted_rms = np.full_like(observed_rms, np.nan)
        for file_index, values in forecast_rms.iterrows():
            predicted_rms[int(file_index) - 1] = values.to_numpy(dtype=float)
        if np.isnan(predicted_rms[args.reference_files:]).any():
            raise ValueError(f"{bearing_dir.name}: missing predicted RMS values after reference")

        observed_lhi = compute_lhi(observed_rms, args.reference_files)
        predicted_input = np.vstack([observed_rms[:args.reference_files], predicted_rms[args.reference_files:]])
        predicted_lhi = compute_lhi(predicted_input, args.reference_files)
        comparison = pd.DataFrame({
            "measurement": np.arange(1, n_files + 1),
            "observed_horizontal_rms": observed_rms[:, 0],
            "observed_vertical_rms": observed_rms[:, 1],
            "predicted_horizontal_rms_q50": predicted_rms[:, 0],
            "predicted_vertical_rms_q50": predicted_rms[:, 1],
            "observed_lhi_mae": observed_lhi["lhi_mae"],
            "observed_lhi_rmse": observed_lhi["lhi_rmse"],
            "forecast_lhi_mae_q50": np.r_[np.full(args.reference_files, np.nan), predicted_lhi["lhi_mae"][args.reference_files:]],
            "forecast_lhi_rmse_q50": np.r_[np.full(args.reference_files, np.nan), predicted_lhi["lhi_rmse"][args.reference_files:]],
        })
        prefix = bearing_dir.name
        comparison.to_csv(bearing_dir / f"{prefix}_abs_middle256_lhi_comparison.csv", index=False)

        fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, dpi=180)
        for ax, metric, label in [(axes[0], "rmse", "LHI_RMSE"), (axes[1], "mae", "LHI_MAE")]:
            ax.axvspan(1, args.reference_files, color="#dbeafe", alpha=0.8, label=f"health reference: first {args.reference_files} files")
            ax.plot(comparison["measurement"], comparison[f"observed_lhi_{metric}"], color="#111827", linewidth=1.2, label="observed sampled LHI")
            ax.plot(comparison["measurement"], comparison[f"forecast_lhi_{metric}_q50"], color="#15803d", linewidth=1.45, linestyle="--", label="Chronos-2 forecast LHI q50")
            ax.axhline(0.0, color="#6b7280", linewidth=0.9)
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("measurement index; one CSV per minute")
        axes[-1].set_xlim(1, n_files)
        tick_step = max(1, n_files // 10)
        axes[-1].set_xticks(np.arange(1, n_files + 1, tick_step))
        fig.suptitle(f"{prefix}: observed sampled LHI and Chronos-2 forecast LHI", fontsize=15)
        fig.text(0.01, 0.01, "LHI is computed from per-file RMS of abs(middle-256) sensor readings; both observed and forecast use the first 20 files as reference.", fontsize=8.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        plot_path = bearing_dir / f"{prefix}_abs_middle256_lhi_comparison.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        valid = comparison["forecast_lhi_rmse_q50"].notna()
        summary_rows.append({
            "condition": bearing_dir.name.split("_", 1)[0],
            "bearing": bearing_dir.name.split("_", 1)[1],
            "measurement_count": n_files,
            "forecast_measurements": int(valid.sum()),
            "forecast_lhi_rmse_mae_vs_observed": float(np.mean(np.abs(comparison.loc[valid, "forecast_lhi_rmse_q50"] - comparison.loc[valid, "observed_lhi_rmse"]))),
            "forecast_lhi_rmse_rmse_vs_observed": float(np.sqrt(np.mean((comparison.loc[valid, "forecast_lhi_rmse_q50"] - comparison.loc[valid, "observed_lhi_rmse"]) ** 2))),
            "plot": str(plot_path),
        })

    summary = pd.DataFrame(summary_rows).sort_values(["condition", "bearing"])
    summary.to_csv(args.output_root / "all_bearings_abs_middle256_lhi_summary.csv", index=False)
    metadata = {
        "reference_files": args.reference_files,
        "center_points": args.center_points,
        "lhi_definition": "log ratio of condition-free per-file sensor RMS drift to the first-20-file baseline",
        "preprocessing": "abs applied before middle-256 extraction",
        "summary_file": str(args.output_root / "all_bearings_abs_middle256_lhi_summary.csv"),
    }
    (args.output_root / "all_bearings_abs_middle256_lhi_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(summary.to_string(index=False))
    print(f"Saved LHI comparisons to: {args.output_root}")


if __name__ == "__main__":
    main()

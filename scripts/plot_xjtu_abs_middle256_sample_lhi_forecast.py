#!/usr/bin/env python3
"""Plot custom sample-level LHI for observed and Chronos-2 XJTU-SY forecasts.

The LHI follows the project's sample-level definition, without reducing each
CSV measurement to RMS first:

1. use all abs(middle-256) points in the first reference files as health data;
2. calculate a per-sensor reference mean and min-max range;
3. calculate absolute min-max-normalized sensor drift at every time sample;
4. aggregate the sensor drifts into D_MAE and D_RMSE;
5. calculate LHI = log((D + eps) / (B + eps)), where B is the reference-period
   drift baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SENSOR_COLUMNS = ["horizontal_abs", "vertical_abs"]


def compute_sample_lhi(
    values: np.ndarray,
    reference_points: int,
    epsilon: float = 1e-6,
    range_epsilon: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    """Compute the project's custom LHI directly for every sensor sample."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(f"Expected an (n, {len(SENSOR_COLUMNS)}) sensor array, got {values.shape}")
    if reference_points <= 0 or reference_points > len(values):
        raise ValueError("reference_points must be within the available sample range")

    reference = values[:reference_points]
    reference_mean = np.mean(reference, axis=0)
    reference_range = np.max(reference, axis=0) - np.min(reference, axis=0)
    usable = np.isfinite(reference_range) & (reference_range > range_epsilon)
    if not np.any(usable):
        raise ValueError("No usable sensor min-max range in the health reference")

    normalized_drift = np.full_like(values, np.nan, dtype=float)
    normalized_drift[:, usable] = np.abs(
        (values[:, usable] - reference_mean[usable]) / reference_range[usable]
    )
    valid_sensor_count = np.isfinite(normalized_drift).sum(axis=1)
    d_mae = np.divide(
        np.nansum(normalized_drift, axis=1),
        valid_sensor_count,
        out=np.full(len(values), np.nan, dtype=float),
        where=valid_sensor_count > 0,
    )
    d_rmse = np.sqrt(
        np.divide(
            np.nansum(normalized_drift**2, axis=1),
            valid_sensor_count,
            out=np.full(len(values), np.nan, dtype=float),
            where=valid_sensor_count > 0,
        )
    )

    baseline_mae = float(np.nanmean(d_mae[:reference_points]))
    baseline_rmse = float(np.sqrt(np.nanmean(d_rmse[:reference_points] ** 2)))
    lhi_mae = np.log((d_mae + epsilon) / (baseline_mae + epsilon))
    lhi_rmse = np.log((d_rmse + epsilon) / (baseline_rmse + epsilon))
    return {
        "d_mae": d_mae,
        "d_rmse": d_rmse,
        "lhi_mae": lhi_mae,
        "lhi_rmse": lhi_rmse,
        "reference_mean": reference_mean,
        "reference_range": reference_range,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "usable": usable,
    }


def plot_comparison(
    comparison: pd.DataFrame,
    output: Path,
    prefix: str,
    reference_files: int,
    center_points: int,
) -> None:
    x = comparison["time_minute"].to_numpy(dtype=float)
    reference_end = float(reference_files)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, dpi=180)
    for ax, metric, label in [
        (axes[0], "rmse", "LHI_RMSE"),
        (axes[1], "mae", "LHI_MAE"),
    ]:
        ax.axvspan(0, reference_end, color="#dbeafe", alpha=0.8, label=f"health reference: first {reference_files} files")
        ax.plot(
            x,
            comparison[f"observed_lhi_{metric}"],
            color="#111827",
            linewidth=0.28,
            alpha=0.78,
            label="observed sampled LHI",
        )
        ax.plot(
            x,
            comparison[f"forecast_lhi_{metric}_q50"],
            color="#15803d",
            linewidth=0.34,
            alpha=0.9,
            linestyle="--",
            label="Chronos-2 forecast LHI q50",
        )
        ax.axhline(0.0, color="#6b7280", linewidth=0.9)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time since first CSV (minutes; 256 sampled points per CSV)")
    axes[-1].set_xlim(0, float(comparison["time_minute"].max()))
    fig.suptitle(f"{prefix}: observed sampled LHI and Chronos-2 forecast LHI", fontsize=15)
    fig.text(
        0.01,
        0.01,
        "LHI is computed directly at every abs(middle-256) sensor sample; no RMS reduction is used.",
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
    parser.add_argument("--lhi-epsilon", type=float, default=1e-6)
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
        n_points = n_files * args.center_points
        reference_points = args.reference_files * args.center_points
        if reference_points >= n_points:
            raise ValueError(f"{bearing_dir.name}: not enough measurements after the reference")

        actual_values = actual[SENSOR_COLUMNS].to_numpy(dtype=float)
        predicted_values = np.full_like(actual_values, np.nan)
        for row in forecast.itertuples(index=False):
            point = (int(row.target_file_index) - 1) * args.center_points + int(row.sample_in_file)
            if 0 <= point < n_points:
                predicted_values[point] = [
                    float(row.predicted_horizontal_abs_q50),
                    float(row.predicted_vertical_abs_q50),
                ]
        if np.isnan(predicted_values[reference_points:]).any():
            raise ValueError(f"{bearing_dir.name}: forecast does not cover every post-reference point")

        observed_lhi = compute_sample_lhi(actual_values, reference_points, args.lhi_epsilon)
        predicted_input = np.vstack([actual_values[:reference_points], predicted_values[reference_points:]])
        forecast_lhi = compute_sample_lhi(predicted_input, reference_points, args.lhi_epsilon)
        comparison = pd.DataFrame({
            "measurement_index": np.arange(1, n_points + 1),
            "file_index": np.repeat(np.arange(1, n_files + 1), args.center_points),
            "sample_in_file": np.tile(np.arange(args.center_points), n_files),
            "time_minute": np.arange(n_points, dtype=float) / args.center_points,
            "observed_horizontal_abs": actual_values[:, 0],
            "observed_vertical_abs": actual_values[:, 1],
            "predicted_horizontal_abs_q50": predicted_values[:, 0],
            "predicted_vertical_abs_q50": predicted_values[:, 1],
            "observed_d_mae": observed_lhi["d_mae"],
            "observed_d_rmse": observed_lhi["d_rmse"],
            "observed_lhi_mae": observed_lhi["lhi_mae"],
            "observed_lhi_rmse": observed_lhi["lhi_rmse"],
            "forecast_lhi_mae_q50": np.r_[np.full(reference_points, np.nan), forecast_lhi["lhi_mae"][reference_points:]],
            "forecast_lhi_rmse_q50": np.r_[np.full(reference_points, np.nan), forecast_lhi["lhi_rmse"][reference_points:]],
        })
        prefix = bearing_dir.name
        csv_path = bearing_dir / f"{prefix}_abs_middle256_sample_lhi_comparison.csv"
        plot_path = bearing_dir / f"{prefix}_abs_middle256_sample_lhi_comparison.png"
        comparison.to_csv(csv_path, index=False)
        plot_comparison(comparison, plot_path, prefix, args.reference_files, args.center_points)

        valid = comparison["forecast_lhi_rmse_q50"].notna()
        summary_rows.append({
            "condition": prefix.split("_", 1)[0],
            "bearing": prefix.split("_", 1)[1],
            "measurement_count": n_files,
            "sample_count": n_points,
            "reference_files": args.reference_files,
            "reference_points": reference_points,
            "forecast_points": int(valid.sum()),
            "forecast_lhi_rmse_mae_vs_observed": float(np.mean(np.abs(comparison.loc[valid, "forecast_lhi_rmse_q50"] - comparison.loc[valid, "observed_lhi_rmse"]))),
            "forecast_lhi_rmse_rmse_vs_observed": float(np.sqrt(np.mean((comparison.loc[valid, "forecast_lhi_rmse_q50"] - comparison.loc[valid, "observed_lhi_rmse"]) ** 2))),
            "forecast_lhi_mae_mae_vs_observed": float(np.mean(np.abs(comparison.loc[valid, "forecast_lhi_mae_q50"] - comparison.loc[valid, "observed_lhi_mae"]))),
            "forecast_lhi_mae_rmse_vs_observed": float(np.sqrt(np.mean((comparison.loc[valid, "forecast_lhi_mae_q50"] - comparison.loc[valid, "observed_lhi_mae"]) ** 2))),
            "baseline_mae": float(observed_lhi["baseline_mae"]),
            "baseline_rmse": float(observed_lhi["baseline_rmse"]),
            "plot": str(plot_path),
            "comparison": str(csv_path),
        })

    summary = pd.DataFrame(summary_rows).sort_values(["condition", "bearing"])
    summary_path = args.output_root / "all_bearings_abs_middle256_sample_lhi_summary.csv"
    summary.to_csv(summary_path, index=False)
    metadata = {
        "reference_files": args.reference_files,
        "reference_points_per_bearing": args.reference_files * args.center_points,
        "center_points": args.center_points,
        "preprocessing": "absolute values, then middle-256 points from every CSV",
        "lhi_definition": "direct sample-level custom LHI; per-sensor reference mean and min-max range, absolute normalized drift, D_MAE/D_RMSE aggregation, log ratio to reference baseline",
        "rms_used_for_lhi": False,
        "forecast": "Chronos-2 q50 predictions of the abs(middle-256) sensor points",
        "summary_file": str(summary_path),
    }
    metadata_path = args.output_root / "all_bearings_abs_middle256_sample_lhi_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved direct sample-level LHI comparisons to: {args.output_root}")


if __name__ == "__main__":
    main()

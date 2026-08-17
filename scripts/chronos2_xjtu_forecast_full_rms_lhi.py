#!/usr/bin/env python3
"""Forecast full-sample XJTU-SY RMS and LHI series with Chronos-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronos2_xjtu_rms_forecast import load_pipeline
from plot_xjtu_sy_all_bearings_middle128 import compute_lhi_from_rms


SENSORS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]


def numeric_csvs(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("*.csv"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def load_series(directory: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    files = numeric_csvs(directory)
    if not files:
        raise FileNotFoundError(f"No numeric CSV files found in {directory}")
    sensor_rms: list[list[float]] = []
    for path in files:
        frame = pd.read_csv(path)
        values = frame[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite sensor values found")
        sensor_rms.append(np.sqrt(np.mean(values ** 2, axis=0)).tolist())
    sensor_rms_array = np.asarray(sensor_rms, dtype=float)
    combined_rms = np.sqrt(np.mean(sensor_rms_array ** 2, axis=1))
    return files, sensor_rms_array, combined_rms


def make_inputs(series: np.ndarray, reference_minutes: int, forecast_length: int) -> tuple[list[dict[str, np.ndarray]], list[int]]:
    if len(series) <= reference_minutes:
        raise ValueError("The series has no data after the reference period.")
    inputs: list[dict[str, np.ndarray]] = []
    origins: list[int] = []
    # Non-overlapping 20-minute blocks. Since all bearings have fewer than
    # 8192 minute-level observations, this passes all history before each
    # cutoff to Chronos-2 without truncation.
    for origin in range(reference_minutes, len(series), forecast_length):
        inputs.append({"target": series[:origin].astype(np.float32)[None, :]})
        origins.append(origin)
    return inputs, origins


def forecast_series(
    pipeline,
    series: np.ndarray,
    reference_minutes: int,
    forecast_length: int,
    batch_size: int,
) -> pd.DataFrame:
    inputs, origins = make_inputs(series, reference_minutes, forecast_length)
    rows: list[dict[str, float | int]] = []
    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_origins = origins[start : start + batch_size]
        print(f"Forecasting blocks {start + 1}-{start + len(batch_inputs)} / {len(inputs)}", flush=True)
        _, point_forecasts = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=forecast_length,
            quantile_levels=[0.5],
            batch_size=len(batch_inputs),
        )
        for origin, point_tensor in zip(batch_origins, point_forecasts):
            prediction = point_tensor.detach().float().cpu().numpy()[0]
            available = min(forecast_length, len(series) - origin)
            for horizon in range(available):
                rows.append({
                    "forecast_origin_minute": origin,
                    "forecast_horizon_minute": horizon + 1,
                    "target_minute": origin + horizon + 1,
                    "prediction_q50": float(prediction[horizon]),
                    "ground_truth": float(series[origin + horizon]),
                })
    return pd.DataFrame(rows).sort_values("target_minute")


def plot_comparison(
    output: Path,
    title_prefix: str,
    measurement: np.ndarray,
    rms_actual: np.ndarray,
    rms_forecast: np.ndarray,
    lhi_actual: np.ndarray,
    lhi_forecast: np.ndarray,
    reference_minutes: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, dpi=180)
    for ax, actual, forecast, ylabel, title in [
        (axes[0], rms_actual, rms_forecast, "Combined RMS", "Combined RMS"),
        (axes[1], lhi_actual, lhi_forecast, "LHI_RMSE", "LHI_RMSE"),
    ]:
        ax.axvspan(1, reference_minutes, color="#dbeafe", alpha=0.8, label=f"reference: first {reference_minutes} minutes")
        ax.plot(measurement, actual, color="#111827", linewidth=1.35, label="ground truth")
        ax.plot(measurement, forecast, color="#15803d", linewidth=1.45, linestyle="--", label="Chronos-2 q50")
        if ylabel == "LHI_RMSE":
            ax.axhline(0.0, color="#6b7280", linewidth=0.9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("measurement index; adjacent CSV files are sampled 1 minute apart")
    axes[-1].set_xlim(1, len(measurement))
    tick_step = max(1, len(measurement) // 10)
    axes[-1].set_xticks(np.arange(1, len(measurement) + 1, tick_step))
    fig.suptitle(f"{title_prefix}: full-sample RMS and LHI forecasting", fontsize=15)
    fig.text(0.01, 0.01, "Each CSV uses all 32768 sensor samples; Chronos-2 uses q50 point forecasts with prediction length 20.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--reference-minutes", type=int, default=20)
    parser.add_argument("--forecast-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    bearing_dirs = sorted(
        (path for condition in args.input_root.iterdir() if condition.is_dir()
         for path in condition.iterdir() if path.is_dir()),
        key=lambda path: (path.parent.name, path.name),
    )
    pipeline = load_pipeline(args.model_id, args.device, args.torch_dtype, args.local_files_only)
    summary_rows: list[dict[str, object]] = []

    for bearing_dir in bearing_dirs:
        condition = bearing_dir.parent.name
        bearing = bearing_dir.name
        prefix = f"{condition}_{bearing}"
        output_dir = args.output_root / prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{prefix}] loading full-sample RMS", flush=True)

        files, sensor_rms, combined_rms = load_series(bearing_dir)
        lhi_actual = compute_lhi_from_rms(sensor_rms, args.reference_minutes)["lhi_rmse"]
        rms_forecast = forecast_series(pipeline, combined_rms, args.reference_minutes, args.forecast_length, args.batch_size)
        lhi_forecast = forecast_series(pipeline, lhi_actual, args.reference_minutes, args.forecast_length, args.batch_size)

        result = pd.DataFrame({
            "measurement": np.arange(1, len(files) + 1),
            "ground_truth_combined_rms": combined_rms,
            "ground_truth_lhi_rmse": lhi_actual,
            "chronos2_combined_rms_q50": np.nan,
            "chronos2_lhi_rmse_q50": np.nan,
        })
        result.loc[rms_forecast["target_minute"].to_numpy() - 1, "chronos2_combined_rms_q50"] = rms_forecast["prediction_q50"].to_numpy()
        result.loc[lhi_forecast["target_minute"].to_numpy() - 1, "chronos2_lhi_rmse_q50"] = lhi_forecast["prediction_q50"].to_numpy()
        result.to_csv(output_dir / f"{prefix}_chronos2_full_rms_lhi_forecast_q50.csv", index=False)
        rms_forecast.to_csv(output_dir / f"{prefix}_chronos2_full_rms_forecast_q50.csv", index=False)
        lhi_forecast.to_csv(output_dir / f"{prefix}_chronos2_lhi_forecast_q50.csv", index=False)

        rms_plot = np.full(len(files), np.nan)
        lhi_plot = np.full(len(files), np.nan)
        rms_plot[rms_forecast["target_minute"].to_numpy() - 1] = rms_forecast["prediction_q50"].to_numpy()
        lhi_plot[lhi_forecast["target_minute"].to_numpy() - 1] = lhi_forecast["prediction_q50"].to_numpy()
        plot_path = output_dir / f"{prefix}_chronos2_full_rms_lhi_forecast_q50.png"
        plot_comparison(
            plot_path,
            prefix,
            np.arange(1, len(files) + 1),
            combined_rms,
            rms_plot,
            lhi_actual,
            lhi_plot,
            args.reference_minutes,
        )

        rms_valid = np.isfinite(rms_plot)
        lhi_valid = np.isfinite(lhi_plot)
        summary_rows.append({
            "condition": condition,
            "bearing": bearing,
            "file_count": len(files),
            "reference_minutes": args.reference_minutes,
            "forecast_length": args.forecast_length,
            "rms_forecast_points": int(rms_valid.sum()),
            "lhi_forecast_points": int(lhi_valid.sum()),
            "rms_mae": float(np.mean(np.abs(rms_plot[rms_valid] - combined_rms[rms_valid]))),
            "rms_rmse": float(np.sqrt(np.mean((rms_plot[rms_valid] - combined_rms[rms_valid]) ** 2))),
            "lhi_mae": float(np.mean(np.abs(lhi_plot[lhi_valid] - lhi_actual[lhi_valid]))),
            "lhi_rmse": float(np.sqrt(np.mean((lhi_plot[lhi_valid] - lhi_actual[lhi_valid]) ** 2))),
            "plot": str(plot_path),
            "output_dir": str(output_dir),
        })
        metadata = {
            "input_dir": str(bearing_dir),
            "file_count": len(files),
            "samples_per_file": 32768,
            "reference_minutes": args.reference_minutes,
            "forecast_length": args.forecast_length,
            "context_policy": "all minute-level history before each cutoff; the series is shorter than 8192 points",
            "forecast_policy": "non-overlapping 20-minute blocks starting after minute 20",
            "rms_definition": "combined RMS over both sensors and all 32768 samples in each CSV",
            "lhi_definition": "LHI_RMSE from the two full-sample sensor RMS series using the first 20 minutes as reference",
            "point_forecast": "Chronos-2 q50",
            "model_id": args.model_id,
            "plot": str(plot_path),
        }
        (output_dir / f"{prefix}_chronos2_full_rms_lhi_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[{prefix}] saved RMS/LHI forecast plot", flush=True)

    pd.DataFrame(summary_rows).to_csv(args.output_root / "all_bearings_full_rms_lhi_chronos2_summary.csv", index=False)
    print(f"Saved summary to: {args.output_root / 'all_bearings_full_rms_lhi_chronos2_summary.csv'}")


if __name__ == "__main__":
    main()

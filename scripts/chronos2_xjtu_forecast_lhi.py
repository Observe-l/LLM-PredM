#!/usr/bin/env python3
"""Forecast XJTU-SY center waveforms with Chronos-2 and compute forecast LHI."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--center-points", type=int, default=128)
    parser.add_argument("--reference-minutes", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--prediction-length", type=int, default=1024)
    parser.add_argument("--forecast-block-minutes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def numeric_csvs(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("*.csv"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def load_center_waveforms(directory: Path, center_points: int) -> tuple[list[Path], np.ndarray, np.ndarray]:
    files = numeric_csvs(directory)
    if not files:
        raise FileNotFoundError(f"No numeric CSV files found in {directory}")
    selected_rows: list[np.ndarray] = []
    full_rms_rows: list[list[float]] = []
    for path in files:
        frame = pd.read_csv(path)
        missing = [sensor for sensor in SENSORS if sensor not in frame.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        values = frame[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite sensor values found")
        if center_points > len(values):
            raise ValueError(f"{path}: center-points exceeds file length")
        start = (len(values) - center_points) // 2
        selected = values[start : start + center_points]
        selected_rows.append(selected)
        full_rms_rows.append(np.sqrt(np.mean(values ** 2, axis=0)).tolist())
    return files, np.stack(selected_rows, axis=0), np.asarray(full_rms_rows, dtype=float)


def make_forecast_inputs(
    selected: np.ndarray,
    reference_minutes: int,
    context_length: int,
    prediction_length: int,
    block_minutes: int,
) -> tuple[list[dict[str, np.ndarray]], list[int]]:
    n_minutes, points_per_minute, n_sensors = selected.shape
    if prediction_length != block_minutes * points_per_minute:
        raise ValueError(
            f"prediction-length ({prediction_length}) must equal "
            f"forecast-block-minutes ({block_minutes}) * center-points ({points_per_minute})."
        )
    if n_minutes <= reference_minutes:
        raise ValueError("The bearing has no measurements after the reference period.")

    inputs: list[dict[str, np.ndarray]] = []
    origins: list[int] = []
    # Non-overlapping 8-minute forecast blocks create one continuous predicted
    # RMS/LHI sequence. Each origin still uses the observed history available
    # at that origin, truncated to the model's maximum context length.
    for origin in range(reference_minutes, n_minutes, block_minutes):
        history = selected[:origin].reshape(origin * points_per_minute, n_sensors).T
        history = history[:, -context_length:]
        inputs.append({"target": history.astype(np.float32)})
        origins.append(origin)
    return inputs, origins


def forecast_blocks(
    pipeline,
    inputs: list[dict[str, np.ndarray]],
    origins: list[int],
    n_minutes: int,
    points_per_minute: int,
    prediction_length: int,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    minute_rows: list[dict[str, float | int]] = []
    raw_blocks: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_origins = origins[start : start + batch_size]
        print(
            f"Forecasting blocks {start + 1}-{start + len(batch_inputs)} / {len(inputs)}",
            flush=True,
        )
        _, point_forecasts = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=prediction_length,
            quantile_levels=[0.5],
            batch_size=len(batch_inputs),
        )
        for origin, point_tensor in zip(batch_origins, point_forecasts):
            prediction = point_tensor.detach().float().cpu().numpy()
            if prediction.shape != (2, prediction_length):
                raise ValueError(f"Unexpected Chronos-2 forecast shape: {prediction.shape}")
            raw_blocks.append(prediction)
            available_minutes = min(
                (n_minutes - origin),
                prediction_length // points_per_minute,
            )
            for minute_offset in range(available_minutes):
                start_point = minute_offset * points_per_minute
                stop_point = start_point + points_per_minute
                chunk = prediction[:, start_point:stop_point]
                sensor_rms = np.sqrt(np.mean(chunk ** 2, axis=1))
                target_minute = origin + minute_offset + 1
                minute_rows.append({
                    "forecast_origin_minute": origin,
                    "forecast_horizon_minute": minute_offset + 1,
                    "target_minute": target_minute,
                    "predicted_horizontal_rms": float(sensor_rms[0]),
                    "predicted_vertical_rms": float(sensor_rms[1]),
                    "predicted_combined_rms": float(np.sqrt(np.mean(chunk ** 2))),
                })
    return pd.DataFrame(minute_rows).sort_values("target_minute"), np.stack(raw_blocks, axis=0)


def save_lhi_comparison(
    output_dir: Path,
    prefix: str,
    full_rms: np.ndarray,
    sampled_rms: np.ndarray,
    forecast_rms: pd.DataFrame,
    reference_minutes: int,
) -> dict[str, float | str]:
    observed_full = compute_lhi_from_rms(full_rms, reference_minutes)
    observed_sampled = compute_lhi_from_rms(sampled_rms, reference_minutes)

    predicted_sensor_rms = forecast_rms[["predicted_horizontal_rms", "predicted_vertical_rms"]].to_numpy(dtype=float)
    predicted_with_reference = np.vstack([sampled_rms[:reference_minutes], predicted_sensor_rms])
    predicted_lhi = compute_lhi_from_rms(predicted_with_reference, reference_minutes)

    n_minutes = len(sampled_rms)
    comparison = pd.DataFrame({
        "measurement": np.arange(1, n_minutes + 1),
        "full_lhi_mae": observed_full["lhi_mae"],
        "full_lhi_rmse": observed_full["lhi_rmse"],
        "sampled_lhi_mae": observed_sampled["lhi_mae"],
        "sampled_lhi_rmse": observed_sampled["lhi_rmse"],
        "forecast_lhi_mae": np.nan,
        "forecast_lhi_rmse": np.nan,
    })
    predicted_minutes = forecast_rms["target_minute"].to_numpy(dtype=int)
    comparison.loc[predicted_minutes - 1, "forecast_lhi_mae"] = predicted_lhi["lhi_mae"][reference_minutes:]
    comparison.loc[predicted_minutes - 1, "forecast_lhi_rmse"] = predicted_lhi["lhi_rmse"][reference_minutes:]
    comparison.to_csv(output_dir / f"{prefix}_chronos2_lhi_comparison.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, dpi=180)
    for ax, metric, label in [
        (axes[0], "lhi_rmse", "LHI_RMSE"),
        (axes[1], "lhi_mae", "LHI_MAE"),
    ]:
        ax.axvspan(1, reference_minutes, color="#dbeafe", alpha=0.8, label=f"health reference: first {reference_minutes} minutes")
        ax.plot(comparison["measurement"], comparison[f"full_{metric}"], color="#111827", linewidth=1.35, label="full 32768 samples")
        ax.plot(comparison["measurement"], comparison[f"sampled_{metric}"], color="#b45f06", linewidth=1.15, label="middle 128 samples")
        ax.plot(comparison["measurement"], comparison[f"forecast_{metric}"], color="#15803d", linewidth=1.5, linestyle="--", label="Chronos-2 forecast")
        ax.axhline(0.0, color="#6b7280", linewidth=0.9)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("measurement index; adjacent CSV files are sampled 1 minute apart")
    axes[-1].set_xlim(1, n_minutes)
    tick_step = max(1, n_minutes // 10)
    axes[-1].set_xticks(np.arange(1, n_minutes + 1, tick_step))
    fig.suptitle(f"{prefix}: observed and Chronos-2 forecast LHI", fontsize=15)
    fig.text(0.01, 0.01, "Chronos-2 uses q50 point forecasts of the middle-128 raw waveform; forecast blocks are 8 minutes.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    plot_path = output_dir / f"{prefix}_chronos2_lhi.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    valid = comparison["forecast_lhi_rmse"].notna()
    metrics = {
        "forecast_lhi_rmse_mae_vs_sampled_observed": float(np.mean(np.abs(comparison.loc[valid, "forecast_lhi_rmse"] - comparison.loc[valid, "sampled_lhi_rmse"]))),
        "forecast_lhi_rmse_rmse_vs_sampled_observed": float(np.sqrt(np.mean((comparison.loc[valid, "forecast_lhi_rmse"] - comparison.loc[valid, "sampled_lhi_rmse"]) ** 2))),
        "forecast_lhi_mae_mae_vs_sampled_observed": float(np.mean(np.abs(comparison.loc[valid, "forecast_lhi_mae"] - comparison.loc[valid, "sampled_lhi_mae"]))),
        "forecast_lhi_mae_rmse_vs_sampled_observed": float(np.sqrt(np.mean((comparison.loc[valid, "forecast_lhi_mae"] - comparison.loc[valid, "sampled_lhi_mae"]) ** 2))),
        "lhi_plot": str(plot_path),
        "lhi_comparison": str(output_dir / f"{prefix}_chronos2_lhi_comparison.csv"),
    }
    return metrics


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the selected environment.")
    if args.prediction_length != 1024 or args.context_length != 8192:
        raise ValueError("This experiment is configured for Chronos-2 context=8192 and prediction=1024.")
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
        print(f"\n[{prefix}] loading center waveforms", flush=True)
        files, selected, full_rms = load_center_waveforms(bearing_dir, args.center_points)
        sampled_rms = np.sqrt(np.mean(selected ** 2, axis=1))
        inputs, origins = make_forecast_inputs(
            selected,
            args.reference_minutes,
            args.context_length,
            args.prediction_length,
            args.forecast_block_minutes,
        )
        forecast_rms, raw_blocks = forecast_blocks(
            pipeline,
            inputs,
            origins,
            len(files),
            args.center_points,
            args.prediction_length,
            args.batch_size,
        )
        forecast_rms.to_csv(output_dir / f"{prefix}_chronos2_forecast_minute_rms_q50.csv", index=False)
        np.savez_compressed(
            output_dir / f"{prefix}_chronos2_forecast_raw_q50.npz",
            forecast_origins=np.asarray(origins),
            forecasts=raw_blocks,
        )
        metrics = save_lhi_comparison(
            output_dir,
            prefix,
            full_rms,
            sampled_rms,
            forecast_rms,
            args.reference_minutes,
        )
        metadata = {
            "input_dir": str(bearing_dir),
            "condition": condition,
            "bearing": bearing,
            "file_count": len(files),
            "center_points_per_minute": args.center_points,
            "context_length_points": args.context_length,
            "context_length_minutes": args.context_length // args.center_points,
            "prediction_length_points": args.prediction_length,
            "prediction_length_minutes": args.prediction_length // args.center_points,
            "reference_minutes": args.reference_minutes,
            "forecast_block_minutes": args.forecast_block_minutes,
            "forecast_origins": origins,
            "forecast_point_count": int(len(forecast_rms)),
            "context_policy": "all observed center-waveform history before each origin, truncated to the latest 8192 points",
            "forecast_policy": "non-overlapping 8-minute blocks, using q50 point forecasts",
            "lhi_policy": "forecast sensor RMS is compared with the sampled first-20-minute RMS reference",
            "model_id": args.model_id,
            "device": args.device,
            **metrics,
        }
        (output_dir / f"{prefix}_chronos2_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary_rows.append({
            "condition": condition,
            "bearing": bearing,
            "file_count": len(files),
            "forecast_origins": len(origins),
            "forecast_target_start": int(forecast_rms["target_minute"].min()),
            "forecast_target_end": int(forecast_rms["target_minute"].max()),
            **metrics,
            "output_dir": str(output_dir),
        })
        print(f"[{prefix}] forecast minutes={forecast_rms['target_minute'].min()}..{forecast_rms['target_minute'].max()}, LHI plot saved", flush=True)

    pd.DataFrame(summary_rows).to_csv(args.output_root / "all_bearings_chronos2_lhi_summary.csv", index=False)
    print(f"Saved summary to: {args.output_root / 'all_bearings_chronos2_lhi_summary.csv'}")


if __name__ == "__main__":
    main()

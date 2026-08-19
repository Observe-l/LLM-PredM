#!/usr/bin/env python3
"""Forecast absolute-valued middle-256 XJTU-SY sensor readings with Chronos-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronos2_xjtu_rms_forecast import load_pipeline


SENSORS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]
LABELS = ["Horizontal vibration | abs(sampled)", "Vertical vibration | abs(sampled)"]
COLORS = ["#2f855a", "#b45f06"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--center-points", type=int, default=256)
    parser.add_argument("--reference-files", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--prediction-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def numeric_csvs(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("*.csv"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def load_abs_middle_waveforms(directory: Path, center_points: int) -> tuple[list[Path], np.ndarray]:
    files = numeric_csvs(directory)
    if not files:
        raise FileNotFoundError(f"No numeric CSV files found in {directory}")
    selected_rows: list[np.ndarray] = []
    full_length: int | None = None
    for path in files:
        frame = pd.read_csv(path)
        missing = [sensor for sensor in SENSORS if sensor not in frame.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        values = frame[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite sensor values found")
        if full_length is None:
            full_length = len(values)
        if len(values) != full_length:
            raise ValueError(f"{path}: length {len(values)} differs from {full_length}")
        if center_points > len(values):
            raise ValueError(f"{path}: center-points exceeds file length")
        values = np.abs(values)
        start = (len(values) - center_points) // 2
        selected_rows.append(values[start : start + center_points])
    return files, np.stack(selected_rows, axis=0)


def make_forecast_inputs(
    selected: np.ndarray,
    reference_files: int,
    context_length: int,
    prediction_length: int,
    center_points: int,
) -> tuple[list[dict[str, np.ndarray]], list[int], int]:
    n_files, points_per_file, n_sensors = selected.shape
    if prediction_length % points_per_file != 0:
        raise ValueError("prediction-length must be an integer number of sampled CSV files")
    block_files = prediction_length // points_per_file
    if n_files <= reference_files:
        raise ValueError("The bearing has no measurements after the reference files.")
    inputs: list[dict[str, np.ndarray]] = []
    origins: list[int] = []
    for origin in range(reference_files, n_files, block_files):
        history = selected[:origin].reshape(origin * points_per_file, n_sensors).T
        history = history[:, -context_length:]
        inputs.append({"target": history.astype(np.float32)})
        origins.append(origin)
    return inputs, origins, block_files


def forecast_blocks(pipeline, inputs: list[dict[str, np.ndarray]], origins: list[int], prediction_length: int, batch_size: int) -> np.ndarray:
    raw_blocks: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_origins = origins[start : start + batch_size]
        print(f"Forecasting blocks {start + 1}-{start + len(batch_inputs)} / {len(inputs)}", flush=True)
        _, point_forecasts = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=prediction_length,
            quantile_levels=[0.5],
            batch_size=len(batch_inputs),
        )
        for origin, point_tensor in zip(batch_origins, point_forecasts):
            prediction = point_tensor.detach().float().cpu().numpy()
            if prediction.shape != (2, prediction_length):
                raise ValueError(f"Unexpected Chronos-2 forecast shape at origin {origin}: {prediction.shape}")
            raw_blocks.append(prediction)
    return np.stack(raw_blocks, axis=0)


def save_outputs(
    output_dir: Path,
    prefix: str,
    files: list[Path],
    selected: np.ndarray,
    origins: list[int],
    forecasts: np.ndarray,
    reference_files: int,
    center_points: int,
    context_length: int,
    prediction_length: int,
    block_files: int,
) -> dict[str, object]:
    n_files = len(files)
    total_points = n_files * center_points
    actual = selected.reshape(total_points, 2)
    time_minutes = np.arange(total_points, dtype=float) / center_points
    actual_frame = pd.DataFrame({
        "file_index": np.repeat(np.arange(1, n_files + 1), center_points),
        "file": np.repeat([path.name for path in files], center_points),
        "sample_in_file": np.tile(np.arange(center_points), n_files),
        "time_minute": time_minutes,
        "horizontal_abs": actual[:, 0],
        "vertical_abs": actual[:, 1],
    })
    actual_frame.to_csv(output_dir / f"{prefix}_abs_middle{center_points}_sensor_readings.csv", index=False)

    forecast_rows: list[dict[str, float | int]] = []
    for origin, block in zip(origins, forecasts):
        available = min(prediction_length, total_points - origin * center_points)
        for point_offset in range(available):
            global_point = origin * center_points + point_offset
            forecast_rows.append({
                "forecast_origin_file": origin,
                "forecast_origin_minute": origin,
                "horizon_sample": point_offset + 1,
                "target_file_index": global_point // center_points + 1,
                "sample_in_file": global_point % center_points,
                "time_minute": global_point / center_points,
                "predicted_horizontal_abs_q50": float(block[0, point_offset]),
                "predicted_vertical_abs_q50": float(block[1, point_offset]),
            })
    forecast_frame = pd.DataFrame(forecast_rows)
    forecast_frame.to_csv(output_dir / f"{prefix}_chronos2_abs_middle{center_points}_sensor_forecast_q50.csv", index=False)
    np.savez_compressed(
        output_dir / f"{prefix}_chronos2_abs_middle{center_points}_forecast_raw_q50.npz",
        forecast_origins=np.asarray(origins),
        forecasts=forecasts,
    )

    predicted = np.full((total_points, 2), np.nan, dtype=float)
    for origin, block in zip(origins, forecasts):
        start = origin * center_points
        stop = min(total_points, start + prediction_length)
        predicted[start:stop] = block[:, : stop - start].T

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, dpi=180)
    for sensor_index, (ax, label, color) in enumerate(zip(axes, LABELS, COLORS)):
        ax.axvspan(0, reference_files, color="#dbeafe", alpha=0.8, label=f"health reference: first {reference_files} files")
        ax.plot(time_minutes, actual[:, sensor_index], color="#111827", linewidth=0.28, alpha=0.60, label="ground truth: abs(middle-256)")
        ax.plot(time_minutes, predicted[:, sensor_index], color=color, linewidth=0.45, alpha=0.88, label="Chronos-2 q50 forecast")
        ax.set_ylabel("absolute amplitude")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time since first CSV (minutes; one CSV per minute)")
    axes[-1].set_xlim(0, n_files)
    tick_step = max(1, n_files // 10)
    axes[-1].set_xticks(np.arange(0, n_files + 1, tick_step))
    fig.suptitle(f"{prefix}: abs middle-256 sensor readings and Chronos-2 forecast", fontsize=15)
    fig.text(
        0.01,
        0.01,
        f"Input: abs(sensor), middle {center_points} points/file; context={context_length} points ({context_length / center_points:.0f} min); "
        f"prediction={prediction_length} points ({prediction_length / center_points:.0f} min); q50.",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    plot_path = output_dir / f"{prefix}_abs_middle{center_points}_chronos2_sensor_forecast_q50.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "plot": str(plot_path),
        "actual_points": int(total_points),
        "forecast_points": int(len(forecast_frame)),
        "forecast_origins": int(len(origins)),
        "forecast_start_file": int(reference_files + 1),
        "forecast_end_file": int(n_files),
        "block_files": int(block_files),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the selected environment.")
    if args.center_points != 256 or args.context_length != 8192 or args.prediction_length != 1024:
        raise ValueError("This experiment is configured for center-points=256, context=8192, prediction=1024.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    bearing_dirs = sorted(
        (path for condition in args.input_root.iterdir() if condition.is_dir() for path in condition.iterdir() if path.is_dir()),
        key=lambda path: (path.parent.name, path.name),
    )
    if not bearing_dirs:
        raise FileNotFoundError(f"No bearing directories found under {args.input_root}")

    pipeline = load_pipeline(args.model_id, args.device, args.torch_dtype, args.local_files_only)
    summary_rows: list[dict[str, object]] = []
    for bearing_dir in bearing_dirs:
        condition = bearing_dir.parent.name
        bearing = bearing_dir.name
        prefix = f"{condition}_{bearing}"
        output_dir = args.output_root / prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{prefix}] loading abs middle-256 waveforms", flush=True)
        files, selected = load_abs_middle_waveforms(bearing_dir, args.center_points)
        if len(files) <= args.reference_files:
            print(f"[{prefix}] skipped: only {len(files)} files", flush=True)
            continue
        inputs, origins, block_files = make_forecast_inputs(selected, args.reference_files, args.context_length, args.prediction_length, args.center_points)
        forecasts = forecast_blocks(pipeline, inputs, origins, args.prediction_length, args.batch_size)
        metrics = save_outputs(output_dir, prefix, files, selected, origins, forecasts, args.reference_files, args.center_points, args.context_length, args.prediction_length, block_files)
        metadata = {
            "input_dir": str(bearing_dir),
            "condition": condition,
            "bearing": bearing,
            "file_count": len(files),
            "samples_per_file": 32768,
            "center_points_per_file": args.center_points,
            "preprocessing": "absolute value applied to both sensor channels before middle-256 extraction",
            "reference_files": args.reference_files,
            "context_length_points": args.context_length,
            "context_length_minutes_at_1_file_per_minute": args.context_length / args.center_points,
            "prediction_length_points": args.prediction_length,
            "prediction_length_minutes_at_1_file_per_minute": args.prediction_length / args.center_points,
            "forecast_block_files": block_files,
            "forecast_block_minutes_at_1_file_per_minute": block_files,
            "forecast_origins": origins,
            "context_policy": "all observed abs middle-256 history before each origin, truncated to latest 8192 points",
            "forecast_policy": "non-overlapping blocks with q50 point forecasts",
            "model_id": args.model_id,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "time_conversion_note": "With one CSV per minute and 256 retained points per CSV, 8192 points=32 minutes and 1024 points=4 minutes.",
            **metrics,
        }
        (output_dir / f"{prefix}_chronos2_abs_middle{args.center_points}_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_rows.append({"condition": condition, "bearing": bearing, "file_count": len(files), **metrics, "output_dir": str(output_dir)})
        print(f"[{prefix}] files={len(files)}, origins={len(origins)}, forecast points={metrics['forecast_points']}", flush=True)

    pd.DataFrame(summary_rows).to_csv(args.output_root / "all_bearings_abs_middle256_chronos2_summary.csv", index=False)
    print(f"Saved summary to: {args.output_root / 'all_bearings_abs_middle256_chronos2_summary.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare forecast-then-evaluate and evaluate-then-forecast LHI on XJTU-SY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronos2_xjtu_rms_forecast import load_pipeline


RAW_SENSORS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]
SAMPLED_COLUMNS = ["observed_lhi_rmse", "observed_lhi_mae", "forecast_lhi_rmse_q50", "forecast_lhi_mae_q50"]


def numeric_csvs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.csv"), key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)


def load_abs_sensor_values(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    missing = [name for name in RAW_SENSORS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing sensor columns {missing}")
    values = frame[RAW_SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite sensor values")
    return np.abs(values)


def compute_reference_stats(
    raw_files: list[Path],
    reference_files: int,
    epsilon: float,
    range_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    reference_values = np.concatenate([load_abs_sensor_values(path) for path in raw_files[:reference_files]], axis=0)
    mean = reference_values.mean(axis=0)
    ranges = reference_values.max(axis=0) - reference_values.min(axis=0)
    usable = np.isfinite(ranges) & (ranges > range_epsilon)
    if not np.any(usable):
        raise ValueError("No usable raw-sensor min-max range in the health reference")
    drift = np.full_like(reference_values, np.nan, dtype=float)
    drift[:, usable] = np.abs((reference_values[:, usable] - mean[usable]) / ranges[usable])
    d_mae = np.nanmean(drift, axis=1)
    d_rmse = np.sqrt(np.nanmean(drift**2, axis=1))
    baseline_mae = float(np.nanmean(d_mae))
    baseline_rmse = float(np.sqrt(np.nanmean(d_rmse**2)))
    return mean, ranges, baseline_mae, baseline_rmse


def compute_raw_minute_lhi(
    raw_files: list[Path],
    reference_files: int,
    epsilon: float = 1e-6,
    range_epsilon: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, object]]:
    mean, ranges, baseline_mae, baseline_rmse = compute_reference_stats(
        raw_files, reference_files, epsilon, range_epsilon
    )
    usable = np.isfinite(ranges) & (ranges > range_epsilon)
    rows: list[dict[str, float | int | str]] = []
    for file_index, path in enumerate(raw_files, start=1):
        values = load_abs_sensor_values(path)
        drift = np.full_like(values, np.nan, dtype=float)
        drift[:, usable] = np.abs((values[:, usable] - mean[usable]) / ranges[usable])
        d_mae = np.nanmean(drift, axis=1)
        d_rmse = np.sqrt(np.nanmean(drift**2, axis=1))
        lhi_mae = np.log((d_mae + epsilon) / (baseline_mae + epsilon))
        lhi_rmse = np.log((d_rmse + epsilon) / (baseline_rmse + epsilon))
        rows.append({
            "file_index": file_index,
            "time_minute": float(file_index - 1),
            "file": path.name,
            "raw_sample_count": int(len(values)),
            "raw_lhi_mae": float(np.mean(lhi_mae)),
            "raw_lhi_rmse": float(np.mean(lhi_rmse)),
            "raw_d_mae": float(np.mean(d_mae)),
            "raw_d_rmse": float(np.mean(d_rmse)),
        })
    frame = pd.DataFrame(rows)
    metadata = {
        "reference_files": reference_files,
        "reference_raw_points": int(reference_files * rows[0]["raw_sample_count"]),
        "sensor_names": RAW_SENSORS,
        "preprocessing": "absolute value applied to all raw sensor readings before LHI",
        "normalization": "per-sensor min-max range computed from all raw readings in the first reference files",
        "lhi_definition": "per-raw-sample absolute normalized sensor drift, D_MAE/D_RMSE aggregation, then log ratio to health baseline; plotted value is the mean LHI within each CSV/minute",
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
    }
    return frame, metadata


def load_sampled_forecast_lhi(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"file_index", *SAMPLED_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    grouped = frame.groupby("file_index", sort=True)[SAMPLED_COLUMNS].mean().reset_index()
    grouped = grouped.rename(columns={
        "observed_lhi_rmse": "sampled_observed_lhi_rmse",
        "observed_lhi_mae": "sampled_observed_lhi_mae",
        "forecast_lhi_rmse_q50": "sampled_forecast_lhi_rmse_q50",
        "forecast_lhi_mae_q50": "sampled_forecast_lhi_mae_q50",
    })
    return grouped


def make_lhi_forecast_inputs(series: np.ndarray, reference_files: int, forecast_window: int, context_mode: str, context_minutes: int) -> tuple[list[dict[str, np.ndarray]], list[int]]:
    inputs: list[dict[str, np.ndarray]] = []
    origins: list[int] = []
    for origin in range(reference_files, len(series), forecast_window):
        history = series[:origin]
        if context_mode == "context32":
            history = history[-context_minutes:]
        target = history.astype(np.float32)[None, :] if history.ndim == 1 else history.astype(np.float32).T
        inputs.append({"target": target})
        origins.append(origin)
    return inputs, origins


def forecast_lhi_series(
    pipeline,
    series: np.ndarray,
    reference_files: int,
    forecast_window: int,
    context_mode: str,
    context_minutes: int,
    batch_size: int,
) -> pd.DataFrame:
    inputs, origins = make_lhi_forecast_inputs(series, reference_files, forecast_window, context_mode, context_minutes)
    rows: list[dict[str, float | int | str]] = []
    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_origins = origins[start : start + batch_size]
        print(f"  {context_mode}: forecasting windows {start + 1}-{start + len(batch_inputs)} / {len(inputs)}", flush=True)
        quantiles, _ = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=forecast_window,
            quantile_levels=[0.5],
            batch_size=len(batch_inputs),
        )
        for origin, quantile_tensor in zip(batch_origins, quantiles):
            values = quantile_tensor.detach().float().cpu().numpy()
            values = values[:, :, 0] if values.ndim == 3 else values[None, :]
            for horizon in range(1, min(forecast_window, len(series) - origin) + 1):
                target = origin + horizon
                rows.append({
                    "context_mode": context_mode,
                    "forecast_origin_minute": origin,
                    "forecast_horizon_minute": horizon,
                    "target_minute": target,
                    "forecast_lhi_rmse_q50": float(values[0, horizon - 1]),
                    "forecast_lhi_mae_q50": float(values[1, horizon - 1]) if values.shape[0] > 1 else float(values[0, horizon - 1]),
                })
    return pd.DataFrame(rows).sort_values("target_minute")


def plot_comparison(comparison: pd.DataFrame, output: Path, prefix: str, reference_files: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, dpi=180)
    series = {
        "rmse": [
            ("raw_lhi_rmse", "raw all-sample LHI", "#111827", "-", 1.35),
            ("sampled_forecast_lhi_rmse_q50", "sampled sensor forecast → LHI", "#b45f06", "--", 1.15),
            ("lhi_forecast_context32_rmse_q50", "LHI forecast; past 32 min", "#15803d", "-.", 1.25),
            ("lhi_forecast_all_history_rmse_q50", "LHI forecast; all history", "#7a3e9d", ":", 1.35),
        ],
        "mae": [
            ("raw_lhi_mae", "raw all-sample LHI", "#111827", "-", 1.35),
            ("sampled_forecast_lhi_mae_q50", "sampled sensor forecast → LHI", "#b45f06", "--", 1.15),
            ("lhi_forecast_context32_mae_q50", "LHI forecast; past 32 min", "#15803d", "-.", 1.25),
            ("lhi_forecast_all_history_mae_q50", "LHI forecast; all history", "#7a3e9d", ":", 1.35),
        ],
    }
    x = comparison["file_index"].to_numpy(dtype=int)
    for ax, metric, label in [(axes[0], "rmse", "LHI_RMSE"), (axes[1], "mae", "LHI_MAE")]:
        ax.axvspan(1, reference_files, color="#dbeafe", alpha=0.8, label=f"health reference: first {reference_files} files")
        for column, legend, color, style, width in series[metric]:
            ax.plot(x, comparison[column], color=color, linestyle=style, linewidth=width, label=legend)
        ax.axhline(0.0, color="#6b7280", linewidth=0.9)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("measurement index; one CSV per minute")
    axes[-1].set_xlim(1, int(x.max()))
    fig.suptitle(f"{prefix}: LHI forecast/evaluation order comparison", fontsize=15)
    fig.text(0.01, 0.01, "Raw LHI uses all 32768 readings per CSV; forecast curves are q50 and predict 8 minutes per block.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/XJTU-SY_Bearing_Datasets"))
    parser.add_argument("--sensor-forecast-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--reference-files", type=int, default=20)
    parser.add_argument("--forecast-window", type=int, default=8)
    parser.add_argument("--context-minutes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    bearing_dirs = sorted(
        (path for condition in args.raw_root.iterdir() if condition.is_dir() for path in condition.iterdir() if path.is_dir()),
        key=lambda path: (path.parent.name, path.name),
    )
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
    pipeline = load_pipeline(args.model_id, args.device, args.torch_dtype, args.local_files_only)
    summary_rows: list[dict[str, object]] = []

    for bearing_dir in bearing_dirs:
        condition = bearing_dir.parent.name
        bearing = bearing_dir.name
        prefix = f"{condition}_{bearing}"
        output_dir = args.output_root / prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        sensor_dir = args.sensor_forecast_root / prefix
        raw_files = numeric_csvs(bearing_dir)
        sampled_lhi_path = sensor_dir / f"{prefix}_abs_middle256_sample_lhi_comparison.csv"
        if len(raw_files) <= args.reference_files or not sampled_lhi_path.exists():
            print(f"[{prefix}] skipped: insufficient files or missing sampled LHI output", flush=True)
            continue
        print(f"\n[{prefix}] computing LHI from all raw readings", flush=True)
        raw_lhi, raw_metadata = compute_raw_minute_lhi(raw_files, args.reference_files)
        raw_lhi.to_csv(output_dir / f"{prefix}_raw_all_samples_lhi_by_minute.csv", index=False)
        sampled = load_sampled_forecast_lhi(sampled_lhi_path)
        base = raw_lhi.merge(sampled, on="file_index", how="left")
        raw_series = raw_lhi[["raw_lhi_rmse", "raw_lhi_mae"]].to_numpy(dtype=np.float32)

        forecast_context32 = forecast_lhi_series(
            pipeline, raw_series, args.reference_files, args.forecast_window,
            "context32", args.context_minutes, args.batch_size,
        )
        forecast_all = forecast_lhi_series(
            pipeline, raw_series, args.reference_files, args.forecast_window,
            "all_history", args.context_minutes, args.batch_size,
        )
        forecast_context32.to_csv(output_dir / f"{prefix}_chronos2_raw_lhi_forecast_context32_q50.csv", index=False)
        forecast_all.to_csv(output_dir / f"{prefix}_chronos2_raw_lhi_forecast_all_history_q50.csv", index=False)

        for forecast, suffix in [(forecast_context32, "context32"), (forecast_all, "all_history")]:
            forecast = forecast.rename(columns={
                "forecast_lhi_rmse_q50": f"lhi_forecast_{suffix}_rmse_q50",
                "forecast_lhi_mae_q50": f"lhi_forecast_{suffix}_mae_q50",
            })
            base = base.merge(
                forecast[["target_minute", f"lhi_forecast_{suffix}_rmse_q50", f"lhi_forecast_{suffix}_mae_q50"]],
                left_on="file_index",
                right_on="target_minute",
                how="left",
            )
            base = base.drop(columns=["target_minute"])
        comparison = base.rename(columns={
            "sampled_forecast_lhi_rmse_q50": "sampled_forecast_lhi_rmse_q50",
            "sampled_forecast_lhi_mae_q50": "sampled_forecast_lhi_mae_q50",
        })
        comparison["file_index"] = comparison["file_index"].astype(int)
        comparison.to_csv(output_dir / f"{prefix}_lhi_order_comparison.csv", index=False)
        plot_path = output_dir / f"{prefix}_lhi_order_comparison.png"
        plot_comparison(comparison, plot_path, prefix, args.reference_files)

        valid_context = comparison["lhi_forecast_context32_rmse_q50"].notna()
        valid_all = comparison["lhi_forecast_all_history_rmse_q50"].notna()
        summary_rows.append({
            "condition": condition,
            "bearing": bearing,
            "measurement_count": len(raw_lhi),
            "reference_files": args.reference_files,
            "raw_samples_per_file": int(raw_lhi["raw_sample_count"].iloc[0]),
            "context32_forecast_points": int(valid_context.sum()),
            "all_history_forecast_points": int(valid_all.sum()),
            "context32_lhi_rmse_mae_vs_raw": float(np.mean(np.abs(comparison.loc[valid_context, "lhi_forecast_context32_rmse_q50"] - comparison.loc[valid_context, "raw_lhi_rmse"]))),
            "all_history_lhi_rmse_mae_vs_raw": float(np.mean(np.abs(comparison.loc[valid_all, "lhi_forecast_all_history_rmse_q50"] - comparison.loc[valid_all, "raw_lhi_rmse"]))),
            "plot": str(plot_path),
            "comparison": str(output_dir / f"{prefix}_lhi_order_comparison.csv"),
        })
        metadata = {
            "prefix": prefix,
            "reference_files": args.reference_files,
            "forecast_window_minutes": args.forecast_window,
            "context_minutes": args.context_minutes,
            "raw_lhi": raw_metadata,
            "series_3": "Chronos-2 q50 forecast of raw all-sample per-minute LHI using the previous 32 LHI values",
            "series_4": "Chronos-2 q50 forecast of raw all-sample per-minute LHI using all available historical LHI values",
            "series_2": "Chronos-2 sensor q50 forecast from abs(middle-256) readings, then direct sample-level LHI aggregated by minute",
        }
        (output_dir / f"{prefix}_lhi_order_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = pd.DataFrame(summary_rows).sort_values(["condition", "bearing"])
    summary.to_csv(args.output_root / "all_bearings_lhi_order_summary.csv", index=False)
    (args.output_root / "all_bearings_lhi_order_metadata.json").write_text(json.dumps({
        "reference_files": args.reference_files,
        "forecast_window_minutes": args.forecast_window,
        "context_minutes": args.context_minutes,
        "raw_lhi_uses_rms": False,
        "raw_lhi_aggregation_for_plot": "mean of sample-level LHI over all raw readings within each CSV/minute",
        "comparison_series": [
            "raw all-sample LHI",
            "sampled sensor forecast then LHI",
            "raw LHI forecast with past 32 minutes",
            "raw LHI forecast with all history",
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved LHI order comparison outputs to: {args.output_root}")


if __name__ == "__main__":
    main()

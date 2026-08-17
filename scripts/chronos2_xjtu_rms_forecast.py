#!/usr/bin/env python3
"""Run rolling zero-shot Chronos-2 forecasts on XJTU-SY bearing RMS values."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SENSORS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--forecast-window", type=int, default=20)
    parser.add_argument("--minimum-history", type=int, default=20)
    parser.add_argument(
        "--cutoffs",
        default="20,40,60,80,100",
        help="Comma-separated forecast cutoffs. Each cutoff uses all RMS history before it.",
    )
    parser.add_argument(
        "--plot-quantile",
        choices=["q10", "q50", "q90"],
        default="q50",
        help="Quantile to plot and score.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_rms_series(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"), key=lambda path: int(path.stem))
    if not files:
        raise FileNotFoundError(f"No numeric CSV files found in {input_dir}")
    rows = []
    for measurement, path in enumerate(files, start=1):
        frame = pd.read_csv(path)
        missing = [column for column in SENSORS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        values = frame[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite values found")
        rows.append({
            "measurement": measurement,
            "file": path.name,
            "horizontal_rms": float(np.sqrt(np.mean(values[:, 0] ** 2))),
            "vertical_rms": float(np.sqrt(np.mean(values[:, 1] ** 2))),
            # One RMS per CSV: RMS over both vibration channels and all samples.
            "rms": float(np.sqrt(np.mean(values ** 2))),
            "samples_per_file": int(len(values)),
        })
    return pd.DataFrame(rows)


def load_pipeline(model_id: str, device: str, torch_dtype: str, local_files_only: bool):
    from chronos import Chronos2Pipeline

    kwargs = {"dtype": dtype_from_name(torch_dtype), "device_map": device}
    if local_files_only:
        kwargs["local_files_only"] = True
    print(f"Loading Chronos-2 model {model_id!r} on {device}...", flush=True)
    return Chronos2Pipeline.from_pretrained(model_id, **kwargs)


def parse_cutoffs(value: str) -> list[int]:
    cutoffs = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cutoffs:
        raise ValueError("At least one cutoff is required.")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError(f"Cutoffs must be unique: {cutoffs}")
    return cutoffs


def make_forecast_inputs(
    series: np.ndarray,
    minimum_history: int,
    forecast_window: int,
    requested_cutoffs: list[int],
):
    # A cutoff t predicts t+1 ... t+H, using all observations before t.
    last_complete_cutoff = len(series) - forecast_window
    cutoffs = sorted(requested_cutoffs)
    invalid = [
        cutoff for cutoff in cutoffs
        if cutoff < minimum_history or cutoff > last_complete_cutoff
    ]
    if invalid:
        raise ValueError(
            f"Invalid cutoff(s) {invalid}; valid range is "
            f"{minimum_history}..{last_complete_cutoff} for a complete forecast window."
        )
    inputs = []
    metadata = []
    for cutoff in cutoffs:
        inputs.append({"target": series[:cutoff].astype(np.float32)[None, :]})
        metadata.append(cutoff)
    return inputs, metadata


def run_forecasts(
    pipeline,
    series: np.ndarray,
    minimum_history: int,
    forecast_window: int,
    requested_cutoffs: list[int],
    batch_size: int,
) -> pd.DataFrame:
    inputs, cutoffs = make_forecast_inputs(
        series, minimum_history, forecast_window, requested_cutoffs
    )
    rows = []
    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_cutoffs = cutoffs[start : start + batch_size]
        print(f"Forecasting windows {start + 1}-{start + len(batch_inputs)} / {len(inputs)}", flush=True)
        quantiles, point_forecasts = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=forecast_window,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=len(batch_inputs),
        )
        for cutoff, quantile_tensor, point_tensor in zip(batch_cutoffs, quantiles, point_forecasts):
            q = quantile_tensor.detach().float().cpu().numpy()
            pred = point_tensor.detach().float().cpu().numpy()
            # Chronos-2 output is (n_variates=1, horizon, quantiles).
            q = q[0]
            pred = pred[0]
            for horizon_index in range(forecast_window):
                target_measurement = cutoff + horizon_index + 1
                rows.append({
                    "forecast_cutoff": cutoff,
                    "forecast_origin_measurement": cutoff,
                    "horizon": horizon_index + 1,
                    "target_measurement": target_measurement,
                    "y_true": float(series[target_measurement - 1]),
                    "y_pred": float(pred[horizon_index]),
                    "q10": float(q[horizon_index, 0]),
                    "q50": float(q[horizon_index, 1]),
                    "q90": float(q[horizon_index, 2]),
                })
    return pd.DataFrame(rows)


def plot_forecasts(
    series_frame: pd.DataFrame,
    forecasts: pd.DataFrame,
    output: Path,
    minimum_history: int,
    forecast_window: int,
    plot_quantile: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7), dpi=180)
    x = series_frame["measurement"].to_numpy()
    ax.plot(x, series_frame["rms"], color="#111827", linewidth=1.8, label="ground truth RMS")

    # Plot one selected quantile and concatenate all selected forecast windows.
    forecast_line = forecasts.sort_values(["forecast_cutoff", "horizon"])
    ax.plot(
        forecast_line["target_measurement"],
        forecast_line[plot_quantile],
        color="#d97706",
        linewidth=2.0,
        label="Chronos-2",
    )
    ax.set_title(f"XJTU-SY Bearing RMS: Chronos-2 {plot_quantile} forecast")
    ax.set_xlabel("measurement time (minutes; one CSV per minute)")
    ax.set_ylabel("combined RMS amplitude")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    cutoffs = np.sort(forecasts["forecast_cutoff"].unique())
    cutoff_step = int(np.median(np.diff(cutoffs))) if len(cutoffs) > 1 else "single"
    fig.text(
        0.01,
        0.01,
        f"Forecast window={forecast_window}; {plot_quantile} only; cutoff step={cutoff_step} minutes; "
        "each segment uses all RMS history before its cutoff.",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the selected environment.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series = load_rms_series(args.input_dir)
    series.to_csv(args.output_dir / "rms_series.csv", index=False)
    requested_cutoffs = parse_cutoffs(args.cutoffs)
    pipeline = load_pipeline(args.model_id, args.device, args.torch_dtype, args.local_files_only)
    forecasts = run_forecasts(
        pipeline,
        series["rms"].to_numpy(dtype=np.float32),
        args.minimum_history,
        args.forecast_window,
        requested_cutoffs,
        args.batch_size,
    )
    forecasts.to_csv(args.output_dir / "chronos2_rolling_forecasts.csv", index=False)
    forecasts[
        ["forecast_cutoff", "forecast_origin_measurement", "horizon", "target_measurement", "y_true", args.plot_quantile]
    ].to_csv(args.output_dir / f"chronos2_{args.plot_quantile}_forecasts.csv", index=False)
    plot_forecasts(
        series,
        forecasts,
        args.output_dir / "chronos2_vs_ground_truth_rms.png",
        args.minimum_history,
        args.forecast_window,
        args.plot_quantile,
    )

    scored = forecasts.copy()
    scored["abs_error"] = np.abs(scored[args.plot_quantile] - scored["y_true"])
    scored["squared_error"] = (scored[args.plot_quantile] - scored["y_true"]) ** 2
    metrics = scored.groupby("horizon", sort=True).agg(
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
        n=("abs_error", "size"),
    ).reset_index()
    metrics.to_csv(args.output_dir / "chronos2_metrics_by_horizon.csv", index=False)
    metadata = {
        "input_dir": str(args.input_dir),
        "model_id": args.model_id,
        "device": args.device,
        "file_count": int(len(series)),
        "samples_per_file": int(series["samples_per_file"].iloc[0]),
        "rms_definition": "RMS over both horizontal and vertical vibration channels and all 32768 samples in each CSV",
        "forecast_window": int(args.forecast_window),
        "forecast_cutoffs": requested_cutoffs,
        "minimum_history": int(args.minimum_history),
        "forecast_origins": int(forecasts["forecast_cutoff"].nunique()),
        "forecast_rows": int(len(forecasts)),
        "context_policy": "all RMS observations before each cutoff",
        "reported_forecast": args.plot_quantile,
        "complete_windows_only": True,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved Chronos-2 forecasts to: {args.output_dir}")
    print(metrics.head().to_string(index=False))


if __name__ == "__main__":
    main()

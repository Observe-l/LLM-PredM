#!/usr/bin/env python3
"""Run Chronos-2 RMS forecasting separately for every XJTU-SY bearing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chronos2_xjtu_rms_forecast import (
    load_pipeline,
    load_rms_series,
    plot_forecasts,
    run_forecasts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--forecast-window", type=int, default=20)
    parser.add_argument("--minimum-history", type=int, default=20)
    parser.add_argument("--cutoff-step", type=int, default=20)
    parser.add_argument("--plot-quantile", choices=["q10", "q50", "q90"], default="q50")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cutoff_step <= 0:
        raise ValueError("cutoff-step must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    bearing_dirs = sorted(
        (path for condition in args.input_root.iterdir() if condition.is_dir()
         for path in condition.iterdir() if path.is_dir()),
        key=lambda path: (path.parent.name, path.name),
    )
    if not bearing_dirs:
        raise FileNotFoundError(f"No bearing directories found under {args.input_root}")

    pipeline = load_pipeline(args.model_id, args.device, args.torch_dtype, args.local_files_only)
    summary_rows = []

    for bearing_dir in bearing_dirs:
        relative_name = f"{bearing_dir.parent.name}_{bearing_dir.name}"
        output_dir = args.output_root / relative_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{relative_name}] loading RMS series", flush=True)

        series = load_rms_series(bearing_dir)
        series.to_csv(output_dir / "rms_series.csv", index=False)
        last_complete_cutoff = len(series) - args.forecast_window
        cutoffs = list(range(args.minimum_history, last_complete_cutoff + 1, args.cutoff_step))
        if not cutoffs:
            print(f"[{relative_name}] skipped: not enough data for one complete forecast window", flush=True)
            continue

        forecasts = run_forecasts(
            pipeline,
            series["rms"].to_numpy(dtype=np.float32),
            args.minimum_history,
            args.forecast_window,
            cutoffs,
            args.batch_size,
        )
        forecasts.to_csv(output_dir / "chronos2_rolling_forecasts.csv", index=False)
        forecasts[
            ["forecast_cutoff", "forecast_origin_measurement", "horizon", "target_measurement", "y_true", args.plot_quantile]
        ].to_csv(output_dir / f"chronos2_{args.plot_quantile}_forecasts.csv", index=False)
        plot_forecasts(
            series,
            forecasts,
            output_dir / "chronos2_vs_ground_truth_rms.png",
            args.minimum_history,
            args.forecast_window,
            args.plot_quantile,
        )

        scored = forecasts.copy()
        scored["abs_error"] = np.abs(scored[args.plot_quantile] - scored["y_true"])
        scored["squared_error"] = (scored[args.plot_quantile] - scored["y_true"]) ** 2
        metrics = scored.groupby("horizon", sort=True).agg(
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            n=("abs_error", "size"),
        ).reset_index()
        metrics.to_csv(output_dir / "chronos2_metrics_by_horizon.csv", index=False)

        metadata = {
            "input_dir": str(bearing_dir),
            "model_id": args.model_id,
            "device": args.device,
            "file_count": int(len(series)),
            "samples_per_file": int(series["samples_per_file"].iloc[0]),
            "rms_definition": "RMS over both horizontal and vertical vibration channels and all samples in each CSV",
            "forecast_window": int(args.forecast_window),
            "minimum_history": int(args.minimum_history),
            "cutoff_step": int(args.cutoff_step),
            "forecast_cutoffs": cutoffs,
            "forecast_origins": int(forecasts["forecast_cutoff"].nunique()),
            "forecast_rows": int(len(forecasts)),
            "context_policy": "all RMS observations before each cutoff",
            "reported_forecast": args.plot_quantile,
            "complete_windows_only": True,
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        overall_mae = float(scored["abs_error"].mean())
        overall_rmse = float(np.sqrt(scored["squared_error"].mean()))
        summary_rows.append({
            "condition": bearing_dir.parent.name,
            "bearing": bearing_dir.name,
            "file_count": len(series),
            "forecast_origins": len(cutoffs),
            "forecast_rows": len(forecasts),
            "first_cutoff": cutoffs[0],
            "last_cutoff": cutoffs[-1],
            "forecast_start_measurement": int(forecasts["target_measurement"].min()),
            "forecast_end_measurement": int(forecasts["target_measurement"].max()),
            f"{args.plot_quantile}_mae": overall_mae,
            f"{args.plot_quantile}_rmse": overall_rmse,
            "output_dir": str(output_dir),
        })
        print(
            f"[{relative_name}] files={len(series)}, cutoffs={cutoffs[0]}..{cutoffs[-1]} "
            f"({len(cutoffs)} origins), {args.plot_quantile} MAE={overall_mae:.6f}, RMSE={overall_rmse:.6f}",
            flush=True,
        )

    pd.DataFrame(summary_rows).to_csv(args.output_root / "all_bearings_summary.csv", index=False)
    print(f"\nSaved all-bearing summary to: {args.output_root / 'all_bearings_summary.csv'}")


if __name__ == "__main__":
    main()

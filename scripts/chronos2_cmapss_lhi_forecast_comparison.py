#!/usr/bin/env python3
"""Compare direct LHI forecasting with the earlier sensor-forecasting path.

The script uses the same C-MAPSS sensor set and log-ratio LHI implementation as
the existing zero-shot experiments.  It produces, for the first N engines of
each FD:

* raw sensor -> LHI (with and without operating-condition matching);
* existing Chronos-2 sensor forecast -> LHI results;
* new Chronos-2 LHI -> LHI forecast results.

The new LHI forecast passes the complete history available at each origin to
Chronos-2 and explicitly sets ``context_length=20``; Chronos therefore uses
the last 20 LHI observations while the input object still contains all past
observations. Forecast origins are non-overlapping 20-cycle blocks so the
forecast curve is easy to read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.zero_shot_cmapss.lhi_indicator import (  # noqa: E402
    add_lhi_columns,
    build_condition_means,
    compute_lhi_scores,
    compute_past_sensor_ranges,
    load_eval_frames,
)
from src.zero_shot_cmapss.raw_lhi_indicator import (  # noqa: E402
    build_raw_observation_rows,
    compute_healthy_target_cycle_baselines,
)
from src.zero_shot_cmapss.plot_operating_condition_clusters import (  # noqa: E402
    DEFAULT_SENSORS,
    FD_NAMES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("dataset/CMAPSSData"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/CMAPSS/lhi_then_forecast_first10_context20_h20"))
    p.add_argument("--model-id", default="amazon/chronos-2")
    p.add_argument("--fds", nargs="+", default=list(FD_NAMES), choices=list(FD_NAMES))
    p.add_argument("--first-n-units", type=int, default=10)
    p.add_argument("--healthy-cycles", type=int, default=50)
    p.add_argument("--context-length", type=int, default=20)
    p.add_argument("--prediction-length", type=int, default=20)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--local-files-only", action="store_true", default=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lhi-epsilon", type=float, default=1e-6)
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_pipeline(args: argparse.Namespace):
    from chronos import Chronos2Pipeline

    kwargs = {"dtype": dtype_from_name(args.torch_dtype), "device_map": args.device}
    if args.local_files_only:
        kwargs["local_files_only"] = True
    print(f"Loading Chronos-2 {args.model_id!r} on {args.device}...", flush=True)
    return Chronos2Pipeline.from_pretrained(args.model_id, **kwargs)


def pooled_frames(data_dir: Path, fds: Sequence[str]) -> dict[str, pd.DataFrame]:
    frames = load_eval_frames(data_dir, "train", fds)
    for frame in frames.values():
        frame["op_condition_key"] = "pooled"
    return frames


def restrict_to_first_units(frames: dict[str, pd.DataFrame], n: int) -> dict[str, pd.DataFrame]:
    """Keep only the requested engines before the relatively expensive LHI pass."""
    result = {}
    for fd, frame in frames.items():
        units = sorted(frame["unit_id"].unique())[:n]
        result[fd] = frame[frame["unit_id"].isin(units)].copy()
    return result


def calculate_raw_lhi(
    frames: dict[str, pd.DataFrame],
    sensors: Sequence[str],
    healthy_cycles: int,
    prediction_length: int,
    stride: int,
    lhi_epsilon: float,
) -> pd.DataFrame:
    """Reproduce raw_lhi_indicator.py with either real or pooled OC keys."""
    means = build_condition_means(frames, sensors, healthy_cycles)
    observations = build_raw_observation_rows(
        frames,
        sensors,
        prediction_length=prediction_length,
        stride=stride,
        min_context_cycles=0,
        forecast_start_cycle=20,
    )
    ranges = compute_past_sensor_ranges(
        frames,
        observations,
        sensors,
        range_epsilon=1e-6,
        minmax_scope="past_and_forecast",
    )
    all_scores, _ = compute_lhi_scores(observations, ranges, means)
    _, baselines = compute_healthy_target_cycle_baselines(
        frames, means, sensors, healthy_cycles, range_epsilon=1e-6
    )
    scores = add_lhi_columns(all_scores, baselines, lhi_epsilon)

    # Add cycles 1..19 so the first Chronos origin has a complete 19-point
    # history. These are the same health-reference quantities used to build B.
    health = compute_health_point_lhi(frames, means, sensors, healthy_cycles, lhi_epsilon)
    health = health[health["cycle"] < 20]
    cols = ["covariate_mode", "fd", "unit_id", "cycle", "lhi_rmse", "d_rmse", "b_rmse"]
    result = pd.concat([health[cols], scores[cols]], ignore_index=True)
    return result.sort_values(["fd", "unit_id", "cycle"]).reset_index(drop=True)


def compute_health_point_lhi(
    frames: dict[str, pd.DataFrame],
    condition_means: pd.DataFrame,
    sensors: Sequence[str],
    healthy_cycles: int,
    lhi_epsilon: float,
) -> pd.DataFrame:
    rows = []
    for fd, frame in frames.items():
        healthy = frame[frame["cycle"] <= healthy_cycles].copy()
        for unit_id, unit in healthy.groupby("unit_id", sort=True):
            values = unit.loc[:, sensors]
            minimum = values.min()
            value_range = values.max() - minimum
            long = unit.melt(
                id_vars=["unit_id", "cycle", "op_condition_key"],
                value_vars=list(sensors),
                var_name="sensor",
                value_name="y_pred",
            )
            long["fd"] = fd
            keyed = long.merge(
                condition_means,
                on=["fd", "unit_id", "op_condition_key", "sensor"],
                how="left",
                validate="many_to_one",
            )
            keyed["past_min"] = keyed["sensor"].map(minimum.to_dict())
            keyed["past_range"] = keyed["sensor"].map(value_range.to_dict())
            keyed = keyed[
                keyed["healthy_condition_mean_raw"].notna()
                & (keyed["past_range"] > 1e-6)
            ].copy()
            if keyed.empty:
                continue
            keyed["drift"] = (
                (keyed["y_pred"] - keyed["healthy_condition_mean_raw"]).abs()
                / keyed["past_range"]
            )
            point = keyed.groupby(["fd", "unit_id", "cycle"], sort=True).agg(
                d_rmse=("drift", lambda s: float(np.sqrt(np.mean(np.square(s)))))
            ).reset_index()
            point["covariate_mode"] = "raw_observed"
            point["b_rmse"] = point.groupby(["fd", "unit_id"])["d_rmse"].transform("mean")
            point["lhi_rmse"] = np.log(
                (point["d_rmse"] + lhi_epsilon) / (point["b_rmse"] + lhi_epsilon)
            )
            rows.append(point)
    if not rows:
        raise ValueError("No health point LHI values were computed.")
    return pd.concat(rows, ignore_index=True)


def select_first_units(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    units = (
        frame[["fd", "unit_id"]]
        .drop_duplicates()
        .sort_values(["fd", "unit_id"])
        .groupby("fd", sort=False)
        .head(n)
    )
    return frame.merge(units, on=["fd", "unit_id"], how="inner")


def load_existing_sensor_lhi(
    fds: Sequence[str],
    first_n_units: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the previously generated sensor->LHI curves, selecting 20-cycle blocks."""
    with_parts = []
    for fd in fds:
        if fd in {"FD002", "FD004"}:
            path = Path("outputs/CMAPSS/history_condition_h20_fd002_fd004/lhi/lhi_scores.csv")
        else:
            path = Path("outputs/CMAPSS/cluster_20/lhi_fix/lhi_scores.csv")
        df = pd.read_csv(path, usecols=["fd", "unit_id", "forecast_start_cycle", "cycle", "lhi_rmse"])
        df = df[(df["fd"] == fd) & (df["forecast_start_cycle"] >= 20)]
        df = df[(df["forecast_start_cycle"] - 20) % 20 == 0]
        with_parts.append(df)
    with_oc = select_first_units(pd.concat(with_parts, ignore_index=True), first_n_units)

    path = Path("outputs/CMAPSS/no_cov_20/lhi_fix/lhi_scores.csv")
    no_oc = pd.read_csv(path, usecols=["fd", "unit_id", "forecast_start_cycle", "cycle", "lhi_rmse"])
    no_oc = no_oc[no_oc["fd"].isin(fds) & (no_oc["forecast_start_cycle"] >= 20)]
    no_oc = no_oc[(no_oc["forecast_start_cycle"] - 20) % 20 == 0]
    no_oc = select_first_units(no_oc, first_n_units)
    return with_oc, no_oc


def add_health_lhi_history(raw: pd.DataFrame, healthy_cycles: int) -> pd.DataFrame:
    """The raw block implementation starts at cycle 20; retain its early history."""
    # The raw calculation already includes early cycles from compute_health_point_lhi.
    return raw[raw["cycle"] <= raw.groupby(["fd", "unit_id"])["cycle"].transform("max")].copy()


def prepare_lhi_series(raw: pd.DataFrame, first_n_units: int) -> dict[tuple[str, int], np.ndarray]:
    raw = select_first_units(raw, first_n_units)
    out = {}
    for (fd, unit), group in raw.groupby(["fd", "unit_id"], sort=True):
        group = group.sort_values("cycle").drop_duplicates("cycle")
        out[(str(fd), int(unit))] = group["lhi_rmse"].to_numpy(dtype=np.float32)
    return out


def run_lhi_forecasts(
    pipeline,
    raw_with: pd.DataFrame,
    raw_no: pd.DataFrame,
    first_n_units: int,
    context_length: int,
    prediction_length: int,
    stride: int,
    batch_size: int,
) -> pd.DataFrame:
    tasks = []
    for mode, raw in [("with_oc", raw_with), ("without_oc", raw_no)]:
        raw = select_first_units(raw, first_n_units)
        for (fd, unit), group in raw.groupby(["fd", "unit_id"], sort=True):
            series = group.sort_values("cycle").drop_duplicates("cycle")
            cycles = series["cycle"].to_numpy(dtype=int)
            values = series["lhi_rmse"].to_numpy(dtype=np.float32)
            if len(values) < 2:
                continue
            # Start with cycle 20 (cutoff 19), then use non-overlapping blocks.
            for target_start_idx in range(19, len(values), stride):
                cutoff_idx = target_start_idx
                target_end_idx = min(cutoff_idx + prediction_length, len(values))
                if target_end_idx <= cutoff_idx:
                    continue
                tasks.append({
                    "mode": mode,
                    "fd": str(fd),
                    "unit_id": int(unit),
                    "cutoff_cycle": int(cycles[cutoff_idx - 1]),
                    "forecast_start_cycle": int(cycles[cutoff_idx]),
                    "target_cycles": cycles[cutoff_idx:target_end_idx],
                    "target_values": values[cutoff_idx:target_end_idx],
                    "input": {"target": values[:cutoff_idx][None, :]},
                })

    rows = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        _, point_forecasts = pipeline.predict_quantiles(
            [task["input"] for task in batch],
            prediction_length=prediction_length,
            quantile_levels=[0.5],
            batch_size=len(batch),
            context_length=context_length,
        )
        for task, tensor in zip(batch, point_forecasts):
            prediction = tensor.detach().float().cpu().numpy()[0]
            for idx, cycle in enumerate(task["target_cycles"]):
                rows.append({
                    "mode": task["mode"],
                    "fd": task["fd"],
                    "unit_id": task["unit_id"],
                    "cutoff_cycle": task["cutoff_cycle"],
                    "forecast_start_cycle": task["forecast_start_cycle"],
                    "cycle": int(cycle),
                    "horizon": idx + 1,
                    "lhi_true": float(task["target_values"][idx]),
                    "lhi_forecast_q50": float(prediction[idx]),
                    "context_points_supplied": int(task["input"]["target"].shape[1]),
                    "context_length_used": int(context_length),
                    "prediction_length": int(prediction_length),
                })
        print(f"  LHI forecast batches {min(start + batch_size, len(tasks))}/{len(tasks)}", flush=True)
    return pd.DataFrame(rows)


def block_select(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    if "forecast_start_cycle" in result.columns:
        result = result[result["forecast_start_cycle"] >= 20]
        result = result[(result["forecast_start_cycle"] - 20) % 20 == 0]
    return result.sort_values(["fd", "unit_id", "cycle"]).drop_duplicates(["fd", "unit_id", "cycle"])


def plot_engine(
    fd: str,
    unit: int,
    raw_with: pd.DataFrame,
    raw_no: pd.DataFrame,
    sensor_with: pd.DataFrame,
    sensor_no: pd.DataFrame,
    lhi_forecasts: pd.DataFrame,
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, dpi=170)
    styles = {
        "raw": ("#111827", "Raw sensor → LHI", "-", 1.8),
        "sensor": ("#2563EB", "Chronos sensor forecast → LHI", "--", 1.5),
        "lhi": ("#DC2626", "Chronos LHI forecast (q50)", ":", 1.8),
    }
    for ax, mode, raw, sensor in [
        (axes[0], "with_oc", raw_with, sensor_with),
        (axes[1], "without_oc", raw_no, sensor_no),
    ]:
        raw = raw[(raw["fd"] == fd) & (raw["unit_id"] == unit)].sort_values("cycle")
        sensor = block_select(sensor[(sensor["fd"] == fd) & (sensor["unit_id"] == unit)], "lhi_rmse")
        forecast = lhi_forecasts[(lhi_forecasts["fd"] == fd) & (lhi_forecasts["unit_id"] == unit) & (lhi_forecasts["mode"] == mode)]
        forecast = forecast.sort_values("cycle")
        ax.plot(raw["cycle"], raw["lhi_rmse"], color=styles["raw"][0], linestyle=styles["raw"][2], linewidth=styles["raw"][3], label=styles["raw"][1])
        ax.plot(sensor["cycle"], sensor["lhi_rmse"], color=styles["sensor"][0], linestyle=styles["sensor"][2], linewidth=styles["sensor"][3], label=styles["sensor"][1])
        if not forecast.empty:
            ax.plot(forecast["cycle"], forecast["lhi_forecast_q50"], color=styles["lhi"][0], linestyle=styles["lhi"][2], linewidth=styles["lhi"][3], label=styles["lhi"][1])
        ax.axvline(19.5, color="#6B7280", linewidth=0.8, alpha=0.7)
        ax.set_ylabel("LHI (RMSE log-ratio)")
        ax.set_title("with operating-condition matching" if mode == "with_oc" else "without operating-condition matching", loc="left", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, frameon=True)
    axes[-1].set_xlabel("Cycle")
    fig.suptitle(f"C-MAPSS {fd} unit {unit}: raw LHI vs Chronos-2 forecasts", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Health reference: cycles 1–50; LHI forecast uses all supplied history with Chronos context_length=20 and horizon=20; q50.", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.first_n_units < 1 or args.context_length < 1 or args.prediction_length < 1 or args.stride < 1:
        raise ValueError("first_n_units, context_length, prediction_length, and stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing raw LHI with operating-condition matching...", flush=True)
    frames_with = restrict_to_first_units(
        load_eval_frames(args.data_dir, "train", args.fds), args.first_n_units
    )
    raw_with = calculate_raw_lhi(frames_with, DEFAULT_SENSORS, args.healthy_cycles, args.prediction_length, args.stride, args.lhi_epsilon)
    print("Computing raw LHI without operating-condition matching...", flush=True)
    frames_no = restrict_to_first_units(pooled_frames(args.data_dir, args.fds), args.first_n_units)
    raw_no = calculate_raw_lhi(frames_no, DEFAULT_SENSORS, args.healthy_cycles, args.prediction_length, args.stride, args.lhi_epsilon)

    raw_with = select_first_units(raw_with, args.first_n_units)
    raw_no = select_first_units(raw_no, args.first_n_units)
    raw_with.to_csv(args.output_dir / "raw_lhi_with_oc.csv", index=False)
    raw_no.to_csv(args.output_dir / "raw_lhi_without_oc.csv", index=False)

    print("Loading existing sensor-forecast → LHI results...", flush=True)
    sensor_with, sensor_no = load_existing_sensor_lhi(args.fds, args.first_n_units)
    sensor_with.to_csv(args.output_dir / "sensor_forecast_lhi_with_oc.csv", index=False)
    sensor_no.to_csv(args.output_dir / "sensor_forecast_lhi_without_oc.csv", index=False)

    pipeline = load_pipeline(args)
    print("Forecasting LHI series...", flush=True)
    forecasts = run_lhi_forecasts(
        pipeline, raw_with, raw_no, args.first_n_units, args.context_length,
        args.prediction_length, args.stride, args.batch_size,
    )
    forecasts.to_csv(args.output_dir / "lhi_forecasts_q50.csv", index=False)

    for fd in args.fds:
        for unit in range(1, args.first_n_units + 1):
            plot_engine(
                fd, unit, raw_with, raw_no, sensor_with, sensor_no, forecasts,
                args.output_dir / "plots" / fd / f"{fd}_unit{unit}_lhi_comparison.png",
            )

    summary = []
    for fd in args.fds:
        for unit in range(1, args.first_n_units + 1):
            for mode in ["with_oc", "without_oc"]:
                part = forecasts[(forecasts["fd"] == fd) & (forecasts["unit_id"] == unit) & (forecasts["mode"] == mode)]
                summary.append({
                    "fd": fd, "unit_id": unit, "mode": mode,
                    "forecast_points": int(len(part)),
                    "forecast_origins": int(part["cutoff_cycle"].nunique()) if not part.empty else 0,
                    "forecast_start_cycle": int(part["cycle"].min()) if not part.empty else np.nan,
                    "forecast_end_cycle": int(part["cycle"].max()) if not part.empty else np.nan,
                })
    pd.DataFrame(summary).to_csv(args.output_dir / "forecast_summary.csv", index=False)

    metadata = {
        "data_dir": str(args.data_dir),
        "fds": list(args.fds),
        "first_n_units_per_fd": args.first_n_units,
        "selected_sensors": list(DEFAULT_SENSORS),
        "healthy_reference_cycles": f"1..{args.healthy_cycles}",
        "lhi_formula": "LHI_RMSE = log((D_RMSE + epsilon)/(B_RMSE + epsilon)); D_RMSE aggregates per-sensor absolute normalized drift.",
        "with_oc_raw_lhi": "raw sensor values grouped by canonical C-MAPSS op_condition_key",
        "without_oc_raw_lhi": "same calculation with one pooled op_condition_key='pooled'",
        "sensor_forecast_lhi_sources": {"FD001_FD003": "outputs/CMAPSS/cluster_20/lhi_fix", "FD002_FD004": "outputs/CMAPSS/history_condition_h20_fd002_fd004/lhi", "without_oc": "outputs/CMAPSS/no_cov_20/lhi_fix"},
        "model_id": args.model_id,
        "forecast_input": "full LHI history supplied at each origin",
        "chronos_context_length": args.context_length,
        "forecast_horizon": args.prediction_length,
        "forecast_origin_stride": args.stride,
        "quantile": "q50",
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved comparison outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

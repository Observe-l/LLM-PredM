from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

FD_NAMES = ("FD001", "FD002", "FD003", "FD004")
SETTING_COLUMNS = ["setting1", "setting2", "setting3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = ["unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]
SELECTED_SENSORS = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot Chronos-2 sensor forecasting on C-MAPSS FD001-FD004."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/chronos2_cmapss"))
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--fds", nargs="+", default=list(FD_NAMES), choices=list(FD_NAMES))
    parser.add_argument(
        "--eval_split",
        choices=["train", "test"],
        default="train",
        help="C-MAPSS split used for zero-shot forecasting evaluation.",
    )
    parser.add_argument("--context_length", type=int, default=40)
    parser.add_argument("--prediction_length", type=int, default=10)
    parser.add_argument(
        "--normalization",
        choices=["none", "zscore"],
        default="none",
        help="External preprocessing applied before Chronos-2. Default is none. zscore is only kept for ablation/reproduction.",
    )
    parser.add_argument(
        "--target_transform",
        choices=["none", "context_robust"],
        default="none",
        help=(
            "Leakage-free per-window target transform. context_robust forecasts "
            "(sensor - last_context_value) / context_MAD and restores with the same context statistics."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Rolling backtest stride within each test engine sequence.",
    )
    parser.add_argument(
        "--max_windows_per_fd",
        type=int,
        default=0,
        help="Optional cap for quick smoke tests. 0 means use every eligible rolling window.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load Chronos-2 from the local Hugging Face cache only. Useful after the model has been downloaded once.",
    )
    parser.add_argument(
        "--covariate_modes",
        nargs="+",
        choices=["none", "past_only", "known_future"],
        default=["past_only", "known_future"],
        help=(
            "Operating-condition covariate modes to evaluate. "
            "past_only gives setting1-3 only in the context; known_future also gives setting1-3 over the forecast horizon."
        ),
    )
    parser.add_argument("--cross_learning", action="store_true")
    parser.add_argument(
        "--aggregate_method",
        choices=["mean", "median"],
        default="mean",
        help="How to combine multiple horizon predictions for the same unit/sensor/cycle in the full forecast curve.",
    )
    parser.add_argument(
        "--plot_units",
        nargs="*",
        default=None,
        help="Specific full-curve unit plots to create, e.g. FD004:1 FD001:12. Defaults to the first units per FD.",
    )
    parser.add_argument(
        "--plot_examples",
        type=int,
        default=3,
        help="Number of full-unit plots to create per FD when --plot_units is not set. Use 0 to skip plots.",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow CPU inference. By default this script requires CUDA because the project request says to use GPU.",
    )
    parser.add_argument(
        "--anomaly_eps",
        type=float,
        default=1e-6,
        help="Small denominator added to train-split sensor MAD when computing normalized forecasting-error scores.",
    )
    return parser.parse_args()


def load_cmapss_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


def compute_train_stats(train_df: pd.DataFrame, sensors: Sequence[str]) -> pd.DataFrame:
    stats = train_df.loc[:, sensors].agg(["mean", "std"]).T
    stats["std"] = stats["std"].replace(0.0, np.nan).fillna(1.0)
    medians = train_df.loc[:, sensors].median()
    mad = (train_df.loc[:, sensors] - medians).abs().median()
    stats["median"] = medians
    stats["mad"] = mad.replace(0.0, np.nan).fillna(1.0)
    return stats


def normalize_frame(df: pd.DataFrame, sensors: Sequence[str], stats: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    means = stats.loc[sensors, "mean"]
    stds = stats.loc[sensors, "std"]
    sensor_values = normalized.loc[:, sensors].astype(np.float32)
    normalized_values = (sensor_values - means) / stds
    for sensor in sensors:
        normalized[sensor] = normalized_values[sensor].astype(np.float32)
    normalized.loc[:, sensors] = normalized.loc[:, sensors].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return normalized


def compute_context_center_scale(context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = context[-1].astype(np.float32)
    median = np.median(context, axis=0).astype(np.float32)
    mad = np.median(np.abs(context - median[None, :]), axis=0).astype(np.float32)
    fallback = np.std(context, axis=0).astype(np.float32)
    scale = np.where(mad > 1e-6, mad, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    return center, scale


def iter_windows(
    fd_name: str,
    eval_df: pd.DataFrame,
    model_input_df: pd.DataFrame,
    sensors: Sequence[str],
    covariate_mode: str,
    target_transform: str,
    context_length: int,
    prediction_length: int,
    stride: int,
) -> Iterable[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]]:
    for unit_id, unit_model in model_input_df.groupby("unit_id", sort=True):
        unit_model = unit_model.sort_values("cycle").reset_index(drop=True)
        unit_raw = (
            eval_df[eval_df["unit_id"] == unit_id]
            .sort_values("cycle")
            .reset_index(drop=True)
        )
        n_rows = len(unit_model)
        last_start = n_rows - prediction_length
        if last_start < context_length:
            continue

        for forecast_start in range(context_length, last_start + 1, stride):
            context = unit_model.loc[
                forecast_start - context_length : forecast_start - 1, sensors
            ].to_numpy(dtype=np.float32)
            truth = unit_raw.loc[
                forecast_start : forecast_start + prediction_length - 1, sensors
            ].to_numpy(dtype=np.float32)
            transform_center = np.zeros(len(sensors), dtype=np.float32)
            transform_scale = np.ones(len(sensors), dtype=np.float32)
            model_context = context
            if target_transform == "context_robust":
                transform_center, transform_scale = compute_context_center_scale(context)
                model_context = (context - transform_center[None, :]) / transform_scale[None, :]
            past_covariates = {
                col: unit_raw.loc[forecast_start - context_length : forecast_start - 1, col].to_numpy(dtype=np.float32)
                for col in SETTING_COLUMNS
            }
            future_covariates = {
                col: unit_raw.loc[forecast_start : forecast_start + prediction_length - 1, col].to_numpy(dtype=np.float32)
                for col in SETTING_COLUMNS
            }
            cycles = unit_raw.loc[
                forecast_start : forecast_start + prediction_length - 1, "cycle"
            ].to_numpy(dtype=np.int64)
            meta = {
                "covariate_mode": covariate_mode,
                "fd": fd_name,
                "unit_id": int(unit_id),
                "cutoff_cycle": int(unit_raw.loc[forecast_start - 1, "cycle"]),
                "forecast_start_cycle": int(cycles[0]),
            }
            chronos_input: Dict[str, Any] = {"target": model_context.T}
            if covariate_mode in {"past_only", "known_future"}:
                chronos_input["past_covariates"] = past_covariates
            if covariate_mode == "known_future":
                chronos_input["future_covariates"] = future_covariates
            transform = {
                "center": transform_center,
                "scale": transform_scale,
            }
            yield meta, chronos_input, truth.T, transform


def inverse_normalize(
    values: np.ndarray,
    sensors: Sequence[str],
    stats: pd.DataFrame,
) -> np.ndarray:
    means = stats.loc[sensors, "mean"].to_numpy(dtype=np.float32)[:, None]
    stds = stats.loc[sensors, "std"].to_numpy(dtype=np.float32)[:, None]
    return values * stds + means


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_pipeline(model_id: str, device: str, torch_dtype: str, local_files_only: bool):
    from chronos import Chronos2Pipeline

    kwargs = {"dtype": get_torch_dtype(torch_dtype)}
    if device == "cuda":
        kwargs["device_map"] = "cuda"
    elif device == "cpu":
        kwargs["device_map"] = "cpu"
    else:
        kwargs["device_map"] = device
    if local_files_only:
        kwargs["local_files_only"] = True
    print(f"Loading Chronos-2 model from {model_id!r} on {device}...", flush=True)
    return Chronos2Pipeline.from_pretrained(model_id, **kwargs)


def forecast_windows(
    pipeline,
    windows: List[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]],
    sensors: Sequence[str],
    stats: pd.DataFrame,
    prediction_length: int,
    batch_size: int,
    context_length: int,
    cross_learning: bool,
    normalization: str,
    target_transform: str,
    anomaly_eps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float | int | str]] = []
    score_rows: List[Dict[str, float | int | str]] = []
    total_batches = math.ceil(len(windows) / batch_size)
    if windows:
        n_variates, history_length = windows[0][1]["target"].shape
        covariate_mode = str(windows[0][0]["covariate_mode"])
        covariate_text = "no operating-condition covariates"
        if covariate_mode == "past_only":
            covariate_text = "past-only setting1-3 covariates"
        elif covariate_mode == "known_future":
            covariate_text = "past + known-future setting1-3 covariates"
        print(
            f"  Chronos input: multivariate windows with shape "
            f"({n_variates} sensors, {history_length} time steps), {covariate_text}",
            flush=True,
        )

    for batch_idx, start in enumerate(range(0, len(windows), batch_size), start=1):
        batch = windows[start : start + batch_size]
        inputs = [item[1] for item in batch]
        _, point_forecasts = pipeline.predict_quantiles(
            inputs,
            prediction_length=prediction_length,
            quantile_levels=[0.5],
            batch_size=batch_size,
            context_length=context_length,
            cross_learning=cross_learning,
        )
        print(f"  batch {batch_idx}/{total_batches}: {len(batch)} windows in this inference batch", flush=True)

        for (meta, _chronos_input, truth, transform), pred_tensor in zip(batch, point_forecasts):
            pred_model_scale = pred_tensor.numpy().astype(np.float32)
            if target_transform == "context_robust":
                center = transform["center"].astype(np.float32)[:, None]
                scale = transform["scale"].astype(np.float32)[:, None]
                pred = pred_model_scale * scale + center
            elif normalization == "zscore":
                pred = inverse_normalize(pred_model_scale, sensors, stats)
            else:
                pred = pred_model_scale
            mad = stats.loc[sensors, "mad"].to_numpy(dtype=np.float32)[:, None]
            # S_t = mean_{sensor,horizon} |x_{t+h,j} - median_forecast_{t+h,j}| / (MAD_j + eps).
            normalized_abs_error = np.abs(truth - pred) / (mad + float(anomaly_eps))
            score_rows.append(
                {
                    **meta,
                    "anomaly_score": float(np.mean(normalized_abs_error)),
                    "mean_abs_error": float(np.mean(np.abs(truth - pred))),
                    "prediction_length": int(prediction_length),
                    "num_sensors": int(len(sensors)),
                }
            )
            for sensor_idx, sensor in enumerate(sensors):
                for horizon_idx in range(prediction_length):
                    rows.append(
                        {
                            **meta,
                            "sensor": sensor,
                            "horizon": horizon_idx + 1,
                            "cycle": int(meta["forecast_start_cycle"]) + horizon_idx,
                            "y_true": float(truth[sensor_idx, horizon_idx]),
                            "y_pred": float(pred[sensor_idx, horizon_idx]),
                            "y_pred_model_scale": float(pred_model_scale[sensor_idx, horizon_idx]),
                            "target_transform": target_transform,
                            "normalized_abs_error": float(normalized_abs_error[sensor_idx, horizon_idx]),
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    metric_rows: List[Dict[str, float | int | str]] = []
    group_keys = ["covariate_mode", "fd"] if "covariate_mode" in predictions.columns else ["fd"]
    for key, fd_group in predictions.groupby(group_keys, sort=True):
        if isinstance(key, tuple):
            covariate_mode, fd_name = key
        else:
            covariate_mode, fd_name = "", key
        metric_rows.append(
            {
                "covariate_mode": covariate_mode,
                "fd": fd_name,
                "sensor": "ALL",
                "mae": float(np.mean(np.abs(fd_group["y_pred"] - fd_group["y_true"]))),
                "mse": float(np.mean((fd_group["y_pred"] - fd_group["y_true"]) ** 2)),
                "rmse": float(np.sqrt(np.mean((fd_group["y_pred"] - fd_group["y_true"]) ** 2))),
                "n": int(len(fd_group)),
            }
        )
        for sensor, sensor_group in fd_group.groupby("sensor", sort=True):
            metric_rows.append(
                {
                    "covariate_mode": covariate_mode,
                    "fd": fd_name,
                    "sensor": sensor,
                    "mae": float(np.mean(np.abs(sensor_group["y_pred"] - sensor_group["y_true"]))),
                    "mse": float(np.mean((sensor_group["y_pred"] - sensor_group["y_true"]) ** 2)),
                    "rmse": float(np.sqrt(np.mean((sensor_group["y_pred"] - sensor_group["y_true"]) ** 2))),
                    "n": int(len(sensor_group)),
                }
            )

    return pd.DataFrame(metric_rows)


def aggregate_forecast_curve(window_predictions: pd.DataFrame, method: str) -> pd.DataFrame:
    group_cols = ["covariate_mode", "fd", "unit_id", "sensor", "cycle"]
    if method == "mean":
        pred_agg = "mean"
    elif method == "median":
        pred_agg = "median"
    else:
        raise ValueError(f"Unsupported aggregate method: {method}")

    return (
        window_predictions.groupby(group_cols, sort=True)
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", pred_agg),
            n_window_predictions=("y_pred", "size"),
            first_forecast_start_cycle=("forecast_start_cycle", "min"),
            last_forecast_start_cycle=("forecast_start_cycle", "max"),
        )
        .reset_index()
    )


def summarize_sensor_anomaly_scores(window_predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "sensor"]
    return (
        window_predictions.groupby(group_cols, sort=True)
        .agg(
            sensor_anomaly_score=("normalized_abs_error", "mean"),
            sensor_mae=("y_pred", lambda s: float(np.mean(np.abs(s - window_predictions.loc[s.index, "y_true"])))),
            prediction_length=("horizon", "size"),
        )
        .reset_index()
    )


def parse_plot_units(items: Sequence[str] | None) -> List[Tuple[str, int]]:
    if not items:
        return []
    parsed: List[Tuple[str, int]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"--plot_units entries must look like FD004:1, got {item!r}")
        fd_name, unit_text = item.split(":", 1)
        if fd_name not in FD_NAMES:
            raise ValueError(f"Unknown FD in --plot_units: {fd_name!r}")
        parsed.append((fd_name, int(unit_text)))
    return parsed


def plot_examples(predictions: pd.DataFrame, output_dir: Path, examples_per_fd: int, plot_units: Sequence[str] | None) -> None:
    if predictions.empty:
        return
    if examples_per_fd <= 0 and not plot_units:
        return

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    explicit_units = parse_plot_units(plot_units)
    if explicit_units:
        modes = sorted(predictions["covariate_mode"].unique()) if "covariate_mode" in predictions.columns else [""]
        keys = pd.DataFrame(
            [{"covariate_mode": mode, "fd": fd_name, "unit_id": unit_id} for mode in modes for fd_name, unit_id in explicit_units]
        )
    else:
        keys = (
            predictions[["covariate_mode", "fd", "unit_id"]]
            .drop_duplicates()
            .sort_values(["covariate_mode", "fd", "unit_id"])
            .groupby(["covariate_mode", "fd"], as_index=False)
            .head(examples_per_fd)
        )

    for _, key in keys.iterrows():
        covariate_mode = str(key["covariate_mode"])
        fd_name = str(key["fd"])
        unit_id = int(key["unit_id"])
        unit_df = predictions[
            (predictions["covariate_mode"] == covariate_mode)
            & (predictions["fd"] == fd_name)
            & (predictions["unit_id"] == unit_id)
        ]
        if unit_df.empty:
            print(f"  plot skipped: no predictions for {covariate_mode} {fd_name} unit {unit_id}", flush=True)
            continue

        fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
        axes_flat = axes.ravel()
        for ax_idx, sensor in enumerate(SELECTED_SENSORS):
            ax = axes_flat[ax_idx]
            sensor_df = unit_df[unit_df["sensor"] == sensor].sort_values("cycle")
            ax.plot(sensor_df["cycle"], sensor_df["y_true"], label="ground truth", linewidth=1.5)
            ax.plot(sensor_df["cycle"], sensor_df["y_pred"], label="zero-shot forecast", linewidth=1.5)
            ax.set_title(sensor)
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[len(SELECTED_SENSORS) :]:
            ax.axis("off")
        axes_flat[0].legend(loc="best")
        first_cycle = int(unit_df["cycle"].min())
        last_cycle = int(unit_df["cycle"].max())
        fig.suptitle(f"{covariate_mode} {fd_name} unit {unit_id} full forecast curve, cycles {first_cycle}-{last_cycle}")
        fig.supxlabel("cycle")
        fig.supylabel("sensor reading")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{covariate_mode}_{fd_name}_unit{unit_id}_full_curve.png", dpi=160)
        plt.close(fig)


def save_stats(stats_by_fd: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    payload = {
        fd: {
            sensor: {
                key: float(row[key])
                for key in row.index
            }
            for sensor, row in stats.iterrows()
        }
        for fd, stats in stats_by_fd.items()
    }
    with open(output_dir / "normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available in this Python environment. The request requires GPU. "
            "Use a CUDA-enabled conda env, or pass --device cpu --allow_cpu only for debugging."
        )

    pipeline = load_pipeline(args.model_id, args.device, args.torch_dtype, args.local_files_only)
    print("Chronos-2 model loaded.", flush=True)
    all_predictions: List[pd.DataFrame] = []
    all_anomaly_scores: List[pd.DataFrame] = []
    stats_by_fd: Dict[str, pd.DataFrame] = {}

    for fd_name in args.fds:
        print(f"\n=== {fd_name} ===", flush=True)
        train_df = load_cmapss_file(args.data_dir / fd_name / f"train_{fd_name}.txt")
        test_df = load_cmapss_file(args.data_dir / fd_name / f"test_{fd_name}.txt")
        eval_df = train_df if args.eval_split == "train" else test_df
        stats = compute_train_stats(train_df, SELECTED_SENSORS)
        stats_by_fd[fd_name] = stats
        if args.normalization == "zscore":
            model_input_df = normalize_frame(eval_df, SELECTED_SENSORS, stats)
        else:
            model_input_df = eval_df

        for covariate_mode in args.covariate_modes:
            print(f"  covariate mode: {covariate_mode}", flush=True)
            windows = list(
                iter_windows(
                    fd_name=fd_name,
                    eval_df=eval_df,
                    model_input_df=model_input_df,
                    sensors=SELECTED_SENSORS,
                    covariate_mode=covariate_mode,
                    target_transform=args.target_transform,
                    context_length=args.context_length,
                    prediction_length=args.prediction_length,
                    stride=args.stride,
                )
            )
            if args.max_windows_per_fd > 0:
                windows = windows[: args.max_windows_per_fd]
            if not windows:
                print("  no eligible windows, skipped", flush=True)
                continue

            print(f"  forecasting {len(windows)} rolling windows", flush=True)
            fd_predictions, fd_anomaly_scores = forecast_windows(
                pipeline=pipeline,
                windows=windows,
                sensors=SELECTED_SENSORS,
                stats=stats,
                prediction_length=args.prediction_length,
                batch_size=args.batch_size,
                context_length=args.context_length,
                cross_learning=args.cross_learning,
                normalization=args.normalization,
                target_transform=args.target_transform,
                anomaly_eps=args.anomaly_eps,
            )
            all_predictions.append(fd_predictions)
            all_anomaly_scores.append(fd_anomaly_scores)

            fd_metrics = summarize_metrics(fd_predictions)
            fd_all = fd_metrics[fd_metrics["sensor"] == "ALL"].iloc[0]
            print(f"  MAE={fd_all['mae']:.6f} RMSE={fd_all['rmse']:.6f}", flush=True)

    if not all_predictions:
        raise RuntimeError("No forecasts were produced.")

    window_predictions = pd.concat(all_predictions, ignore_index=True)
    anomaly_scores = pd.concat(all_anomaly_scores, ignore_index=True)
    sensor_anomaly_scores = summarize_sensor_anomaly_scores(window_predictions)
    curve_predictions = aggregate_forecast_curve(window_predictions, args.aggregate_method)
    metrics = summarize_metrics(curve_predictions)
    window_metrics = summarize_metrics(window_predictions)
    curve_predictions.to_csv(args.output_dir / "forecasts.csv", index=False)
    window_predictions.to_csv(args.output_dir / "window_forecasts.csv", index=False)
    anomaly_scores.to_csv(args.output_dir / "anomaly_scores.csv", index=False)
    sensor_anomaly_scores.to_csv(args.output_dir / "sensor_anomaly_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    window_metrics.to_csv(args.output_dir / "window_metrics.csv", index=False)
    save_stats(stats_by_fd, args.output_dir)
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"selected_sensors": SELECTED_SENSORS}, f, indent=2, default=str)
    plot_examples(curve_predictions, args.output_dir, args.plot_examples, args.plot_units)

    print("\nOverall metrics by FD:", flush=True)
    print(metrics[metrics["sensor"] == "ALL"].to_string(index=False), flush=True)
    print(f"\nSaved outputs to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

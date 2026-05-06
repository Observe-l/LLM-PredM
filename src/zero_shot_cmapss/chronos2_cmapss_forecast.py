from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
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
EXPERIMENT_MODES = ("cluster_covariate", "future_covariate", "no_covariate")
MODE_ALIASES = {
    "known_future": "cluster_covariate",
    "none": "no_covariate",
}


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
    parser.add_argument(
        "--context_length",
        type=int,
        default=0,
        help=(
            "Maximum historical context length. 0 means all available past context per engine: "
            "cutoff cycle 10 uses 10 points, cutoff cycle 50 uses 50 points."
        ),
    )
    parser.add_argument("--prediction_length", type=int, default=10)
    parser.add_argument(
        "--target_transform",
        choices=["none", "context_minmax"],
        default="context_minmax",
        help=(
            "Leakage-free per-window target transform. context_minmax scales each sensor with "
            "the current past context min/max and restores predictions with the same context statistics."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Rolling backtest stride in forecast-start cycles. Use 1 to create a forecast at every eligible cycle.",
    )
    parser.add_argument(
        "--forecast_start_cycle",
        type=int,
        default=20,
        help="Earliest target cycle used as the start of a prediction_length forecast window.",
    )
    parser.add_argument(
        "--forecast_end_cycle",
        type=int,
        default=0,
        help=(
            "Latest target cycle used as a forecast-start cycle. "
            "0 means use the final observed cycle, even though later horizon steps have no ground truth."
        ),
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
        choices=["cluster_covariate", "future_covariate", "no_covariate", "known_future", "none"],
        default=["cluster_covariate"],
        help=(
            "Experiment modes. cluster_covariate first groups by operating condition and passes past/future "
            "setting1-3 covariates; future_covariate passes raw multivariate sensors plus past/future setting1-3 "
            "without grouping; no_covariate uses only raw multivariate sensors. Deprecated aliases: "
            "known_future=cluster_covariate, none=no_covariate."
        ),
    )
    parser.add_argument("--cross_learning", action="store_true")
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
    stats = pd.DataFrame(index=sensors)
    medians = train_df.loc[:, sensors].median()
    mad = (train_df.loc[:, sensors] - medians).abs().median()
    stats["median"] = medians
    stats["mad"] = mad.replace(0.0, np.nan).fillna(1.0)
    return stats


def compute_context_minmax(context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    minimum = np.min(context, axis=0).astype(np.float32)
    maximum = np.max(context, axis=0).astype(np.float32)
    value_range = (maximum - minimum).astype(np.float32)
    value_range = np.where(value_range > 1e-6, value_range, 1.0).astype(np.float32)
    return minimum, value_range


def make_condition_keys(frame: pd.DataFrame) -> pd.Series:
    canonical = pd.DataFrame(index=frame.index)
    canonical["setting1"] = np.rint(frame["setting1"]).astype(int)
    canonical["setting2"] = np.rint(frame["setting2"] * 100).astype(int)
    canonical["setting3"] = np.rint(frame["setting3"]).astype(int)
    return canonical.astype(str).agg("|".join, axis=1)


def add_window_condition_labels(history_raw: pd.DataFrame, future_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    history = history_raw.copy()
    future = future_raw.copy()
    history["op_condition_key"] = make_condition_keys(history)
    future["op_condition_key"] = make_condition_keys(future)
    condition_order = sorted(pd.concat([history["op_condition_key"], future["op_condition_key"]]).unique())
    condition_to_id = {condition_key: idx for idx, condition_key in enumerate(condition_order)}
    history["op_condition"] = history["op_condition_key"].map(condition_to_id).astype(int)
    future["op_condition"] = future["op_condition_key"].map(condition_to_id).astype(int)
    return history, future, len(condition_to_id)


def normalize_covariate_modes(modes: Sequence[str]) -> List[str]:
    normalized = []
    for mode in modes:
        normalized_mode = MODE_ALIASES.get(str(mode), str(mode))
        if normalized_mode not in EXPERIMENT_MODES:
            raise ValueError(f"Unsupported covariate mode: {mode!r}")
        if normalized_mode not in normalized:
            normalized.append(normalized_mode)
    return normalized


def build_future_frame(unit_raw: pd.DataFrame, forecast_start: int, prediction_length: int) -> pd.DataFrame:
    rows = []
    start_cycle = int(unit_raw.loc[forecast_start, "cycle"])
    last_row = unit_raw.iloc[-1]
    for horizon_idx in range(prediction_length):
        row_idx = forecast_start + horizon_idx
        if row_idx < len(unit_raw):
            row = unit_raw.loc[row_idx].copy()
            row["has_ground_truth"] = True
        else:
            row = last_row.copy()
            row["cycle"] = start_cycle + horizon_idx
            for sensor in SENSOR_COLUMNS:
                row[sensor] = np.nan
            row["has_ground_truth"] = False
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


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
    forecast_start_cycle: int,
    forecast_end_cycle: int,
) -> Iterable[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, Any]]]:
    if context_length < 0:
        raise ValueError("--context_length must be >= 0. Use 0 for all-past context.")
    context_cap = None if context_length == 0 else int(context_length)
    for unit_id, unit_model in model_input_df.groupby("unit_id", sort=True):
        unit_model = unit_model.sort_values("cycle").reset_index(drop=True)
        unit_raw = (
            eval_df[eval_df["unit_id"] == unit_id]
            .sort_values("cycle")
            .reset_index(drop=True)
        )
        n_rows = len(unit_model)
        last_observed_start = n_rows - 1
        if last_observed_start < 1:
            continue

        cycle_values = unit_raw["cycle"].to_numpy(dtype=np.int64)
        first_candidates = np.flatnonzero(cycle_values >= int(forecast_start_cycle))
        if len(first_candidates) == 0:
            continue
        first_forecast_start = int(first_candidates[0])
        if forecast_end_cycle > 0:
            end_candidates = np.flatnonzero(cycle_values <= int(forecast_end_cycle))
            if len(end_candidates) == 0:
                continue
            last_requested_start = int(end_candidates[-1])
        else:
            last_requested_start = last_observed_start
        last_start = min(last_observed_start, last_requested_start)
        if last_start < first_forecast_start:
            continue

        for forecast_start in range(first_forecast_start, last_start + 1, stride):
            context_start = 0 if context_cap is None else max(0, forecast_start - context_cap)
            history_model = unit_model.loc[context_start : forecast_start - 1].copy()
            history_raw = unit_raw.loc[context_start : forecast_start - 1].copy()
            future_raw = build_future_frame(unit_raw, forecast_start, prediction_length)
            history_labeled, future_labeled, n_conditions = add_window_condition_labels(history_raw, future_raw)
            cycles = future_labeled["cycle"].to_numpy(dtype=np.int64)
            base_meta = {
                "covariate_mode": covariate_mode,
                "fd": fd_name,
                "unit_id": int(unit_id),
                "cutoff_cycle": int(unit_raw.loc[forecast_start - 1, "cycle"]),
                "forecast_start_cycle": int(cycles[0]),
                "context_start_cycle": int(unit_raw.loc[context_start, "cycle"]),
                "context_length": int(len(history_raw)),
                "total_prediction_length": int(prediction_length),
                "n_operating_conditions": int(n_conditions),
            }

            if covariate_mode == "cluster_covariate":
                future_groups = [
                    (int(op_condition), future_group.copy())
                    for op_condition, future_group in future_labeled.groupby("op_condition", sort=True)
                ]
            else:
                future_groups = [(-1, future_labeled.copy())]

            for op_condition, future_group in future_groups:
                op_condition = int(op_condition)
                if covariate_mode == "cluster_covariate":
                    history_mask = history_labeled["op_condition"] == op_condition
                    group_history_model = history_model.loc[history_mask.to_numpy()].copy()
                    group_history_raw = history_labeled.loc[history_mask].copy()
                else:
                    group_history_model = history_model.copy()
                    group_history_raw = history_labeled.copy()
                used_full_context_fallback = False
                if group_history_model.empty:
                    group_history_model = history_model
                    group_history_raw = history_labeled
                    used_full_context_fallback = True

                context = group_history_model.loc[:, sensors].to_numpy(dtype=np.float32)
                truth = future_group.loc[:, sensors].to_numpy(dtype=np.float32)
                transform_offset = np.zeros(len(sensors), dtype=np.float32)
                transform_scale = np.ones(len(sensors), dtype=np.float32)
                model_context = context
                if target_transform == "context_minmax":
                    transform_offset, transform_scale = compute_context_minmax(context)
                    model_context = (context - transform_offset[None, :]) / transform_scale[None, :]

                past_covariates = {
                    col: group_history_raw.loc[:, col].to_numpy(dtype=np.float32)
                    for col in SETTING_COLUMNS
                }
                future_covariates = {
                    col: future_group.loc[:, col].to_numpy(dtype=np.float32)
                    for col in SETTING_COLUMNS
                }
                group_prediction_length = int(len(future_group))
                meta = {
                    **base_meta,
                    "op_condition": op_condition,
                    "op_condition_key": str(future_group["op_condition_key"].iloc[0])
                    if future_group["op_condition_key"].nunique() == 1
                    else "mixed",
                    "group_context_length": int(len(group_history_model)),
                    "group_prediction_length": group_prediction_length,
                    "used_full_context_fallback": int(used_full_context_fallback),
                }
                chronos_input: Dict[str, Any] = {"target": model_context.T}
                if covariate_mode in {"cluster_covariate", "future_covariate"}:
                    chronos_input["past_covariates"] = past_covariates
                    chronos_input["future_covariates"] = future_covariates
                transform = {
                    "offset": transform_offset,
                    "scale": transform_scale,
                    "future_cycles": future_group["cycle"].to_numpy(dtype=np.int64),
                    "future_horizons": future_group.index.to_numpy(dtype=np.int64) + 1,
                    "future_has_ground_truth": future_group["has_ground_truth"].to_numpy(dtype=bool),
                    "future_op_condition_keys": future_group["op_condition_key"].astype(str).to_numpy(),
                }
                yield meta, chronos_input, truth.T, transform


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
    windows: List[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, Any]]],
    sensors: Sequence[str],
    stats: pd.DataFrame,
    prediction_length: int,
    batch_size: int,
    cross_learning: bool,
    target_transform: str,
    anomaly_eps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float | int | str]] = []
    if windows:
        n_variates = windows[0][1]["target"].shape[0]
        history_lengths = [item[1]["target"].shape[1] for item in windows]
        group_prediction_lengths = [int(item[0]["group_prediction_length"]) for item in windows]
        covariate_mode = str(windows[0][0]["covariate_mode"])
        covariate_text = "no operating-condition covariates"
        if covariate_mode == "cluster_covariate":
            covariate_text = "condition-clustered past + known-future setting1-3 covariates"
        elif covariate_mode == "future_covariate":
            covariate_text = "raw multivariate target with past + known-future setting1-3 covariates"
        print(
            f"  Chronos input: multivariate windows with shape "
            f"({n_variates} sensors, variable history {min(history_lengths)}-{max(history_lengths)} time steps), "
            f"grouped future length {min(group_prediction_lengths)}-{max(group_prediction_lengths)}, {covariate_text}",
            flush=True,
        )

    tasks_by_prediction_length: Dict[int, List[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, Any]]]] = defaultdict(list)
    for task in windows:
        tasks_by_prediction_length[int(task[0]["group_prediction_length"])].append(task)

    total_batches = sum(math.ceil(len(tasks) / batch_size) for tasks in tasks_by_prediction_length.values())
    batch_idx = 0
    for group_prediction_length, tasks in sorted(tasks_by_prediction_length.items()):
        for start in range(0, len(tasks), batch_size):
            batch_idx += 1
            batch = tasks[start : start + batch_size]
            inputs = [item[1] for item in batch]
            batch_context_length = max(int(item[1]["target"].shape[1]) for item in batch)
            _, point_forecasts = pipeline.predict_quantiles(
                inputs,
                prediction_length=group_prediction_length,
                quantile_levels=[0.5],
                batch_size=batch_size,
                context_length=batch_context_length,
                cross_learning=cross_learning,
            )
            print(
                f"  batch {batch_idx}/{total_batches}: {len(batch)} grouped tasks, "
                f"group prediction length={group_prediction_length}, batch context length={batch_context_length}",
                flush=True,
            )

            for (meta, _chronos_input, truth, transform), pred_tensor in zip(batch, point_forecasts):
                pred_model_scale = pred_tensor.numpy().astype(np.float32)
                if target_transform == "context_minmax":
                    center = transform["offset"].astype(np.float32)[:, None]
                    scale = transform["scale"].astype(np.float32)[:, None]
                    pred = pred_model_scale * scale + center
                else:
                    pred = pred_model_scale
                mad = stats.loc[sensors, "mad"].to_numpy(dtype=np.float32)[:, None]
                normalized_abs_error = np.abs(truth - pred) / (mad + float(anomaly_eps))
                future_cycles = transform["future_cycles"].astype(np.int64)
                future_horizons = transform["future_horizons"].astype(np.int64)
                future_has_ground_truth = transform["future_has_ground_truth"].astype(bool)
                future_op_condition_keys = transform["future_op_condition_keys"]
                for sensor_idx, sensor in enumerate(sensors):
                    for group_horizon_idx in range(group_prediction_length):
                        horizon = int(future_horizons[group_horizon_idx])
                        cycle = int(future_cycles[group_horizon_idx])
                        row_meta = dict(meta)
                        row_meta.pop("group_prediction_length", None)
                        row_meta["op_condition_key"] = str(future_op_condition_keys[group_horizon_idx])
                        rows.append(
                            {
                                **row_meta,
                                "sensor": sensor,
                                "horizon": horizon,
                                "cycle": cycle,
                                "group_horizon": group_horizon_idx + 1,
                                "group_prediction_length": group_prediction_length,
                                "prediction_length": int(prediction_length),
                                "has_ground_truth": int(future_has_ground_truth[group_horizon_idx]),
                                "y_true": float(truth[sensor_idx, group_horizon_idx]),
                                "y_pred": float(pred[sensor_idx, group_horizon_idx]),
                                "y_pred_model_scale": float(pred_model_scale[sensor_idx, group_horizon_idx]),
                                "target_transform": target_transform,
                                "normalized_abs_error": float(normalized_abs_error[sensor_idx, group_horizon_idx]),
                            }
                        )

    predictions = pd.DataFrame(rows)
    if not predictions.empty:
        predictions = predictions.sort_values(
            ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "sensor", "horizon", "op_condition"]
        ).reset_index(drop=True)
    score_rows: List[Dict[str, float | int | str]] = []
    if not predictions.empty:
        score_cols = [
            "covariate_mode",
            "fd",
            "unit_id",
            "cutoff_cycle",
            "forecast_start_cycle",
            "context_start_cycle",
            "context_length",
            "total_prediction_length",
            "n_operating_conditions",
        ]
        for key, group in predictions.groupby(score_cols, sort=True):
            available = group[group["has_ground_truth"].astype(bool)].copy()
            score_rows.append(
                {
                    **dict(zip(score_cols, key)),
                    "prediction_length": int(group["prediction_length"].iloc[0]),
                    "anomaly_score": float(available["normalized_abs_error"].mean()) if not available.empty else np.nan,
                    "mean_abs_error": float(np.mean(np.abs(available["y_true"] - available["y_pred"])))
                    if not available.empty
                    else np.nan,
                    "ground_truth_rows": int(len(available)),
                    "num_sensors": int(len(sensors)),
                    "num_condition_tasks": int(group["op_condition"].nunique()),
                    "used_full_context_fallback_tasks": int(
                        group[["op_condition", "used_full_context_fallback"]]
                        .drop_duplicates()["used_full_context_fallback"]
                        .sum()
                    ),
                }
            )

    return predictions, pd.DataFrame(score_rows)


def limit_tasks_by_rolling_windows(
    tasks: List[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, Any]]],
    max_windows: int,
) -> List[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, Any]]]:
    if max_windows <= 0:
        return tasks
    selected_keys = []
    selected_key_set = set()
    for meta, *_ in tasks:
        key = (
            meta["covariate_mode"],
            meta["fd"],
            meta["unit_id"],
            meta["cutoff_cycle"],
            meta["forecast_start_cycle"],
        )
        if key not in selected_key_set:
            selected_keys.append(key)
            selected_key_set.add(key)
        if len(selected_keys) >= max_windows:
            break
    keep_keys = set(selected_keys)
    return [
        task
        for task in tasks
        if (
            task[0]["covariate_mode"],
            task[0]["fd"],
            task[0]["unit_id"],
            task[0]["cutoff_cycle"],
            task[0]["forecast_start_cycle"],
        )
        in keep_keys
    ]


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions[predictions["y_true"].notna()].copy()
    metric_rows: List[Dict[str, float | int | str]] = []
    if predictions.empty:
        return pd.DataFrame(columns=["covariate_mode", "fd", "sensor", "mae", "mse", "rmse", "n"])
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


def select_metric_windows(window_predictions: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    key_cols = ["covariate_mode", "fd", "unit_id", "forecast_start_cycle"]
    horizon_counts = (
        window_predictions[window_predictions["has_ground_truth"].astype(bool)]
        .groupby(key_cols, sort=True)["horizon"]
        .nunique()
        .reset_index(name="ground_truth_horizon_count")
    )
    full_keys = horizon_counts[horizon_counts["ground_truth_horizon_count"] >= prediction_length].copy()
    if full_keys.empty:
        return window_predictions.iloc[0:0].copy()

    selected_rows = []
    for _, group in full_keys.groupby(["covariate_mode", "fd", "unit_id"], sort=True):
        first_start = int(group["forecast_start_cycle"].min())
        keep = group[(group["forecast_start_cycle"] - first_start) % prediction_length == 0]
        selected_rows.append(keep[key_cols])
    selected_keys = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(columns=key_cols)
    metric_predictions = window_predictions.merge(selected_keys, on=key_cols, how="inner")
    metric_predictions = metric_predictions[metric_predictions["has_ground_truth"].astype(bool)].copy()
    return metric_predictions


def summarize_sensor_anomaly_scores(window_predictions: pd.DataFrame) -> pd.DataFrame:
    available = window_predictions[window_predictions["has_ground_truth"].astype(bool)].copy()
    group_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "sensor"]
    return (
        available.groupby(group_cols, sort=True)
        .agg(
            sensor_anomaly_score=("normalized_abs_error", "mean"),
            sensor_mae=("y_pred", lambda s: float(np.mean(np.abs(s - available.loc[s.index, "y_true"])))),
            prediction_length=("horizon", "size"),
        )
        .reset_index()
    )


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
    with open(output_dir / "sensor_error_stats.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.forecast_start_cycle <= 1:
        raise ValueError("--forecast_start_cycle must be > 1 so at least one past context point exists.")
    if args.forecast_end_cycle and args.forecast_end_cycle < args.forecast_start_cycle:
        raise ValueError("--forecast_end_cycle must be 0 or >= --forecast_start_cycle.")
    args.covariate_modes = normalize_covariate_modes(args.covariate_modes)
    if args.context_length < 0:
        raise ValueError("--context_length must be >= 0. Use 0 for all-past context.")
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
                    forecast_start_cycle=args.forecast_start_cycle,
                    forecast_end_cycle=args.forecast_end_cycle,
                )
            )
            if args.max_windows_per_fd > 0:
                windows = limit_tasks_by_rolling_windows(windows, args.max_windows_per_fd)
            if not windows:
                print("  no eligible windows, skipped", flush=True)
                continue

            num_rolling_windows = len(
                {
                    (
                        task[0]["fd"],
                        task[0]["unit_id"],
                        task[0]["cutoff_cycle"],
                        task[0]["forecast_start_cycle"],
                    )
                    for task in windows
                }
            )
            print(
                f"  forecasting {num_rolling_windows} rolling windows as {len(windows)} condition-grouped tasks",
                flush=True,
            )
            fd_predictions, fd_anomaly_scores = forecast_windows(
                pipeline=pipeline,
                windows=windows,
                sensors=SELECTED_SENSORS,
                stats=stats,
                prediction_length=args.prediction_length,
                batch_size=args.batch_size,
                cross_learning=args.cross_learning,
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
    metric_window_predictions = select_metric_windows(window_predictions, args.prediction_length)
    sensor_anomaly_scores = summarize_sensor_anomaly_scores(metric_window_predictions)
    metrics = summarize_metrics(metric_window_predictions)
    window_metrics = summarize_metrics(metric_window_predictions)
    window_predictions.to_csv(args.output_dir / "window_forecasts.csv", index=False)
    metric_window_predictions.to_csv(args.output_dir / "metric_window_forecasts.csv", index=False)
    anomaly_scores.to_csv(args.output_dir / "anomaly_scores.csv", index=False)
    sensor_anomaly_scores.to_csv(args.output_dir / "sensor_anomaly_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    window_metrics.to_csv(args.output_dir / "window_metrics.csv", index=False)
    save_stats(stats_by_fd, args.output_dir)
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            vars(args) | {"selected_sensors": SELECTED_SENSORS, "condition_group_forecasting": True},
            f,
            indent=2,
            default=str,
        )

    print("\nOverall metrics by FD:", flush=True)
    print(metrics[metrics["sensor"] == "ALL"].to_string(index=False), flush=True)
    print(f"\nSaved outputs to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

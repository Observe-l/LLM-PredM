from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

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
EXPERIMENT_MODES = ("cluster_covariate", "no_covariate")
MODE_ALIASES = {
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
    parser.add_argument(
        "--task_chunk_size",
        type=int,
        default=1024,
        help=(
            "Maximum condition tasks materialized before forecasts are appended to CSV. "
            "This bounds memory for full stride-1 FD002/FD004 experiments."
        ),
    )
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
        choices=["cluster_covariate", "no_covariate", "none"],
        default=["cluster_covariate"],
        help=(
            "Experiment modes. cluster_covariate extracts conditions from past context only and forecasts "
            "a full horizon independently under every observed historical condition. no_covariate uses only "
            "raw multivariate sensors. Deprecated alias: none=no_covariate."
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


def add_history_condition_labels(history_raw: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    history = history_raw.copy()
    history["op_condition_key"] = make_condition_keys(history)
    condition_order = sorted(history["op_condition_key"].unique())
    condition_to_id = {condition_key: idx for idx, condition_key in enumerate(condition_order)}
    history["op_condition"] = history["op_condition_key"].map(condition_to_id).astype(int)
    return history, len(condition_to_id)


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
    for horizon_idx in range(prediction_length):
        row_idx = forecast_start + horizon_idx
        if row_idx < len(unit_raw):
            row = unit_raw.loc[row_idx, ["cycle", *SENSOR_COLUMNS]].copy()
            row["has_ground_truth"] = True
        else:
            row = pd.Series(
                {
                    "cycle": start_cycle + horizon_idx,
                    **{sensor: np.nan for sensor in SENSOR_COLUMNS},
                    "has_ground_truth": False,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def build_metric_condition_keys(
    unit_raw: pd.DataFrame,
    forecast_start: int,
    prediction_length: int,
) -> np.ndarray:
    """Return realized future condition keys for retrospective metrics only.

    These values must never be added to a Chronos input. They are kept separate
    from ``build_future_frame`` so the forecasting path cannot accidentally use
    realized future operating settings.
    """
    keys = np.full(prediction_length, "", dtype=object)
    observed_end = min(forecast_start + prediction_length, len(unit_raw))
    if observed_end > forecast_start:
        observed_future = unit_raw.iloc[forecast_start:observed_end]
        observed_keys = make_condition_keys(observed_future).astype(str).to_numpy()
        keys[: len(observed_keys)] = observed_keys
    return keys


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

        first_complete_metric_start: int | None = None
        for forecast_start in range(first_forecast_start, last_start + 1, stride):
            context_start = 0 if context_cap is None else max(0, forecast_start - context_cap)
            history_model = unit_model.loc[context_start : forecast_start - 1].copy()
            history_raw = unit_raw.loc[context_start : forecast_start - 1].copy()
            future_raw = build_future_frame(unit_raw, forecast_start, prediction_length)
            metric_condition_keys = build_metric_condition_keys(
                unit_raw,
                forecast_start,
                prediction_length,
            )
            history_labeled, n_conditions = add_history_condition_labels(history_raw)
            cycles = future_raw["cycle"].to_numpy(dtype=np.int64)
            has_full_ground_truth = bool(future_raw["has_ground_truth"].astype(bool).all())
            if covariate_mode == "cluster_covariate":
                historical_keys = set(history_labeled["op_condition_key"].astype(str))
                realized_keys = {str(key) for key in metric_condition_keys if str(key)}
                has_all_realized_condition_forecasts = realized_keys.issubset(historical_keys)
            else:
                has_all_realized_condition_forecasts = True
            metric_window_complete_match = (
                has_full_ground_truth and has_all_realized_condition_forecasts
            )
            if metric_window_complete_match and first_complete_metric_start is None:
                first_complete_metric_start = int(cycles[0])
            metric_window_selected = bool(
                metric_window_complete_match
                and first_complete_metric_start is not None
                and (int(cycles[0]) - first_complete_metric_start) % prediction_length == 0
            )
            n_condition_tasks = n_conditions if covariate_mode == "cluster_covariate" else 1
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
                "n_condition_forecast_tasks": int(n_condition_tasks),
                "condition_prediction_points": int(n_condition_tasks * prediction_length),
                "condition_source": "past_context_only",
                "metric_window_complete_match": int(metric_window_complete_match),
                "metric_window_selected": int(metric_window_selected),
            }

            if covariate_mode == "cluster_covariate":
                history_groups = [
                    (int(op_condition), history_group.copy())
                    for op_condition, history_group in history_labeled.groupby("op_condition", sort=True)
                ]
            else:
                history_groups = [(-1, history_labeled.copy())]

            for op_condition, group_history_raw in history_groups:
                op_condition = int(op_condition)
                if covariate_mode == "cluster_covariate":
                    history_mask = history_labeled["op_condition"] == op_condition
                    group_history_model = history_model.loc[history_mask.to_numpy()].copy()
                else:
                    group_history_model = history_model.copy()
                if group_history_model.empty or group_history_raw.empty:
                    raise RuntimeError("Historical operating-condition group unexpectedly has no context rows.")

                context = group_history_model.loc[:, sensors].to_numpy(dtype=np.float32)
                truth = future_raw.loc[:, sensors].to_numpy(dtype=np.float32)
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
                last_observed_settings = group_history_raw.iloc[-1].loc[SETTING_COLUMNS]
                repeated_condition_covariates = {
                    col: np.repeat(np.float32(last_observed_settings[col]), prediction_length)
                    for col in SETTING_COLUMNS
                }
                group_prediction_length = int(prediction_length)
                condition_key = (
                    str(group_history_raw["op_condition_key"].iloc[-1])
                    if covariate_mode == "cluster_covariate"
                    else str(history_labeled["op_condition_key"].iloc[-1])
                )
                meta = {
                    **base_meta,
                    "op_condition": op_condition,
                    "op_condition_key": condition_key,
                    "group_context_length": int(len(group_history_model)),
                    "group_prediction_length": group_prediction_length,
                    "future_condition_policy": "repeat_last_observed_settings",
                }
                chronos_input: Dict[str, Any] = {"target": model_context.T}
                if covariate_mode == "cluster_covariate":
                    chronos_input["past_covariates"] = past_covariates
                    # "future_covariates" is the Chronos API field name.  Its
                    # values here are hypothetical repeats of past observations.
                    chronos_input["future_covariates"] = repeated_condition_covariates
                transform = {
                    "offset": transform_offset,
                    "scale": transform_scale,
                    "future_cycles": future_raw["cycle"].to_numpy(dtype=np.int64),
                    "future_horizons": np.arange(1, prediction_length + 1, dtype=np.int64),
                    "future_has_ground_truth": future_raw["has_ground_truth"].to_numpy(dtype=bool),
                    "future_op_condition_keys": np.repeat(condition_key, prediction_length),
                    "metric_op_condition_keys": metric_condition_keys,
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
            covariate_text = (
                "historical-condition context + repeated last-observed setting1-3 covariates"
            )
        print(
            f"  Chronos input: multivariate windows with shape "
            f"({n_variates} sensors, variable history {min(history_lengths)}-{max(history_lengths)} time steps), "
            f"independent condition horizon {min(group_prediction_lengths)}-{max(group_prediction_lengths)}, "
            f"{covariate_text}",
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
                metric_op_condition_keys = transform["metric_op_condition_keys"]
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
                                "metric_op_condition_key": str(
                                    metric_op_condition_keys[group_horizon_idx]
                                ),
                                "is_metric_condition_match": int(
                                    covariate_mode != "cluster_covariate"
                                    or str(future_op_condition_keys[group_horizon_idx])
                                    == str(metric_op_condition_keys[group_horizon_idx])
                                ),
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
            "condition_prediction_points",
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


def iter_task_chunks(
    tasks: Iterable[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, Any]]],
    task_chunk_size: int,
    max_windows: int = 0,
) -> Iterator[List[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, Any]]]]:
    """Yield bounded task chunks without splitting a rolling window."""
    if task_chunk_size < 1:
        raise ValueError("--task_chunk_size must be >= 1.")

    chunk: List[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, Any]]] = []
    window_group: List[Tuple[Dict[str, int | str], Dict[str, Any], np.ndarray, Dict[str, Any]]] = []
    current_key = None
    window_count = 0

    def task_key(task):
        meta = task[0]
        return (
            meta["covariate_mode"],
            meta["fd"],
            meta["unit_id"],
            meta["cutoff_cycle"],
            meta["forecast_start_cycle"],
        )

    def flush_group():
        nonlocal chunk, window_group, window_count
        if not window_group:
            return None
        if max_windows > 0 and window_count >= max_windows:
            return False
        if chunk and len(chunk) + len(window_group) > task_chunk_size:
            ready = chunk
            chunk = window_group
            window_group = []
            window_count += 1
            return ready
        chunk.extend(window_group)
        window_group = []
        window_count += 1
        return None

    for task in tasks:
        key = task_key(task)
        if current_key is None:
            current_key = key
        if key != current_key:
            ready = flush_group()
            if ready is False:
                break
            if ready:
                yield ready
            current_key = key
        window_group.append(task)
    else:
        ready = flush_group()
        if ready:
            yield ready

    if chunk:
        yield chunk


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
    if "metric_window_selected" in window_predictions.columns:
        selected = window_predictions[
            window_predictions["metric_window_selected"].astype(bool)
        ].copy()
        if "is_metric_condition_match" in selected.columns:
            selected = selected[selected["is_metric_condition_match"].astype(bool)].copy()
        return selected[selected["has_ground_truth"].astype(bool)].copy()

    key_cols = ["covariate_mode", "fd", "unit_id", "forecast_start_cycle"]
    if "is_metric_condition_match" in window_predictions.columns:
        metric_candidates = window_predictions[
            window_predictions["is_metric_condition_match"].astype(bool)
        ].copy()
    else:
        metric_candidates = window_predictions.copy()
    horizon_counts = (
        metric_candidates[metric_candidates["has_ground_truth"].astype(bool)]
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
    metric_predictions = metric_candidates.merge(selected_keys, on=key_cols, how="inner")
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
    if args.task_chunk_size < 1:
        raise ValueError("--task_chunk_size must be >= 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available in this Python environment. The request requires GPU. "
            "Use a CUDA-enabled conda env, or pass --device cpu --allow_cpu only for debugging."
        )

    pipeline = load_pipeline(args.model_id, args.device, args.torch_dtype, args.local_files_only)
    print("Chronos-2 model loaded.", flush=True)
    window_forecast_path = args.output_dir / "window_forecasts.csv"
    for generated_name in (
        "window_forecasts.csv",
        "metric_window_forecasts.csv",
        "anomaly_scores.csv",
        "sensor_anomaly_scores.csv",
        "metrics.csv",
        "window_metrics.csv",
    ):
        generated_path = args.output_dir / generated_name
        if generated_path.exists():
            generated_path.unlink()

    wrote_window_header = False
    metric_prediction_parts: List[pd.DataFrame] = []
    all_anomaly_scores: List[pd.DataFrame] = []
    stats_by_fd: Dict[str, pd.DataFrame] = {}
    total_prediction_rows = 0

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
            task_source = iter_windows(
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
            mode_metric_parts: List[pd.DataFrame] = []
            mode_window_count = 0
            mode_task_count = 0
            for chunk_index, windows in enumerate(
                iter_task_chunks(
                    task_source,
                    task_chunk_size=args.task_chunk_size,
                    max_windows=args.max_windows_per_fd,
                ),
                start=1,
            ):
                chunk_window_count = len(
                    {
                        (
                            task[0]["unit_id"],
                            task[0]["cutoff_cycle"],
                            task[0]["forecast_start_cycle"],
                        )
                        for task in windows
                    }
                )
                mode_window_count += chunk_window_count
                mode_task_count += len(windows)
                print(
                    f"  task chunk {chunk_index}: {chunk_window_count} rolling windows, "
                    f"{len(windows)} condition tasks",
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
                fd_predictions.to_csv(
                    window_forecast_path,
                    mode="a",
                    header=not wrote_window_header,
                    index=False,
                )
                wrote_window_header = True
                total_prediction_rows += len(fd_predictions)
                all_anomaly_scores.append(fd_anomaly_scores)
                metric_part = select_metric_windows(
                    fd_predictions,
                    args.prediction_length,
                )
                if not metric_part.empty:
                    metric_prediction_parts.append(metric_part)
                    mode_metric_parts.append(metric_part)

            if mode_window_count == 0:
                print("  no eligible windows, skipped", flush=True)
                continue
            print(
                f"  completed {mode_window_count} rolling windows as "
                f"{mode_task_count} condition tasks",
                flush=True,
            )
            if mode_metric_parts:
                fd_metrics = summarize_metrics(pd.concat(mode_metric_parts, ignore_index=True))
                fd_all = fd_metrics[fd_metrics["sensor"] == "ALL"].iloc[0]
                print(
                    f"  condition-matched MAE={fd_all['mae']:.6f} "
                    f"RMSE={fd_all['rmse']:.6f} n={int(fd_all['n'])}",
                    flush=True,
                )

    if not wrote_window_header or not metric_prediction_parts:
        raise RuntimeError("No forecasts were produced.")

    anomaly_scores = pd.concat(all_anomaly_scores, ignore_index=True)
    metric_window_predictions = pd.concat(metric_prediction_parts, ignore_index=True)
    sensor_anomaly_scores = summarize_sensor_anomaly_scores(metric_window_predictions)
    metrics = summarize_metrics(metric_window_predictions)
    window_metrics = summarize_metrics(metric_window_predictions)
    metric_window_predictions.to_csv(args.output_dir / "metric_window_forecasts.csv", index=False)
    anomaly_scores.to_csv(args.output_dir / "anomaly_scores.csv", index=False)
    sensor_anomaly_scores.to_csv(args.output_dir / "sensor_anomaly_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    window_metrics.to_csv(args.output_dir / "window_metrics.csv", index=False)
    save_stats(stats_by_fd, args.output_dir)
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            vars(args)
            | {
                "selected_sensors": SELECTED_SENSORS,
                "condition_group_forecasting": True,
                "operating_condition_source": "past_context_only",
                "condition_forecast_policy": "full_horizon_per_historical_condition",
                "future_operating_condition_used": False,
                "future_operating_condition_used_for_forecasting": False,
                "future_operating_condition_used_for_retrospective_metrics": True,
                "metric_condition_policy": "realized_future_condition_match",
                "streamed_forecast_output": True,
                "total_prediction_rows": total_prediction_rows,
            },
            f,
            indent=2,
            default=str,
        )

    print("\nOverall metrics by FD:", flush=True)
    print(metrics[metrics["sensor"] == "ALL"].to_string(index=False), flush=True)
    print(f"\nSaved outputs to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


RECOMMENDATION_ACTIONS = [
    "regular_maintenance",
    "schedule_component_inspection",
    "targeted_component_diagnostic",
    "schedule_component_replacement",
    "immediate_replace_component",
]
ACTION_PRIORITY = {
    "immediate_replace_component": 0,
    "schedule_component_replacement": 1,
    "targeted_component_diagnostic": 2,
    "schedule_component_inspection": 3,
    "regular_maintenance": 4,
}


@dataclass
class PreparedVehicle:
    vehicle_id: str
    sensor_values: np.ndarray
    time_values: np.ndarray
    spec_ids: np.ndarray
    study_end: float
    has_repair: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_tables(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ops = pd.read_csv(dataset_dir / "train_operational_readouts.csv")
    train_specs = pd.read_csv(dataset_dir / "train_specifications.csv")
    train_tte = pd.read_csv(dataset_dir / "train_tte.csv")
    maintenance_log = pd.read_csv(dataset_dir / "synthetic_maintenance_log.csv")
    return train_ops, train_specs, train_tte, maintenance_log


def build_spec_vocab(train_specs: pd.DataFrame) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    spec_cols = [c for c in train_specs.columns if c != "vehicle_id"]
    mapping: Dict[str, Dict[str, int]] = {}
    for col in spec_cols:
        values = sorted(str(v) for v in train_specs[col].dropna().unique())
        mapping[col] = {value: idx for idx, value in enumerate(values)}
    return mapping, spec_cols


def compute_sensor_stats(train_ops: pd.DataFrame, sensor_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in sensor_cols:
        mean = float(train_ops[col].mean())
        std = float(train_ops[col].std())
        if np.isnan(mean):
            mean = 0.0
        if np.isnan(std) or std <= 1e-8:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def normalize_sensor_frame(frame: pd.DataFrame, sensor_cols: List[str], stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    filled = frame[sensor_cols].copy()
    for col in sensor_cols:
        filled[col] = filled[col].fillna(stats[col]["mean"])
    values = filled.to_numpy(dtype=np.float32)
    means = np.asarray([stats[c]["mean"] for c in sensor_cols], dtype=np.float32)
    stds = np.asarray([stats[c]["std"] for c in sensor_cols], dtype=np.float32)
    normalized = (values - means[None, :]) / stds[None, :]
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def build_time_features(time_steps: np.ndarray, study_end: float) -> np.ndarray:
    time_steps = time_steps.astype(np.float32)
    horizon = max(float(study_end), 1.0)
    normalized_time = time_steps / horizon
    delta = np.diff(time_steps, prepend=time_steps[:1]).astype(np.float32)
    delta_scale = max(float(np.abs(delta).mean()), 1.0)
    normalized_delta = delta / delta_scale
    return np.stack([normalized_time, normalized_delta], axis=1).astype(np.float32)


def build_vehicle_store(
    train_ops: pd.DataFrame,
    train_specs: pd.DataFrame,
    train_tte: pd.DataFrame,
    spec_vocab: Dict[str, Dict[str, int]],
    spec_cols: List[str],
    sensor_stats: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    sensor_cols = [c for c in train_ops.columns if c not in {"vehicle_id", "time_step"}]
    spec_lookup = train_specs.set_index("vehicle_id")
    tte_lookup = train_tte.set_index("vehicle_id")

    vehicle_store: Dict[str, Dict[str, Any]] = {}
    for vehicle_id, group in train_ops.groupby("vehicle_id", sort=True):
        group = group.sort_values("time_step").reset_index(drop=True)
        tte_row = tte_lookup.loc[vehicle_id]
        study_end = float(tte_row["length_of_study_time_step"])
        spec_row = spec_lookup.loc[vehicle_id]
        spec_ids = np.asarray(
            [spec_vocab[col][str(spec_row[col])] for col in spec_cols],
            dtype=np.int64,
        )
        time_steps = group["time_step"].to_numpy(dtype=np.float32)
        vehicle_store[str(int(vehicle_id))] = {
            "sensor_values": normalize_sensor_frame(group, sensor_cols, sensor_stats),
            "time_values": build_time_features(time_steps, study_end),
            "time_steps": time_steps,
            "spec_ids": spec_ids,
            "study_end": study_end,
            "has_repair": int(tte_row["in_study_repair"]),
        }
    return vehicle_store, sensor_cols


def build_action_vocab(maintenance_log: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    actions = sorted(str(v) for v in maintenance_log["action_taken"].dropna().unique())
    action_to_id = {name: idx for idx, name in enumerate(actions)}
    id_to_action = {idx: name for name, idx in action_to_id.items()}
    return action_to_id, id_to_action


def filter_action_supervision_log(maintenance_log: pd.DataFrame) -> pd.DataFrame:
    mask = maintenance_log["action_taken"].astype(str).isin(RECOMMENDATION_ACTIONS)
    if "maintenance_note" in maintenance_log.columns:
        recommendation_note = maintenance_log["maintenance_note"].fillna("").astype(str).str.startswith("Recommendation:")
        mask = mask | recommendation_note
    filtered = maintenance_log[mask].copy()
    return filtered.sort_values(["vehicle_id", "event_time_step", "event_id"]).reset_index(drop=True)


def format_sensor_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.2f}"
    return str(value)


def make_all_sensor_snapshot(row: pd.Series, sensor_cols: List[str]) -> str:
    return "|".join(f"{col}={format_sensor_value(row[col])}" for col in sensor_cols)


def build_sensor_snapshot_lookup(
    train_ops: pd.DataFrame,
    sensor_cols: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for vehicle_id, group in train_ops.groupby("vehicle_id", sort=True):
        group = group.sort_values("time_step").reset_index(drop=True)
        rows: List[Dict[str, Any]] = []
        for idx, row in group.iterrows():
            rows.append(
                {
                    "row_pos": int(idx),
                    "time_step": float(row["time_step"]),
                    "snapshot": make_all_sensor_snapshot(row, sensor_cols),
                }
            )
        lookup[str(int(vehicle_id))] = rows
    return lookup


def build_base_sample_index(
    vehicle_store: Dict[str, Dict[str, Any]],
    window_len: int,
    stride: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for vehicle_id, item in vehicle_store.items():
        time_steps = item["time_steps"]
        study_end = float(item["study_end"])
        for endpoint_pos in range(0, len(time_steps), stride):
            endpoint_time = float(time_steps[endpoint_pos])
            rul = max(study_end - endpoint_time, 0.0)
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "split": "train",
                    "endpoint_pos": int(endpoint_pos),
                    "endpoint_time": endpoint_time,
                    "rul": rul,
                    "action_id": -100,
                    "action_name": "",
                    "has_action_label": 0,
                    "sample_type": "rul_only",
                    "event_id": "",
                    "label_time": endpoint_time,
                }
            )
    return pd.DataFrame(rows)


def match_action_row_position(
    row_candidates: List[Dict[str, Any]],
    event_time_step: float,
    all_sensor_readings: Optional[str],
) -> Optional[int]:
    eligible = [item for item in row_candidates if float(item["time_step"]) <= float(event_time_step)]
    if not eligible:
        return None
    if all_sensor_readings and isinstance(all_sensor_readings, str):
        matching = [item for item in eligible if item["snapshot"] == all_sensor_readings]
        if matching:
            return int(matching[-1]["row_pos"])
    return int(eligible[-1]["row_pos"])


def build_action_supervision_index(
    vehicle_store: Dict[str, Dict[str, Any]],
    supervision_log: pd.DataFrame,
    snapshot_lookup: Dict[str, List[Dict[str, Any]]],
    action_to_id: Dict[str, int],
    id_to_action: Dict[int, str],
    window_len: int,
    action_train_ratio: float,
) -> pd.DataFrame:
    ordered_log = supervision_log.sort_values(["vehicle_id", "event_time_step", "event_id"]).reset_index(drop=True)
    action_supervised_count = int(len(ordered_log) * action_train_ratio)
    selected_log = ordered_log.iloc[:action_supervised_count].copy()

    rows: List[Dict[str, Any]] = []
    for log_row in selected_log.itertuples(index=False):
        vehicle_id = str(int(log_row.vehicle_id))
        if vehicle_id not in vehicle_store:
            continue
        row_candidates = snapshot_lookup.get(vehicle_id, [])
        matched_row_pos = match_action_row_position(
            row_candidates=row_candidates,
            event_time_step=float(log_row.event_time_step),
            all_sensor_readings=getattr(log_row, "all_sensor_readings", None),
        )
        if matched_row_pos is None:
            continue
        endpoint_pos = int(matched_row_pos)

        item = vehicle_store[vehicle_id]
        time_steps = item["time_steps"]
        if endpoint_pos >= len(time_steps):
            continue
        endpoint_time = float(time_steps[endpoint_pos])
        rul = max(float(item["study_end"]) - endpoint_time, 0.0)
        action_name = str(log_row.action_taken)
        action_id = int(action_to_id[action_name])
        rows.append(
            {
                "vehicle_id": vehicle_id,
                "split": "train",
                "endpoint_pos": int(endpoint_pos),
                "endpoint_time": endpoint_time,
                "rul": rul,
                "action_id": action_id,
                "action_name": id_to_action[action_id],
                "has_action_label": 1,
                "sample_type": "action_supervised",
                "event_id": str(getattr(log_row, "event_id", "")),
                "label_time": float(log_row.event_time_step),
            }
        )
    return pd.DataFrame(rows)


def apply_action_supervision(
    base_index: pd.DataFrame,
    action_index: pd.DataFrame,
) -> pd.DataFrame:
    sample_index = base_index.copy()
    if action_index.empty:
        return sample_index

    row_lookup = {
        (str(row.vehicle_id), int(row.endpoint_pos)): idx
        for idx, row in sample_index.iterrows()
    }
    best_rows: Dict[Tuple[str, int], Any] = {}
    for row in action_index.itertuples(index=False):
        key = (str(row.vehicle_id), int(row.endpoint_pos))
        current = best_rows.get(key)
        if current is None:
            best_rows[key] = row
            continue
        current_priority = ACTION_PRIORITY.get(str(current.action_name), 10)
        next_priority = ACTION_PRIORITY.get(str(row.action_name), 10)
        if next_priority < current_priority:
            best_rows[key] = row

    for row in best_rows.values():
        key = (str(row.vehicle_id), int(row.endpoint_pos))
        if key not in row_lookup:
            continue
        target_idx = row_lookup[key]
        sample_index.at[target_idx, "action_id"] = int(row.action_id)
        sample_index.at[target_idx, "action_name"] = str(row.action_name)
        sample_index.at[target_idx, "has_action_label"] = 1
        sample_index.at[target_idx, "sample_type"] = "action_supervised"
        sample_index.at[target_idx, "event_id"] = str(row.event_id)
        sample_index.at[target_idx, "label_time"] = float(row.label_time)
    return sample_index


def build_domain_prompt(sensor_cols: List[str], spec_cols: List[str], action_names: List[str]) -> str:
    action_text = ", ".join(action_names)
    return (
        "You are assisting a predictive maintenance task for commercial vehicles. "
        f"The input contains a vehicle specification profile with {len(spec_cols)} categorical fields and "
        f"a current multivariate sensor observation with {len(sensor_cols)} sensor channels and optional recent context. "
        "Use the current observation and limited operating context to estimate remaining useful life (RUL) and select the most appropriate maintenance recommendation. "
        f"Valid maintenance recommendations are: {action_text}. "
        "Return internal representations useful for regression and recommendation classification."
    )


def save_outputs(
    output_dir: Path,
    vehicle_store: Dict[str, Dict[str, Any]],
    sample_index: pd.DataFrame,
    sensor_cols: List[str],
    spec_cols: List[str],
    spec_vocab: Dict[str, Dict[str, int]],
    sensor_stats: Dict[str, Dict[str, float]],
    action_to_id: Dict[str, int],
    id_to_action: Dict[int, str],
    domain_prompt: str,
    window_len: int,
    stride: int,
    num_missing_sensor_values_filled: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "vehicle_store.pkl", "wb") as f:
        pickle.dump(vehicle_store, f)
    sample_index.to_csv(output_dir / "sample_index.csv", index=False)

    feature_schema = {
        "window_len": int(window_len),
        "stride": int(stride),
        "sensor_feature_names": sensor_cols,
        "time_feature_names": ["normalized_time", "normalized_delta"],
        "spec_feature_names": spec_cols,
        "spec_vocab": spec_vocab,
        "sensor_stats": sensor_stats,
        "action_to_id": action_to_id,
        "id_to_action": id_to_action,
        "domain_prompt": domain_prompt,
    }
    with open(output_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2, ensure_ascii=False)

    summary = {
        "num_vehicles": len(vehicle_store),
        "num_samples": int(len(sample_index)),
        "num_train_samples": int((sample_index["split"] == "train").sum()),
        "num_test_samples": int((sample_index["split"] == "test").sum()) if "test" in sample_index["split"].values else 0,
        "num_action_supervised_samples": int(sample_index["has_action_label"].sum()),
        "num_recommendation_actions": int((sample_index["sample_type"] == "action_supervised").sum()),
        "num_missing_sensor_values_filled": int(num_missing_sensor_values_filled),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare windowed predictive-maintenance samples from raw CSV files.")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="artifacts/qwen_predm")
    parser.add_argument("--window_len", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--action_train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    train_ops, train_specs, train_tte, maintenance_log = load_tables(dataset_dir)

    spec_vocab, spec_cols = build_spec_vocab(train_specs)
    sensor_cols = [c for c in train_ops.columns if c not in {"vehicle_id", "time_step"}]
    num_missing_sensor_values_filled = int(train_ops[sensor_cols].isna().sum().sum())
    sensor_stats = compute_sensor_stats(train_ops, sensor_cols)
    vehicle_store, sensor_cols = build_vehicle_store(
        train_ops=train_ops,
        train_specs=train_specs,
        train_tte=train_tte,
        spec_vocab=spec_vocab,
        spec_cols=spec_cols,
        sensor_stats=sensor_stats,
    )

    supervision_log = filter_action_supervision_log(maintenance_log)
    action_to_id, id_to_action = build_action_vocab(supervision_log)
    snapshot_lookup = build_sensor_snapshot_lookup(train_ops, sensor_cols)
    base_index = build_base_sample_index(
        vehicle_store=vehicle_store,
        window_len=args.window_len,
        stride=args.stride,
    )
    action_index = build_action_supervision_index(
        vehicle_store=vehicle_store,
        supervision_log=supervision_log,
        snapshot_lookup=snapshot_lookup,
        action_to_id=action_to_id,
        id_to_action=id_to_action,
        window_len=args.window_len,
        action_train_ratio=args.action_train_ratio,
    )
    sample_index = apply_action_supervision(base_index, action_index)
    domain_prompt = build_domain_prompt(sensor_cols, spec_cols, [id_to_action[i] for i in sorted(id_to_action)])
    save_outputs(
        output_dir=output_dir,
        vehicle_store=vehicle_store,
        sample_index=sample_index,
        sensor_cols=sensor_cols,
        spec_cols=spec_cols,
        spec_vocab=spec_vocab,
        sensor_stats=sensor_stats,
        action_to_id=action_to_id,
        id_to_action=id_to_action,
        domain_prompt=domain_prompt,
        window_len=args.window_len,
        stride=args.stride,
        num_missing_sensor_values_filled=num_missing_sensor_values_filled,
    )

    print(json.dumps({"output_dir": str(output_dir), "num_samples": int(len(sample_index))}, indent=2))


if __name__ == "__main__":
    main()

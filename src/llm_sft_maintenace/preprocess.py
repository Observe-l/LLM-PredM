from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PAIR_PATTERN = re.compile(r"([^=|]+)=([^|]*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare maintenance SFT artifacts from train/validation/test logs.")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="artifacts/maintenance_sft")
    return parser.parse_args()


def load_split_tables(dataset_dir: Path) -> Dict[str, pd.DataFrame]:
    split_files = {
        "train": "synthetic_maintenance_log_train.csv",
        "validation": "synthetic_maintenance_log_validation.csv",
        "test": "synthetic_maintenance_log_test.csv",
    }
    tables: Dict[str, pd.DataFrame] = {}
    for split, filename in split_files.items():
        path = dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required split file: {path}")
        tables[split] = pd.read_csv(path)
    return tables


def parse_key_value_string(value: Any) -> Dict[str, float]:
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    output: Dict[str, float] = {}
    for key, raw_value in PAIR_PATTERN.findall(text):
        cleaned = raw_value.strip()
        if cleaned.upper() == "NA" or cleaned == "":
            output[key.strip()] = np.nan
            continue
        if cleaned.startswith("Cat"):
            output[key.strip()] = float(cleaned[3:])
        else:
            output[key.strip()] = float(cleaned)
    return output


def collect_feature_names(tables: Dict[str, pd.DataFrame]) -> Tuple[List[str], List[str]]:
    spec_names = set()
    sensor_names = set()
    for table in tables.values():
        for value in table["spec_profile"]:
            spec_names.update(parse_key_value_string(value).keys())
        for value in table["sensor_readings"]:
            sensor_names.update(parse_key_value_string(value).keys())
    return sorted(spec_names), sorted(sensor_names)


def build_label_vocab(tables: Dict[str, pd.DataFrame], column: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    names = sorted({str(v) for table in tables.values() for v in table[column].dropna().unique()})
    name_to_id = {name: idx for idx, name in enumerate(names)}
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    return name_to_id, id_to_name


def build_numeric_stats(train_frame: pd.DataFrame, numeric_feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for name in numeric_feature_names:
        series = train_frame[name].astype(float)
        mean = float(series.mean()) if len(series) > 0 else 0.0
        std = float(series.std()) if len(series) > 0 else 1.0
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std <= 1e-8:
            std = 1.0
        stats[name] = {"mean": mean, "std": std}
    return stats


def build_domain_prompt(action_names: List[str], priority_names: List[str]) -> str:
    action_text = ", ".join(action_names)
    priority_text = ", ".join(priority_names)
    return (
        "You are assisting a maintenance work-order recommendation task for commercial vehicles. "
        "Given event metadata, maintenance notes, and structured numeric observations, estimate the correct maintenance action and action priority. "
        f"Valid actions are: {action_text}. "
        f"Valid priorities are: {priority_text}. "
        "Use the domain context, event identifiers, component information, predicted RUL, specification profile, and sensor readings."
    )


def prepare_split_records(
    split: str,
    table: pd.DataFrame,
    spec_feature_names: List[str],
    sensor_feature_names: List[str],
    numeric_stats: Dict[str, Dict[str, float]],
    action_to_id: Dict[str, int],
    priority_to_id: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    numeric_feature_names = [
        *spec_feature_names,
        "event_time_step",
        "predicted_rul",
        *sensor_feature_names,
    ]
    for row_idx, row in enumerate(table.itertuples(index=False)):
        spec_map = parse_key_value_string(getattr(row, "spec_profile"))
        sensor_map = parse_key_value_string(getattr(row, "sensor_readings"))
        numeric_raw: Dict[str, float] = {}
        for name in spec_feature_names:
            numeric_raw[name] = float(spec_map.get(name, np.nan))
        numeric_raw["event_time_step"] = float(getattr(row, "event_time_step"))
        numeric_raw["predicted_rul"] = float(getattr(row, "predicted_rul"))
        for name in sensor_feature_names:
            numeric_raw[name] = float(sensor_map.get(name, np.nan))

        numeric_values: List[float] = []
        for name in numeric_feature_names:
            stats = numeric_stats[name]
            value = numeric_raw[name]
            if not np.isfinite(value):
                value = stats["mean"]
            standardized = (value - stats["mean"]) / stats["std"]
            numeric_values.append(float(standardized))

        record = {
            "split": split,
            "event_id": str(getattr(row, "event_id")),
            "case_id": str(getattr(row, "case_id")),
            "work_order_id": str(getattr(row, "work_order_id")),
            "vehicle_id": str(getattr(row, "vehicle_id")),
            "system": str(getattr(row, "system")),
            "subsystem": str(getattr(row, "subsystem")),
            "component": str(getattr(row, "component")),
            "maintenance_note": str(getattr(row, "maintenance_note")),
            "numeric_values": np.asarray(numeric_values, dtype=np.float32),
            "action_label": int(action_to_id[str(getattr(row, "action_taken"))]),
            "priority_label": int(priority_to_id[str(getattr(row, "action_priority"))]),
        }
        records.append(record)
        index_rows.append(
            {
                "split": split,
                "row_idx": row_idx,
                "event_id": record["event_id"],
                "case_id": record["case_id"],
                "work_order_id": record["work_order_id"],
                "vehicle_id": record["vehicle_id"],
                "action_taken": str(getattr(row, "action_taken")),
                "action_priority": str(getattr(row, "action_priority")),
            }
        )
    return records, index_rows


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_split_tables(dataset_dir)
    spec_feature_names, sensor_feature_names = collect_feature_names(tables)
    action_to_id, id_to_action = build_label_vocab(tables, "action_taken")
    priority_to_id, id_to_priority = build_label_vocab(tables, "action_priority")

    numeric_feature_names = [
        *spec_feature_names,
        "event_time_step",
        "predicted_rul",
        *sensor_feature_names,
    ]

    train_numeric_rows: List[Dict[str, float]] = []
    for row in tables["train"].itertuples(index=False):
        spec_map = parse_key_value_string(getattr(row, "spec_profile"))
        sensor_map = parse_key_value_string(getattr(row, "sensor_readings"))
        entry: Dict[str, float] = {}
        for name in spec_feature_names:
            entry[name] = float(spec_map.get(name, np.nan))
        entry["event_time_step"] = float(getattr(row, "event_time_step"))
        entry["predicted_rul"] = float(getattr(row, "predicted_rul"))
        for name in sensor_feature_names:
            entry[name] = float(sensor_map.get(name, np.nan))
        train_numeric_rows.append(entry)
    train_numeric_frame = pd.DataFrame(train_numeric_rows, columns=numeric_feature_names)
    numeric_stats = build_numeric_stats(train_numeric_frame, numeric_feature_names)

    all_records: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    split_counts: Dict[str, int] = {}
    for split, table in tables.items():
        records, rows = prepare_split_records(
            split=split,
            table=table,
            spec_feature_names=spec_feature_names,
            sensor_feature_names=sensor_feature_names,
            numeric_stats=numeric_stats,
            action_to_id=action_to_id,
            priority_to_id=priority_to_id,
        )
        all_records.extend(records)
        index_rows.extend(rows)
        split_counts[split] = len(records)

    with open(output_dir / "records.pkl", "wb") as f:
        pickle.dump(all_records, f)
    pd.DataFrame(index_rows).to_csv(output_dir / "sample_index.csv", index=False)

    feature_schema = {
        "text_fields": [
            "event_id",
            "case_id",
            "work_order_id",
            "vehicle_id",
            "system",
            "subsystem",
            "component",
            "maintenance_note",
        ],
        "numeric_feature_names": numeric_feature_names,
        "spec_feature_names": spec_feature_names,
        "sensor_feature_names": sensor_feature_names,
        "action_to_id": action_to_id,
        "id_to_action": id_to_action,
        "priority_to_id": priority_to_id,
        "id_to_priority": id_to_priority,
        "numeric_stats": numeric_stats,
        "domain_prompt": build_domain_prompt(
            [id_to_action[idx] for idx in sorted(id_to_action)],
            [id_to_priority[idx] for idx in sorted(id_to_priority)],
        ),
    }
    with open(output_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2, ensure_ascii=False)

    summary = {
        "num_train_samples": split_counts.get("train", 0),
        "num_validation_samples": split_counts.get("validation", 0),
        "num_test_samples": split_counts.get("test", 0),
        "num_numeric_features": len(numeric_feature_names),
        "num_action_classes": len(action_to_id),
        "num_priority_classes": len(priority_to_id),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()

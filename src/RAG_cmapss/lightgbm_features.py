from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .reflection_memory import REFLECTION_COLUMNS


POSITIVE_RISK_LABELS = {
    "correct_maintenance",
    "missed_HPC_maintenance",
    "missed_fan_maintenance",
    "missed_maintenance_unknown",
}
NEGATIVE_RISK_LABELS = {"too_early", "over_maintenance"}

NUMERIC_FEATURES = [
    "peak_score",
    "final_score",
    "unit_past_q95",
    "unit_past_q99",
    "peak_minus_unit_q95",
    "peak_minus_unit_q99",
    "duration_above_unit_q95",
    "duration_above_unit_q99",
    "area_above_unit_q95",
    "area_above_unit_q99",
    "persistent_high_risk_duration",
    "slope",
    "delta_score",
    "monotonicity",
    "volatility",
    "peak_lhi",
    "peak_d_rmse",
    "lhi_slope",
    "d_rmse_slope",
    "hpc_sensor_presence_ratio",
    "fan_sensor_presence_ratio",
    "conflict_sensor_presence_ratio",
    "sensor_pattern_stability",
    "hpc_path_score",
    "fan_path_score",
    "uncertain_path_score",
    "component_conflict_score",
    "dominance_margin",
    "action_to_peak_gap",
    "action_to_warning_gap",
    "action_to_persistence_gap",
    "similar_correct_count",
    "similar_too_early_count",
    "similar_missed_count",
    "similar_over_maintenance_count",
    "similar_wrong_component_count",
    "max_retrieval_similarity",
    "mean_retrieval_similarity",
    "reflection_supports_maintenance",
    "reflection_warns_too_early",
    "risk_gate_statistical_candidate",
    "risk_gate_maintenance_candidate",
    "component_gate_supported",
]

CATEGORICAL_FEATURES = [
    "dataset_subset_code",
    "candidate_action_code",
    "previous_action_type_code",
    "dominant_component_code",
    "d_rmse_lhi_consistency_code",
    "feedback_label_code",
]

FEATURE_COLUMNS = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]

# Risk prediction runs before an action has been executed and before feedback exists.
# Keep this schema limited to features that are available both at online inference
# time and when rebuilding the same forecast state from reflection rows.
RISK_EXCLUDED_FEATURES = {
    "action_to_peak_gap",
    "action_to_warning_gap",
    "action_to_persistence_gap",
    "previous_action_type_code",
    "feedback_label_code",
}
RISK_FEATURE_COLUMNS = [name for name in FEATURE_COLUMNS if name not in RISK_EXCLUDED_FEATURES]

ACTION_CODES = {
    "": 0,
    "continue_normal_operation": 1,
    "schedule_monitoring": 2,
    "schedule_HPC_maintenance": 3,
    "schedule_fan_maintenance": 4,
}
COMPONENT_CODES = {
    "": 0,
    "none": 0,
    "HPC_related_degradation": 1,
    "Fan_related_degradation": 2,
    "uncertain_component_degradation": 3,
}
CONSISTENCY_CODES = {
    "": 0,
    "unknown": 0,
    "mixed": 1,
    "consistent_increasing": 2,
    "consistent_decreasing": 3,
}
FEEDBACK_CODES = {
    "": 0,
    "correct_maintenance": 1,
    "too_early": 2,
    "over_maintenance": 3,
    "missed_HPC_maintenance": 4,
    "missed_fan_maintenance": 5,
    "missed_maintenance_unknown": 6,
    "wrong_component": 7,
}


def extract_lightgbm_features(
    case: dict[str, Any],
    context: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    feedback: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Flatten a ForecastCase plus prepared RAG context into tabular features."""
    context = context or {}
    summary = case.get("forecast_summary", {})
    risk = case.get("risk_statistics", {})
    trend = case.get("trend_statistics", {})
    multi = case.get("multi_score_statistics", {})
    sensor = case.get("sensor_evidence_statistics", {})
    component_stats = context.get("component_evidence_statistics") or case.get("component_evidence_statistics", {})
    risk_gate = context.get("risk_gate", {})
    component_gate = context.get("component_gate", {})
    reflection_stats = summarize_reflection_rules(context.get("reflection_rules", []))

    lhi_name = case.get("score_source", {}).get("lhi_name", "lhi_rmse_roll_mean")
    raw_name = case.get("score_source", {}).get("raw_score_name", "d_rmse")
    lhi_stats = multi.get("lhi_rmse_roll_mean") or multi.get(lhi_name, {})
    raw_stats = multi.get("d_rmse") or multi.get(raw_name, {})

    candidate_action = str(context.get("candidate_action") or "")
    previous_action = str((action or {}).get("action_type") or "")
    feedback_label = str((feedback or {}).get("feedback_label") or "")
    dominant_component = str(component_stats.get("dominant_component") or "")
    consistency = str(multi.get("d_rmse_lhi_consistency") or "")

    features: dict[str, float | int] = {
        "peak_score": _num(risk.get("peak_score") or summary.get("peak_score")),
        "final_score": _num(risk.get("final_score") or summary.get("final_cycle_score")),
        "unit_past_q95": _num(risk.get("unit_past_q95")),
        "unit_past_q99": _num(risk.get("unit_past_q99")),
        "peak_minus_unit_q95": _num(risk.get("peak_minus_unit_q95")),
        "peak_minus_unit_q99": _num(risk.get("peak_minus_unit_q99")),
        "duration_above_unit_q95": _num(trend.get("duration_above_unit_q95")),
        "duration_above_unit_q99": _num(trend.get("duration_above_unit_q99")),
        "area_above_unit_q95": _num(trend.get("area_above_unit_q95")),
        "area_above_unit_q99": _num(trend.get("area_above_unit_q99")),
        "persistent_high_risk_duration": _num(summary.get("persistent_high_risk_duration")),
        "slope": _num(trend.get("slope")),
        "delta_score": _num(trend.get("delta_score")),
        "monotonicity": _num(trend.get("monotonicity")),
        "volatility": _num(trend.get("volatility")),
        "peak_lhi": _num(lhi_stats.get("peak") or risk.get("peak_score") or summary.get("peak_score")),
        "peak_d_rmse": _num(raw_stats.get("peak") or risk.get("peak_d_rmse")),
        "lhi_slope": _num(lhi_stats.get("slope") or trend.get("slope")),
        "d_rmse_slope": _num(raw_stats.get("slope")),
        "hpc_sensor_presence_ratio": _num(sensor.get("hpc_sensor_presence_ratio")),
        "fan_sensor_presence_ratio": _num(sensor.get("fan_sensor_presence_ratio")),
        "conflict_sensor_presence_ratio": _num(sensor.get("conflict_sensor_presence_ratio")),
        "sensor_pattern_stability": _num(sensor.get("sensor_pattern_stability")),
        "hpc_path_score": _num(component_stats.get("hpc_path_score")),
        "fan_path_score": _num(component_stats.get("fan_path_score")),
        "uncertain_path_score": _num(component_stats.get("uncertain_path_score")),
        "component_conflict_score": _num(component_stats.get("component_conflict_score")),
        "dominance_margin": _num(component_stats.get("dominance_margin")),
        "action_to_peak_gap": _num(_cycle_gap((action or {}).get("action_time"), summary.get("peak_score_cycle"))),
        "action_to_warning_gap": _num(_cycle_gap((action or {}).get("action_time"), summary.get("first_warning_crossing_cycle"))),
        "action_to_persistence_gap": _num(_cycle_gap((action or {}).get("action_time"), summary.get("first_persistent_pattern_cycle"))),
        "risk_gate_statistical_candidate": float(bool(risk_gate.get("statistical_candidate"))),
        "risk_gate_maintenance_candidate": float(bool(risk_gate.get("maintenance_candidate"))),
        "component_gate_supported": float(bool(component_gate.get("component_supported"))),
        "dataset_subset_code": _dataset_code(case.get("dataset_subset")),
        "candidate_action_code": ACTION_CODES.get(candidate_action, 0),
        "previous_action_type_code": ACTION_CODES.get(previous_action, 0),
        "dominant_component_code": COMPONENT_CODES.get(dominant_component, 0),
        "d_rmse_lhi_consistency_code": CONSISTENCY_CODES.get(consistency, 0),
        "feedback_label_code": FEEDBACK_CODES.get(feedback_label, 0),
        **reflection_stats,
    }
    return {name: features.get(name, 0.0) for name in FEATURE_COLUMNS}


def feature_frame(rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or FEATURE_COLUMNS
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in frame:
            frame[col] = 0.0
    return frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def read_reflection_training_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != REFLECTION_COLUMNS:
            raise ValueError(f"Unsupported reflection schema in {path}: {reader.fieldnames}")
        return [row for row in reader if any(str(value).strip() for value in row.values())]


def training_features_from_reflection_rows(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_rows = [_features_from_reflection_row(row) for row in rows]
    return feature_frame(feature_rows, columns=columns)


def risk_labels_from_reflection_rows(rows: list[dict[str, Any]]) -> list[int]:
    labels: list[int] = []
    for row in rows:
        label = str(row.get("feedback_label", ""))
        if label in POSITIVE_RISK_LABELS:
            labels.append(1)
        elif label in NEGATIVE_RISK_LABELS:
            labels.append(0)
    return labels


def filter_risk_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("feedback_label", "")) in POSITIVE_RISK_LABELS | NEGATIVE_RISK_LABELS]


def update_labels_from_reflection_rows(rows: list[dict[str, Any]], target: str) -> list[str]:
    labels = []
    for row in rows:
        value = str(row.get(target, "")).strip()
        if value:
            labels.append(value)
    return labels


def filter_update_training_rows(rows: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get(target, "")).strip()]


def summarize_reflection_rules(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = Counter(str(row.get("feedback_label", "")) for row in rows)
    similarities = [_num(row.get("retrieval_similarity"), default=math.nan) for row in rows]
    similarities = [value for value in similarities if math.isfinite(value)]
    supports = labels.get("correct_maintenance", 0) + labels.get("missed_HPC_maintenance", 0) + labels.get("missed_fan_maintenance", 0)
    warns = labels.get("too_early", 0) + labels.get("over_maintenance", 0)
    return {
        "similar_correct_count": float(labels.get("correct_maintenance", 0)),
        "similar_too_early_count": float(labels.get("too_early", 0)),
        "similar_missed_count": float(labels.get("missed_HPC_maintenance", 0) + labels.get("missed_fan_maintenance", 0)),
        "similar_over_maintenance_count": float(labels.get("over_maintenance", 0)),
        "similar_wrong_component_count": float(labels.get("wrong_component", 0)),
        "max_retrieval_similarity": max(similarities) if similarities else 0.0,
        "mean_retrieval_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
        "reflection_supports_maintenance": float(supports > warns and supports > 0),
        "reflection_warns_too_early": float(warns > 0 and warns >= supports),
    }


def top_feature_names(features: dict[str, Any], limit: int = 5) -> list[str]:
    ignore = {name for name in CATEGORICAL_FEATURES}
    scored = []
    for key, value in features.items():
        if key in ignore:
            continue
        number = _num(value)
        if number != 0.0:
            scored.append((abs(number), key))
    return [name for _, name in sorted(scored, reverse=True)[:limit]]


def _features_from_reflection_row(row: dict[str, Any]) -> dict[str, Any]:
    feedback_label = str(row.get("feedback_label", ""))
    previous_action = str(row.get("previous_action_type", ""))
    dominant_component = str(row.get("dominant_component", ""))
    consistency = str(row.get("d_rmse_lhi_consistency", ""))
    base = {name: _num(row.get(name)) for name in NUMERIC_FEATURES}
    base.update(
        {
            "dataset_subset_code": _dataset_code(row.get("applies_to_dataset")),
            "candidate_action_code": ACTION_CODES.get(previous_action, 0),
            "previous_action_type_code": ACTION_CODES.get(previous_action, 0),
            "dominant_component_code": COMPONENT_CODES.get(dominant_component, 0),
            "d_rmse_lhi_consistency_code": CONSISTENCY_CODES.get(consistency, 0),
            "feedback_label_code": FEEDBACK_CODES.get(feedback_label, 0),
        }
    )
    return base


def _dataset_code(value: Any) -> int:
    text = str(value or "")
    if text.startswith("FD"):
        try:
            return int(text[2:])
        except ValueError:
            return 0
    return 0


def _cycle_gap(start: Any, end: Any) -> int | None:
    start_num = _parse_t_plus(start)
    end_num = _parse_t_plus(end)
    if start_num is None or end_num is None:
        return None
    return end_num - start_num


def _parse_t_plus(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    text = str(value)
    if not text.startswith("t+"):
        return None
    try:
        return int(text[2:])
    except ValueError:
        return None


def _num(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number

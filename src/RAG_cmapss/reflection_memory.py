from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .action_validator import STRONG_FAN_SENSORS, STRONG_HPC_SENSORS, parse_t_plus


REFLECTION_COLUMNS = [
    "rule_id",
    "applies_to_dataset",
    "if_pattern",
    "previous_action_type",
    "previous_action_time",
    "feedback_label",
    "feedback_reason",
    "peak_score",
    "peak_score_cycle",
    "score_at_action_time",
    "score_trend",
    "first_warning_crossing_cycle",
    "first_critical_crossing_cycle",
    "persistent_high_risk_duration",
    "first_persistent_pattern_cycle",
    "action_to_peak_gap",
    "action_to_warning_gap",
    "action_to_persistence_gap",
    "dominant_top_sensors",
    "top_sensor_at_action_time",
    "top_sensor_at_peak_cycle",
    "component_evidence_strength",
    "then_revise_action_type",
    "recommended_time_rule",
    "then_adjust_threshold",
    "then_adjust_component_preference",
]


def infer_forecast_pattern(forecast_summary: dict[str, Any]) -> str:
    sensors = [str(s) for s in forecast_summary.get("dominant_top_sensors", [])]
    persistent = forecast_summary.get("first_persistent_pattern_cycle") is not None

    if len(sensors) >= 2 and sensors[0] == "S7" and sensors[1] == "S11":
        return "Pattern_S7_S11_persistent_high_drift" if persistent else "Pattern_S7_S11_high_drift"
    if len(sensors) >= 2 and sensors[0] == "S8" and sensors[1] == "S13":
        return "Pattern_S8_S13_persistent_high_drift" if persistent else "Pattern_S8_S13_high_drift"
    if any(s in {"S4", "S12", "S20", "S21"} for s in sensors[:3]):
        return "Pattern_score_increasing_sensor_evidence_conflict"
    if forecast_summary.get("peak_score", 0) and not persistent:
        return "Pattern_high_score_without_persistent_top_sensor_pattern"
    sensor_part = "_".join(sensors[:3]) if sensors else "unknown"
    prefix = "persistent" if persistent else "nonpersistent"
    return f"{prefix}_{sensor_part}_high_drift"


def retrieve_reflection_rules(
    kg_dir: str | Path,
    dataset_subset: str,
    forecast_summary: dict[str, Any],
    candidate_action: str,
    max_rules: int = 5,
) -> list[dict[str, Any]]:
    path = Path(kg_dir) / "reflection_rules.csv"
    if not path.exists():
        return []

    rows = _read_reflection_rows(path)
    if not rows:
        return []

    pattern = infer_forecast_pattern(forecast_summary)
    query = _feature_vector_from_summary(forecast_summary)
    query_sensors = _as_sensor_set(forecast_summary.get("dominant_top_sensors", [])) | _as_sensor_set(
        forecast_summary.get("top_sensor_at_peak_cycle", [])
    )
    results: list[tuple[tuple[float, float, float, float], dict[str, Any]]] = []
    for row in rows:
        applies = str(row.get("applies_to_dataset", ""))
        rule_pattern = str(row.get("if_pattern", ""))

        if applies not in {dataset_subset, "ALL"}:
            continue
        pattern_match = rule_pattern == pattern or rule_pattern in pattern or pattern in rule_pattern
        peak_similarity = _peak_similarity(query, _feature_vector_from_row(row))
        persistence_similarity = _persistence_similarity(query, _feature_vector_from_row(row))
        sensor_similarity = _jaccard(query_sensors, _as_sensor_set(row.get("dominant_top_sensors", "")) | _as_sensor_set(row.get("top_sensor_at_peak_cycle", "")))
        rank_score = (
            10.0 * peak_similarity
            + 0.5 * persistence_similarity
            + 0.25 * sensor_similarity
            + 0.1 * float(pattern_match)
        )
        sort_key = (
            round(peak_similarity, 6),
            round(persistence_similarity, 6),
            round(sensor_similarity, 6),
            float(pattern_match),
        )
        compact = _compact_reflection_row(row)
        compact["retrieval_similarity"] = round(rank_score, 4)
        compact["peak_similarity"] = round(peak_similarity, 4)
        compact["persistence_similarity"] = round(persistence_similarity, 4)
        compact["sensor_similarity"] = round(sensor_similarity, 4)
        compact["matched_pattern"] = pattern_match
        results.append((sort_key, compact))

    return [row for _, row in sorted(results, key=lambda x: x[0], reverse=True)[:max_rules]]


def feedback_to_rule(feedback: dict[str, Any], case: dict[str, Any], action: dict[str, Any]) -> dict[str, str] | None:
    dataset = str(case["dataset_subset"])
    pattern = infer_forecast_pattern(case["forecast_summary"])
    previous_action = str(action.get("action_type", ""))
    label = str(feedback["feedback_label"])
    feedback_id = str(feedback.get("feedback_id") or feedback.get("case_id") or "unknown")
    features = reflection_features(case, action)

    base = {
        "applies_to_dataset": dataset,
        "if_pattern": pattern,
        "previous_action_type": previous_action,
        "feedback_label": label,
        "feedback_reason": _feedback_reason(label, features, previous_action, str(feedback.get("component_feedback", ""))),
        **features,
    }

    if label == "over_maintenance":
        return {
            **base,
            "rule_id": f"ReflectionRule_{dataset}_auto_over_{feedback_id}",
            "then_revise_action_type": "schedule_monitoring",
            "recommended_time_rule": "schedule_monitoring_at_peak_cycle",
            "then_adjust_threshold": "higher",
            "then_adjust_component_preference": "unchanged",
        }
    if label == "correct_maintenance":
        return {
            **base,
            "rule_id": f"ReflectionRule_{dataset}_auto_correct_{feedback_id}",
            "then_revise_action_type": previous_action,
            "recommended_time_rule": "keep_similar_timing",
            "then_adjust_threshold": "unchanged",
            "then_adjust_component_preference": _component_from_action(previous_action),
        }
    if label == "missed_HPC_maintenance":
        return {
            **base,
            "rule_id": f"ReflectionRule_{dataset}_auto_missed_hpc_{feedback_id}",
            "then_revise_action_type": "schedule_HPC_maintenance",
            "recommended_time_rule": _missed_time_rule(features),
            "then_adjust_threshold": "lower",
            "then_adjust_component_preference": "HPC",
        }
    if label == "missed_fan_maintenance":
        return {
            **base,
            "rule_id": f"ReflectionRule_{dataset}_auto_missed_fan_{feedback_id}",
            "then_revise_action_type": "schedule_fan_maintenance",
            "recommended_time_rule": _missed_time_rule(features),
            "then_adjust_threshold": "lower",
            "then_adjust_component_preference": "FAN",
        }
    if label == "missed_maintenance_unknown":
        return {
            **base,
            "rule_id": f"ReflectionRule_{dataset}_auto_missed_unknown_{feedback_id}",
            "then_revise_action_type": "schedule_monitoring",
            "recommended_time_rule": _missed_time_rule(features),
            "then_adjust_threshold": "lower",
            "then_adjust_component_preference": "unchanged",
        }
    if label == "wrong_component":
        return {
            **base,
            "rule_id": f"ReflectionRule_{dataset}_auto_wrong_component_{feedback_id}",
            "then_revise_action_type": "schedule_monitoring",
            "recommended_time_rule": "schedule_monitoring_at_peak_cycle",
            "then_adjust_threshold": "unchanged",
            "then_adjust_component_preference": "unchanged",
        }
    if label == "too_early":
        if str(feedback.get("component_feedback", "")).lower() == "correct" and _maintenance_still_supported(
            features, previous_action
        ):
            revise_action = previous_action
            time_rule = "peak_score_cycle_minus_margin"
            threshold = "unchanged"
            component = _component_from_action(previous_action)
        else:
            revise_action = "schedule_monitoring"
            time_rule = "schedule_monitoring_at_peak_cycle"
            threshold = "higher"
            component = "unchanged"
        return {
            **base,
            "rule_id": f"ReflectionRule_{dataset}_auto_too_early_{feedback_id}",
            "then_revise_action_type": revise_action,
            "recommended_time_rule": time_rule,
            "then_adjust_threshold": threshold,
            "then_adjust_component_preference": component,
        }
    return None


def _component_from_action(action_type: str) -> str:
    if action_type == "schedule_HPC_maintenance":
        return "HPC"
    if action_type == "schedule_fan_maintenance":
        return "FAN"
    return "unchanged"


def append_reflection_rule(kg_dir: str | Path, rule: dict[str, str]) -> Path:
    path = Path(kg_dir) / "reflection_rules.csv"
    if not path.exists() or path.stat().st_size == 0:
        initialize_reflection_file(path)
    else:
        _assert_reflection_schema(path)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REFLECTION_COLUMNS)
        writer.writerow({col: rule.get(col, "") for col in REFLECTION_COLUMNS})
    return path


def initialize_reflection_file(path: str | Path) -> None:
    path = Path(path)
    path.write_text(",".join(REFLECTION_COLUMNS) + "\n")


def reflection_features(case: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    summary = case.get("forecast_summary", {})
    peak = _to_float(summary.get("peak_score"))
    action_time = action.get("action_time")
    peak_cycle = str(summary.get("peak_score_cycle") or "")
    warning_cycle = summary.get("first_warning_crossing_cycle")
    persistence_cycle = summary.get("first_persistent_pattern_cycle")
    score_at_action, sensors_at_action = _score_and_sensors_at_relative_cycle(case, action_time)
    dominant_sensors = [str(s).upper() for s in summary.get("dominant_top_sensors", [])]
    peak_sensors = [str(s).upper() for s in summary.get("top_sensor_at_peak_cycle", [])]
    component_evidence = _component_evidence_strength(previous_action=str(action.get("action_type", "")), sensors=dominant_sensors + peak_sensors + sensors_at_action)
    return {
        "previous_action_time": _csv_value(action_time),
        "peak_score": _csv_value(peak),
        "peak_score_cycle": peak_cycle,
        "score_at_action_time": _csv_value(score_at_action),
        "score_trend": _csv_value(summary.get("score_trend")),
        "first_warning_crossing_cycle": _csv_value(warning_cycle),
        "first_critical_crossing_cycle": _csv_value(summary.get("first_critical_crossing_cycle")),
        "persistent_high_risk_duration": _csv_value(summary.get("persistent_high_risk_duration")),
        "first_persistent_pattern_cycle": _csv_value(persistence_cycle),
        "action_to_peak_gap": _csv_value(_cycle_gap(action_time, peak_cycle)),
        "action_to_warning_gap": _csv_value(_cycle_gap(action_time, warning_cycle)),
        "action_to_persistence_gap": _csv_value(_cycle_gap(action_time, persistence_cycle)),
        "dominant_top_sensors": "|".join(dominant_sensors),
        "top_sensor_at_action_time": "|".join(sensors_at_action),
        "top_sensor_at_peak_cycle": "|".join(peak_sensors),
        "component_evidence_strength": component_evidence,
    }


def _score_and_sensors_at_relative_cycle(case: dict[str, Any], action_time: Any) -> tuple[float | None, list[str]]:
    action_rel = _parse_rel_cycle(action_time)
    if action_rel is None:
        return None, []
    for item in case.get("key_cycles", []):
        if _parse_rel_cycle(item.get("relative_cycle")) == action_rel:
            return _to_float(item.get("score")), [str(s).upper() for s in item.get("top_sensors", [])]

    detail_path = case.get("forecast_window_detail_path")
    if detail_path and Path(str(detail_path)).exists():
        try:
            detail = json.loads(Path(str(detail_path)).read_text())
        except Exception:
            detail = {}
        for item in detail.get("score_series", []):
            if _parse_rel_cycle(item.get("relative_cycle")) == action_rel:
                return _to_float(item.get("score")), [str(s).upper() for s in item.get("top_drift_sensors", [])]
    return None, []


def _maintenance_still_supported(features: dict[str, Any], action_type: str) -> bool:
    peak = _to_float(features.get("peak_score")) or 0.0
    persistent = int(_to_float(features.get("persistent_high_risk_duration")) or 0)
    component_strength = str(features.get("component_evidence_strength", ""))
    action_to_peak_gap = _to_float(features.get("action_to_peak_gap"))
    later_peak = action_to_peak_gap is None or action_to_peak_gap >= 0
    return (
        action_type in {"schedule_HPC_maintenance", "schedule_fan_maintenance"}
        and peak >= 1.5
        and persistent >= 5
        and component_strength.startswith("strong")
        and later_peak
    )


def _missed_time_rule(features: dict[str, Any]) -> str:
    if features.get("first_persistent_pattern_cycle"):
        return "first_persistent_pattern_cycle"
    if features.get("first_warning_crossing_cycle"):
        return "first_warning_crossing_cycle"
    return "peak_score_cycle_minus_margin"


def _feedback_reason(label: str, features: dict[str, Any], previous_action: str, component_feedback: str) -> str:
    if label == "too_early":
        if component_feedback.lower() == "correct" and _maintenance_still_supported(features, previous_action):
            return "maintenance direction may be plausible, but timing should move closer to persistent/peak risk"
        return "maintenance evidence was insufficient; prefer monitoring until peak or persistent risk evidence strengthens"
    if label == "correct_maintenance":
        return "positive anchor: similar score, persistence, timing, and component evidence supported maintenance"
    if label.startswith("missed_"):
        return "missed maintenance: similar early evidence should trigger earlier monitoring or maintenance"
    return "historical feedback case"


def _component_evidence_strength(previous_action: str, sensors: list[str]) -> str:
    sensor_set = {str(s).upper() for s in sensors}
    hpc_count = len(sensor_set & STRONG_HPC_SENSORS)
    fan_count = len(sensor_set & STRONG_FAN_SENSORS)
    if previous_action == "schedule_HPC_maintenance":
        if hpc_count >= 2 and hpc_count >= fan_count:
            return "strong_HPC"
        if hpc_count > 0:
            return "weak_HPC"
        return "conflicting_or_uncertain"
    if previous_action == "schedule_fan_maintenance":
        if fan_count >= 2 and fan_count >= hpc_count:
            return "strong_FAN"
        if fan_count > 0:
            return "weak_FAN"
        return "conflicting_or_uncertain"
    if hpc_count >= 2:
        return "strong_HPC"
    if fan_count >= 2:
        return "strong_FAN"
    return "uncertain"


def _conflicts_with_candidate_component(row: dict[str, Any], candidate_action: str) -> bool:
    strength = str(row.get("component_evidence_strength", ""))
    revised_action = str(row.get("then_revise_action_type", ""))
    if candidate_action == "schedule_HPC_maintenance" and revised_action == candidate_action:
        return strength.startswith("strong_FAN")
    if candidate_action == "schedule_fan_maintenance" and revised_action == candidate_action:
        return strength.startswith("strong_HPC")
    return False


def _feature_vector_from_summary(summary: dict[str, Any]) -> dict[str, float | None]:
    return {
        "peak_score": _to_float(summary.get("peak_score")),
        "persistent_high_risk_duration": _to_float(summary.get("persistent_high_risk_duration")),
        "peak_cycle_num": _to_float(_parse_rel_cycle(summary.get("peak_score_cycle"))),
        "warning_cycle_num": _to_float(_parse_rel_cycle(summary.get("first_warning_crossing_cycle"))),
        "persistence_cycle_num": _to_float(_parse_rel_cycle(summary.get("first_persistent_pattern_cycle"))),
    }


def _feature_vector_from_row(row: dict[str, Any]) -> dict[str, float | None]:
    return {
        "peak_score": _to_float(row.get("peak_score")),
        "persistent_high_risk_duration": _to_float(row.get("persistent_high_risk_duration")),
        "peak_cycle_num": _to_float(_parse_rel_cycle(row.get("peak_score_cycle"))),
        "warning_cycle_num": _to_float(_parse_rel_cycle(row.get("first_warning_crossing_cycle"))),
        "persistence_cycle_num": _to_float(_parse_rel_cycle(row.get("first_persistent_pattern_cycle"))),
    }


def _peak_similarity(query: dict[str, float | None], row: dict[str, float | None]) -> float:
    query_peak = query.get("peak_score")
    row_peak = row.get("peak_score")
    if query_peak is None or row_peak is None:
        return 0.0
    return max(0.0, 1.0 - abs(float(query_peak) - float(row_peak)) / 1.0)


def _persistence_similarity(query: dict[str, float | None], row: dict[str, float | None]) -> float:
    query_persistence = query.get("persistent_high_risk_duration")
    row_persistence = row.get("persistent_high_risk_duration")
    if query_persistence is None or row_persistence is None:
        return 0.0
    return max(0.0, 1.0 - abs(float(query_persistence) - float(row_persistence)) / 20.0)


def _compact_reflection_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rule_id",
        "applies_to_dataset",
        "if_pattern",
        "previous_action_type",
        "previous_action_time",
        "feedback_label",
        "feedback_reason",
        "peak_score",
        "peak_score_cycle",
        "score_at_action_time",
        "score_trend",
        "first_warning_crossing_cycle",
        "first_critical_crossing_cycle",
        "persistent_high_risk_duration",
        "first_persistent_pattern_cycle",
        "action_to_peak_gap",
        "action_to_warning_gap",
        "action_to_persistence_gap",
        "dominant_top_sensors",
        "top_sensor_at_action_time",
        "top_sensor_at_peak_cycle",
        "component_evidence_strength",
        "then_revise_action_type",
        "recommended_time_rule",
        "then_adjust_threshold",
        "then_adjust_component_preference",
    ]
    return {key: _parse_compact_value(key, row.get(key, "")) for key in keys if row.get(key, "") != ""}


def _parse_compact_value(key: str, value: Any) -> Any:
    if key in {"dominant_top_sensors", "top_sensor_at_action_time", "top_sensor_at_peak_cycle"}:
        return [item for item in str(value).split("|") if item]
    if key in {
        "peak_score",
        "score_at_action_time",
        "action_to_peak_gap",
        "action_to_warning_gap",
        "action_to_persistence_gap",
    }:
        return _to_float(value)
    if key == "persistent_high_risk_duration":
        number = _to_float(value)
        return int(number) if number is not None else None
    return value


def _read_reflection_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != REFLECTION_COLUMNS:
            raise ValueError(
                f"Unsupported reflection schema in {path}. Expected exactly: {REFLECTION_COLUMNS}; "
                f"got: {reader.fieldnames}"
            )
        return list(reader)


def _assert_reflection_schema(path: Path) -> None:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if header != REFLECTION_COLUMNS:
        raise ValueError(
            f"Unsupported reflection schema in {path}. Expected exactly: {REFLECTION_COLUMNS}; got: {header}"
        )


def _cycle_gap(start: Any, end: Any) -> int | None:
    start_num = _parse_rel_cycle(start)
    end_num = _parse_rel_cycle(end)
    if start_num is None or end_num is None:
        return None
    return end_num - start_num


def _parse_rel_cycle(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return parse_t_plus(value)
    except Exception:
        return None


def _as_sensor_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item).upper() for item in value if str(item)}
    return {item.strip().upper() for item in str(value).replace(",", "|").split("|") if item.strip()}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

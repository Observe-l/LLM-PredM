from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
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
    "final_score",
    "unit_past_q95",
    "unit_past_q99",
    "peak_minus_unit_q95",
    "peak_minus_unit_q99",
    "peak_score_cycle",
    "score_at_action_time",
    "score_trend",
    "slope",
    "delta_score",
    "monotonicity",
    "volatility",
    "duration_above_unit_q95",
    "duration_above_unit_q99",
    "area_above_unit_q95",
    "area_above_unit_q99",
    "peak_lhi",
    "peak_d_rmse",
    "lhi_slope",
    "d_rmse_slope",
    "d_rmse_lhi_consistency",
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
    "sensor_pattern_stability",
    "hpc_sensor_presence_ratio",
    "fan_sensor_presence_ratio",
    "conflict_sensor_presence_ratio",
    "component_evidence_strength",
    "hpc_path_score",
    "fan_path_score",
    "uncertain_path_score",
    "component_conflict_score",
    "dominant_component",
    "dominance_margin",
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
    case: dict[str, Any] | None = None,
    component_stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    path = Path(kg_dir) / "reflection_rules.csv"
    if not path.exists():
        return []

    rows = _read_reflection_rows(path)
    if not rows:
        return []

    query = build_reflection_feature(
        case
        or {
            "dataset_subset": dataset_subset,
            "forecast_summary": forecast_summary,
        },
        component_stats=component_stats,
        candidate_action=candidate_action,
    )
    results: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        applies = str(row.get("applies_to_dataset", ""))

        if applies not in {dataset_subset, "ALL"}:
            continue
        memory = _feature_vector_from_row(row)
        similarity = reflection_similarity(query, memory)
        compact = _compact_reflection_row(row)
        compact["retrieval_similarity"] = round(similarity["total"], 4)
        compact["similarity_breakdown"] = {k: round(v, 4) for k, v in similarity.items()}
        compact["helpfulness_score"] = round(
            similarity["total"] + _helpfulness_bonus(query, memory, compact),
            4,
        )
        results.append((float(compact["helpfulness_score"]), compact))

    return _balanced_topk([row for _, row in sorted(results, key=lambda x: x[0], reverse=True)], max_rules=max_rules)


def feedback_to_rule(
    feedback: dict[str, Any],
    case: dict[str, Any],
    action: dict[str, Any],
    component_stats: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    dataset = str(case["dataset_subset"])
    pattern = infer_forecast_pattern(case["forecast_summary"])
    previous_action = str(action.get("action_type", ""))
    label = str(feedback["feedback_label"])
    feedback_id = str(feedback.get("feedback_id") or feedback.get("case_id") or "unknown")
    features = reflection_features(case, action, component_stats=component_stats)

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


def reflection_features(
    case: dict[str, Any],
    action: dict[str, Any],
    component_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = case.get("forecast_summary", {})
    risk = case.get("risk_statistics", {})
    trend = case.get("trend_statistics", {})
    multi = case.get("multi_score_statistics", {})
    sensor = case.get("sensor_evidence_statistics", {})
    peak = _to_float(summary.get("peak_score"))
    action_time = action.get("action_time")
    peak_cycle = str(summary.get("peak_score_cycle") or "")
    warning_cycle = summary.get("first_warning_crossing_cycle")
    persistence_cycle = summary.get("first_persistent_pattern_cycle")
    score_at_action, sensors_at_action = _score_and_sensors_at_relative_cycle(case, action_time)
    dominant_sensors = [str(s).upper() for s in sensor.get("dominant_top_sensors") or summary.get("dominant_top_sensors", [])]
    peak_sensors = [str(s).upper() for s in sensor.get("top_sensor_at_peak_cycle") or summary.get("top_sensor_at_peak_cycle", [])]
    component_evidence = _component_evidence_strength(previous_action=str(action.get("action_type", "")), sensors=dominant_sensors + peak_sensors + sensors_at_action)
    component_stats = component_stats or case.get("component_evidence_statistics", {})
    lhi_stats = multi.get("lhi_rmse_roll_mean") or multi.get(case.get("score_source", {}).get("lhi_name", ""), {})
    d_stats = multi.get("d_rmse") or multi.get(case.get("score_source", {}).get("raw_score_name", ""), {})
    return {
        "previous_action_time": _csv_value(action_time),
        "peak_score": _csv_value(peak),
        "final_score": _csv_value(risk.get("final_score") or summary.get("final_cycle_score")),
        "unit_past_q95": _csv_value(risk.get("unit_past_q95")),
        "unit_past_q99": _csv_value(risk.get("unit_past_q99")),
        "peak_minus_unit_q95": _csv_value(risk.get("peak_minus_unit_q95")),
        "peak_minus_unit_q99": _csv_value(risk.get("peak_minus_unit_q99")),
        "peak_score_cycle": peak_cycle,
        "score_at_action_time": _csv_value(score_at_action),
        "score_trend": _csv_value(summary.get("score_trend")),
        "slope": _csv_value(trend.get("slope")),
        "delta_score": _csv_value(trend.get("delta_score")),
        "monotonicity": _csv_value(trend.get("monotonicity")),
        "volatility": _csv_value(trend.get("volatility")),
        "duration_above_unit_q95": _csv_value(trend.get("duration_above_unit_q95")),
        "duration_above_unit_q99": _csv_value(trend.get("duration_above_unit_q99")),
        "area_above_unit_q95": _csv_value(trend.get("area_above_unit_q95")),
        "area_above_unit_q99": _csv_value(trend.get("area_above_unit_q99")),
        "peak_lhi": _csv_value(lhi_stats.get("peak")),
        "peak_d_rmse": _csv_value(d_stats.get("peak")),
        "lhi_slope": _csv_value(lhi_stats.get("slope")),
        "d_rmse_slope": _csv_value(d_stats.get("slope")),
        "d_rmse_lhi_consistency": _csv_value(multi.get("d_rmse_lhi_consistency")),
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
        "sensor_pattern_stability": _csv_value(sensor.get("sensor_pattern_stability")),
        "hpc_sensor_presence_ratio": _csv_value(sensor.get("hpc_sensor_presence_ratio")),
        "fan_sensor_presence_ratio": _csv_value(sensor.get("fan_sensor_presence_ratio")),
        "conflict_sensor_presence_ratio": _csv_value(sensor.get("conflict_sensor_presence_ratio")),
        "component_evidence_strength": component_evidence,
        "hpc_path_score": _csv_value(component_stats.get("hpc_path_score")),
        "fan_path_score": _csv_value(component_stats.get("fan_path_score")),
        "uncertain_path_score": _csv_value(component_stats.get("uncertain_path_score")),
        "component_conflict_score": _csv_value(component_stats.get("component_conflict_score")),
        "dominant_component": _csv_value(component_stats.get("dominant_component")),
        "dominance_margin": _csv_value(component_stats.get("dominance_margin")),
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


def build_reflection_feature(
    case: dict[str, Any],
    component_stats: dict[str, Any] | None = None,
    candidate_action: str | None = None,
) -> dict[str, Any]:
    summary = case.get("forecast_summary", {})
    risk = case.get("risk_statistics", {})
    trend = case.get("trend_statistics", {})
    multi = case.get("multi_score_statistics", {})
    sensor = case.get("sensor_evidence_statistics", {})
    component_stats = component_stats or case.get("component_evidence_statistics", {})
    lhi_stats = multi.get("lhi_rmse_roll_mean") or multi.get(case.get("score_source", {}).get("lhi_name", ""), {})
    d_stats = multi.get("d_rmse") or multi.get(case.get("score_source", {}).get("raw_score_name", ""), {})
    return {
        "dataset": case.get("dataset_subset"),
        "candidate_action": candidate_action,
        "peak_score": _to_float(risk.get("peak_score") or summary.get("peak_score")),
        "final_score": _to_float(risk.get("final_score") or summary.get("final_cycle_score")),
        "peak_minus_unit_q95": _to_float(risk.get("peak_minus_unit_q95")),
        "peak_minus_unit_q99": _to_float(risk.get("peak_minus_unit_q99")),
        "unit_past_context_reliable": bool(risk.get("unit_past_context_reliable")),
        "slope": _to_float(trend.get("slope")),
        "delta_score": _to_float(trend.get("delta_score")),
        "monotonicity": _to_float(trend.get("monotonicity")),
        "volatility": _to_float(trend.get("volatility")),
        "duration_above_unit_q95": _to_float(trend.get("duration_above_unit_q95")),
        "duration_above_unit_q99": _to_float(trend.get("duration_above_unit_q99")),
        "area_above_unit_q95": _to_float(trend.get("area_above_unit_q95")),
        "area_above_unit_q99": _to_float(trend.get("area_above_unit_q99")),
        "peak_lhi": _to_float(lhi_stats.get("peak") or risk.get("peak_score") or summary.get("peak_score")),
        "peak_d_rmse": _to_float(d_stats.get("peak")),
        "lhi_slope": _to_float(lhi_stats.get("slope") or trend.get("slope")),
        "d_rmse_slope": _to_float(d_stats.get("slope")),
        "d_rmse_lhi_consistency": multi.get("d_rmse_lhi_consistency"),
        "dominant_top_sensors": sensor.get("dominant_top_sensors") or summary.get("dominant_top_sensors", []),
        "top_sensor_at_peak_cycle": sensor.get("top_sensor_at_peak_cycle") or summary.get("top_sensor_at_peak_cycle", []),
        "sensor_pattern_stability": _to_float(sensor.get("sensor_pattern_stability")),
        "hpc_sensor_presence_ratio": _to_float(sensor.get("hpc_sensor_presence_ratio")),
        "fan_sensor_presence_ratio": _to_float(sensor.get("fan_sensor_presence_ratio")),
        "conflict_sensor_presence_ratio": _to_float(sensor.get("conflict_sensor_presence_ratio")),
        "hpc_path_score": _to_float(component_stats.get("hpc_path_score")),
        "fan_path_score": _to_float(component_stats.get("fan_path_score")),
        "uncertain_path_score": _to_float(component_stats.get("uncertain_path_score")),
        "component_conflict_score": _to_float(component_stats.get("component_conflict_score")),
        "dominant_component": component_stats.get("dominant_component"),
        "dominance_margin": _to_float(component_stats.get("dominance_margin")),
        "action_to_peak_gap": None,
        "action_to_persistence_gap": None,
        "action_to_warning_gap": None,
    }


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
        "final_score": _to_float(row.get("final_score")),
        "peak_minus_unit_q95": _to_float(row.get("peak_minus_unit_q95")),
        "peak_minus_unit_q99": _to_float(row.get("peak_minus_unit_q99")),
        "slope": _to_float(row.get("slope")),
        "delta_score": _to_float(row.get("delta_score")),
        "monotonicity": _to_float(row.get("monotonicity")),
        "volatility": _to_float(row.get("volatility")),
        "duration_above_unit_q95": _to_float(row.get("duration_above_unit_q95")),
        "duration_above_unit_q99": _to_float(row.get("duration_above_unit_q99")),
        "area_above_unit_q95": _to_float(row.get("area_above_unit_q95")),
        "area_above_unit_q99": _to_float(row.get("area_above_unit_q99")),
        "peak_lhi": _to_float(row.get("peak_lhi") or row.get("peak_score")),
        "peak_d_rmse": _to_float(row.get("peak_d_rmse")),
        "lhi_slope": _to_float(row.get("lhi_slope")),
        "d_rmse_slope": _to_float(row.get("d_rmse_slope")),
        "d_rmse_lhi_consistency": row.get("d_rmse_lhi_consistency"),
        "dominant_top_sensors": _as_sensor_set(row.get("dominant_top_sensors", "")),
        "top_sensor_at_peak_cycle": _as_sensor_set(row.get("top_sensor_at_peak_cycle", "")),
        "sensor_pattern_stability": _to_float(row.get("sensor_pattern_stability")),
        "hpc_sensor_presence_ratio": _to_float(row.get("hpc_sensor_presence_ratio")),
        "fan_sensor_presence_ratio": _to_float(row.get("fan_sensor_presence_ratio")),
        "conflict_sensor_presence_ratio": _to_float(row.get("conflict_sensor_presence_ratio")),
        "hpc_path_score": _to_float(row.get("hpc_path_score")),
        "fan_path_score": _to_float(row.get("fan_path_score")),
        "uncertain_path_score": _to_float(row.get("uncertain_path_score")),
        "component_conflict_score": _to_float(row.get("component_conflict_score")),
        "dominant_component": row.get("dominant_component"),
        "dominance_margin": _to_float(row.get("dominance_margin")),
        "action_to_peak_gap": _to_float(row.get("action_to_peak_gap")),
        "action_to_persistence_gap": _to_float(row.get("action_to_persistence_gap")),
        "action_to_warning_gap": _to_float(row.get("action_to_warning_gap")),
        "persistent_high_risk_duration": _to_float(row.get("persistent_high_risk_duration")),
        "peak_cycle_num": _to_float(_parse_rel_cycle(row.get("peak_score_cycle"))),
        "warning_cycle_num": _to_float(_parse_rel_cycle(row.get("first_warning_crossing_cycle"))),
        "persistence_cycle_num": _to_float(_parse_rel_cycle(row.get("first_persistent_pattern_cycle"))),
    }


def reflection_similarity(query: dict[str, Any], memory: dict[str, Any]) -> dict[str, float]:
    s_risk = _risk_similarity(query, memory)
    s_trend = _trend_similarity(query, memory)
    s_multi = _multi_score_similarity(query, memory)
    s_sensor = _sensor_similarity(query, memory)
    s_component = _component_similarity(query, memory)
    s_timing = _timing_similarity(query, memory)
    total = (
        0.20 * s_risk
        + 0.20 * s_trend
        + 0.15 * s_multi
        + 0.15 * s_sensor
        + 0.20 * s_component
        + 0.10 * s_timing
    )
    return {
        "total": total,
        "risk": s_risk,
        "trend": s_trend,
        "multi_score": s_multi,
        "sensor": s_sensor,
        "component": s_component,
        "timing": s_timing,
    }


def _risk_similarity(query: dict[str, Any], memory: dict[str, Any]) -> float:
    if query.get("unit_past_context_reliable"):
        return _mean(
            [
                _numeric_sim(query.get("peak_score"), memory.get("peak_score"), 1.0),
                _numeric_sim(query.get("final_score"), memory.get("final_score"), 1.0),
                _numeric_sim(query.get("peak_minus_unit_q95"), memory.get("peak_minus_unit_q95"), 1.0),
                _numeric_sim(query.get("peak_minus_unit_q99"), memory.get("peak_minus_unit_q99"), 1.0),
            ]
        )
    return _mean(
        [
            _numeric_sim(query.get("peak_score"), memory.get("peak_score"), 1.0),
            _numeric_sim(query.get("final_score"), memory.get("final_score"), 1.0),
        ]
    )


def _trend_similarity(query: dict[str, Any], memory: dict[str, Any]) -> float:
    return _mean(
        [
            _numeric_sim(query.get("slope"), memory.get("slope"), 0.05),
            _numeric_sim(query.get("delta_score"), memory.get("delta_score"), 1.0),
            _numeric_sim(query.get("monotonicity"), memory.get("monotonicity"), 0.5),
            _numeric_sim(query.get("volatility"), memory.get("volatility"), 0.1),
            _numeric_sim(query.get("duration_above_unit_q95"), memory.get("duration_above_unit_q95"), 20.0),
            _numeric_sim(query.get("duration_above_unit_q99"), memory.get("duration_above_unit_q99"), 20.0),
            _numeric_sim(query.get("area_above_unit_q95"), memory.get("area_above_unit_q95"), 10.0),
            _numeric_sim(query.get("area_above_unit_q99"), memory.get("area_above_unit_q99"), 10.0),
        ]
    )


def _multi_score_similarity(query: dict[str, Any], memory: dict[str, Any]) -> float:
    consistency = 1.0 if query.get("d_rmse_lhi_consistency") == memory.get("d_rmse_lhi_consistency") else 0.0
    return _mean(
        [
            _numeric_sim(query.get("peak_lhi"), memory.get("peak_lhi"), 1.0),
            _numeric_sim(query.get("peak_d_rmse"), memory.get("peak_d_rmse"), 0.2),
            _numeric_sim(query.get("lhi_slope"), memory.get("lhi_slope"), 0.05),
            _numeric_sim(query.get("d_rmse_slope"), memory.get("d_rmse_slope"), 0.02),
            consistency,
        ]
    )


def _sensor_similarity(query: dict[str, Any], memory: dict[str, Any]) -> float:
    return _mean(
        [
            _jaccard(_as_sensor_set(query.get("dominant_top_sensors", [])), _as_sensor_set(memory.get("dominant_top_sensors", []))),
            _jaccard(
                _as_sensor_set(query.get("top_sensor_at_peak_cycle", [])),
                _as_sensor_set(memory.get("top_sensor_at_peak_cycle", [])),
            ),
            _numeric_sim(query.get("hpc_sensor_presence_ratio"), memory.get("hpc_sensor_presence_ratio"), 0.5),
            _numeric_sim(query.get("fan_sensor_presence_ratio"), memory.get("fan_sensor_presence_ratio"), 0.5),
            _numeric_sim(query.get("conflict_sensor_presence_ratio"), memory.get("conflict_sensor_presence_ratio"), 0.5),
            _numeric_sim(query.get("sensor_pattern_stability"), memory.get("sensor_pattern_stability"), 0.5),
        ]
    )


def _component_similarity(query: dict[str, Any], memory: dict[str, Any]) -> float:
    dominant_match = 1.0 if query.get("dominant_component") and query.get("dominant_component") == memory.get("dominant_component") else 0.0
    return _mean(
        [
            _numeric_sim(query.get("hpc_path_score"), memory.get("hpc_path_score"), 0.4),
            _numeric_sim(query.get("fan_path_score"), memory.get("fan_path_score"), 0.4),
            _numeric_sim(query.get("uncertain_path_score"), memory.get("uncertain_path_score"), 0.4),
            _numeric_sim(query.get("component_conflict_score"), memory.get("component_conflict_score"), 0.5),
            _numeric_sim(query.get("dominance_margin"), memory.get("dominance_margin"), 0.4),
            dominant_match,
        ]
    )


def _timing_similarity(query: dict[str, Any], memory: dict[str, Any]) -> float:
    return _mean(
        [
            _numeric_sim(query.get("action_to_peak_gap"), memory.get("action_to_peak_gap"), 20.0),
            _numeric_sim(query.get("action_to_persistence_gap"), memory.get("action_to_persistence_gap"), 20.0),
            _numeric_sim(query.get("action_to_warning_gap"), memory.get("action_to_warning_gap"), 20.0),
        ]
    )


def _numeric_sim(left: Any, right: Any, scale: float) -> float:
    left = _to_float(left)
    right = _to_float(right)
    if left is None or right is None:
        return 0.0
    return math.exp(-abs(left - right) / max(scale, 1e-9))


def _mean(values: list[float]) -> float:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _helpfulness_bonus(query: dict[str, Any], memory: dict[str, Any], compact: dict[str, Any]) -> float:
    bonus = 0.0
    if query.get("candidate_action") and compact.get("then_revise_action_type") == query.get("candidate_action"):
        bonus += 0.05
    if query.get("dominant_component") and query.get("dominant_component") == memory.get("dominant_component"):
        bonus += 0.05
    if compact.get("feedback_label") in {"correct_maintenance", "missed_HPC_maintenance", "missed_fan_maintenance"}:
        bonus += 0.05
    return bonus


def _balanced_topk(rows: list[dict[str, Any]], max_rules: int) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("feedback_label", "unknown"))].append(row)
    selected: list[dict[str, Any]] = []
    for label in ["correct_maintenance", "too_early", "missed_HPC_maintenance", "missed_fan_maintenance", "over_maintenance", "wrong_component"]:
        selected.extend(by_label.get(label, [])[:2])
    if len(selected) < max_rules:
        used = {str(item.get("rule_id")) for item in selected}
        for row in rows:
            if str(row.get("rule_id")) not in used:
                selected.append(row)
            if len(selected) >= max_rules:
                break
    return selected[:max_rules]


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
        "final_score",
        "unit_past_q95",
        "unit_past_q99",
        "peak_minus_unit_q95",
        "peak_minus_unit_q99",
        "peak_score_cycle",
        "score_at_action_time",
        "score_trend",
        "slope",
        "delta_score",
        "monotonicity",
        "volatility",
        "duration_above_unit_q95",
        "duration_above_unit_q99",
        "area_above_unit_q95",
        "area_above_unit_q99",
        "peak_lhi",
        "peak_d_rmse",
        "lhi_slope",
        "d_rmse_slope",
        "d_rmse_lhi_consistency",
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
        "sensor_pattern_stability",
        "hpc_sensor_presence_ratio",
        "fan_sensor_presence_ratio",
        "conflict_sensor_presence_ratio",
        "component_evidence_strength",
        "hpc_path_score",
        "fan_path_score",
        "uncertain_path_score",
        "component_conflict_score",
        "dominant_component",
        "dominance_margin",
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
        "final_score",
        "unit_past_q95",
        "unit_past_q99",
        "peak_minus_unit_q95",
        "peak_minus_unit_q99",
        "score_at_action_time",
        "slope",
        "delta_score",
        "monotonicity",
        "volatility",
        "duration_above_unit_q95",
        "duration_above_unit_q99",
        "area_above_unit_q95",
        "area_above_unit_q99",
        "peak_lhi",
        "peak_d_rmse",
        "lhi_slope",
        "d_rmse_slope",
        "action_to_peak_gap",
        "action_to_warning_gap",
        "action_to_persistence_gap",
        "sensor_pattern_stability",
        "hpc_sensor_presence_ratio",
        "fan_sensor_presence_ratio",
        "conflict_sensor_presence_ratio",
        "hpc_path_score",
        "fan_path_score",
        "uncertain_path_score",
        "component_conflict_score",
        "dominance_margin",
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

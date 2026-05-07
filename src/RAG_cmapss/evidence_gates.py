from __future__ import annotations

from collections import defaultdict
from typing import Any


CORRECT_MAINTENANCE_LABELS = {"correct_maintenance"}
MISSED_MAINTENANCE_LABELS = {"missed_HPC_maintenance", "missed_fan_maintenance"}
POSITIVE_MAINTENANCE_LABELS = CORRECT_MAINTENANCE_LABELS | MISSED_MAINTENANCE_LABELS
EARLY_MAINTENANCE_LABELS = {"too_early", "over_maintenance"}
WARMUP_Q95_GAP_MIN = 0.35
WARMUP_Q99_GAP_MIN = 0.25
WARMUP_DURATION_MIN = 10.0


def build_risk_gate(case: dict[str, Any], reflection_rules: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    risk = case.get("risk_statistics", {})
    trend = case.get("trend_statistics", {})
    peak = _to_float(risk.get("peak_score"))
    q95_gap = _to_float(risk.get("peak_minus_unit_q95"))
    q99_gap = _to_float(risk.get("peak_minus_unit_q99"))
    duration_q95 = _to_float(trend.get("duration_above_unit_q95"))
    duration_q99 = _to_float(trend.get("duration_above_unit_q99"))
    slope = _to_float(trend.get("slope"))
    consistency = case.get("multi_score_statistics", {}).get("d_rmse_lhi_consistency")
    reliable = bool(risk.get("unit_past_context_reliable"))

    duration_q95_value = duration_q95 or 0.0
    duration_q99_value = duration_q99 or 0.0
    persistent_duration = case.get("forecast_summary", {}).get("persistent_high_risk_duration") or 0
    increasing = bool(consistency == "consistent_increasing" and slope is not None and slope > 0)
    reflection_rules = reflection_rules or []
    calibration = _reflection_peak_calibration(peak, reflection_rules)
    q95_gap_calibration = _reflection_lower_bound_calibration(
        current_value=q95_gap,
        reflection_rules=reflection_rules,
        field="peak_minus_unit_q95",
        warmup_min=WARMUP_Q95_GAP_MIN,
    )
    q99_gap_calibration = _reflection_lower_bound_calibration(
        current_value=q99_gap,
        reflection_rules=reflection_rules,
        field="peak_minus_unit_q99",
        warmup_min=WARMUP_Q99_GAP_MIN,
    )
    q95_duration_calibration = _reflection_lower_bound_calibration(
        current_value=duration_q95,
        reflection_rules=reflection_rules,
        field="duration_above_unit_q95",
        warmup_min=WARMUP_DURATION_MIN,
    )
    q99_duration_calibration = _reflection_lower_bound_calibration(
        current_value=duration_q99,
        reflection_rules=reflection_rules,
        field="duration_above_unit_q99",
        warmup_min=WARMUP_DURATION_MIN,
    )
    strong_q95_excess = q95_gap_calibration["decision"] == "supports_maintenance"
    strong_q99_excess = q99_gap_calibration["decision"] == "supports_maintenance"
    persistent_q95 = q95_duration_calibration["decision"] == "supports_maintenance"
    persistent_q99 = q99_duration_calibration["decision"] == "supports_maintenance"

    if reliable and strong_q99_excess and persistent_q99 and increasing:
        level = "high_persistent"
    elif reliable and strong_q95_excess and persistent_q95 and increasing:
        level = "persistent_warning"
    elif reliable and q95_gap is not None and q95_gap > 0:
        level = "transient_warning"
    elif not reliable and persistent_duration >= 10 and increasing:
        level = "high_persistent_uncalibrated"
    elif not reliable and peak is not None and peak >= 1.0:
        level = "warning_uncalibrated"
    else:
        level = "low"

    statistical_candidate = level in {"high_persistent", "persistent_warning", "high_persistent_uncalibrated"}
    reflection_allows = calibration["decision"] == "supports_maintenance"
    maintenance_candidate = bool(statistical_candidate and reflection_allows)

    return {
        "risk_level": level,
        "maintenance_candidate": maintenance_candidate,
        "unit_past_context_reliable": reliable,
        "peak_score": peak,
        "peak_minus_unit_q95": q95_gap,
        "peak_minus_unit_q99": q99_gap,
        "duration_above_unit_q95": duration_q95,
        "duration_above_unit_q99": duration_q99,
        "slope": slope,
        "d_rmse_lhi_consistency": consistency,
        "statistical_candidate": statistical_candidate,
        "reflection_peak_calibration": calibration,
        "reflection_q95_gap_calibration": q95_gap_calibration,
        "reflection_q99_gap_calibration": q99_gap_calibration,
        "reflection_q95_duration_calibration": q95_duration_calibration,
        "reflection_q99_duration_calibration": q99_duration_calibration,
        "thresholds": {
            "absolute_high_peak_min": None,
            "q95_excess_min": q95_gap_calibration["learned_min"],
            "q99_excess_min": q99_gap_calibration["learned_min"],
            "persistent_duration_min": min(
                q95_duration_calibration["learned_min"],
                q99_duration_calibration["learned_min"],
            ),
            "requires_increasing_trend": True,
            "absolute_peak_source": "learned_online_from_reflection_memory",
            "gap_duration_source": "reflection_memory_with_warmup_prior",
        },
        "reason": _risk_reason(level, reliable, q95_gap, q99_gap, duration_q95, duration_q99, consistency),
    }


def _reflection_peak_calibration(current_peak: float | None, reflection_rules: list[dict[str, Any]]) -> dict[str, Any]:
    correct_peaks: list[float] = []
    missed_peaks: list[float] = []
    early_peaks: list[float] = []
    for item in reflection_rules:
        peak = _to_float(item.get("peak_score"))
        if peak is None:
            continue
        label = str(item.get("feedback_label", ""))
        if label in CORRECT_MAINTENANCE_LABELS:
            correct_peaks.append(peak)
        elif label in MISSED_MAINTENANCE_LABELS:
            missed_peaks.append(peak)
        elif label in EARLY_MAINTENANCE_LABELS:
            early_peaks.append(peak)

    positive_peaks = _positive_values_with_weak_missed(correct_peaks, missed_peaks, early_peaks)
    positive_min = min(positive_peaks) if positive_peaks else None
    early_max = max(early_peaks) if early_peaks else None
    learned_min: float | None = None
    source = "cold_start_no_reflection"
    separability = "unknown"

    if positive_min is not None and early_max is not None:
        if early_max < positive_min:
            learned_min = (early_max + positive_min) / 2.0
            source = "midpoint_between_early_max_and_positive_min"
            separability = "separable"
        else:
            learned_min = positive_min
            source = "overlap_use_positive_min"
            separability = "overlap"
    elif positive_min is not None:
        learned_min = positive_min
        source = "positive_anchor_min"
    elif early_max is not None:
        learned_min = early_max
        source = "above_early_anchor_max"

    if learned_min is None or current_peak is None:
        decision = "blocks_maintenance"
    elif current_peak >= learned_min:
        decision = "supports_maintenance"
    else:
        decision = "blocks_maintenance"

    return {
        "decision": decision,
        "learned_peak_score_min": round(learned_min, 6) if learned_min is not None else None,
        "boundary_source": source,
        "separability": separability,
        "current_peak_score": current_peak,
        "positive_peak_min": round(positive_min, 6) if positive_min is not None else None,
        "early_peak_max": round(early_max, 6) if early_max is not None else None,
        "positive_anchor_count": len(positive_peaks),
        "correct_anchor_count": len(correct_peaks),
        "missed_anchor_count": len(missed_peaks),
        "early_anchor_count": len(early_peaks),
    }


def _reflection_lower_bound_calibration(
    current_value: float | None,
    reflection_rules: list[dict[str, Any]],
    field: str,
    warmup_min: float,
) -> dict[str, Any]:
    correct_values: list[float] = []
    missed_values: list[float] = []
    early_values: list[float] = []
    for item in reflection_rules:
        value = _to_float(item.get(field))
        if value is None:
            continue
        label = str(item.get("feedback_label", ""))
        if label in CORRECT_MAINTENANCE_LABELS:
            correct_values.append(value)
        elif label in MISSED_MAINTENANCE_LABELS:
            missed_values.append(value)
        elif label in EARLY_MAINTENANCE_LABELS:
            early_values.append(value)

    positive_values = _positive_values_with_weak_missed(correct_values, missed_values, early_values)
    positive_min = min(positive_values) if positive_values else None
    early_max = max(early_values) if early_values else None
    learned_min = warmup_min
    source = "warmup_prior"
    separability = "unknown"

    if positive_min is not None and early_max is not None:
        if early_max < positive_min:
            learned_min = (early_max + positive_min) / 2.0
            source = "midpoint_between_early_max_and_positive_min"
            separability = "separable"
        else:
            learned_min = positive_min
            source = "overlap_use_positive_min"
            separability = "overlap"
    elif positive_min is not None:
        learned_min = min(warmup_min, positive_min)
        source = "positive_anchor_min_with_warmup_prior"
    elif early_max is not None:
        learned_min = max(warmup_min, early_max)
        source = "above_early_anchor_max_with_warmup_prior"

    if current_value is None:
        decision = "blocks_maintenance"
    elif current_value >= learned_min:
        decision = "supports_maintenance"
    else:
        decision = "blocks_maintenance"

    return {
        "field": field,
        "decision": decision,
        "current_value": current_value,
        "learned_min": round(learned_min, 6),
        "warmup_min": warmup_min,
        "boundary_source": source,
        "separability": separability,
        "positive_min": round(positive_min, 6) if positive_min is not None else None,
        "early_max": round(early_max, 6) if early_max is not None else None,
        "positive_anchor_count": len(positive_values),
        "correct_anchor_count": len(correct_values),
        "missed_anchor_count": len(missed_values),
        "early_anchor_count": len(early_values),
    }


def _positive_values_with_weak_missed(
    correct_values: list[float],
    missed_values: list[float],
    early_values: list[float],
) -> list[float]:
    if not missed_values:
        return list(correct_values)
    if not early_values:
        return [*correct_values, *missed_values]
    early_max = max(early_values)
    compatible_missed = [value for value in missed_values if value >= early_max]
    if correct_values or compatible_missed:
        return [*correct_values, *compatible_missed]
    return list(correct_values)


def build_component_gate(component_stats: dict[str, Any], dataset_rules: dict[str, Any]) -> dict[str, Any]:
    dominant = component_stats.get("dominant_component")
    margin = _to_float(component_stats.get("dominance_margin")) or 0.0
    conflict = _to_float(component_stats.get("component_conflict_score")) or 0.0
    allowed_hypotheses = set(dataset_rules.get("allowed_hypotheses", []))
    hpc_score = _to_float(component_stats.get("hpc_path_score")) or 0.0
    fan_score = _to_float(component_stats.get("fan_path_score")) or 0.0
    explicit_components = {"HPC_related_degradation", "Fan_related_degradation"}
    dominant_score = hpc_score if dominant == "HPC_related_degradation" else fan_score
    competing_score = fan_score if dominant == "HPC_related_degradation" else hpc_score
    component_supported = bool(
        dominant in explicit_components
        and dominant in allowed_hypotheses
        and dominant_score >= 0.9
        and competing_score <= 0.65
        and (margin >= 0.02 or conflict < 0.75)
    )
    if dominant == "HPC_related_degradation":
        action = "schedule_HPC_maintenance"
    elif dominant == "Fan_related_degradation":
        action = "schedule_fan_maintenance"
    else:
        action = "schedule_monitoring"
    if action in set(dataset_rules.get("disallowed_actions", [])):
        action = "schedule_monitoring"
        component_supported = False
    return {
        "dominant_component": dominant,
        "component_supported": component_supported,
        "component_conflict_score": conflict,
        "dominance_margin": margin,
        "suggested_component_action": action,
        "hpc_path_score": hpc_score,
        "fan_path_score": fan_score,
        "uncertain_path_score": component_stats.get("uncertain_path_score"),
        "recommendation": "component_supported" if component_supported else "monitor_due_to_component_uncertainty",
    }


def build_reflection_gate(reflection_rules: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in reflection_rules:
        grouped[str(item.get("feedback_label", "unknown"))].append(item)
    labels = {label: _label_summary(items) for label, items in grouped.items()}
    return {
        "label_summaries": labels,
        "recommendation": _reflection_recommendation(labels),
        "note": "Reflection is auxiliary; gates should not follow it blindly.",
    }


def _label_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    similarities = [_to_float(item.get("retrieval_similarity")) for item in items]
    similarities = [item for item in similarities if item is not None]
    peaks = [_to_float(item.get("peak_score")) for item in items]
    peaks = [item for item in peaks if item is not None]
    return {
        "count": len(items),
        "mean_similarity": round(sum(similarities) / len(similarities), 4) if similarities else None,
        "best_similarity": round(max(similarities), 4) if similarities else None,
        "mean_peak_score": round(sum(peaks) / len(peaks), 4) if peaks else None,
        "top_suggested_action": items[0].get("then_revise_action_type") if items else None,
        "top_recommended_time_rule": items[0].get("recommended_time_rule") if items else None,
    }


def _reflection_recommendation(labels: dict[str, Any]) -> str:
    correct = labels.get("correct_maintenance", {}).get("best_similarity") or 0.0
    too_early = labels.get("too_early", {}).get("best_similarity") or 0.0
    missed = max(
        labels.get("missed_HPC_maintenance", {}).get("best_similarity") or 0.0,
        labels.get("missed_fan_maintenance", {}).get("best_similarity") or 0.0,
    )
    if correct > too_early + 0.05 and correct >= missed:
        return "supports_maintenance_anchor"
    if missed > too_early + 0.05 and missed > correct:
        return "warns_against_repeated_monitoring"
    if too_early > 0:
        return "warns_against_premature_maintenance"
    return "insufficient_reflection_evidence"


def _risk_reason(
    level: str,
    reliable: bool,
    q95_gap: float | None,
    q99_gap: float | None,
    duration_q95: float | None,
    duration_q99: float | None,
    consistency: Any,
) -> str:
    if reliable:
        return (
            f"{level}: q95_gap={q95_gap}, q99_gap={q99_gap}, "
            f"duration_q95={duration_q95}, duration_q99={duration_q99}, consistency={consistency}"
        )
    return f"{level}: unit-past context is not reliable; using forecast trend and score persistence only"


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

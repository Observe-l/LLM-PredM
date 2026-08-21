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
MIN_CORRECT_ONLY_ANCHORS = 3


def build_risk_gate(
    case: dict[str, Any],
    reflection_rules: list[dict[str, Any]] | None = None,
    threshold_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    threshold_overrides = threshold_overrides or {}
    prior_q95_gap_min = _override_float(threshold_overrides, "q95_excess_min", WARMUP_Q95_GAP_MIN)
    prior_q99_gap_min = _override_float(threshold_overrides, "q99_excess_min", WARMUP_Q99_GAP_MIN)
    prior_duration_min = _override_float(threshold_overrides, "persistent_duration_min", WARMUP_DURATION_MIN)

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
        prior_min=prior_q95_gap_min,
    )
    q99_gap_calibration = _reflection_lower_bound_calibration(
        current_value=q99_gap,
        reflection_rules=reflection_rules,
        field="peak_minus_unit_q99",
        prior_min=prior_q99_gap_min,
    )
    q95_duration_calibration = _reflection_lower_bound_calibration(
        current_value=duration_q95,
        reflection_rules=reflection_rules,
        field="duration_above_unit_q95",
        prior_min=prior_duration_min,
    )
    q99_duration_calibration = _reflection_lower_bound_calibration(
        current_value=duration_q99,
        reflection_rules=reflection_rules,
        field="duration_above_unit_q99",
        prior_min=prior_duration_min,
    )
    strong_q95_excess = q95_gap_calibration["decision"] == "supports_maintenance"
    strong_q99_excess = q99_gap_calibration["decision"] == "supports_maintenance"
    persistent_q95 = q95_duration_calibration["decision"] == "supports_maintenance"
    persistent_q99 = q99_duration_calibration["decision"] == "supports_maintenance"
    learned_peak_support = calibration["decision"] == "supports_maintenance"

    if reliable and strong_q99_excess and persistent_q99 and increasing:
        level = "high_persistent"
    elif reliable and strong_q95_excess and persistent_q95 and increasing:
        level = "persistent_warning"
    elif reliable and learned_peak_support and persistent_q95 and increasing and q95_gap is not None and q95_gap > 0:
        level = "learned_peak_persistent"
    elif reliable and q95_gap is not None and q95_gap > 0:
        level = "transient_warning"
    elif not reliable and persistent_duration >= 10 and increasing:
        level = "high_persistent_uncalibrated"
    elif not reliable and peak is not None and peak >= 1.0:
        level = "warning_uncalibrated"
    else:
        level = "low"

    statistical_candidate = level in {
        "high_persistent",
        "persistent_warning",
        "learned_peak_persistent",
        "high_persistent_uncalibrated",
    }
    reflection_allows = calibration["decision"] == "supports_maintenance"
    # Reflection memory is auxiliary calibration. It must not suppress a
    # statistically persistent case during cold start; this is the behavior
    # recorded by the historical consensus_v2 output and also used by the
    # current-LHI path.
    maintenance_candidate = bool(statistical_candidate)

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
        "reflection_allows": reflection_allows,
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
            "gap_duration_source": "online_update_override" if threshold_overrides else "reflection_memory_with_prior",
            "runtime_threshold_overrides": threshold_overrides,
        },
        "reason": _risk_reason(level, reliable, q95_gap, q99_gap, duration_q95, duration_q99, consistency),
    }


def _reflection_peak_calibration(current_peak: float | None, reflection_rules: list[dict[str, Any]]) -> dict[str, Any]:
    correct_peaks: list[float] = []
    missed_peaks: list[float] = []
    early_peaks: list[float] = []
    for item in reflection_rules:
        label = str(item.get("feedback_label", ""))
        peak = _reflection_peak_anchor_value(item, label)
        if peak is None:
            continue
        if label in CORRECT_MAINTENANCE_LABELS:
            correct_peaks.append(peak)
        elif label in MISSED_MAINTENANCE_LABELS:
            missed_peaks.append(peak)
        elif label in EARLY_MAINTENANCE_LABELS:
            early_peaks.append(peak)

    positive_peaks = _positive_values_with_weak_missed(correct_peaks, missed_peaks, early_peaks)
    positive_boundary = _robust_lower_boundary(positive_peaks)
    positive_min = min(positive_peaks) if positive_peaks else None
    early_max = max(early_peaks) if early_peaks else None
    learned_min: float | None = None
    source = "cold_start_no_reflection"
    separability = "unknown"

    if positive_boundary is None:
        learned_min = None
    elif not missed_peaks and not early_peaks and len(correct_peaks) < MIN_CORRECT_ONLY_ANCHORS:
        learned_min = None
        source = "exploration_required_correct_only_memory"
    elif early_max is not None:
        if early_max < positive_boundary:
            learned_min = (early_max + positive_boundary) / 2.0
            source = "midpoint_between_early_max_and_positive_min"
            separability = "separable"
        else:
            learned_min = max(positive_boundary, _median(positive_peaks))
            source = "overlap_use_robust_positive_boundary"
            separability = "overlap"
    elif positive_boundary is not None:
        learned_min = positive_boundary
        if missed_peaks and not correct_peaks:
            source = "missed_anchor_boundary"
        else:
            source = "robust_positive_anchor_boundary"
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
        "positive_peak_boundary": round(positive_boundary, 6) if positive_boundary is not None else None,
        "early_peak_max": round(early_max, 6) if early_max is not None else None,
        "positive_anchor_count": len(positive_peaks),
        "correct_anchor_count": len(correct_peaks),
        "missed_anchor_count": len(missed_peaks),
        "early_anchor_count": len(early_peaks),
    }


def _override_float(overrides: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(overrides.get(key, default))
    except (TypeError, ValueError):
        value = default
    if key == "persistent_duration_min":
        return max(1.0, value)
    return max(0.0, value)


def _reflection_lower_bound_calibration(
    current_value: float | None,
    reflection_rules: list[dict[str, Any]],
    field: str,
    prior_min: float,
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
    positive_boundary = _robust_lower_boundary(positive_values)
    positive_min = min(positive_values) if positive_values else None
    early_max = max(early_values) if early_values else None
    learned_min = prior_min
    source = "prior"
    separability = "unknown"

    if positive_boundary is not None and not missed_values and not early_values and len(correct_values) < MIN_CORRECT_ONLY_ANCHORS:
        learned_min = prior_min
        source = "exploration_required_correct_only_memory_with_prior"
    elif positive_boundary is not None and early_max is not None:
        if early_max < positive_boundary:
            learned_min = (early_max + positive_boundary) / 2.0
            source = "midpoint_between_early_max_and_positive_min"
            separability = "separable"
        else:
            learned_min = max(positive_boundary, _median(positive_values))
            source = "overlap_use_robust_positive_boundary"
            separability = "overlap"
    elif positive_boundary is not None:
        if missed_values and not correct_values:
            learned_min = min(prior_min, positive_boundary)
            source = "missed_anchor_boundary_with_prior"
        else:
            learned_min = min(prior_min, positive_boundary)
            source = "robust_positive_anchor_boundary_with_prior"
    elif early_max is not None:
        learned_min = max(prior_min, early_max)
        source = "above_early_anchor_max_with_prior"

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
        "prior_min": prior_min,
        "boundary_source": source,
        "separability": separability,
        "positive_min": round(positive_min, 6) if positive_min is not None else None,
        "positive_boundary": round(positive_boundary, 6) if positive_boundary is not None else None,
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


def _reflection_peak_anchor_value(item: dict[str, Any], label: str) -> float | None:
    peak = _to_float(item.get("peak_score"))
    if peak is None:
        return None
    action_score = _to_float(item.get("score_at_action_time"))
    if label in CORRECT_MAINTENANCE_LABELS and action_score is not None:
        return action_score
    if label in EARLY_MAINTENANCE_LABELS and action_score is not None:
        return action_score
    if label in MISSED_MAINTENANCE_LABELS:
        q95_gap = max(_to_float(item.get("peak_minus_unit_q95")) or 0.0, 0.0)
        q99_gap = max(_to_float(item.get("peak_minus_unit_q99")) or 0.0, 0.0)
        lower_margin = max(q95_gap, q99_gap)
        if lower_margin > 0:
            return peak - lower_margin
    return peak


def _robust_lower_boundary(values: list[float]) -> float | None:
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    if len(values) < 3:
        return values[0]
    return _quantile(values, 0.25)


def _median(values: list[float]) -> float:
    return _quantile(sorted(float(value) for value in values), 0.5)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def build_component_gate(component_stats: dict[str, Any], dataset_rules: dict[str, Any]) -> dict[str, Any]:
    """Build the scored component gate used by the legacy forecast path.

    This is deliberately not called by the current-LHI path.  It preserves
    the component evidence contract recorded in the consensus_v2 artifacts:
    scores are derived from KG paths, while FD-specific action restrictions
    are already removed by ``get_dataset_rules(..., mixed_fleet=True)``.
    """
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
        "explicit_component": dominant,
        "explicit_component_score": dominant_score,
        "explicit_competing_score": competing_score,
        "explicit_component_margin": round(dominant_score - competing_score, 4),
        "uncertain_is_non_veto": True,
        "component_supported": component_supported,
        "component_conflict_score": conflict,
        "dominance_margin": margin,
        "suggested_component_action": action,
        "hpc_path_score": hpc_score,
        "fan_path_score": fan_score,
        "uncertain_path_score": component_stats.get("uncertain_path_score"),
        "recommendation": "component_supported" if component_supported else "monitor_due_to_component_uncertainty",
    }


def apply_engine_escalation(
    component_gate: dict[str, Any],
    component_stats: dict[str, Any],
    risk_gate: dict[str, Any],
    dataset_rules: dict[str, Any],
    prior_monitoring_count: int = 0,
) -> dict[str, Any]:
    """Compatibility no-op: escalation must not select a component."""
    return dict(component_gate or {})


def update_component_consensus(
    consensus_state: dict[str, Any] | None,
    component_stats: dict[str, Any] | None,
    component_gate: dict[str, Any] | None = None,
    *,
    max_history: int = 20,
) -> dict[str, Any]:
    # consensus_v2 recorded the state in the context but did not feed a
    # synthetic component vote back into the next LLM decision. Keep this
    # compatibility no-op for exact behavioral separation from current-LHI.
    return dict(consensus_state or {})


def apply_component_consensus(
    component_gate: dict[str, Any],
    component_stats: dict[str, Any],
    dataset_rules: dict[str, Any],
    consensus_state: dict[str, Any] | None,
) -> dict[str, Any]:
    return dict(component_gate or {})


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

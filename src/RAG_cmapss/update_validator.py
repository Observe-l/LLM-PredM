from __future__ import annotations

from collections import Counter
from typing import Any


def validate_update(
    update_result: dict[str, Any],
    feedback: dict[str, Any],
    context: dict[str, Any] | None = None,
    min_confidence: float = 0.55,
    min_supporting_reflections: int = 0,
) -> dict[str, Any]:
    context = context or {}
    label = str(feedback.get("feedback_label", ""))
    threshold = str(update_result.get("threshold_update", ""))
    timing = str(update_result.get("timing_update", ""))
    violations: list[str] = []

    expected_threshold = {
        "too_early": "higher",
        "over_maintenance": "higher",
        "missed_HPC_maintenance": "lower",
        "missed_fan_maintenance": "lower",
        "missed_maintenance_unknown": "lower",
        "correct_maintenance": "unchanged",
    }.get(label)
    if label.startswith("missed_") and str(feedback.get("missed_maintenance_cause", "")) in {
        "maintenance_scheduled_at_or_after_failure",
        "monitoring_without_maintenance",
        "continued_operation_without_maintenance",
        "lhi_gate_not_triggered_before_failure",
    }:
        expected_threshold = "unchanged"
    expected_timing = {
        "too_early": "delay",
        "over_maintenance": "delay",
        "missed_HPC_maintenance": "earlier",
        "missed_fan_maintenance": "earlier",
        "missed_maintenance_unknown": "earlier",
        "correct_maintenance": "keep",
    }.get(label)

    if expected_threshold and threshold != expected_threshold:
        violations.append(f"threshold_update={threshold} conflicts with feedback_label={label}")
    if expected_timing and timing != expected_timing:
        violations.append(f"timing_update={timing} conflicts with feedback_label={label}")
    if float(update_result.get("confidence", 0.0)) < min_confidence:
        violations.append("update confidence below minimum")

    if min_supporting_reflections > 0:
        labels = Counter(str(row.get("feedback_label", "")) for row in context.get("reflection_rules", []))
        if labels.get(label, 0) < min_supporting_reflections:
            violations.append("not enough same-label reflection support")

    return {
        "valid": not violations,
        "violations": violations,
        "feedback_label": label,
        "expected_threshold_update": expected_threshold,
        "expected_timing_update": expected_timing,
    }

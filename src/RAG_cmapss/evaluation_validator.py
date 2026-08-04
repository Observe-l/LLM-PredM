from __future__ import annotations

from typing import Any

from .llm_policy_risk_tool import validate_policy


ALLOWED_VALUES = {
    "action_escalation_policy": {
        "neutral",
        "maintenance_when_risk_activated_and_component_supported",
    },
    "peak_offset_level": {"none", "small", "median", "large"},
    "monitoring_interval": {5, 10, 20},
}

def validate_and_apply_evaluation(
    *,
    decision: dict[str, Any],
    report: dict[str, Any],
    current_policy: dict[str, Any],
    minimum_support: int = 3,
) -> dict[str, Any]:
    """Apply the evaluation agent's policy decision.

    The evaluator selects the policy change. This function protects the
    executor from malformed fields/values;
    it does not impose causal-support, one-field, or cooldown rules.
    ``minimum_support`` is retained for CLI/backward compatibility and is
    intentionally unused.
    """
    del minimum_support
    policy = validate_policy(current_policy)
    patch = decision.get("policy_patch")
    patch = dict(patch) if isinstance(patch, dict) else {}
    violations: list[str] = []
    completed = int(report.get("engines_completed", 0))
    unknown = set(patch) - set(ALLOWED_VALUES)
    if unknown:
        violations.append(f"unsupported adaptive-policy fields: {sorted(unknown)}")
    for key, value in patch.items():
        normalized = int(value) if key == "monitoring_interval" and str(value).isdigit() else value
        patch[key] = normalized
        if key in ALLOWED_VALUES and normalized not in ALLOWED_VALUES[key]:
            violations.append(f"unsupported value for {key}: {value!r}")

    effective_patch = {
        key: value for key, value in patch.items() if policy.get(key) != value
    }
    applied = bool(effective_patch) and not violations
    updated = dict(policy)
    if applied:
        updated.update(effective_patch)
        updated["policy_revision"] = int(policy.get("policy_revision", 0) or 0) + 1
        updated["effective_from_engine"] = completed + 1
        updated["source"] = "periodic_evaluation_agent"
        recent = report.get("recent_window") or {}
        entry = {
            "checkpoint_id": report.get("checkpoint_id"),
            "engines_completed": completed,
            "score_status": report.get("score_status"),
            "recent_correct_rate": recent.get("correct_maintenance_rate"),
            "previous_correct_rate": (
                (report.get("previous_window") or {}).get("correct_maintenance_rate")
            ),
            "policy_patch": effective_patch,
            "reason": decision.get("reason"),
            "confidence": decision.get("confidence"),
            "observed_support": decision.get("observed_support"),
            "evidence_strength": report.get("evidence_strength"),
            "previous_peak_offset_level": policy.get("peak_offset_level"),
            "updated_peak_offset_level": updated.get("peak_offset_level"),
        }
        updated["evaluation_updates"] = list(policy.get("evaluation_updates", [])) + [entry]
    updated = validate_policy(updated)
    return {
        "valid": not violations,
        "applied": applied,
        "violations": violations,
        "requested_patch": patch,
        "applied_patch": effective_patch if applied else {},
        "decision_authority": "evaluation_agent",
        "structural_validation_only": True,
        "transition_validation": "none",
        "previous_policy_revision": policy.get("policy_revision", 0),
        "updated_policy_revision": updated.get("policy_revision", 0),
        "effective_from_engine": updated.get("effective_from_engine"),
        "updated_policy": updated,
    }

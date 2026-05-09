from __future__ import annotations

import re
from typing import Any


ALLOWED_ACTIONS = {
    "continue_normal_operation",
    "schedule_monitoring",
    "schedule_fan_maintenance",
    "schedule_HPC_maintenance",
}
STRONG_HPC_SENSORS = {"S7", "S11", "S3", "S9", "S14"}
STRONG_FAN_SENSORS = {"S8", "S13", "S15"}


def parse_t_plus(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"t\+(\d+)", value.strip())
        if match:
            return int(match.group(1))
    raise ValueError(f"Invalid action_time: {value!r}; expected null or t+N.")


def validate_action(
    action: dict[str, Any],
    case: dict[str, Any],
    dataset_rules: dict[str, Any],
    sensor_paths: list[dict[str, Any]],
    risk_gate: dict[str, Any] | None = None,
    lightgbm_risk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action_type = action.get("action_type")
    action_time = action.get("action_time")
    violations: list[str] = []
    warnings: list[str] = []

    if action_type not in ALLOWED_ACTIONS:
        violations.append(f"Unknown action_type: {action_type}")

    horizon = case.get("forecast_horizon", {})
    h_start = int(horizon.get("start", 1))
    h_end = int(horizon.get("end", 10**9))

    if action_type == "continue_normal_operation":
        if action_time is not None:
            violations.append("continue_normal_operation must have null action_time")
    elif action_type in ALLOWED_ACTIONS:
        if action_time is None:
            violations.append(f"{action_type} must have non-null action_time")
        else:
            try:
                t = parse_t_plus(action_time)
                if t is None or not (h_start <= t <= h_end):
                    violations.append("action_time is outside forecast_horizon")
            except ValueError as exc:
                violations.append(str(exc))

    if action_type in set(dataset_rules.get("disallowed_actions", [])):
        violations.append(f"{action_type} is disallowed by dataset policy")
    if action_type not in set(dataset_rules.get("allowed_actions", [])) and action_type in ALLOWED_ACTIONS:
        violations.append(f"{action_type} is not allowed by dataset policy")

    path_hypotheses = {str(p.get("hypothesis")) for p in sensor_paths}
    if action_type == "schedule_HPC_maintenance" and not _has_strong_sensor_evidence(
        sensor_paths, "HPC_related_degradation", STRONG_HPC_SENSORS
    ):
        violations.append("schedule_HPC_maintenance requires strong HPC sensor evidence from S7/S11/S3/S9/S14")
    if action_type == "schedule_fan_maintenance" and not _has_strong_sensor_evidence(
        sensor_paths, "Fan_related_degradation", STRONG_FAN_SENSORS
    ):
        violations.append("schedule_fan_maintenance requires strong Fan sensor evidence from S8/S13/S15")

    if action_type in {"schedule_fan_maintenance", "schedule_HPC_maintenance"}:
        if not path_hypotheses.intersection({"HPC_related_degradation", "Fan_related_degradation"}):
            violations.append("maintenance action lacks component evidence")
        if action.get("risk_hypothesis") == "critical_risk_hypothesis" and action.get("degradation_hypothesis") in {
            None,
            "",
            "uncertain_component_degradation",
        }:
            violations.append("critical_risk_hypothesis alone is insufficient to trigger fan/HPC maintenance")
        if risk_gate and not risk_gate.get("maintenance_candidate", False) and not _learned_risk_supports_maintenance(lightgbm_risk):
            violations.append("maintenance action lacks LightGBM or statistical risk support")

    if "uncertain_component_degradation" in path_hypotheses and not path_hypotheses.intersection(
        {"HPC_related_degradation", "Fan_related_degradation"}
    ):
        if action_type != "schedule_monitoring":
            violations.append("uncertain component evidence should choose schedule_monitoring")

    return {"valid": len(violations) == 0, "violations": violations, "warnings": warnings}


def _has_strong_sensor_evidence(
    sensor_paths: list[dict[str, Any]],
    hypothesis: str,
    strong_sensors: set[str],
) -> bool:
    return any(
        str(path.get("hypothesis")) == hypothesis and str(path.get("sensor", "")).upper() in strong_sensors
        for path in sensor_paths
    )


def _learned_risk_supports_maintenance(lightgbm_risk: dict[str, Any] | None) -> bool:
    if not lightgbm_risk:
        return False
    score = _to_float(lightgbm_risk.get("maintenance_risk_score")) or 0.0
    theta = _to_float(lightgbm_risk.get("theta_low")) or 0.4
    stage = str(lightgbm_risk.get("predicted_risk_stage", ""))
    decision = str(lightgbm_risk.get("risk_decision", ""))
    return bool(
        score >= max(theta, 0.60)
        and decision in {"activate_llm_agent", "activate_llm_agent_uncertain"}
        and stage in {"maintenance_window", "late_or_missed"}
    )


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

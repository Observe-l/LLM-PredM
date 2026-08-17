from __future__ import annotations

import re
from typing import Any

from .timing_policy import recommended_maintenance_time, recommended_monitoring_time


ALLOWED_ACTIONS = {
    "continue_normal_operation",
    "schedule_monitoring",
    "schedule_maintenance",
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
    component_gate: dict[str, Any] | None = None,
    llm_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action_type = action.get("action_type")
    action_time = action.get("action_time")
    violations: list[str] = []
    warnings: list[str] = []
    component_gate = component_gate or {}
    llm_policy = llm_policy or {}

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

    if action_type == "schedule_monitoring" and action_time is not None:
        recommended_monitoring = recommended_monitoring_time(case, llm_policy)
        if action_time != recommended_monitoring:
            violations.append(
                "schedule_monitoring action_time must equal "
                f"recommended_monitoring_time={recommended_monitoring}"
            )

    if action_type in {
        "schedule_maintenance",
        "schedule_HPC_maintenance",
        "schedule_fan_maintenance",
    } and action_time is not None:
        recommended = recommended_maintenance_time(case, llm_policy)
        if action_time != recommended:
            violations.append(
                f"maintenance action_time must equal recommended_maintenance_time={recommended}"
            )

    if action_type in set(dataset_rules.get("disallowed_actions", [])):
        violations.append(f"{action_type} is disallowed by dataset policy")
    if action_type not in set(dataset_rules.get("allowed_actions", [])) and action_type in ALLOWED_ACTIONS:
        violations.append(f"{action_type} is not allowed by dataset policy")

    # Component selection is intentionally left to the LLM's interpretation
    # of the supplied evidence paths.  The validator checks action schema,
    # timing, and dataset action permissions only; it must not reproduce a
    # numeric component gate.

    return {"valid": len(violations) == 0, "violations": violations, "warnings": warnings}

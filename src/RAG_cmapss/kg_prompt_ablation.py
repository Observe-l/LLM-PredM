from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .action_validator import ALLOWED_ACTIONS, validate_action
from .prompt_builder import build_risk_profile, compact_llm_policy, compact_risk_gate


NO_KG_SYSTEM_PROMPT = """You are a maintenance timing decision agent for zero-shot C-MAPSS predictive maintenance.

Use only the supplied forecast statistics, raw sensor-error statistics, and risk-tool output.
No component identity or graph-derived evidence is available in this experiment.

Choose exactly one action:
1. continue_normal_operation
2. schedule_monitoring
3. schedule_maintenance

Constraints:
- continue_normal_operation must have action_time = null.
- Every other action must use "t+N" within the forecast horizon.
- Use the active risk-tool result as the primary risk-stage evidence.
- Raw sensor-error statistics can support degradation risk, persistence, and timing, but not a subsystem identity.
- Choose schedule_maintenance when the risk and timing evidence justify maintenance; it requires no component selection.
- Use degradation_hypothesis = "unspecified_component_degradation".
- Return evidence_paths as an empty list because no retrieved paths are supplied.
- Keep reason under 30 words.
- Return only valid JSON without markdown.
"""


def build_no_kg_prompt(
    case: dict[str, Any],
    risk_gate: dict[str, Any],
    lightgbm_risk: dict[str, Any] | None = None,
    llm_policy: dict[str, Any] | None = None,
) -> str:
    sensor_profile = build_raw_sensor_profile(case)
    risk_tool = sanitize_risk_tool(lightgbm_risk)
    return f"""Forecast case:
{json.dumps({"case_id": case.get("case_id"), "forecast_horizon": case.get("forecast_horizon")}, indent=2)}

Forecast calibrated risk profile:
{json.dumps(build_risk_profile(case), indent=2)}

Trend and persistence profile:
{json.dumps(build_non_domain_trend_profile(case), indent=2)}

Raw sensor-error evidence:
{json.dumps(sensor_profile, indent=2)}

Statistical risk diagnostic:
{json.dumps(compact_risk_gate(risk_gate), indent=2)}

Active risk-tool evidence:
{json.dumps(risk_tool, indent=2)}

Risk policy:
{json.dumps(compact_llm_policy(llm_policy), indent=2)}

Return exactly one JSON object:
{{
  "action_type": "...",
  "action_time": "t+N or null",
  "risk_hypothesis": "...",
  "degradation_hypothesis": "unspecified_component_degradation",
  "confidence": 0.0,
  "evidence_paths": [],
  "reason": "under 30 words",
  "validation_status": "valid"
}}

Use peak_score_cycle for timing when it is inside the forecast horizon. Base the decision only on information shown."""


def build_raw_sensor_profile(case: dict[str, Any]) -> dict[str, Any]:
    sensors = case.get("sensor_evidence_statistics", {})
    keys = [
        "dominant_top_sensors",
        "top_sensor_at_peak_cycle",
        "sensor_presence_ratio",
        "sensor_mean_rank",
        "sensor_pattern_stability",
    ]
    return {key: sensors.get(key) for key in keys if key in sensors}


def build_non_domain_trend_profile(case: dict[str, Any]) -> dict[str, Any]:
    trend = case.get("trend_statistics", {})
    keys = [
        "slope",
        "delta_score",
        "relative_increase",
        "monotonicity",
        "volatility",
        "duration_above_unit_q95",
        "duration_above_unit_q99",
        "area_above_unit_q95",
        "area_above_unit_q99",
        "first_unit_q95_crossing_cycle",
        "first_unit_q99_crossing_cycle",
    ]
    profile = {key: trend.get(key) for key in keys if key in trend}
    stability = case.get("sensor_evidence_statistics", {}).get("sensor_pattern_stability")
    if stability is not None:
        profile["sensor_pattern_stability"] = stability
    return profile


def sanitize_risk_tool(risk_tool: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(risk_tool, dict):
        return {}
    keys = [
        "tool_name",
        "model_source",
        "maintenance_risk_score",
        "predicted_risk_stage",
        "confidence",
        "risk_decision",
        "theta_low",
        "theta_conf",
    ]
    return {key: risk_tool.get(key) for key in keys if key in risk_tool}


def neutral_validation(action: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    generic_rules = {
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "disallowed_actions": [],
    }
    result = validate_action(action, case, generic_rules, [])
    if action.get("evidence_paths") != []:
        result["violations"].append("evidence_paths must be empty when no paths are supplied")
        result["valid"] = False
    confidence = action.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        result["violations"].append("confidence must be a number in [0, 1]")
        result["valid"] = False
    return result


def production_validation(
    action: dict[str, Any],
    case: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Audit only; this KG-aware result must not modify the primary ablation action."""
    return validate_action(
        action,
        case,
        context.get("dataset_rules", {}),
        context.get("sensor_paths", []),
        risk_gate=context.get("risk_gate"),
        lightgbm_risk=context.get("lightgbm_risk"),
    )


def summarize_ablation(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if isinstance(record.get("ablation_action"), dict)]
    baseline_counts = Counter(record.get("baseline_action", {}).get("action_type") for record in completed)
    ablation_counts = Counter(record.get("ablation_action", {}).get("action_type") for record in completed)
    changed = [
        record
        for record in completed
        if record.get("baseline_action", {}).get("action_type")
        != record.get("ablation_action", {}).get("action_type")
    ]
    maintenance = {"schedule_fan_maintenance", "schedule_HPC_maintenance"}
    maintenance_to_monitoring = [
        record
        for record in completed
        if record.get("baseline_action", {}).get("action_type") in maintenance
        and record.get("ablation_action", {}).get("action_type") == "schedule_monitoring"
    ]
    neutral_valid = [
        bool(record.get("neutral_validation", {}).get("valid"))
        for record in completed
    ]
    production_valid = [
        bool(record.get("production_validation", {}).get("valid"))
        for record in completed
    ]
    n = len(completed)
    return {
        "selected_cases": len(records),
        "completed_cases": n,
        "failed_cases": len(records) - n,
        "baseline_action_counts": _clean_counter(baseline_counts),
        "no_kg_action_counts": _clean_counter(ablation_counts),
        "action_type_changed_count": len(changed),
        "action_type_changed_rate": round(len(changed) / n, 6) if n else None,
        "baseline_maintenance_to_monitoring_count": len(maintenance_to_monitoring),
        "baseline_maintenance_to_monitoring_rate": (
            round(len(maintenance_to_monitoring) / n, 6) if n else None
        ),
        "neutral_validation_pass_rate": (
            round(sum(neutral_valid) / n, 6) if n else None
        ),
        "production_policy_audit_pass_rate": (
            round(sum(production_valid) / n, 6) if n else None
        ),
    }


def _clean_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
        if key is not None
    }

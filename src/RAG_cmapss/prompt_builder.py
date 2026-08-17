from __future__ import annotations

import json
from typing import Any

from .timing_policy import maintenance_timing_profile


SYSTEM_PROMPT = """You are a KG-grounded maintenance decision agent for zero-shot C-MAPSS predictive maintenance.

You must choose exactly one action:
1. continue_normal_operation
2. schedule_monitoring
3. schedule_fan_maintenance
4. schedule_HPC_maintenance

Hard constraints:
- Do not choose an action disallowed by the dataset policy.
- continue_normal_operation must have action_time = null.
- schedule_monitoring must use the supplied recommended_monitoring_time.
- schedule_fan_maintenance and schedule_HPC_maintenance must set action_time exactly to the supplied recommended_maintenance_time.
- Maintenance component selection must be justified by the supplied KG evidence paths.
- The active risk tool is named in the provided evidence. Use that actual tool_name/model_source as the risk perception source.
- Do not choose a maintenance action that is disallowed by the dataset policy.
    - Do not use a numeric component ranking, component gate, FD identity, or deterministic component preference. Compare the evidence paths themselves.
- If the evidence paths do not support one component clearly, choose schedule_monitoring.
- Reflection memory is not action evidence. It is used only to train/update LightGBM tools after feedback.
- Maintenance requires an appropriate risk stage and evidence paths supporting the selected component.
- Historical feedback contains forecast-state features only; do not infer or mention hidden RUL.
- In evidence_paths, return only evidence IDs such as ["E1", "E3"], not full path text.
- Keep reason under 30 words.
- Return only valid JSON. Do not output markdown.
"""


def build_prompt(
    case: dict[str, Any],
    dataset_rules: dict[str, Any],
    sensor_paths: list[dict[str, Any]],
    action_paths: list[dict[str, Any]],
    component_evidence_statistics: dict[str, Any],
    risk_gate: dict[str, Any],
    component_gate: dict[str, Any],
    lightgbm_risk: dict[str, Any] | None = None,
    llm_policy: dict[str, Any] | None = None,
) -> str:
    top_sensor_paths = [
        {
            "id": f"E{idx}",
            "sensor": p.get("sensor"),
            "hypothesis": p.get("hypothesis"),
            "component": p.get("component"),
            "path": p.get("path_text"),
        }
        for idx, p in enumerate(sensor_paths[:8], start=1)
    ]
    action_path_texts = [
        {
            "id": f"A{idx}",
            "hypothesis": p.get("hypothesis"),
            "action_type": p.get("action_type"),
            "weight": p.get("weight"),
            "path": p.get("path_text"),
        }
        for idx, p in enumerate(action_paths[:6], start=1)
    ]
    risk_profile = build_risk_profile(case)
    trend_profile = build_trend_profile(case)
    timing_profile = maintenance_timing_profile(case, llm_policy)
    risk_tool = lightgbm_risk or {}
    risk_tool_name = risk_tool.get("tool_name") or "unknown_risk_tool"
    risk_model_source = risk_tool.get("model_source") or "unknown_source"
    mixed_fleet_instruction = (
        "Mixed-fleet rule: do not use the FD name to suppress a component. "
        "Select one explicit component from the strongest per-engine evidence paths. "
        "Uncertain evidence alone does not veto maintenance. If maintenance_escalation_required is true, "
        "the selected explicit component must be maintained."
        if dataset_rules.get("mixed_fleet")
        else ""
    )
    return f"""Forecast case:
{json.dumps({"case_id": case.get("case_id"), "forecast_horizon": case.get("forecast_horizon")}, indent=2)}

Forecast calibrated risk profile:
{json.dumps(risk_profile, indent=2)}

Trend and persistence profile:
{json.dumps(trend_profile, indent=2)}

Maintenance timing policy:
{json.dumps(timing_profile, indent=2)}

Dataset rules:
{json.dumps(dataset_rules, indent=2)}

Component evidence inventory (descriptive only; no numeric ranking):
{json.dumps(component_evidence_statistics, indent=2)}

Risk gate:
{json.dumps(compact_risk_gate(risk_gate), indent=2)}

Active risk-tool evidence from {risk_tool_name} / {risk_model_source}:
{json.dumps(lightgbm_risk or {}, indent=2)}

LLM policy:
{json.dumps(compact_llm_policy(llm_policy), indent=2)}

Top graph evidence paths:
{json.dumps(top_sensor_paths, indent=2)}

Retrieved action paths:
{json.dumps(action_path_texts, indent=2)}

Choose exactly one action and return JSON with this schema:
{{
  "action_type": "...",
  "action_time": "t+N or null",
  "risk_hypothesis": "...",
  "degradation_hypothesis": "...",
  "confidence": 0.0,
  "evidence_paths": ["E1", "E2"],
  "reason": "under 30 words",
  "validation_status": "valid"
}}

Important:
- {mixed_fleet_instruction}
- evidence_paths must contain only evidence IDs, not full path strings.
- action_time must be null only for continue_normal_operation.
- For schedule_monitoring, action_time must equal {timing_profile["recommended_monitoring_time"]}.
- For schedule_fan_maintenance or schedule_HPC_maintenance, action_time must equal {timing_profile["recommended_maintenance_time"]}; do not default to the horizon end.
- The action_type must agree with the reason: if the reason says maintenance is required, choose the corresponding maintenance action rather than schedule_monitoring.
- Use the Graph RAG paths to choose one explicit degradation component. This is an LLM judgment from evidence, not a numeric gate.
- Use {risk_tool_name} / {risk_model_source} as the primary risk-stage evidence. Do not cite a different risk tool.
- If risk_level is high_persistent, persistent_warning, or high_persistent_uncalibrated, weigh maintenance more strongly when the evidence paths support a component.
- The supplied maintenance_timing_policy determines recommended_maintenance_time and is mandatory.
- risk_decision only describes LLM activation and predicted_risk_stage is advisory. Do not infer late_or_missed from activation alone; use risk_gate for timing.
- Do not use FD001/FD002/FD003/FD004 names as component restrictions in mixed-fleet mode.
- If choosing schedule_monitoring, set action_time to {timing_profile["recommended_monitoring_time"]}.
- Use risk_gate only for timing/risk context; reflection memory and component metadata are not action gates.
- Do not use reflection anchors, historical feedback, or hidden RUL in action reasoning.
- Maintenance timing is forecast-grounded, not freely chosen: use {timing_profile["recommended_maintenance_time"]}.
"""


def build_risk_profile(case: dict[str, Any]) -> dict[str, Any]:
    risk = case.get("risk_statistics", {})
    multi = case.get("multi_score_statistics", {})
    return {
        "peak_score": risk.get("peak_score"),
        "peak_score_cycle": risk.get("peak_score_cycle"),
        "final_score": risk.get("final_score"),
        "current_d_rmse": risk.get("current_d_rmse"),
        "peak_d_rmse": risk.get("peak_d_rmse"),
        "unit_past_context_reliable": risk.get("unit_past_context_reliable"),
        "unit_past_q95": risk.get("unit_past_q95"),
        "unit_past_q99": risk.get("unit_past_q99"),
        "peak_minus_unit_q95": risk.get("peak_minus_unit_q95"),
        "peak_minus_unit_q99": risk.get("peak_minus_unit_q99"),
        "d_rmse_lhi_consistency": multi.get("d_rmse_lhi_consistency"),
    }


def build_trend_profile(case: dict[str, Any]) -> dict[str, Any]:
    trend = case.get("trend_statistics", {})
    sensors = case.get("sensor_evidence_statistics", {})
    return {
        "slope": trend.get("slope"),
        "delta_score": trend.get("delta_score"),
        "relative_increase": trend.get("relative_increase"),
        "monotonicity": trend.get("monotonicity"),
        "volatility": trend.get("volatility"),
        "duration_above_unit_q95": trend.get("duration_above_unit_q95"),
        "duration_above_unit_q99": trend.get("duration_above_unit_q99"),
        "area_above_unit_q95": trend.get("area_above_unit_q95"),
        "area_above_unit_q99": trend.get("area_above_unit_q99"),
        "first_unit_q95_crossing_cycle": trend.get("first_unit_q95_crossing_cycle"),
        "first_unit_q99_crossing_cycle": trend.get("first_unit_q99_crossing_cycle"),
        "sensor_pattern_stability": sensors.get("sensor_pattern_stability"),
        "hpc_sensor_presence_ratio": sensors.get("hpc_sensor_presence_ratio"),
        "fan_sensor_presence_ratio": sensors.get("fan_sensor_presence_ratio"),
        "conflict_sensor_presence_ratio": sensors.get("conflict_sensor_presence_ratio"),
    }


def compact_risk_gate(risk_gate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "risk_level",
        "maintenance_candidate",
        "unit_past_context_reliable",
        "peak_score",
        "peak_minus_unit_q95",
        "peak_minus_unit_q99",
        "duration_above_unit_q95",
        "duration_above_unit_q99",
        "slope",
        "d_rmse_lhi_consistency",
        "statistical_candidate",
        "reason",
    ]
    return {key: risk_gate.get(key) for key in keys if key in risk_gate}


def compact_llm_policy(llm_policy: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(llm_policy, dict):
        return {}
    keys = [
        "tool_name",
        "version",
        "source",
        "policy_type",
        "policy_revision",
        "effective_from_engine",
        "theta_conf",
        "maintenance_timing_policy",
        "peak_offset_level",
        "monitoring_interval",
        "missed_cause_counts",
        "reason",
    ]
    return {key: llm_policy.get(key) for key in keys if key in llm_policy}

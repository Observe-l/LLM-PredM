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
- schedule_monitoring must use the forecast-horizon end as its next review time.
- schedule_fan_maintenance and schedule_HPC_maintenance must set action_time exactly to the supplied recommended_maintenance_time.
- schedule_fan_maintenance requires retrieved evidence supporting Fan_related_degradation.
- schedule_HPC_maintenance requires retrieved evidence supporting HPC_related_degradation.
- The active risk tool is named in the provided evidence. Use that actual tool_name/model_source as the risk perception source.
- Do not choose a maintenance action that is disallowed by the dataset policy.
- A high or critical score alone is insufficient to choose the component; use graph/component evidence to select the maintainable component.
- Component_gate is diagnostic, not a hard veto. If risk-tool evidence indicates maintenance_window or late_or_missed and a dataset-allowed component has strong graph evidence, you may choose that maintenance action even when uncertain_component_degradation is also strong.
- In late_or_missed cases, strong graph evidence for a dataset-allowed maintainable component should outweigh uncertain_component_degradation unless there is stronger conflicting maintainable-component evidence.
- Reflection memory is not action evidence. It is used only to train/update LightGBM tools after feedback.
- Maintenance requires active risk-tool support and graph evidence for a dataset-allowed component.
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
            "score": p.get("score"),
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
    timing_profile = maintenance_timing_profile(case)
    risk_tool = lightgbm_risk or {}
    risk_tool_name = risk_tool.get("tool_name") or "unknown_risk_tool"
    risk_model_source = risk_tool.get("model_source") or "unknown_source"
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

Component evidence profile:
{json.dumps(component_evidence_statistics, indent=2)}

Decision gates:
{json.dumps({"risk_gate": compact_risk_gate(risk_gate), "component_gate": component_gate}, indent=2)}

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
- evidence_paths must contain only evidence IDs, not full path strings.
- action_time must be null only for continue_normal_operation.
- For schedule_monitoring, action_time must equal the forecast-horizon end.
- For schedule_fan_maintenance or schedule_HPC_maintenance, action_time must equal {timing_profile["recommended_maintenance_time"]}; do not default to the horizon end.
- The action_type must agree with the reason: if the reason says maintenance is required, choose the corresponding maintenance action rather than schedule_monitoring.
- Use Graph RAG paths and component_gate to choose the degradation hypothesis.
- Use {risk_tool_name} / {risk_model_source} as the primary risk-stage evidence. Do not cite a different risk tool.
- If LLM policy is present, use its peak_score boundary as risk/timing bias while still obeying graph evidence, dataset rules, and validation constraints.
- If LLM policy action_escalation_policy is maintenance_when_risk_activated_and_component_supported, do not repeat monitoring when the risk tool activated reasoning and strong graph evidence supports an allowed component; choose that maintenance action.
- maintenance_timing_policy=peak_score_cycle means maintenance must use the supplied recommended_maintenance_time.
- If the risk tool says normal or monitor_without_llm, choose schedule_monitoring or continue_normal_operation.
- If the risk tool score is below theta_low, or risk_decision is monitor_without_llm, do not choose fan/HPC maintenance.
- If the risk tool says maintenance_window or late_or_missed, prefer a dataset-allowed maintenance action when graph evidence strongly supports a maintainable component.
- Do not let uncertain_component_degradation alone override strong evidence for a dataset-allowed maintainable component in a late_or_missed case.
- For FD001, schedule_HPC_maintenance is allowed; if late_or_missed and HPC graph evidence is strong, choose schedule_HPC_maintenance even if uncertain_component_degradation has a higher path score.
- If FD001 disallows fan maintenance, do not choose schedule_fan_maintenance; use monitoring unless HPC graph evidence is strong enough to support schedule_HPC_maintenance.
- If choosing schedule_monitoring, set action_time to the forecast horizon end, e.g. "t+50".
- Use risk_gate only as a transparent statistical diagnostic, not as reflection memory.
- Do not use reflection anchors, historical feedback, or hidden RUL in action reasoning.
- Maintenance timing is forecast-grounded, not freely chosen: use {timing_profile["recommended_maintenance_time"]}, derived primarily from peak_score_cycle.
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
        "theta_conf",
        "action_escalation_policy",
        "maintenance_timing_policy",
        "missed_cause_counts",
        "reason",
    ]
    return {key: llm_policy.get(key) for key in keys if key in llm_policy}

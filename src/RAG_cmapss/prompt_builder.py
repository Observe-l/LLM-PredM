from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a KG-grounded maintenance decision agent for zero-shot C-MAPSS predictive maintenance.

You must choose exactly one action:
1. continue_normal_operation
2. schedule_monitoring
3. schedule_fan_maintenance
4. schedule_HPC_maintenance

Hard constraints:
- Do not choose an action disallowed by the dataset policy.
- continue_normal_operation must have action_time = null.
- schedule_monitoring, schedule_fan_maintenance, and schedule_HPC_maintenance must have action_time as a string like "t+20" within the forecast horizon.
- schedule_fan_maintenance requires retrieved evidence supporting Fan_related_degradation.
- schedule_HPC_maintenance requires retrieved evidence supporting HPC_related_degradation.
- If component evidence is uncertain and no strong sensor evidence supports a maintainable component, choose schedule_monitoring.
- The active risk tool (LightGBMRiskTool or LLMPolicyRiskTool) is the risk perception tool. Use it to decide whether the case is normal, early warning, maintenance window, or late/missed.
- A high or critical score alone is insufficient to trigger fan/HPC maintenance.
- Reflection memory is not action evidence. It is used only to train/update LightGBM tools after feedback.
- Maintenance requires learned risk support, strong component evidence, and no stronger conflicting evidence.
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
    return f"""Forecast case:
{json.dumps({"case_id": case.get("case_id"), "forecast_horizon": case.get("forecast_horizon")}, indent=2)}

Forecast calibrated risk profile:
{json.dumps(risk_profile, indent=2)}

Trend and persistence profile:
{json.dumps(trend_profile, indent=2)}

Dataset rules:
{json.dumps(dataset_rules, indent=2)}

Component evidence profile:
{json.dumps(component_evidence_statistics, indent=2)}

Decision gates:
{json.dumps({"risk_gate": compact_risk_gate(risk_gate), "component_gate": component_gate}, indent=2)}

Active risk-tool evidence:
{json.dumps(lightgbm_risk or {}, indent=2)}

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
- action_time must be null or a string like "t+20", never a bare number.
- The action_type must agree with the reason: if the reason says maintenance is required, choose the corresponding maintenance action rather than schedule_monitoring.
- Use Graph RAG paths and component_gate to choose the degradation hypothesis.
- Use the active risk tool as the primary risk-stage evidence.
- If the risk tool says normal or monitor_without_llm, choose schedule_monitoring or continue_normal_operation.
- If the risk tool says maintenance_window or late_or_missed and component_gate.component_supported is true, choose the supported maintenance action.
- Use risk_gate only as a transparent statistical diagnostic, not as reflection memory.
- Do not use reflection anchors, historical feedback, or hidden RUL in action reasoning.
- For maintenance timing, use peak_score_cycle unless it conflicts with the forecast horizon.
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

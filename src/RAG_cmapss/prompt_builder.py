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
- If component evidence is uncertain or conflicting, choose schedule_monitoring.
- If risk_gate.maintenance_candidate is false, do not choose schedule_fan_maintenance or schedule_HPC_maintenance.
- A high or critical score alone is insufficient to trigger fan/HPC maintenance.
- Reflection memory is auxiliary evidence; do not follow historical feedback blindly.
- Maintenance requires persistent statistical risk, supported component evidence, and no stronger conflicting evidence.
- Absolute peak-score cutoffs must come from reflection_peak_calibration, not from fixed dataset constants.
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
    reflection_rules: list[dict[str, Any]],
    component_evidence_statistics: dict[str, Any],
    risk_gate: dict[str, Any],
    component_gate: dict[str, Any],
    reflection_gate: dict[str, Any],
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
    timing_guidance = build_reflection_timing_guidance(case, reflection_rules)
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
{json.dumps({"risk_gate": risk_gate, "component_gate": component_gate, "reflection_gate": reflection_gate}, indent=2)}

Top graph evidence paths:
{json.dumps(top_sensor_paths, indent=2)}

Retrieved action paths:
{json.dumps(action_path_texts, indent=2)}

Label-balanced reflection anchors:
{json.dumps(compact_reflection_anchors(reflection_rules), indent=2)}

Reflection timing guidance:
{json.dumps(timing_guidance, indent=2)}

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
- Use risk_gate and component_gate as primary evidence.
- risk_gate.statistical_candidate is the online statistical signal; risk_gate.reflection_peak_calibration is the learned peak-score boundary from retrieved reflection memory.
- q95/q99 gap and persistence duration thresholds are calibrated by reflection memory with warm-up priors; use the reported calibration decisions rather than fixed score constants.
- Use reflection_gate and reflection anchors only as auxiliary calibration; do not copy suggested actions blindly.
- Use recommended_time_rule only after choosing the action type, as guidance for action_time.
- If top reflection timing guidance has a concrete suggested_action_time, use it unless it conflicts with hard constraints.
- For maintenance actions without concrete reflection timing, use peak_score_cycle; do not schedule maintenance at first_persistent_pattern_cycle.
- If risk_gate.maintenance_candidate is false, schedule_fan_maintenance and schedule_HPC_maintenance are invalid; choose schedule_monitoring.
- If reflection_peak_calibration has no learned_peak_score_min, treat this as cold start and choose schedule_monitoring.
- If risk_gate.maintenance_candidate is true and component_gate.component_supported is true, choose the supported maintenance action unless reflection_gate shows stronger too_early evidence.
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


def compact_reflection_anchors(reflection_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "feedback_label": item.get("feedback_label"),
            "peak_score": item.get("peak_score"),
            "retrieval_similarity": item.get("retrieval_similarity"),
            "peak_similarity": item.get("peak_similarity"),
            "suggested_action_type": item.get("then_revise_action_type"),
            "recommended_time_rule": item.get("recommended_time_rule"),
        }
        for item in reflection_rules[:5]
    ]


def build_reflection_timing_guidance(case: dict[str, Any], reflection_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guidance: list[dict[str, Any]] = []
    for item in reflection_rules[:5]:
        time_rule = item.get("recommended_time_rule")
        guidance.append(
            {
                "rule_id": item.get("rule_id"),
                "feedback_label": item.get("feedback_label"),
                "peak_score": item.get("peak_score"),
                "then_revise_action_type": item.get("then_revise_action_type"),
                "recommended_time_rule": time_rule,
                "suggested_action_time": resolve_time_rule(case, str(time_rule or "")),
                "time_rule_interpretation": time_rule_interpretation(str(time_rule or "")),
            }
        )
    return guidance


def resolve_time_rule(case: dict[str, Any], time_rule: str) -> str | None:
    summary = case.get("forecast_summary", {})
    if time_rule in {"schedule_monitoring_at_peak_cycle", "keep_similar_timing"}:
        return _clamp_t_plus(summary.get("peak_score_cycle"), case)
    if time_rule == "peak_score_cycle_minus_margin":
        return _offset_t_plus(summary.get("peak_score_cycle"), case, offset=-3)
    if time_rule == "first_persistent_pattern_cycle":
        return _offset_t_plus(summary.get("peak_score_cycle"), case, offset=-3)
    if time_rule == "first_warning_crossing_cycle":
        return _clamp_t_plus(summary.get("peak_score_cycle"), case)
    if time_rule == "first_critical_crossing_cycle":
        return _clamp_t_plus(summary.get("first_critical_crossing_cycle"), case)
    return None


def time_rule_interpretation(time_rule: str) -> str:
    meanings = {
        "schedule_monitoring_at_peak_cycle": "monitor again near the forecasted peak score cycle",
        "keep_similar_timing": "reuse timing from similar correct maintenance cases; use peak cycle when uncertain",
        "peak_score_cycle_minus_margin": "schedule maintenance shortly before the peak score cycle",
        "first_persistent_pattern_cycle": "schedule maintenance when persistent high-risk evidence first appears",
        "first_warning_crossing_cycle": "schedule at the first warning crossing",
        "first_critical_crossing_cycle": "schedule at the first critical crossing",
    }
    return meanings.get(time_rule, "")


def _clamp_t_plus(value: Any, case: dict[str, Any]) -> str | None:
    number = _parse_t_plus(value)
    if number is None:
        return None
    horizon = case.get("forecast_horizon", {})
    start = int(horizon.get("start", 1))
    end = int(horizon.get("end", number))
    return f"t+{min(max(number, start), end)}"


def _offset_t_plus(value: Any, case: dict[str, Any], offset: int) -> str | None:
    number = _parse_t_plus(value)
    if number is None:
        return None
    return _clamp_t_plus(f"t+{number + offset}", case)


def _parse_t_plus(value: Any) -> int | None:
    if not isinstance(value, str) or not value.startswith("t+"):
        return None
    try:
        return int(value[2:])
    except ValueError:
        return None

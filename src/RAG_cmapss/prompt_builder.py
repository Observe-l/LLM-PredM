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
- A high or critical score alone is insufficient to trigger fan/HPC maintenance.
- Use similar historical feedback cases to distinguish weak-risk early maintenance from maintenance that is still justified but should be timed closer to persistent or peak risk.
- Treat missed maintenance feedback as positive evidence for the reflected maintenance action when the current peak score, persistence, and component evidence are similar.
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
    action_guidance = build_reflection_action_guidance(reflection_rules)
    timing_guidance = build_reflection_timing_guidance(case, reflection_rules)
    return f"""Forecast case:
{json.dumps(case, indent=2)}

Dataset rules:
{json.dumps(dataset_rules, indent=2)}

Retrieved graph evidence paths:
{json.dumps(top_sensor_paths, indent=2)}

Retrieved action paths:
{json.dumps(action_path_texts, indent=2)}

Similar historical feedback cases from reflection memory:
{json.dumps(reflection_rules, indent=2)}

Reflection action guidance:
{json.dumps(action_guidance, indent=2)}

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
- Compare the current peak_score against historical peak_score first; this is the primary reflection matching signal.
- Use Reflection action guidance as the direct action_type reference after comparing peak_score.
- Use recommended_time_rule only after choosing the action type, as guidance for action_time.
- If top reflection timing guidance has a concrete suggested_action_time, use it unless it conflicts with hard constraints.
- Treat reflection records with feedback_label=correct_maintenance as positive anchors.
- Treat feedback_label=too_early with then_revise_action_type=schedule_monitoring as evidence that similar weak or non-persistent cases should be monitored instead of maintained.
- Treat feedback_label starting with missed_ and then_revise_action_type=schedule_HPC_maintenance as evidence that similar high-peak, persistent, strong-HPC cases should choose schedule_HPC_maintenance, not repeated monitoring.
- If a reflection record recommends a time rule, align action_time with that rule when it fits the current forecast horizon.
"""


def build_reflection_action_guidance(reflection_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "feedback_label": item.get("feedback_label"),
            "peak_score": item.get("peak_score"),
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
        return _clamp_t_plus(summary.get("first_persistent_pattern_cycle"), case) or _clamp_t_plus(
            summary.get("first_warning_crossing_cycle"), case
        )
    if time_rule == "first_warning_crossing_cycle":
        return _clamp_t_plus(summary.get("first_warning_crossing_cycle"), case)
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

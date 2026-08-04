from __future__ import annotations

import json
from typing import Any

from .ollama_client import extract_json, ollama_chat


def run_evaluation_agent(
    report: dict[str, Any],
    *,
    model: str,
    ollama_url: str,
    temperature: float,
    timeout: int,
    num_predict: int,
    format_json: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    payload = {
        "task": (
            "Evaluate whether correct_maintenance performance changed relative to the previous "
            "completed-engine window and decide whether to update the adaptive policy."
        ),
        "evaluation_report": report,
        "allowed_policy_values": {
            "action_escalation_policy": [
                "neutral",
                "maintenance_when_risk_activated_and_component_supported",
            ],
            "peak_offset_level": {
                "none": 0,
                "small": 5,
                "median": 10,
                "large": 15,
            },
            "allowed_next_peak_offset_levels": ["none", "small", "median", "large"],
            "monitoring_interval": [5, 10, 20],
        },
        "mandatory_decision_rules": [
            "Use correct_maintenance_rate as the primary score and missed causes as diagnosis.",
            "Maintenance timing is always max(t+1, t_peak - delta); delta levels are none=0, small=5, median=10, large=15 cycles.",
            "The evaluator may directly choose any peak_offset_level; adjacent-level movement is not required.",
            "evidence_strength and Wilson overlap describe uncertainty but are not hard gates.",
            "The first evaluation may update policy when its absolute late/early/monitoring counts are compelling even though no previous window exists.",
            "No cooldown or minimum policy-exposure period is required after an update.",
            "Use recent_evaluation_history to identify repeated directions, reversals, and whether earlier updates helped.",
            "For late scheduled maintenance, choose a larger offset; small is mild, median is clear/repeated, and large is severe/repeated.",
            "For too-early maintenance, choose a smaller offset using the same severity logic.",
            "Do not default to small when the observed severity or repeated history supports median or large.",
            "For every peak_offset_level update, MUST follow timing_error_balance.dominant_observed_timing_direction exactly.",
            "For a timing update, primary_driver MUST equal timing_error_balance.required_primary_driver_for_timing_update.",
            "If dominant_observed_timing_direction is increase_offset_schedule_earlier, only move toward a larger offset.",
            "If dominant_observed_timing_direction is decrease_offset_schedule_later, only move toward a smaller offset.",
            "If dominant_observed_timing_direction is no_timing_change, do not change peak_offset_level based on timing alone.",
            "Do not label late_timing as the primary driver when recent_too_early_count exceeds recent_late_count, and vice versa.",
            "Do not treat a one-engine score change or overlapping Wilson intervals as strong timing evidence.",
            "Monitoring-related missed maintenance may justify action escalation or a shorter monitoring interval.",
            "LHI-gate misses cannot be repaired by these adaptive prompt policies.",
            "You are the sole policy decision maker and may update any combination of allowed policy fields.",
            "Use an empty policy_patch when you decide that no policy change is appropriate.",
            "Do not modify LHI trigger, graph weights, evidence count, or base prompt text.",
        ],
        "required_output": {
            "status": "improving/stable/degrading/insufficient_evidence",
            "primary_driver": (
                "late_timing/early_timing/monitoring/lhi_gate/component/infrastructure/none"
            ),
            "recommended_action": (
                "no_change/update_action_policy/update_timing_policy/"
                "shorten_monitoring_interval"
            ),
            "observed_support": "copy the relevant observed counts from evaluation_report",
            "policy_patch": "object containing zero or more allowed policy fields",
            "reason": "short string",
            "confidence": "float 0..1",
        },
    }
    prompt = (
        "You are the periodic evaluation agent for an online predictive-maintenance experiment.\n"
        "All statistics were computed deterministically from completed engines only.\n"
        "The mandatory_decision_rules are constraints, not optional advice. Return one valid JSON "
        "object that obeys every rule. Do not calculate new metrics and do not rewrite the prompt.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )
    if dry_run:
        return {
            "status": report.get("score_status", "insufficient_evidence"),
            "primary_driver": "none",
            "recommended_action": "no_change",
            "policy_patch": {},
            "reason": "Dry-run evaluation does not update adaptive policy.",
            "confidence": 1.0,
            "source": "dry_run_evaluation_agent",
            "prompt": prompt,
            "raw_output": "",
        }
    try:
        messages = [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        raw_attempts: list[str] = []
        violations: list[str] = []
        parsed: dict[str, Any] = {}
        for repair_index in range(3):
            raw = ollama_chat(
                messages,
                model=model,
                url=ollama_url,
                temperature=temperature,
                timeout=timeout,
                num_predict=num_predict,
                format_json=format_json,
                think=False,
            )
            raw_attempts.append(raw)
            parsed = extract_json(raw)
            violations = _decision_rule_violations(parsed, report)
            if not violations:
                break
            if repair_index < 2:
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your JSON contradicts mandatory_decision_rules:\n- "
                                + "\n- ".join(violations)
                                + "\nRe-evaluate the same report. Return a corrected JSON decision. "
                                "The corrected policy_patch and primary_driver must obey every rule."
                            ),
                        },
                    ]
                )
        if violations:
            parsed = {
                "status": report.get("score_status", "insufficient_evidence"),
                "primary_driver": "infrastructure",
                "recommended_action": "no_change",
                "policy_patch": {},
                "observed_support": {"noncompliance": violations},
                "reason": (
                    "Evaluation LLM remained inconsistent with mandatory rules after two "
                    "self-correction attempts; no policy update was applied."
                ),
                "confidence": 0.0,
            }
        result = {
            "status": str(parsed.get("status", "insufficient_evidence")),
            "primary_driver": str(parsed.get("primary_driver", "none")),
            "recommended_action": str(parsed.get("recommended_action", "no_change")),
            "policy_patch": (
                parsed.get("policy_patch") if isinstance(parsed.get("policy_patch"), dict) else {}
            ),
            "observed_support": parsed.get("observed_support"),
            "reason": str(parsed.get("reason", ""))[:500],
            "confidence": _bounded(parsed.get("confidence"), 0.0),
            "source": (
                "noncompliant_evaluation_agent_no_update"
                if violations else "ollama_evaluation_agent"
            ),
            "prompt": prompt,
            "raw_output": raw_attempts[-1],
            "raw_attempts": raw_attempts,
            "instruction_repair_count": max(len(raw_attempts) - 1, 0),
            "instruction_violations_after_repair": violations,
        }
    except Exception as exc:
        result = {
            "status": "insufficient_evidence",
            "primary_driver": "infrastructure",
            "recommended_action": "no_change",
            "policy_patch": {},
            "observed_support": None,
            "reason": f"Evaluation-agent call failed: {exc}",
            "confidence": 0.0,
            "source": "fallback_evaluation_agent",
            "prompt": prompt,
            "raw_output": "",
        }
    return result


def _bounded(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(min(max(number, 0.0), 1.0), 6)


def _decision_rule_violations(
    decision: dict[str, Any],
    report: dict[str, Any],
) -> list[str]:
    patch = decision.get("policy_patch")
    patch = patch if isinstance(patch, dict) else {}
    requested = patch.get("peak_offset_level")
    current = str((report.get("current_policy") or {}).get("peak_offset_level", "none"))
    if requested is None or str(requested) == current:
        return []

    requested = str(requested)
    violations: list[str] = []
    levels = ["none", "small", "median", "large"]
    if requested not in levels or current not in levels:
        return [f"peak_offset_level transition is unsupported: {current} -> {requested}"]
    balance = report.get("timing_error_balance") or {}
    direction = str(balance.get("dominant_observed_timing_direction", "no_timing_change"))
    step = levels.index(requested) - levels.index(current)
    if direction == "increase_offset_schedule_earlier" and step <= 0:
        violations.append(f"timing direction requires a larger offset, got {current} -> {requested}")
    elif direction == "decrease_offset_schedule_later" and step >= 0:
        violations.append(f"timing direction requires a smaller offset, got {current} -> {requested}")
    elif direction == "no_timing_change":
        violations.append("balanced timing errors must not change peak_offset_level")
    required_driver = str(balance.get("required_primary_driver_for_timing_update", "none"))
    if str(decision.get("primary_driver", "none")) != required_driver:
        violations.append(
            f"primary_driver must be {required_driver} for this timing update"
        )
    return violations

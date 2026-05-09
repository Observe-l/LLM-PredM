from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lightgbm_features import extract_lightgbm_features, read_reflection_training_rows
from .llm_policy_risk_tool import DEFAULT_POLICY, validate_policy
from .ollama_client import extract_json, ollama_chat


RISK_FEATURE_GUIDE = {
    "peak_score": "Highest risk score in the forecast window.",
    "peak_minus_unit_q95": "How far the peak exceeds this engine's q95 baseline.",
    "peak_minus_unit_q99": "How far the peak exceeds this engine's q99 baseline.",
    "duration_above_unit_q95": "How many cycles stay above the engine q95 baseline.",
    "duration_above_unit_q99": "How many cycles stay above the engine q99 baseline.",
    "area_above_unit_q95": "Integrated persistence above q95.",
    "slope": "Trend direction and steepness.",
    "monotonicity": "Whether degradation is consistently moving in one direction.",
    "volatility": "Instability of the forecasted risk signal.",
    "hpc_sensor_presence_ratio": "Fraction of dominant sensors associated with HPC evidence.",
    "fan_sensor_presence_ratio": "Fraction of dominant sensors associated with Fan evidence.",
    "hpc_path_score": "Graph RAG support strength for HPC-related degradation.",
    "fan_path_score": "Graph RAG support strength for Fan-related degradation.",
    "component_conflict_score": "Evidence conflict across candidate components.",
    "dominance_margin": "Gap between the strongest and competing component hypotheses.",
    "risk_gate_statistical_candidate": "Rule-derived statistical warning candidate flag.",
    "risk_gate_maintenance_candidate": "Rule-derived maintenance candidate flag.",
    "component_gate_supported": "Whether graph/component evidence supports a concrete maintenance component.",
}


def llm_design_policy_tool(
    *,
    case: dict[str, Any],
    context: dict[str, Any],
    model: str,
    ollama_url: str,
    temperature: float,
    timeout: int,
    num_predict: int,
    format_json: bool,
    dry_run: bool = False,
    reflection_rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Ask the LLM to create parameters for a reusable LLMPolicyRiskTool."""
    features = extract_lightgbm_features(case, context)
    reflection_context = reflection_policy_context(reflection_rules_path) if reflection_rules_path else {}
    fallback = {
        **DEFAULT_POLICY,
        "source": "default_after_llm_policy_design_unavailable",
        "design_case_id": case.get("case_id"),
    }
    if dry_run:
        fallback["source"] = "dry_run_default_llm_policy_tool"
        return validate_policy(fallback)

    payload = {
        "case_id": case.get("case_id"),
        "task": "Design reusable parameters for LLMPolicyRiskTool. Do not compute a one-off case score.",
        "tool_contract": [
            "The LLMPolicyRiskTool will run deterministically after this step.",
            "Output weights, normalizers, and activation thresholds.",
            "Later cases will call the tool directly; the LLM will not recompute the score each time.",
            "Reflection memory may be used only to calibrate policy parameters and thresholds.",
            "Reflection memory must not be treated as action evidence in Graph RAG maintenance reasoning.",
        ],
        "hard_requirements": [
            "peak_score must be included in weights.",
            "peak_score must have the largest weight and should normally be at least 0.45.",
            "Higher peak_score means higher degradation and failure risk.",
            "Other features are auxiliary: persistence, q95/q99 excess, trend, and component localization.",
            "Component graph scores may help localize a component but must not dominate risk severity.",
        ],
        "available_feature_meanings": RISK_FEATURE_GUIDE,
        "normalizer_schema": {
            "cap": "clamp positive value / scale into [0,1]",
            "positive_cap": "same as cap but explicitly treats negatives as zero",
            "signed_cap": "maps roughly [-scale, +scale] into [0,1]",
            "binary": "0 or 1",
        },
        "current_case_features_for_initial_calibration": {key: features.get(key) for key in RISK_FEATURE_GUIDE},
        "reflection_memory_for_policy_design": reflection_context,
        "suggested_safe_start": DEFAULT_POLICY,
        "required_output": {
            "source": "llm_policy_tool_design",
            "policy_family": "short name",
            "bias": "float 0..0.4",
            "weights": {"peak_score": "dominant float, largest weight", "other_feature": "float"},
            "normalizers": {"feature": {"kind": "cap|positive_cap|signed_cap|binary", "scale": "float if needed"}},
            "theta_low": "activation threshold, usually 0.35..0.55",
            "theta_conf": "uncertainty threshold, usually 0.2..0.4",
            "maintenance_window_threshold": "maintenance-stage threshold, usually 0.55..0.70",
            "score_formula": "human-readable formula that explicitly includes peak_score",
            "reflection_calibration_summary": "how reflection memory changed weights/thresholds",
            "reason": "short design rationale",
        },
    }
    prompt = (
        "You are designing a persistent deterministic risk policy tool for zero-shot predictive maintenance.\n"
        "Return only JSON parameters for LLMPolicyRiskTool. Do not output maintenance_risk_score; "
        "the tool will calculate scores later.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )
    raw = ""
    try:
        raw = ollama_chat(
            [{"role": "system", "content": "Return only valid JSON."}, {"role": "user", "content": prompt}],
            model=model,
            url=ollama_url,
            temperature=temperature,
            timeout=timeout,
            num_predict=num_predict,
            format_json=format_json,
            think=False,
        )
        parsed = extract_json(raw)
        parsed["source"] = "llm_policy_tool_design"
        parsed["design_case_id"] = case.get("case_id")
        parsed["raw_output"] = raw
        parsed["prompt"] = prompt
        parsed["reflection_policy_context"] = reflection_context
        return validate_policy(parsed, fallback=fallback)
    except Exception as exc:
        retry = _retry_compact_policy_tool_design(
            features=features,
            reflection_context=reflection_context,
            model=model,
            ollama_url=ollama_url,
            temperature=temperature,
            timeout=timeout,
            format_json=format_json,
            fallback=fallback,
        )
        if retry is not None:
            retry["first_attempt_error"] = str(exc)
            retry["first_raw_output"] = raw
            return retry
        fallback["error"] = str(exc)
        fallback["raw_output"] = raw
        fallback["prompt"] = prompt
        fallback["reflection_policy_context"] = reflection_context
        return validate_policy(fallback)


def llm_update_policy_tool(
    *,
    current_policy: dict[str, Any],
    feedback: dict[str, Any],
    case: dict[str, Any],
    action: dict[str, Any],
    context: dict[str, Any],
    reflection_rule: dict[str, Any] | None,
    reflection_rules_path: str | Path,
    model: str,
    ollama_url: str,
    temperature: float,
    timeout: int,
    num_predict: int,
    format_json: bool,
    dry_run: bool = False,
    disable_llm: bool = False,
) -> dict[str, Any]:
    """Ask the LLM whether and how to update LLMPolicyRiskTool after non-correct feedback."""
    fallback = _fallback_policy_tool_update(current_policy, feedback, case, context)
    if dry_run or disable_llm:
        return fallback
    features = extract_lightgbm_features(case, context, action=action, feedback=feedback)
    reflection_context = reflection_policy_context(reflection_rules_path)
    payload = {
        "case_id": case.get("case_id"),
        "feedback": feedback,
        "action": {
            "action_type": action.get("action_type"),
            "action_time": action.get("action_time"),
            "validation_status": action.get("validation_status"),
        },
        "current_policy": current_policy,
        "current_tool_output": context.get("lightgbm_risk", {}),
        "current_case_features": {key: features.get(key) for key in RISK_FEATURE_GUIDE},
        "reflection_rule_candidate": reflection_rule,
        "reflection_memory_for_policy_update": reflection_context,
        "hard_requirements": [
            "Return update_policy=false if the current policy already matches the feedback.",
            "For too_early/over_maintenance: usually raise theta_low or reduce auxiliary weights; do not remove peak_score.",
            "For missed maintenance: usually lower theta_low or increase severity/persistence sensitivity.",
            "peak_score must remain in weights, largest, and normally >=0.45.",
            "Do not use reflection rows as future action evidence; use them only to update tool parameters.",
        ],
        "required_output": {
            "update_policy": "bool",
            "updated_policy": "full LLMPolicyRiskTool policy object if update_policy=true, otherwise current policy",
            "reason": "short explanation",
            "confidence": "float 0..1",
        },
    }
    prompt = (
        "You are the LLMPolicyRiskTool update controller.\n"
        "Decide whether this non-correct feedback requires updating the deterministic policy tool. "
        "Return only valid JSON. The updated policy must remain reusable across later cases.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )
    try:
        raw = ollama_chat(
            [{"role": "system", "content": "Return only valid JSON."}, {"role": "user", "content": prompt}],
            model=model,
            url=ollama_url,
            temperature=temperature,
            timeout=timeout,
            num_predict=num_predict,
            format_json=format_json,
            think=False,
        )
        parsed = extract_json(raw)
        update_policy = bool(parsed.get("update_policy", False))
        updated = parsed.get("updated_policy") if update_policy else current_policy
        if not isinstance(updated, dict):
            updated = current_policy
            update_policy = False
        updated = validate_policy(updated, fallback=current_policy or DEFAULT_POLICY)
        updated["source"] = "llm_policy_tool_update" if update_policy else current_policy.get("source", "llm_policy_tool")
        updated["last_update"] = {
            "case_id": case.get("case_id"),
            "feedback_label": feedback.get("feedback_label"),
            "reason": str(parsed.get("reason", ""))[:500],
            "confidence": _clamp_float(parsed.get("confidence"), default=0.5),
        }
        return {
            "update_policy": update_policy,
            "updated_policy": updated,
            "reason": str(parsed.get("reason", ""))[:500],
            "confidence": _clamp_float(parsed.get("confidence"), default=0.5),
            "source": "llm_policy_tool_update_controller",
            "raw_output": raw,
            "prompt": prompt,
        }
    except Exception as exc:
        fallback["error"] = str(exc)
        fallback["prompt"] = prompt
        return fallback


def initial_risk_policy(
    *,
    case: dict[str, Any],
    context: dict[str, Any],
    source: str = "initial_risk_policy",
) -> dict[str, Any]:
    """Deterministic initial risk policy for the non-LLM cold-start experiment arm."""
    features = extract_lightgbm_features(case, context)
    component_gate = context.get("component_gate", {})
    peak_score = _num(features.get("peak_score"))
    q95_gap = max(_num(features.get("peak_minus_unit_q95")), 0.0)
    duration_q95 = max(_num(features.get("duration_above_unit_q95")), 0.0)
    monotonicity = max(_num(features.get("monotonicity")), 0.0)
    component_support = 1.0 if component_gate.get("component_supported") else 0.0
    peak_contribution = min(max(peak_score, 0.0), 1.5) / 1.5 * 0.60
    gap_contribution = min(q95_gap / 0.5, 1.0) * 0.15
    duration_contribution = min(duration_q95 / 20.0, 1.0) * 0.10
    trend_contribution = min(monotonicity, 1.0) * 0.05
    component_contribution = component_support * 0.05
    score = 0.03 + peak_contribution + gap_contribution + duration_contribution + trend_contribution + component_contribution
    components: list[dict[str, Any]] = [
        {"name": "peak_score", "value": peak_score, "contribution": round(peak_contribution, 6)},
        {"name": "peak_minus_unit_q95", "value": q95_gap, "contribution": round(gap_contribution, 6)},
        {"name": "duration_above_unit_q95", "value": duration_q95, "contribution": round(duration_contribution, 6)},
        {"name": "monotonicity", "value": monotonicity, "contribution": round(trend_contribution, 6)},
        {"name": "component_supported", "value": bool(component_support), "contribution": round(component_contribution, 6)},
    ]
    if component_gate.get("component_supported"):
        components.append({"name": "component_role", "value": "localization_only", "contribution": 0.0})
    return {
        "source": source,
        "policy_family": "deterministic_peak_score_initial_policy",
        "maintenance_risk_score": round(min(max(score, 0.02), 0.98), 6),
        "score_formula": (
            "0.03 + 0.60*min(max(peak_score,0),1.5)/1.5 + "
            "0.15*min(max(peak_minus_unit_q95,0)/0.5,1) + "
            "0.10*min(duration_above_unit_q95/20,1) + 0.05*monotonicity + 0.05*component_supported"
        ),
        "score_components": components,
        "risk_threshold_policy": {"theta_low": 0.4, "maintenance_window_threshold": 0.6},
        "confidence": 0.35,
        "severity_timing_assessment": (
            f"peak_score={peak_score:.6g} is the primary severity anchor; auxiliary q95_gap={q95_gap:.6g}, "
            f"duration_q95={duration_q95:.6g}."
        ),
        "reason": "Initial/fallback cold-start risk policy with peak_score as the dominant term.",
    }


def llm_decide_adaptation(
    *,
    feedback: dict[str, Any],
    case: dict[str, Any],
    action: dict[str, Any],
    context: dict[str, Any],
    reflection_rule: dict[str, Any] | None,
    reflection_rules_path: str | Path,
    active_risk_model_path: str | Path | None,
    model: str,
    ollama_url: str,
    temperature: float,
    timeout: int,
    num_predict: int,
    format_json: bool,
    dry_run: bool = False,
    disable_llm: bool = False,
) -> dict[str, Any]:
    """Ask the LLM controller whether this feedback should retrain LightGBM or update thresholds."""
    stats = reflection_memory_stats(reflection_rules_path)
    fallback = _fallback_adaptation_decision(feedback, stats)
    if dry_run or disable_llm:
        return fallback

    payload = {
        "case_id": case.get("case_id"),
        "feedback": feedback,
        "action": {
            "action_type": action.get("action_type"),
            "action_time": action.get("action_time"),
            "validation_status": action.get("validation_status"),
        },
        "lightgbm_risk": context.get("lightgbm_risk", {}),
        "active_risk_model_path": str(active_risk_model_path) if active_risk_model_path else None,
        "reflection_rule_candidate": reflection_rule,
        "reflection_memory_stats": stats,
        "component_evidence_statistics": context.get("component_evidence_statistics", {}),
        "component_gate": context.get("component_gate", {}),
        "decision_options": {
            "retrain_lightgbm_risk": "Train/retrain LightGBMRiskTool from reflection_rules.csv for later decisions.",
            "call_lightgbm_update_tool": "Use LightGBMUpdateTool to produce threshold/timing/component update operations.",
            "do_both": "Use when feedback reveals both a risk-model data need and a threshold/timing correction.",
            "do_neither": "Use when feedback is correct and current model/policy does not need adaptation.",
        },
        "training_guardrails": [
            "Retraining needs at least a few labeled reflection rows and preferably both positive and negative risk labels.",
            "Missed maintenance usually suggests update_tool and may justify retraining.",
            "Too-early or over-maintenance feedback usually suggests threshold/timing update and may justify retraining.",
            "Correct maintenance can still justify retraining if the model is absent and enough diverse labels exist.",
        ],
        "required_output": {
            "retrain_lightgbm_risk": "bool",
            "call_lightgbm_update_tool": "bool",
            "reason": "short string",
            "expected_update_focus": "threshold/timing/component/model/none or combination",
            "min_training_rows": "integer suggested minimum rows before retraining",
            "confidence": "float 0..1",
        },
    }
    prompt = (
        "You are the online adaptation controller in a zero-shot agentic predictive-maintenance system.\n"
        "At every feedback step, decide whether to retrain LightGBMRiskTool, call LightGBMUpdateTool, both, or neither.\n"
        "Reflection memory is a training/update buffer, not direct action evidence.\n"
        "Return only valid JSON matching required_output.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )
    try:
        raw = ollama_chat(
            [{"role": "system", "content": "Return only valid JSON."}, {"role": "user", "content": prompt}],
            model=model,
            url=ollama_url,
            temperature=temperature,
            timeout=timeout,
            num_predict=num_predict,
            format_json=format_json,
            think=False,
        )
        parsed = extract_json(raw)
        return {
            "retrain_lightgbm_risk": bool(parsed.get("retrain_lightgbm_risk", fallback["retrain_lightgbm_risk"])),
            "call_lightgbm_update_tool": bool(parsed.get("call_lightgbm_update_tool", fallback["call_lightgbm_update_tool"])),
            "reason": str(parsed.get("reason", ""))[:300],
            "expected_update_focus": str(parsed.get("expected_update_focus", fallback["expected_update_focus"]))[:120],
            "min_training_rows": max(2, int(parsed.get("min_training_rows", fallback["min_training_rows"]))),
            "confidence": _clamp_float(parsed.get("confidence"), default=0.5),
            "source": "llm_adaptation_controller",
            "raw_output": raw,
            "prompt": prompt,
            "reflection_memory_stats": stats,
        }
    except Exception as exc:
        fallback["source"] = "fallback_after_llm_adaptation_error"
        fallback["error"] = str(exc)
        fallback["prompt"] = prompt
        return fallback


def reflection_memory_stats(path: str | Path) -> dict[str, Any]:
    rows = read_reflection_training_rows(path)
    positive = {"correct_maintenance", "missed_HPC_maintenance", "missed_fan_maintenance", "missed_maintenance_unknown"}
    negative = {"too_early", "over_maintenance"}
    labels: dict[str, int] = {}
    for row in rows:
        label = str(row.get("feedback_label", ""))
        labels[label] = labels.get(label, 0) + 1
    return {
        "row_count": len(rows),
        "label_counts": labels,
        "positive_risk_rows": sum(labels.get(label, 0) for label in positive),
        "negative_risk_rows": sum(labels.get(label, 0) for label in negative),
        "has_two_risk_classes": any(labels.get(label, 0) for label in positive)
        and any(labels.get(label, 0) for label in negative),
    }


def reflection_policy_context(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"row_count": 0, "label_counts": {}, "compact_rules": []}
    rows = read_reflection_training_rows(path)
    stats = reflection_memory_stats(path)
    compact_rules = [_compact_reflection_row(row) for row in rows]
    return {
        **stats,
        "usage": (
            "Use these rows only to calibrate this case-specific risk policy. "
            "They are not direct action evidence for the downstream Graph RAG maintenance decision."
        ),
        "compact_rules": compact_rules,
    }


def _compact_reflection_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "feedback_label",
        "previous_action_type",
        "then_revise_action_type",
        "recommended_time_rule",
        "then_adjust_threshold",
        "then_adjust_component_preference",
        "peak_score",
        "peak_minus_unit_q95",
        "peak_minus_unit_q99",
        "duration_above_unit_q95",
        "duration_above_unit_q99",
        "slope",
        "monotonicity",
        "dominant_component",
        "hpc_path_score",
        "fan_path_score",
        "component_conflict_score",
        "dominance_margin",
    ]
    return {key: row.get(key, "") for key in keys}


def _retry_compact_policy_tool_design(
    *,
    features: dict[str, Any],
    reflection_context: dict[str, Any],
    model: str,
    ollama_url: str,
    temperature: float,
    timeout: int,
    format_json: bool,
    fallback: dict[str, Any],
) -> dict[str, Any] | None:
    prompt = (
        "Return one compact JSON object only. No prose, no markdown.\n"
        "Design reusable parameters for LLMPolicyRiskTool. Do not compute maintenance_risk_score.\n"
        "Required keys: source, policy_family, bias, weights, normalizers, theta_low, theta_conf, "
        "maintenance_window_threshold, score_formula, reason.\n"
        "peak_score must be in weights, must be the largest weight, and should be >= 0.45. "
        "Higher peak_score means higher failure risk. Other factors are auxiliary.\n\n"
        + json.dumps(
            {
                "features_for_calibration": {key: features.get(key) for key in RISK_FEATURE_GUIDE},
                "reflection_memory_for_policy_design": reflection_context,
                "example_policy_shape": DEFAULT_POLICY,
            },
            indent=2,
            default=str,
        )
    )
    try:
        raw = ollama_chat(
            [{"role": "system", "content": "Return only valid JSON."}, {"role": "user", "content": prompt}],
            model=model,
            url=ollama_url,
            temperature=temperature,
            timeout=timeout,
            num_predict=768,
            format_json=format_json,
            think=False,
        )
        parsed = extract_json(raw)
    except Exception:
        return None
    parsed["source"] = "llm_policy_tool_design_retry"
    parsed["raw_output"] = raw
    parsed["prompt"] = prompt
    parsed["reflection_policy_context"] = reflection_context
    return validate_policy(parsed, fallback=fallback)


def _fallback_adaptation_decision(feedback: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    label = str(feedback.get("feedback_label", ""))
    non_correct = label != "correct_maintenance"
    return {
        "retrain_lightgbm_risk": bool(stats.get("row_count", 0) >= 4 and stats.get("has_two_risk_classes")),
        "call_lightgbm_update_tool": non_correct,
        "reason": "Fallback controller: update non-correct feedback; retrain when reflection memory has enough two-class labels.",
        "expected_update_focus": _expected_update_focus(label),
        "min_training_rows": 4,
        "confidence": 0.45,
        "source": "fallback_adaptation_controller",
        "reflection_memory_stats": stats,
    }


def _fallback_policy_tool_update(
    current_policy: dict[str, Any],
    feedback: dict[str, Any],
    case: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    updated = validate_policy(dict(current_policy or DEFAULT_POLICY), fallback=DEFAULT_POLICY)
    label = str(feedback.get("feedback_label", ""))
    if label in {"too_early", "over_maintenance"}:
        updated["theta_low"] = _clamp_float(float(updated.get("theta_low", 0.4)) + 0.05, default=0.45, low=0.1, high=0.8)
        updated["reason"] = "Fallback policy update after early/over-maintenance feedback: raised activation threshold."
    elif label.startswith("missed_"):
        updated["theta_low"] = _clamp_float(float(updated.get("theta_low", 0.4)) - 0.05, default=0.35, low=0.1, high=0.8)
        weights = dict(updated.get("weights") or {})
        weights["peak_score"] = max(float(weights.get("peak_score", 0.6)), 0.6)
        weights["duration_above_unit_q95"] = max(float(weights.get("duration_above_unit_q95", 0.1)), 0.12)
        updated["weights"] = weights
        updated = validate_policy(updated, fallback=DEFAULT_POLICY)
        updated["reason"] = "Fallback policy update after missed maintenance feedback: lowered threshold and strengthened severity sensitivity."
    else:
        return {
            "update_policy": False,
            "updated_policy": updated,
            "reason": f"Fallback policy update skipped for feedback_label={label}.",
            "confidence": 0.35,
            "source": "fallback_policy_tool_update_controller",
        }
    updated["source"] = "fallback_policy_tool_update"
    updated["last_update"] = {
        "case_id": case.get("case_id"),
        "feedback_label": label,
        "tool_score": (context.get("lightgbm_risk") or {}).get("maintenance_risk_score"),
    }
    return {
        "update_policy": True,
        "updated_policy": updated,
        "reason": updated["reason"],
        "confidence": 0.35,
        "source": "fallback_policy_tool_update_controller",
    }


def _expected_update_focus(feedback_label: str) -> str:
    if feedback_label == "correct_maintenance":
        return "model" if feedback_label else "none"
    if feedback_label in {"too_early", "over_maintenance"}:
        return "threshold/timing/model"
    if feedback_label.startswith("missed_"):
        return "threshold/timing/component/model"
    if feedback_label == "wrong_component":
        return "component/model"
    return "threshold/model"


def _clamp_float(value: Any, default: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(min(max(number, low), high), 6)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

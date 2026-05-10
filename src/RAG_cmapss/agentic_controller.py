from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lightgbm_features import read_reflection_training_rows
from .ollama_client import extract_json, ollama_chat


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

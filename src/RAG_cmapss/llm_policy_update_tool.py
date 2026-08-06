from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lightgbm_features import read_reflection_training_rows
from .llm_policy_risk_tool import initial_policy, load_policy, validate_policy
from .ollama_client import extract_json, ollama_chat

UPDATE_STRENGTH_DELTAS = {"none": 0.0, "small": 0.05, "medium": 0.15, "large": 0.30}


class LLMPolicyUpdateTool:
    """LLM updater for the experiment-local LHI activation threshold."""

    def __init__(self, policy_path: str | Path):
        self.policy_path = Path(policy_path)

    def predict(
        self,
        *,
        feedback: dict[str, Any],
        case: dict[str, Any],
        action: dict[str, Any],
        current_policy: dict[str, Any] | None = None,
        reflection_rules_path: str | Path | None = None,
        model: str,
        ollama_url: str,
        temperature: float,
        timeout: int,
        num_predict: int,
        format_json: bool,
        dry_run: bool = False,
        disable_llm: bool = False,
    ) -> dict[str, Any]:
        policy = validate_policy(current_policy or load_policy(self.policy_path) or initial_policy())
        label = str(feedback.get("feedback_label", ""))
        peak = _case_peak_score(case)
        previous_threshold = float(policy.get("peak_threshold", 0.25))
        missed_cause = str(feedback.get("missed_maintenance_cause") or "")
        fallback = _policy_update_result(
            policy=policy,
            label=label,
            missed_cause=missed_cause,
            peak=peak,
            case_id=case.get("case_id"),
            parsed={},
            source=(
                "disabled_llm_policy_update"
                if dry_run or disable_llm
                else "fallback_llm_policy_update"
            ),
            reason="LLM policy update was not run.",
        )
        if dry_run or disable_llm:
            return _persist_and_return(self.policy_path, fallback)

        payload = {
            "task": "Update the LHI activation threshold after non-correct maintenance feedback.",
            "current_case": {
                "case_id": case.get("case_id"),
                "dataset_subset": case.get("dataset_subset"),
                "peak_lhi": peak,
                "previous_peak_threshold": previous_threshold,
                "threshold_was_crossed": bool(peak > previous_threshold),
                "peak_lhi_cycle": _risk_value(case, "peak_score_cycle"),
                "dominant_component": _risk_value(case, "dominant_component"),
            },
            "previous_action": {
                "action_type": action.get("action_type"),
                "action_time": action.get("action_time"),
                "target_component": action.get("target_component"),
                "reason": action.get("reason"),
            },
            "feedback": feedback,
            "current_policy": {
                "policy_type": policy.get("policy_type"),
                "peak_threshold": previous_threshold,
                "missed_cause_counts": policy.get("missed_cause_counts"),
            },
            "online_feedback_statistics": _feedback_history(reflection_rules_path),
            "update_guidance": [
                "Update only peak_threshold; choose lower/higher/unchanged and none/small/medium/large.",
                "For missed maintenance caused by the LHI gate, lower the threshold when the peak did not cross it.",
                "For too_early or over_maintenance, consider raising the threshold when the gate was crossed at a low peak.",
                "If the current case already crossed the threshold, keep it unchanged unless history supports a separate threshold problem.",
                "Do not update after correct_maintenance; this tool is called only for non-correct feedback.",
            ],
            "required_output": {
                "update_policy": "bool",
                "threshold_update": "lower/higher/unchanged",
                "update_strength": "none/small/medium/large",
                "threshold_causality": "blocked_by_threshold/over_activated_by_threshold/action_reasoning_or_component_or_timing/uncertain",
                "reason": "short string",
                "confidence": "float 0..1",
            },
        }
        prompt = (
            "You are the LLM policy update controller for a zero-shot predictive-maintenance experiment.\n"
            "Update the experiment-local LHI activation threshold using the supplied feedback. Return only valid JSON.\n\n"
            + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
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
            result = _policy_update_result(
                policy=policy,
                label=label,
                missed_cause=missed_cause,
                peak=peak,
                case_id=case.get("case_id"),
                parsed=parsed,
                source="llm_policy_update_controller",
                reason=str(parsed.get("reason", ""))[:500],
            )
            result["prompt"] = prompt
            result["raw_output"] = raw
        except Exception as exc:
            result = fallback
            result["source"] = "fallback_after_llm_policy_update_error"
            result["reason"] = f"LLM policy update failed. Error: {exc}"
            result["prompt"] = prompt
        return _persist_and_return(self.policy_path, result)


def _policy_update_result(
    *,
    policy: dict[str, Any],
    label: str,
    missed_cause: str,
    peak: float,
    case_id: Any,
    parsed: dict[str, Any],
    source: str,
    reason: str,
) -> dict[str, Any]:
    threshold_update = str(parsed.get("threshold_update", "unchanged"))
    strength = str(parsed.get("update_strength", "none"))
    if threshold_update not in {"lower", "higher", "unchanged"}:
        threshold_update = "unchanged"
    if strength not in UPDATE_STRENGTH_DELTAS:
        strength = "none"
    previous = float(policy.get("peak_threshold", 0.25))
    crossed = peak > previous
    causality = str(parsed.get("threshold_causality", "uncertain"))[:120]
    is_missed = label.startswith("missed_")
    is_early = label in {"too_early", "over_maintenance"}
    if is_missed and (crossed or causality == "action_reasoning_or_component_or_timing"):
        threshold_update, strength = "unchanged", "none"
    if is_early and threshold_update == "lower":
        threshold_update, strength = "unchanged", "none"
    if strength == "none" or threshold_update == "unchanged":
        proposed = previous
        threshold_update, strength = "unchanged", "none"
    elif threshold_update == "lower":
        proposed = max(1e-6, previous - UPDATE_STRENGTH_DELTAS[strength])
    else:
        proposed = previous + UPDATE_STRENGTH_DELTAS[strength]
    updated = dict(policy)
    _record_feedback(updated, label, peak, missed_cause)
    updated["peak_threshold"] = round(proposed, 6)
    updated["source"] = source
    updated["updates"] = list(updated.get("updates", [])) + [
        {
            "case_id": case_id,
            "feedback_label": label,
            "missed_maintenance_cause": missed_cause or None,
            "peak_lhi": round(float(peak), 6),
            "previous_peak_threshold": round(previous, 6),
            "updated_peak_threshold": round(proposed, 6),
            "threshold_update": threshold_update,
            "update_strength": strength,
            "source": source,
        }
    ]
    updated = validate_policy(updated)
    return {
        "tool_name": "LLMPolicyUpdateTool",
        "update_policy": proposed != previous,
        "update_threshold": proposed != previous,
        "updated_policy": updated,
        "reason": reason,
        "confidence": _bounded(parsed.get("confidence"), 0.0, 0.0, 1.0),
        "missed_maintenance_cause": missed_cause or None,
        "threshold_update": threshold_update,
        "update_strength": strength,
        "threshold_delta": round(proposed - previous, 6),
        "previous_peak_threshold": round(previous, 6),
        "updated_peak_threshold": round(proposed, 6),
        "threshold_causality": causality,
        "source": source,
    }


def _persist_and_return(policy_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    updated = validate_policy(result["updated_policy"])
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
    result["updated_policy"] = updated
    return result


def _record_feedback(policy: dict[str, Any], label: str, peak: float, missed_cause: str) -> None:
    if label in {"too_early", "over_maintenance"}:
        policy["early_anchor_count"] = int(policy.get("early_anchor_count", 0)) + 1
        policy["early_peak_max"] = _max_optional(policy.get("early_peak_max"), peak)
    elif label == "correct_maintenance":
        policy["correct_anchor_count"] = int(policy.get("correct_anchor_count", 0)) + 1
        policy["positive_peak_min"] = _min_optional(policy.get("positive_peak_min"), peak)
    elif label.startswith("missed_"):
        policy["missed_anchor_count"] = int(policy.get("missed_anchor_count", 0)) + 1
        policy["positive_peak_min"] = _min_optional(policy.get("positive_peak_min"), peak)
        if missed_cause:
            counts = dict(policy.get("missed_cause_counts", {}))
            counts[missed_cause] = int(counts.get(missed_cause, 0)) + 1
            policy["missed_cause_counts"] = counts


def _feedback_history(path: str | Path | None, limit: int = 40) -> dict[str, Any]:
    rows = read_reflection_training_rows(path) if path is not None else []
    label_counts: dict[str, int] = {}
    cause_counts: dict[str, int] = {}
    compact: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("feedback_label", ""))
        cause = str(row.get("missed_maintenance_cause") or "")
        label_counts[label] = label_counts.get(label, 0) + 1
        if label.startswith("missed_") and cause:
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
        compact.append(
            {
                "feedback_label": label,
                "previous_action_type": row.get("previous_action_type"),
                "previous_action_time": row.get("previous_action_time"),
                "missed_maintenance_cause": cause or None,
                "maintenance_timing_status": row.get("maintenance_timing_status"),
                "signed_cycle_margin": _num(row.get("signed_cycle_margin"), None),
            }
        )
    total = len(rows)
    missed_total = sum(v for k, v in label_counts.items() if k.startswith("missed_"))
    return {
        "row_count": total,
        "label_counts": label_counts,
        "label_rates": {
            key: round(value / total, 6) if total else 0.0
            for key, value in sorted(label_counts.items())
        },
        "missed_cause_counts": cause_counts,
        "missed_cause_rates_among_missed": {
            key: round(value / missed_total, 6) if missed_total else 0.0
            for key, value in sorted(cause_counts.items())
        },
        "recent_rows": compact[-limit:],
    }


def _case_peak_score(case: dict[str, Any]) -> float:
    value = _risk_value(case, "peak_score")
    return float(_num(value, 0.0) or 0.0)


def _risk_value(case: dict[str, Any], key: str) -> Any:
    risk = case.get("risk_statistics", {})
    value = risk.get(key)
    if value in {None, ""}:
        value = case.get("forecast_summary", {}).get(key)
    return value


def _max_optional(current: Any, value: float) -> float:
    return round(float(value) if current is None else max(float(current), float(value)), 6)


def _min_optional(current: Any, value: float) -> float:
    return round(float(value) if current is None else min(float(current), float(value)), 6)


def _bounded(value: Any, default: float, low: float, high: float) -> float:
    number = _num(value, default)
    return round(min(max(float(number), low), high), 6)


def _num(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

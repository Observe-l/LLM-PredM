from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lightgbm_features import read_reflection_training_rows
from .llm_policy_risk_tool import initial_policy, load_policy, validate_policy
from .ollama_client import extract_json, ollama_chat


class LLMPolicyUpdateTool:
    """Update action and timing policy from experiment-local feedback.

    Score-threshold adaptation intentionally does not live here. The upstream
    LHI trigger is the experiment's only score gate.
    """

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
        missed_cause = str(feedback.get("missed_maintenance_cause") or "")
        peak = _case_peak_score(case)
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
            "task": (
                "Update only maintenance action-escalation and timing policy. "
                "There is no peak-score threshold; the upstream LHI trigger is the only score gate."
            ),
            "current_case": {
                "case_id": case.get("case_id"),
                "dataset_subset": case.get("dataset_subset"),
                "peak_lhi": peak,
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
                "action_escalation_policy": policy.get("action_escalation_policy"),
                "maintenance_timing_policy": policy.get("maintenance_timing_policy"),
                "missed_cause_counts": policy.get("missed_cause_counts"),
            },
            "online_feedback_statistics": _feedback_history(reflection_rules_path),
            "update_guidance": [
                "Do not propose or discuss a peak threshold. It has been removed.",
                "Use missed_maintenance_cause as the primary causal diagnosis.",
                (
                    "For monitoring_without_maintenance or "
                    "continued_operation_without_maintenance, use "
                    "action_policy_update='escalate_when_component_supported'."
                ),
                (
                    "For maintenance_scheduled_at_or_after_failure, use "
                    "timing_policy_update='use_peak_score_cycle'."
                ),
                (
                    "For lhi_gate_not_triggered_before_failure, leave both policies unchanged; "
                    "this updater does not alter the upstream LHI trigger."
                ),
                "For too_early, do not suppress LLM activation; timing must still be grounded in the forecast peak.",
            ],
            "required_output": {
                "update_policy": "bool",
                "action_policy_update": "unchanged/escalate_when_component_supported",
                "timing_policy_update": "unchanged/use_peak_score_cycle",
                "reason": "short string",
                "confidence": "float 0..1",
            },
        }
        prompt = (
            "You are the reflection policy controller for a zero-shot predictive-maintenance experiment.\n"
            "The upstream LHI trigger is the only score gate. Peak threshold has been removed.\n"
            "Update only action escalation and maintenance timing. Return only valid JSON.\n\n"
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
    action_update = str(parsed.get("action_policy_update", "unchanged"))
    timing_update = str(parsed.get("timing_policy_update", "unchanged"))
    if action_update not in {"unchanged", "escalate_when_component_supported"}:
        action_update = "unchanged"
    if timing_update not in {"unchanged", "use_peak_score_cycle"}:
        timing_update = "unchanged"

    # Deterministic causal constraints take precedence over LLM wording.
    if missed_cause in {"monitoring_without_maintenance", "continued_operation_without_maintenance"}:
        action_update = "escalate_when_component_supported"
    if missed_cause == "maintenance_scheduled_at_or_after_failure":
        timing_update = "use_peak_score_cycle"
    if missed_cause == "lhi_gate_not_triggered_before_failure":
        action_update = "unchanged"
        timing_update = "unchanged"

    updated = dict(policy)
    _record_feedback(updated, label, peak, missed_cause)
    if action_update == "escalate_when_component_supported":
        updated["action_escalation_policy"] = (
            "maintenance_when_risk_activated_and_component_supported"
        )
    if timing_update == "use_peak_score_cycle":
        updated["maintenance_timing_policy"] = "peak_score_cycle"
    updated["source"] = source
    updated["updates"] = list(updated.get("updates", [])) + [
        {
            "case_id": case_id,
            "feedback_label": label,
            "missed_maintenance_cause": missed_cause or None,
            "peak_lhi": round(float(peak), 6),
            "action_policy_update": action_update,
            "timing_policy_update": timing_update,
            "source": source,
        }
    ]
    updated = validate_policy(updated)
    changed = action_update != "unchanged" or timing_update != "unchanged"
    return {
        "tool_name": "LLMPolicyUpdateTool",
        "update_policy": changed,
        "updated_policy": updated,
        "reason": reason,
        "confidence": _bounded(parsed.get("confidence"), 0.0, 0.0, 1.0),
        "missed_maintenance_cause": missed_cause or None,
        "action_policy_update": action_update,
        "timing_policy_update": timing_update,
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

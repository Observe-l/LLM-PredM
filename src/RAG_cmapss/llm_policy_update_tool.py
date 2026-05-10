from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lightgbm_features import read_reflection_training_rows
from .llm_policy_risk_tool import load_policy, validate_policy
from .ollama_client import extract_json, ollama_chat


UPDATE_STRENGTH_DELTAS = {
    "none": 0.0,
    "small": 0.05,
    "medium": 0.15,
    "large": 0.3,
}


class LLMPolicyUpdateTool:
    """Ask the LLM how to update the experiment-local peak_score boundary.

    This tool updates only LLMPolicyRiskTool. It does not update LightGBM and it
    does not use q95/q99 threshold parameters. Reflection rows are provided only
    as this experiment's online feedback history.
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
        policy = validate_policy(current_policy or load_policy(self.policy_path) or {"peak_threshold": 0.5})
        label = str(feedback.get("feedback_label", ""))
        peak = _case_peak_score(case)
        history = _peak_history(reflection_rules_path)
        previous_threshold = float(policy["peak_threshold"])
        update_history = _update_history_stats(policy)
        blocked_missed_summary = _blocked_missed_summary(history, previous_threshold)
        early_trigger_summary = _early_trigger_summary(history, previous_threshold)
        is_early_feedback = label in {"too_early", "over_maintenance"}
        threshold_diagnostic = {
            "peak_score": round(float(peak), 6),
            "previous_peak_threshold": round(previous_threshold, 6),
            "peak_minus_threshold": round(float(peak) - previous_threshold, 6),
            "threshold_minus_peak": round(previous_threshold - float(peak), 6),
            "threshold_was_crossed": bool(float(peak) >= previous_threshold),
            "policy_gate_status": (
                "already_activated_by_policy"
                if float(peak) >= previous_threshold
                else "blocked_by_policy_threshold"
            ),
            "counterfactual": (
                "Raising peak_threshold above this peak_score could have prevented this current too-early maintenance activation."
                if is_early_feedback and float(peak) >= previous_threshold
                else
                "Changing peak_threshold would not have changed whether this current case activated LLM reasoning."
                if float(peak) >= previous_threshold
                else "Lowering peak_threshold could have changed whether this current case activated LLM reasoning."
            ),
            "previous_action_type": action.get("action_type"),
            "recent_update_history": update_history,
            "blocked_missed_peak_score_summary_at_current_threshold": blocked_missed_summary,
            "early_maintenance_peak_score_summary_at_current_threshold": early_trigger_summary,
        }
        fallback = _no_update_result(
            policy=policy,
            label=label,
            peak=peak,
            case_id=case.get("case_id"),
            source="disabled_llm_policy_update" if (dry_run or disable_llm) else "fallback_llm_policy_update",
            reason="LLM policy update was not run.",
        )
        if dry_run or disable_llm:
            return _persist_and_return(self.policy_path, fallback)

        payload = {
            "task": "Update the LLMPolicyRiskTool peak_score threshold after non-correct maintenance feedback.",
            "current_case": {
                "case_id": case.get("case_id"),
                "dataset_subset": case.get("dataset_subset"),
                "peak_score": peak,
                "peak_score_cycle": _risk_value(case, "peak_score_cycle"),
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
                "peak_threshold": policy.get("peak_threshold"),
                "positive_peak_min": policy.get("positive_peak_min"),
                "early_peak_max": policy.get("early_peak_max"),
                "missed_anchor_count": policy.get("missed_anchor_count"),
                "early_anchor_count": policy.get("early_anchor_count"),
                "correct_anchor_count": policy.get("correct_anchor_count"),
            },
            "threshold_diagnostic": threshold_diagnostic,
            "reflection_memory_peak_score_history": history,
            "update_strength_options": {
                "none": "do not change peak_threshold",
                "small": "change peak_threshold by 0.05",
                "medium": "change peak_threshold by 0.15",
                "large": "change peak_threshold by 0.30",
            },
            "update_guidance": [
                "Update only peak_threshold for LLMPolicyRiskTool.",
                "Use peak_score as the boundary variable: lower threshold triggers maintenance review more often; higher threshold triggers less often.",
                "threshold_causality='blocked_by_threshold' means the current case stayed below peak_threshold, so the LLM maintenance-reasoning agent was not activated by the policy gate.",
                "If threshold_was_crossed is true, the current case was not blocked by the threshold gate; do not label the current case as blocked_by_threshold.",
                "If feedback is missed maintenance and threshold_causality is action_reasoning_or_component_or_timing, choose update_strength='none', threshold_update='unchanged', and update_policy=false.",
                "If feedback is missed maintenance, never choose threshold_update='higher' or threshold_causality='over_activated_by_threshold' for the current update; raising threshold would make missed maintenance harder to catch.",
                "For missed maintenance, lower peak_threshold only when the threshold likely blocked LLM activation, especially when threshold_was_crossed is false.",
                "For missed maintenance, if threshold_was_crossed is true and previous_action_type was schedule_monitoring or a wrong maintenance action, the default choice should be threshold_update='unchanged' because the failure is more likely LLM action reasoning, component choice, or timing rather than threshold.",
                "For missed maintenance, when threshold_was_crossed is true, lower the threshold only if online reflection history shows separate missed cases below the current threshold; otherwise keep threshold_update='unchanged'.",
                "Do not lower threshold merely to increase review frequency when this current case already crossed the threshold; that would not fix this case's failure mode.",
                "For too_early or over_maintenance, threshold_was_crossed means the policy gate allowed a maintenance action too early; consider raising peak_threshold, especially when peak_score is low or online history shows repeated too_early cases above the current threshold.",
                "For too_early or over_maintenance, do not use threshold_causality='blocked_by_threshold'; use threshold_causality='over_activated_by_threshold' when the threshold was too permissive, or action_reasoning_or_component_or_timing when the threshold was not the likely cause.",
                "Use early_maintenance_peak_score_summary only for too_early or over_maintenance feedback; do not let prior too_early cases override the direction implied by a current missed maintenance feedback.",
                "Use update_policy=false, threshold_update='unchanged', and update_strength='none' when threshold was not the likely cause.",
                "Do not default to small when the current threshold is clearly outside the useful online evidence range; repeated blocked missed cases or a large threshold_minus_peak may justify medium or large.",
                "Use large only for a strong threshold mismatch, such as a threshold far above the missed-maintenance peak_score range; use medium for clear repeated threshold mismatch; use small for isolated or near-boundary evidence.",
                "Choose update_strength from the evidence in the current case and online reflection history; none means no threshold update, small means a cautious adjustment, medium means a meaningful adjustment, and large means a strong adjustment.",
                "Do not treat medium as the default update strength: small and large are equally valid choices when the evidence supports them.",
                "Do not output a numeric threshold. Output threshold_update plus update_strength; the tool will apply the numeric delta.",
                "Use reflection_memory_peak_score_history as online evidence from this experiment only.",
                "Do not use q95, q99, persistent duration, or LightGBM parameters.",
                "Do not update threshold after correct_maintenance; this tool should be called only for non-correct feedback.",
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
            "Your job is to decide whether and how to update the LLMPolicyRiskTool peak_score threshold after feedback.\n"
            "The policy is simple: cases with peak_score >= peak_threshold may activate the maintenance-reasoning agent; "
            "cases below the threshold are monitored without LLM action reasoning.\n"
            "Return only valid JSON matching required_output. Do not output an exact numeric threshold; choose direction and strength only. "
            "Use update_strength='none' when no threshold update is justified.\n\n"
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
            result = _validated_llm_result(
                parsed=parsed,
                policy=policy,
                label=label,
                peak=peak,
                case_id=case.get("case_id"),
                prompt=prompt,
                raw=raw,
            )
        except Exception as exc:
            result = _no_update_result(
                policy=policy,
                label=label,
                peak=peak,
                case_id=case.get("case_id"),
                source="fallback_after_llm_policy_update_error",
                reason=f"LLM policy update failed; kept previous threshold. Error: {exc}",
            )
            result["prompt"] = prompt
        return _persist_and_return(self.policy_path, result)


def _validated_llm_result(
    *,
    parsed: dict[str, Any],
    policy: dict[str, Any],
    label: str,
    peak: float,
    case_id: Any,
    prompt: str,
    raw: str,
) -> dict[str, Any]:
    previous_threshold = float(policy["peak_threshold"])
    update_policy = bool(parsed.get("update_policy", False))
    threshold_update = str(parsed.get("threshold_update", "unchanged"))
    if threshold_update not in {"lower", "higher", "unchanged"}:
        threshold_update = "unchanged"
    update_strength = str(parsed.get("update_strength", "none"))
    if update_strength not in UPDATE_STRENGTH_DELTAS:
        update_strength = "none"
    threshold_causality = str(parsed.get("threshold_causality", "uncertain"))[:120]
    consistency_repair = None
    threshold_was_crossed = float(peak) >= previous_threshold
    is_missed_feedback = label in {"missed_HPC_maintenance", "missed_fan_maintenance", "missed_maintenance_unknown"}
    is_early_feedback = label in {"too_early", "over_maintenance"}
    if is_missed_feedback and (threshold_update == "higher" or threshold_causality == "over_activated_by_threshold"):
        consistency_repair = (
            "LLM proposed raising threshold for missed maintenance; converted to unchanged because that direction "
            "would make missed maintenance harder to catch."
        )
        update_policy = False
        threshold_update = "unchanged"
        update_strength = "none"
        threshold_causality = "action_reasoning_or_component_or_timing" if threshold_was_crossed else "uncertain"
    if is_early_feedback and threshold_update == "lower":
        consistency_repair = (
            "LLM proposed lowering threshold for too-early maintenance; converted to unchanged because that direction "
            "would increase early activations."
        )
        update_policy = False
        threshold_update = "unchanged"
        update_strength = "none"
        threshold_causality = "action_reasoning_or_component_or_timing"
    if is_missed_feedback and threshold_was_crossed and threshold_causality == "action_reasoning_or_component_or_timing":
        if update_policy or threshold_update != "unchanged" or update_strength != "none":
            consistency_repair = (
                "LLM attributed failure to action/component/timing after the case crossed the threshold; "
                "converted threshold update to none."
            )
        update_policy = False
        threshold_update = "unchanged"
        update_strength = "none"
    delta = UPDATE_STRENGTH_DELTAS[update_strength]
    if update_strength == "none":
        update_policy = False
        threshold_update = "unchanged"
    if not update_policy or threshold_update == "unchanged":
        proposed = previous_threshold
        threshold_update = "unchanged"
        update_strength = "none"
        delta = 0.0
    elif threshold_update == "lower":
        proposed = max(0.0, previous_threshold - delta)
    elif threshold_update == "higher":
        proposed = previous_threshold + delta
    else:
        proposed = previous_threshold
        threshold_update = "unchanged"
        update_strength = "none"
        delta = 0.0

    updated = dict(policy)
    _record_anchor(updated, label, peak)
    updated["peak_threshold"] = round(proposed, 6)
    updated["source"] = "llm_policy_update_tool"
    updated.setdefault("updates", [])
    updated["updates"] = list(updated.get("updates", [])) + [
        {
            "case_id": case_id,
            "feedback_label": label,
            "peak_score": round(float(peak), 6),
            "previous_peak_threshold": round(previous_threshold, 6),
            "updated_peak_threshold": round(float(updated["peak_threshold"]), 6),
            "threshold_update": threshold_update,
            "update_strength": update_strength,
            "threshold_delta": round(float(updated["peak_threshold"]) - previous_threshold, 6),
            "source": "llm_policy_update_controller",
        }
    ]
    updated = validate_policy(updated)
    return {
        "tool_name": "LLMPolicyUpdateTool",
        "update_policy": bool(update_policy),
        "update_threshold": bool(update_policy and proposed != previous_threshold),
        "threshold_update": threshold_update,
        "update_strength": update_strength,
        "threshold_delta": round(float(proposed) - previous_threshold, 6),
        "updated_policy": updated,
        "previous_peak_threshold": round(previous_threshold, 6),
        "updated_peak_threshold": updated["peak_threshold"],
        "reason": str(parsed.get("reason", ""))[:500],
        "confidence": _bounded(parsed.get("confidence"), 0.5, 0.0, 1.0),
        "threshold_causality": threshold_causality,
        "consistency_repair": consistency_repair,
        "source": "llm_policy_update_controller",
        "prompt": prompt,
        "raw_output": raw,
    }


def _no_update_result(
    *,
    policy: dict[str, Any],
    label: str,
    peak: float,
    case_id: Any,
    source: str,
    reason: str,
) -> dict[str, Any]:
    previous_threshold = float(policy["peak_threshold"])
    updated = dict(policy)
    _record_anchor(updated, label, peak)
    updated["peak_threshold"] = round(previous_threshold, 6)
    updated["source"] = source
    updated.setdefault("updates", [])
    updated["updates"] = list(updated.get("updates", [])) + [
        {
            "case_id": case_id,
            "feedback_label": label,
            "peak_score": round(float(peak), 6),
            "previous_peak_threshold": round(previous_threshold, 6),
            "updated_peak_threshold": round(previous_threshold, 6),
            "threshold_update": "unchanged",
            "update_strength": "none",
            "threshold_delta": 0.0,
            "source": source,
        }
    ]
    updated = validate_policy(updated)
    return {
        "tool_name": "LLMPolicyUpdateTool",
        "update_policy": False,
        "update_threshold": False,
        "threshold_update": "unchanged",
        "update_strength": "none",
        "threshold_delta": 0.0,
        "updated_policy": updated,
        "previous_peak_threshold": round(previous_threshold, 6),
        "updated_peak_threshold": updated["peak_threshold"],
        "reason": reason,
        "confidence": 0.0,
        "source": source,
    }


def _persist_and_return(policy_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    updated = validate_policy(result["updated_policy"])
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
    result["updated_policy"] = updated
    return result


def _record_anchor(policy: dict[str, Any], label: str, peak: float) -> None:
    if label in {"too_early", "over_maintenance"}:
        policy["early_anchor_count"] = int(policy.get("early_anchor_count", 0)) + 1
        policy["early_peak_max"] = _max_optional(policy.get("early_peak_max"), peak)
    elif label == "correct_maintenance":
        policy["correct_anchor_count"] = int(policy.get("correct_anchor_count", 0)) + 1
        policy["positive_peak_min"] = _min_optional(policy.get("positive_peak_min"), peak)
    elif label in {"missed_HPC_maintenance", "missed_fan_maintenance", "missed_maintenance_unknown"}:
        policy["missed_anchor_count"] = int(policy.get("missed_anchor_count", 0)) + 1
        policy["positive_peak_min"] = _min_optional(policy.get("positive_peak_min"), peak)


def _peak_history(path: str | Path | None, limit: int = 40) -> dict[str, Any]:
    if path is None:
        return {"row_count": 0, "label_counts": {}, "recent_rows": []}
    rows = read_reflection_training_rows(path)
    compact: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    peaks_by_label: dict[str, list[float]] = {}
    for row in rows:
        label = str(row.get("feedback_label", ""))
        peak = _num(row.get("peak_score"), default=None)
        if peak is None:
            continue
        label_counts[label] = label_counts.get(label, 0) + 1
        peaks_by_label.setdefault(label, []).append(float(peak))
        compact.append(
            {
                "feedback_label": label,
                "peak_score": round(float(peak), 6),
                "previous_action_type": row.get("previous_action_type"),
                "previous_action_time": row.get("previous_action_time"),
            }
        )
    return {
        "row_count": len(rows),
        "label_counts": label_counts,
        "peak_score_by_label": {
            label: {
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "mean": round(sum(values) / len(values), 6),
            }
            for label, values in sorted(peaks_by_label.items())
            if values
        },
        "recent_rows": compact[-limit:],
    }


def _blocked_missed_summary(history: dict[str, Any], threshold: float) -> dict[str, Any]:
    rows = history.get("recent_rows", [])
    if not isinstance(rows, list):
        rows = []
    missed_peaks: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("feedback_label", ""))
        peak = _num(row.get("peak_score"), default=None)
        if label.startswith("missed_") and peak is not None and float(peak) < threshold:
            missed_peaks.append(float(peak))
    gaps = [threshold - peak for peak in missed_peaks]
    if not missed_peaks:
        return {
            "blocked_missed_count": 0,
            "interpretation": "No missed-maintenance peak_score in recent reflection history is below the current threshold.",
        }
    return {
        "blocked_missed_count": len(missed_peaks),
        "min_blocked_missed_peak_score": round(min(missed_peaks), 6),
        "max_blocked_missed_peak_score": round(max(missed_peaks), 6),
        "mean_blocked_missed_peak_score": round(sum(missed_peaks) / len(missed_peaks), 6),
        "max_threshold_minus_blocked_missed_peak": round(max(gaps), 6),
        "mean_threshold_minus_blocked_missed_peak": round(sum(gaps) / len(gaps), 6),
        "interpretation": (
            "These missed-maintenance cases would have been blocked by the current peak_threshold; "
            "larger gaps indicate stronger evidence that the threshold is too high."
        ),
    }


def _early_trigger_summary(history: dict[str, Any], threshold: float) -> dict[str, Any]:
    rows = history.get("recent_rows", [])
    if not isinstance(rows, list):
        rows = []
    early_peaks: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("feedback_label", ""))
        peak = _num(row.get("peak_score"), default=None)
        if label in {"too_early", "over_maintenance"} and peak is not None and float(peak) >= threshold:
            early_peaks.append(float(peak))
    margins = [peak - threshold for peak in early_peaks]
    if not early_peaks:
        return {
            "over_activated_early_count": 0,
            "interpretation": "No too-early maintenance peak_score in recent reflection history is above the current threshold.",
        }
    return {
        "over_activated_early_count": len(early_peaks),
        "min_over_activated_early_peak_score": round(min(early_peaks), 6),
        "max_over_activated_early_peak_score": round(max(early_peaks), 6),
        "mean_over_activated_early_peak_score": round(sum(early_peaks) / len(early_peaks), 6),
        "max_early_peak_minus_threshold": round(max(margins), 6),
        "mean_early_peak_minus_threshold": round(sum(margins) / len(margins), 6),
        "interpretation": (
            "These too-early maintenance cases were allowed by the current peak_threshold; "
            "larger margins or repeated cases indicate stronger evidence that the threshold may be too low."
        ),
    }


def _update_history_stats(policy: dict[str, Any], limit: int = 8) -> dict[str, Any]:
    updates = list(policy.get("updates", [])) if isinstance(policy.get("updates"), list) else []
    recent = updates[-limit:]
    return {
        "recent_update_count": len(recent),
        "recent_lower_count": sum(1 for item in recent if item.get("threshold_update") == "lower"),
        "recent_higher_count": sum(1 for item in recent if item.get("threshold_update") == "higher"),
        "recent_unchanged_count": sum(1 for item in recent if item.get("threshold_update") == "unchanged"),
    }


def _case_peak_score(case: dict[str, Any]) -> float:
    value = _risk_value(case, "peak_score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _risk_value(case: dict[str, Any], key: str) -> Any:
    risk = case.get("risk_statistics", {})
    value = risk.get(key)
    if value in {None, ""}:
        value = case.get("forecast_summary", {}).get(key)
    return value


def _max_optional(current: Any, value: float) -> float:
    if current is None:
        return round(float(value), 6)
    return round(max(float(current), float(value)), 6)


def _min_optional(current: Any, value: float) -> float:
    if current is None:
        return round(float(value), 6)
    return round(min(float(current), float(value)), 6)


def _bounded(value: Any, default: float, low: float, high: float) -> float:
    number = _num(value, default)
    return round(min(max(float(number), low), high), 6)


def _num(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

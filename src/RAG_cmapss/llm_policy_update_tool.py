from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lightgbm_features import read_reflection_training_rows
from .llm_policy_risk_tool import initial_policy, load_policy, validate_policy
from .ollama_client import extract_json, ollama_chat

UPDATE_STRENGTH_DELTAS = {
    "none": 0.0,
    "small": 0.05,
    "median": 0.15,
    # Keep medium as a compatibility alias for older LLM outputs/logs.
    "medium": 0.15,
    "large": 0.30,
}


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
        reflection_rows: list[dict[str, Any]] | None = None,
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
        label = _policy_feedback_label(feedback)
        peak = _case_peak_score(case)
        previous_threshold = float(policy.get("peak_threshold", 0.25))
        missed_cause = str(feedback.get("missed_maintenance_cause") or "")
        batch_rows = reflection_rows if reflection_rows is not None else None
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
            # Keep the original full-memory context.  In batch mode the current
            # batch is added as a separate current-event field below.
            "online_feedback_statistics": _feedback_history(reflection_rules_path),
            "update_guidance": [
                "Update only peak_threshold; choose lower/higher/unchanged and none/small/median/large. The legacy token medium is accepted as an alias for median.",
                "For missed maintenance caused by the LHI gate, lower the threshold when the peak did not cross it.",
                "For too_early or over_maintenance, consider raising the threshold when the gate was crossed at a low peak.",
                "If the current case already crossed the threshold, keep it unchanged unless history supports a separate threshold problem.",
                "Do not update after correct_maintenance; this tool is called only for non-correct feedback.",
            ],
            "required_output": {
                "update_policy": "bool",
                "threshold_update": "lower/higher/unchanged",
                "update_strength": "none/small/median/large",
                "threshold_causality": "blocked_by_threshold/over_activated_by_threshold/action_reasoning_or_component_or_timing/uncertain",
                "reason": "short string",
                "confidence": "float 0..1",
            },
        }
        if batch_rows is not None:
            payload["reflection_scope"] = "current_batch_plus_full_reflection_history"
            payload["current_batch_reflections"] = _feedback_history(
                None,
                rows=batch_rows,
            )
            payload["update_guidance"].extend(
                [
                    "In batch mode, use current_batch_reflections as the current event evidence and online_feedback_statistics as the complete historical context.",
                    "Do not replace the complete history with the batch summary; the batch summary replaces only the single current-engine evidence used by sequential mode.",
                ]
            )
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

    def predict_batch(
        self,
        *,
        batch_items: list[dict[str, Any]],
        current_policy: dict[str, Any] | None = None,
        reflection_rules_path: str | Path | None = None,
        reflection_rows: list[dict[str, Any]] | None = None,
        model: str,
        ollama_url: str,
        temperature: float,
        timeout: int,
        num_predict: int,
        format_json: bool,
        dry_run: bool = False,
        disable_llm: bool = False,
    ) -> dict[str, Any]:
        """Perform exactly one theta update for a completed engine batch."""
        policy = validate_policy(current_policy or load_policy(self.policy_path) or initial_policy())
        items = list(batch_items)
        rows = list(reflection_rows or [])
        fallback = _batch_policy_update_result(
            policy=policy,
            batch_items=items,
            parsed={},
            source="disabled_llm_policy_update" if dry_run or disable_llm else "fallback_llm_policy_update",
            reason="LLM batch policy update was not run.",
        )
        if dry_run or disable_llm:
            return _persist_and_return(self.policy_path, fallback)

        batch_feedback = []
        threshold_comparisons = _threshold_update_comparisons(
            items,
            current_threshold=float(policy.get("peak_threshold", 0.25)),
        )
        for item in items:
            feedback = item.get("feedback") or {}
            case = item.get("case") or {}
            action = item.get("action") or {}
            batch_feedback.append(
                {
                    "case_id": case.get("case_id"),
                    "dataset_subset": case.get("dataset_subset"),
                    "feedback_label": _policy_feedback_label(feedback),
                    "missed_maintenance_cause": feedback.get("missed_maintenance_cause"),
                    "maintenance_timing_status": feedback.get("maintenance_timing_status"),
                    "action_type": action.get("action_type"),
                    "action_time": action.get("action_time"),
                    "peak_lhi": _case_peak_score(case),
                    "decision_threshold": item.get("decision_threshold"),
                    "current_threshold": float(policy.get("peak_threshold", 0.25)),
                    "threshold_update_eligible": next(
                        (
                            row["eligible"]
                            for row in threshold_comparisons
                            if row["case_id"] == case.get("case_id")
                        ),
                        None,
                    ),
                    "threshold_update_reason": next(
                        (
                            row["reason"]
                            for row in threshold_comparisons
                            if row["case_id"] == case.get("case_id")
                        ),
                        None,
                    ),
                }
            )
        payload = {
            "task": "Perform one LHI threshold update after a completed engine batch.",
            "batch_size": len(items),
            "reflection_scope": "current_batch_plus_full_reflection_history",
            "current_batch_reflections": _feedback_history(None, rows=rows),
            "online_feedback_statistics": _feedback_history(reflection_rules_path),
            "current_batch_feedback": batch_feedback,
            "current_policy": {
                "peak_threshold": float(policy.get("peak_threshold", 0.25)),
                "missed_cause_counts": policy.get("missed_cause_counts"),
                "correct_anchor_count": policy.get("correct_anchor_count"),
                "missed_anchor_count": policy.get("missed_anchor_count"),
                "early_anchor_count": policy.get("early_anchor_count"),
            },
            "update_guidance": [
                "Make exactly one aggregate threshold decision for the whole batch; do not apply one update per reflection.",
                "Use current_batch_reflections as current evidence and online_feedback_statistics as complete historical context.",
                "Choose threshold_update lower/higher/unchanged and update_strength none/small/median/large.",
                "small=0.05, median=0.15, large=0.3; the selected strength is applied once to the current threshold.",
                "Do not update after a batch containing only correct_maintenance feedback.",
                "If too_early and missed_* both occur in the same batch, do not update theta: use threshold_update=unchanged and update_strength=none.",
                "For too_early, count it for threshold updating when decision_threshold >= current_threshold.",
                "For missed_* maintenance, count it for threshold updating when decision_threshold <= current_threshold.",
                "Ignore timing errors that fail those threshold comparisons for the threshold direction decision; retain them in feedback history.",
                "If only one error type occurs: 1-2 cases require small, 3 through half of the batch require median, and more than half require large.",
                "If too_early is the dominant actionable batch error, threshold_update must be higher or unchanged, never lower.",
                "If missed maintenance is the dominant actionable batch error, threshold_update may be lower; never lower the threshold because of too_early feedback.",
            ],
            "required_output": {
                "threshold_update": "lower/higher/unchanged",
                "update_strength": "none/small/median/large",
                "threshold_causality": "blocked_by_threshold/over_activated_by_threshold/action_reasoning_or_component_or_timing/mixed/uncertain",
                "reason": "short string",
                "confidence": "float 0..1",
            },
        }
        prompt = (
            "You are the aggregate LLM policy update controller for a zero-shot predictive-maintenance experiment.\n"
            "The batch is complete. Return one JSON object and make exactly one threshold update for the entire batch.\n\n"
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
            result = _batch_policy_update_result(
                policy=policy,
                batch_items=items,
                parsed=parsed,
                source="llm_batch_policy_update_controller",
                reason=str(parsed.get("reason", ""))[:500],
            )
            result["prompt"] = prompt
            result["raw_output"] = raw
        except Exception as exc:
            result = fallback
            result["source"] = "fallback_after_llm_batch_policy_update_error"
            result["reason"] = f"LLM batch policy update failed. Error: {exc}"
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
    if strength == "medium":
        strength = "median"
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


def _batch_policy_update_result(
    *,
    policy: dict[str, Any],
    batch_items: list[dict[str, Any]],
    parsed: dict[str, Any],
    source: str,
    reason: str,
) -> dict[str, Any]:
    threshold_update = str(parsed.get("threshold_update", "unchanged"))
    strength = str(parsed.get("update_strength", "none"))
    if strength == "medium":
        strength = "median"
    if threshold_update not in {"lower", "higher", "unchanged"}:
        threshold_update = "unchanged"
    if strength not in {"none", "small", "median", "large"}:
        strength = "none"
    previous = float(policy.get("peak_threshold", 0.25))
    labels: dict[str, int] = {}
    for item in batch_items:
        label = _policy_feedback_label(item.get("feedback") or {})
        labels[label] = labels.get(label, 0) + 1
    (
        required_direction,
        required_strength,
        mixed_errors,
        threshold_comparisons,
    ) = _batch_threshold_policy(batch_items, current_threshold=previous)
    eligible_early = sum(
        1
        for row in threshold_comparisons
        if row["label"] in {"too_early", "over_maintenance"} and row["eligible"]
    )
    eligible_missed = sum(
        1
        for row in threshold_comparisons
        if row["label"].startswith("missed_") and row["eligible"]
    )
    noncorrect = eligible_early + eligible_missed
    # Make direction and strength deterministic from the batch composition.  In
    # particular, mixed early/missed evidence must not move theta at all.
    if mixed_errors:
        threshold_update, strength = "unchanged", "none"
    elif required_direction is not None and noncorrect > 0:
        threshold_update, strength = required_direction, required_strength
    if noncorrect == 0 or threshold_update == "unchanged" or strength == "none":
        proposed = previous
        threshold_update, strength = "unchanged", "none"
    elif threshold_update == "lower":
        proposed = max(1e-6, previous - UPDATE_STRENGTH_DELTAS[strength])
    else:
        proposed = previous + UPDATE_STRENGTH_DELTAS[strength]

    updated = dict(policy)
    comparison_by_case = {
        row["case_id"]: row for row in threshold_comparisons
    }
    for item in batch_items:
        feedback = item.get("feedback") or {}
        case = item.get("case") or {}
        label = _policy_feedback_label(feedback)
        comparison = comparison_by_case.get(case.get("case_id"), {})
        if label in {"too_early", "over_maintenance"} or label.startswith("missed_"):
            if not comparison.get("eligible", False):
                # Keep the real outcome in feedback/reflection history, but do
                # not let a stale-threshold event calibrate future theta gates.
                continue
        _record_feedback(
            updated,
            label,
            _case_peak_score(case),
            str(feedback.get("missed_maintenance_cause") or ""),
        )
    updated["peak_threshold"] = round(proposed, 6)
    updated["source"] = source
    updated["updates"] = list(updated.get("updates", [])) + [
        {
            "batch_case_ids": [
                (item.get("case") or {}).get("case_id")
                for item in batch_items
            ],
            "batch_size": len(batch_items),
            "feedback_label_counts": labels,
            "dominant_batch_error": _dominant_batch_error(labels),
            "required_threshold_direction": required_direction,
            "required_update_strength": required_strength,
            "mixed_timing_errors": mixed_errors,
            "threshold_update_feedback_counts": {
                "too_early": eligible_early,
                "missed_maintenance": eligible_missed,
            },
            "threshold_update_comparisons": threshold_comparisons,
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
        "batch_update": True,
        "batch_size": len(batch_items),
        "batch_case_ids": [
            (item.get("case") or {}).get("case_id")
            for item in batch_items
        ],
        "feedback_label_counts": labels,
        "dominant_batch_error": _dominant_batch_error(labels),
        "required_threshold_direction": required_direction,
        "required_update_strength": required_strength,
        "mixed_timing_errors": mixed_errors,
        "threshold_update_feedback_counts": {
            "too_early": eligible_early,
            "missed_maintenance": eligible_missed,
        },
        "threshold_update_comparisons": threshold_comparisons,
        "update_policy": proposed != previous,
        "update_threshold": proposed != previous,
        "updated_policy": updated,
        "reason": reason,
        "confidence": _bounded(parsed.get("confidence"), 0.0, 0.0, 1.0),
        "threshold_update": threshold_update,
        "update_strength": strength,
        "threshold_delta": round(proposed - previous, 6),
        "previous_peak_threshold": round(previous, 6),
        "updated_peak_threshold": round(proposed, 6),
        "threshold_causality": str(parsed.get("threshold_causality", "uncertain"))[:120],
        "source": source,
    }


def _persist_and_return(policy_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    updated = validate_policy(result["updated_policy"])
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
    result["updated_policy"] = updated
    return result


def _batch_threshold_policy(
    batch_items: list[dict[str, Any]],
    current_threshold: float,
) -> tuple[str | None, str, bool, list[dict[str, Any]]]:
    early = 0
    missed = 0
    comparisons = _threshold_update_comparisons(batch_items, current_threshold)
    for row in comparisons:
        label = row["label"]
        if not row["eligible"]:
            continue
        if label in {"too_early", "over_maintenance"}:
            early += 1
        elif label.startswith("missed_"):
            missed += 1
    if early > 0 and missed > 0:
        return None, "none", True, comparisons
    count = max(early, missed)
    if count == 0:
        return None, "none", False, comparisons
    batch_size = max(early + missed, 1)
    if count <= 2:
        strength = "small"
    elif count > batch_size / 2:
        strength = "large"
    else:
        strength = "median"
    if early > missed:
        return "higher", strength, False, comparisons
    if missed > early:
        return "lower", strength, False, comparisons
    return None, "none", False, comparisons


def _threshold_update_comparisons(
    batch_items: list[dict[str, Any]],
    current_threshold: float,
) -> list[dict[str, Any]]:
    """Decide which timing errors are valid evidence for a theta update.

    A feedback event is generated after the engine's decision-time policy may
    have been replaced by a newer shared policy.  Only count a too-early event
    when it was made under a threshold no lower than the current threshold,
    and only count a missed event when it was made under a threshold no higher
    than the current threshold. Equality is intentionally included so the
    initial threshold can update from the first eligible feedback.
    prevents stale feedback from pushing the shared threshold in the wrong
    direction.
    """
    rows: list[dict[str, Any]] = []
    current = float(current_threshold)
    for item in batch_items:
        feedback = item.get("feedback") or {}
        label = _policy_feedback_label(feedback)
        raw_decision_threshold = item.get("decision_threshold")
        if raw_decision_threshold is None:
            raw_decision_threshold = (item.get("action") or {}).get("decision_threshold")
        try:
            decision_threshold = float(raw_decision_threshold)
        except (TypeError, ValueError):
            decision_threshold = None

        eligible: bool | None = None
        if label in {"too_early", "over_maintenance"}:
            if decision_threshold is None:
                eligible = False
                reason = "missing_decision_threshold"
            elif decision_threshold >= current:
                eligible = True
                reason = "decision_threshold_at_or_above_current"
            else:
                eligible = False
                reason = "decision_threshold_below_current"
        elif label.startswith("missed_"):
            if decision_threshold is None:
                eligible = False
                reason = "missing_decision_threshold"
            elif decision_threshold <= current:
                eligible = True
                reason = "decision_threshold_at_or_below_current"
            else:
                eligible = False
                reason = "decision_threshold_above_current"
        else:
            reason = "non_threshold_feedback"

        rows.append(
            {
                "case_id": (item.get("case") or {}).get("case_id"),
                "label": label,
                "decision_threshold": decision_threshold,
                "current_threshold": current,
                "eligible": eligible,
                "reason": reason,
            }
        )
    return rows


def _dominant_batch_error(labels: dict[str, int]) -> str:
    early = sum(labels.get(label, 0) for label in ("too_early", "over_maintenance"))
    missed = sum(count for label, count in labels.items() if label.startswith("missed_"))
    if early > missed:
        return "too_early"
    if missed > early:
        return "missed_maintenance"
    return "balanced_or_none"


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


def _feedback_history(
    path: str | Path | None,
    limit: int = 40,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = list(rows) if rows is not None else (read_reflection_training_rows(path) if path is not None else [])
    label_counts: dict[str, int] = {}
    cause_counts: dict[str, int] = {}
    compact: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("timing_feedback_label") or row.get("feedback_label", ""))
        cause = str(row.get("missed_maintenance_cause") or "")
        label_counts[label] = label_counts.get(label, 0) + 1
        if label.startswith("missed_") and cause:
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
        compact.append(
            {
                "rule_id": row.get("rule_id"),
                "feedback_label": label,
                "accuracy_feedback_label": str(row.get("feedback_label", "")),
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


def _policy_feedback_label(feedback: dict[str, Any]) -> str:
    """Use timing-only labels for theta updates; accuracy retains component errors."""
    return str(
        feedback.get("policy_feedback_label")
        or feedback.get("timing_feedback_label")
        or feedback.get("feedback_label", "")
    )


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

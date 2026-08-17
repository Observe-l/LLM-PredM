from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence


_ENGINE_RE = re.compile(r"_Engine(\d+)_")


class EvaluationTool:
    """Deterministically evaluate completed-engine maintenance outcomes."""

    def __init__(self, window_size: int = 10):
        if int(window_size) < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = int(window_size)

    def evaluate(
        self,
        *,
        fd: str,
        feedback_logs: Sequence[dict[str, Any]],
        action_hypotheses: Sequence[dict[str, Any]],
        current_policy: dict[str, Any],
        lhi_trigger: float,
    ) -> dict[str, Any]:
        completed = len(feedback_logs)
        recent = list(feedback_logs[-self.window_size :])
        previous = (
            list(feedback_logs[-2 * self.window_size : -self.window_size])
            if completed >= 2 * self.window_size
            else []
        )
        cumulative_stats = _outcome_stats(feedback_logs)
        recent_stats = _outcome_stats(recent)
        previous_stats = _outcome_stats(previous)
        recent_rate = recent_stats["correct_maintenance_rate"]
        previous_rate = previous_stats["correct_maintenance_rate"] if previous else None
        delta = (
            round(float(recent_rate) - float(previous_rate), 6)
            if recent_rate is not None and previous_rate is not None
            else None
        )
        recent_engines = {
            engine
            for engine in (_engine_id(row.get("case_id")) for row in recent)
            if engine is not None
        }
        previous_engines = {
            engine
            for engine in (_engine_id(row.get("case_id")) for row in previous)
            if engine is not None
        }
        recent_actions = [
            row
            for row in action_hypotheses
            if _engine_id(row.get("case_id")) in recent_engines
        ]
        action_counts = Counter(
            str(row.get("action", {}).get("action_type", "unknown"))
            for row in recent_actions
        )
        maintenance_actions = [
            row for row in recent_actions
            if str(row.get("action", {}).get("action_type", "")).startswith("schedule_")
            and "maintenance" in str(row.get("action", {}).get("action_type", ""))
        ]
        peak_matches = 0
        recommendation_matches = 0
        t20_count = 0
        for row in maintenance_actions:
            action_time = row.get("action", {}).get("action_time")
            timing_profile = row.get("context", {}).get("maintenance_timing_profile", {})
            peak_time = timing_profile.get("peak_score_cycle")
            recommended_time = timing_profile.get("recommended_maintenance_time")
            peak_matches += int(action_time == peak_time)
            recommendation_matches += int(action_time == recommended_time)
            t20_count += int(action_time == "t+20")
        kg_nonempty = sum(
            bool(row.get("context", {}).get("sensor_paths"))
            for row in recent_actions
        )
        fallback_count = sum(
            bool(row.get("action", {}).get("llm_fallback_used"))
            for row in recent_actions
        )
        repair_count = sum(
            bool(row.get("action", {}).get("local_validation_repair_used"))
            for row in recent_actions
        )
        invalid_count = sum(
            row.get("action", {}).get("validation_status") != "valid"
            for row in recent_actions
        )
        status = "insufficient_evidence"
        if delta is not None:
            status = "improving" if delta >= 0.05 else "degrading" if delta <= -0.05 else "stable"
        score_change_count = (
            int(recent_stats["correct_maintenance"])
            - int(previous_stats["correct_maintenance"])
            if previous else None
        )
        intervals_overlap = (
            _intervals_overlap(
                recent_stats["correct_rate_wilson_95"],
                previous_stats["correct_rate_wilson_95"],
            )
            if previous else None
        )
        evidence_strength = _evidence_strength(
            score_change_count=score_change_count,
            intervals_overlap=intervals_overlap,
            has_previous=bool(previous),
        )
        recent_timing = _timing_stats(recent)
        previous_timing = _timing_stats(previous) if previous else None
        return {
            "tool_name": "EvaluationTool",
            "fd": str(fd),
            "checkpoint_id": f"{fd}_engine_{completed}",
            "engines_completed": completed,
            "window_size": self.window_size,
            "first_evaluation": not bool(previous),
            "score_status": status,
            "primary_metric": "correct_maintenance_rate",
            "cumulative": cumulative_stats,
            "recent_window": recent_stats,
            "previous_window": previous_stats if previous else None,
            "recent_minus_previous_correct_rate": delta,
            "score_change_engine_count": score_change_count,
            "one_engine_rate_step": round(1 / len(recent), 6) if recent else None,
            "wilson_intervals_overlap": intervals_overlap,
            "evidence_strength": evidence_strength,
            "policy_exposure": {
                "recent_window": _policy_exposure(recent_engines, action_hypotheses),
                "previous_window": (
                    _policy_exposure(previous_engines, action_hypotheses)
                    if previous else None
                ),
            },
            "action_statistics": {
                "decision_points": len(recent_actions),
                "action_counts": dict(action_counts),
                "maintenance_count": len(maintenance_actions),
                "maintenance_peak_cycle_match_rate": (
                    round(peak_matches / len(maintenance_actions), 6)
                    if maintenance_actions else None
                ),
                "maintenance_recommended_time_match_rate": (
                    round(recommendation_matches / len(maintenance_actions), 6)
                    if maintenance_actions else None
                ),
                "maintenance_at_t20_count": t20_count,
                "kg_evidence_coverage_rate": (
                    round(kg_nonempty / len(recent_actions), 6)
                    if recent_actions else None
                ),
                "llm_fallback_count": fallback_count,
                "local_validation_repair_count": repair_count,
                "invalid_action_count": invalid_count,
            },
            "timing_statistics": recent_timing,
            "timing_error_balance": _timing_error_balance(
                recent_timing,
                previous_timing,
            ),
            "lhi_gate": {
                "trigger": float(lhi_trigger),
                "not_triggered_before_failure_count": recent_stats[
                    "missed_cause_counts"
                ].get("lhi_gate_not_triggered_before_failure", 0),
            },
            "current_policy": {
                key: current_policy.get(key)
                for key in [
                    "policy_revision",
                    "effective_from_engine",
                    "action_escalation_policy",
                    "maintenance_timing_policy",
                    "peak_offset_level",
                    "monitoring_interval",
                ]
            },
            "eligible_policy_dimensions": [
                "action_escalation_policy",
                "peak_offset_level",
                "monitoring_interval",
            ],
        }


def compact_evaluation_history(
    evaluation_logs: Sequence[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Return the latest evaluator checkpoints in a prompt-sized form."""
    history: list[dict[str, Any]] = []
    for row in list(evaluation_logs)[-max(int(limit), 0) :]:
        report = row.get("evaluation_report") or {}
        decision = row.get("evaluation_agent_decision") or {}
        validation = row.get("validation") or {}
        recent = report.get("recent_window") or {}
        balance = report.get("timing_error_balance") or {}
        history.append(
            {
                "checkpoint_id": row.get("checkpoint_id"),
                "engines_completed": report.get("engines_completed"),
                "correct_maintenance_rate": recent.get("correct_maintenance_rate"),
                "correct_rate_delta": report.get("recent_minus_previous_correct_rate"),
                "score_status": report.get("score_status"),
                "evidence_strength": report.get("evidence_strength"),
                "late_count": balance.get("recent_late_count"),
                "too_early_count": balance.get("recent_too_early_count"),
                "monitoring_missed_count": (
                    (report.get("timing_statistics") or {}).get(
                        "monitoring_related_missed_count"
                    )
                ),
                "dominant_timing_direction": balance.get(
                    "dominant_observed_timing_direction"
                ),
                "policy_patch": decision.get("policy_patch", {}),
                "patch_applied": bool(validation.get("applied")),
                "applied_patch": validation.get("applied_patch", {}),
                "instruction_repair_count": decision.get("instruction_repair_count", 0),
            }
        )
    return history


def _outcome_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row.get("feedback_label", "unknown")) for row in rows)
    timing_labels = Counter(
        str(row.get("timing_feedback_label") or row.get("feedback_label", "unknown"))
        for row in rows
    )
    total = len(rows)
    correct = labels.get("correct_maintenance", 0)
    early = timing_labels.get("too_early", 0) + timing_labels.get("over_maintenance", 0)
    missed = sum(value for label, value in timing_labels.items() if label.startswith("missed_"))
    causes = Counter(
        str(row.get("missed_maintenance_cause"))
        for row in rows
        if str(row.get("feedback_label", "")).startswith("missed_")
        and row.get("missed_maintenance_cause")
    )
    rate = round(correct / total, 6) if total else None
    low, high = _wilson_interval(correct, total)
    return {
        "total": total,
        "correct_maintenance": correct,
        "too_early": early,
        "missed_maintenance": missed,
        "correct_maintenance_rate": rate,
        "wrong_component": labels.get("wrong_component", 0),
        "correct_rate_wilson_95": {"low": low, "high": high},
        "label_counts": dict(labels),
        "timing_label_counts": dict(timing_labels),
        "missed_cause_counts": dict(causes),
    }


def _timing_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    margins = [
        float(row["signed_cycle_margin"])
        for row in rows
        if row.get("signed_cycle_margin") not in {None, ""}
    ]
    late = sum(
        row.get("missed_maintenance_cause")
        == "maintenance_scheduled_at_or_after_failure"
        for row in rows
    )
    monitoring = sum(
        row.get("missed_maintenance_cause") in {
            "monitoring_without_maintenance",
            "continued_operation_without_maintenance",
        }
        for row in rows
    )
    too_early = sum(
        str(row.get("timing_feedback_label") or row.get("feedback_label", ""))
        in {"too_early", "over_maintenance"}
        for row in rows
    )
    return {
        "late_maintenance_count": late,
        "too_early_maintenance_count": too_early,
        "monitoring_related_missed_count": monitoring,
        "signed_cycle_margin": (
            {
                "min": round(min(margins), 6),
                "max": round(max(margins), 6),
                "mean": round(sum(margins) / len(margins), 6),
            }
            if margins else None
        ),
    }


def _timing_error_balance(
    recent: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    recent_late = int(recent.get("late_maintenance_count", 0))
    recent_early = int(recent.get("too_early_maintenance_count", 0))
    previous_late = int((previous or {}).get("late_maintenance_count", 0))
    previous_early = int((previous or {}).get("too_early_maintenance_count", 0))
    margin = recent_late - recent_early
    direction = (
        "increase_offset_schedule_earlier"
        if margin > 0
        else "decrease_offset_schedule_later"
        if margin < 0
        else "no_timing_change"
    )
    required_driver = (
        "late_timing" if margin > 0 else "early_timing" if margin < 0 else "none"
    )
    return {
        "recent_late_count": recent_late,
        "recent_too_early_count": recent_early,
        "previous_late_count": previous_late if previous is not None else None,
        "previous_too_early_count": previous_early if previous is not None else None,
        "recent_late_minus_too_early": margin,
        "previous_late_minus_too_early": (
            previous_late - previous_early if previous is not None else None
        ),
        "dominant_observed_timing_direction": direction,
        "required_primary_driver_for_timing_update": required_driver,
    }


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return round(max(0.0, centre - radius), 6), round(min(1.0, centre + radius), 6)


def _intervals_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    values = [first.get("low"), first.get("high"), second.get("low"), second.get("high")]
    if any(value is None for value in values):
        return False
    return max(float(first["low"]), float(second["low"])) <= min(
        float(first["high"]), float(second["high"])
    )


def _evidence_strength(
    *,
    score_change_count: int | None,
    intervals_overlap: bool | None,
    has_previous: bool,
) -> str:
    if not has_previous or score_change_count is None:
        return "insufficient"
    magnitude = abs(int(score_change_count))
    if magnitude <= 1:
        return "weak"
    if intervals_overlap:
        return "moderate" if magnitude >= 3 else "weak"
    return "strong"


def _policy_exposure(
    engines: set[int],
    action_hypotheses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the policy actually seen by each completed engine.

    The latest decision record for an engine is used because policy revisions are
    effective between engines, never within an engine.
    """
    latest: dict[int, dict[str, Any]] = {}
    for row in action_hypotheses:
        engine = _engine_id(row.get("case_id"))
        if engine in engines:
            latest[int(engine)] = row
    offsets: Counter[str] = Counter()
    revisions: Counter[str] = Counter()
    for engine in sorted(engines):
        row = latest.get(engine)
        if row is None:
            offsets["no_action"] += 1
            revisions["no_action"] += 1
            continue
        policy = row.get("context", {}).get("llm_policy", {})
        offsets[str(policy.get("peak_offset_level", "none"))] += 1
        revisions[str(policy.get("policy_revision", 0))] += 1
    return {
        "engine_count": len(engines),
        "peak_offset_level_counts": dict(offsets),
        "policy_revision_counts": dict(revisions),
    }


def _engine_id(case_id: Any) -> int | None:
    match = _ENGINE_RE.search(str(case_id or ""))
    return int(match.group(1)) if match else None

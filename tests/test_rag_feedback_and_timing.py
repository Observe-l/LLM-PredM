from __future__ import annotations

from src.RAG_cmapss.action_validator import validate_action
from src.RAG_cmapss.evaluation_tool import EvaluationTool
from src.RAG_cmapss.evaluation_agent import _decision_rule_violations
from src.RAG_cmapss.evaluation_validator import validate_and_apply_evaluation
from src.RAG_cmapss.joint_simulation import (
    breakdown_feedback,
    maintenance_feedback,
    summarize_feedback,
)
from src.RAG_cmapss.llm_policy_risk_tool import initial_policy
from src.RAG_cmapss.llm_policy_update_tool import LLMPolicyUpdateTool
from src.RAG_cmapss.prompt_builder import build_prompt
from src.RAG_cmapss.reflection_memory import feedback_to_rule
from src.RAG_cmapss.timing_policy import (
    recommended_maintenance_time,
    recommended_monitoring_time,
)


def _case(cutoff: int = 100, peak_cycle: str = "t+7") -> dict:
    return {
        "case_id": "ForecastCase_FD002_Engine1_Cycle100",
        "dataset_subset": "FD002",
        "unit_id": 1,
        "cutoff_cycle": cutoff,
        "forecast_horizon": {"start": 1, "end": 20},
        "forecast_summary": {
            "peak_score": 1.2,
            "peak_score_cycle": peak_cycle,
            "dominant_top_sensors": ["S11", "S7"],
        },
        "risk_statistics": {
            "peak_score": 1.2,
            "peak_score_cycle": peak_cycle,
        },
    }


def _maintenance(time: str) -> dict:
    return {
        "action_type": "schedule_HPC_maintenance",
        "action_time": time,
    }


def test_late_and_at_failure_maintenance_are_missed_not_correct() -> None:
    case = _case()
    late = maintenance_feedback(case, _maintenance("t+20"), 0.25, failure_cycle=115)
    at_failure = maintenance_feedback(case, _maintenance("t+15"), 0.25, failure_cycle=115)

    assert late["feedback_label"] == "missed_HPC_maintenance"
    assert late["missed_maintenance_cause"] == "maintenance_scheduled_at_or_after_failure"
    assert late["maintenance_timing_status"] == "after_failure"
    assert late["signed_cycle_margin"] == -5
    assert at_failure["feedback_label"] == "missed_HPC_maintenance"
    assert at_failure["maintenance_timing_status"] == "at_failure"


def test_timely_and_too_early_maintenance_remain_distinct() -> None:
    case = _case()
    timely = maintenance_feedback(case, _maintenance("t+7"), 0.25, failure_cycle=115)
    early = maintenance_feedback(case, _maintenance("t+7"), 0.25, failure_cycle=200)

    assert timely["feedback_label"] == "correct_maintenance"
    assert timely["maintenance_timing_status"] == "timely"
    assert early["feedback_label"] == "too_early"
    assert early["maintenance_timing_status"] == "too_early"


def test_breakdown_feedback_identifies_monitoring_cause_and_statistics() -> None:
    case = _case()
    action = {"action_type": "schedule_monitoring", "action_time": "t+20"}
    feedback = breakdown_feedback(
        case,
        action,
        failure_cycle=115,
        action_history=[
            {
                "case_id": case["case_id"],
                "action_type": "schedule_monitoring",
                "action_time": "t+20",
            }
        ],
    )
    stats = summarize_feedback([feedback])

    assert feedback["missed_maintenance_cause"] == "monitoring_without_maintenance"
    assert feedback["prior_monitoring_count"] == 1
    assert stats["missed_maintenance_cause_counts"] == {"monitoring_without_maintenance": 1}
    assert stats["missed_maintenance_cause_rates_among_missed"] == {
        "monitoring_without_maintenance": 1.0
    }


def test_no_predm_decision_is_attributed_to_upstream_lhi_gate() -> None:
    feedback = breakdown_feedback(
        _case(),
        {"action_type": "continue_normal_operation", "action_time": None},
        failure_cycle=115,
        action_history=[],
    )
    assert feedback["missed_maintenance_cause"] == "lhi_gate_not_triggered_before_failure"


def test_policy_gate_monitoring_is_distinct_from_llm_monitoring() -> None:
    feedback = breakdown_feedback(
        _case(),
        {"action_type": "schedule_monitoring", "action_time": "t+20"},
        failure_cycle=115,
        action_history=[
            {
                "action_type": "schedule_monitoring",
                "action_time": "t+20",
                "action_selection_source": "policy_gate",
            }
        ],
    )
    assert feedback["missed_maintenance_cause"] == "monitoring_due_to_policy_gate"


def test_maintenance_timing_is_peak_grounded_in_prompt_and_validator() -> None:
    case = _case(peak_cycle="t+7")
    rules = {
        "allowed_actions": [
            "continue_normal_operation",
            "schedule_monitoring",
            "schedule_HPC_maintenance",
        ],
        "disallowed_actions": ["schedule_fan_maintenance"],
    }
    prompt = build_prompt(
        case,
        rules,
        sensor_paths=[],
        action_paths=[],
        component_evidence_statistics={},
        risk_gate={},
        component_gate={},
    )

    assert recommended_maintenance_time(case) == "t+7"
    assert '"recommended_maintenance_time": "t+7"' in prompt
    assert "action_time must equal t+7" in prompt

    valid = validate_action(_maintenance("t+7"), case, rules, [])
    invalid = validate_action(_maintenance("t+20"), case, rules, [])
    assert valid["valid"]
    assert not invalid["valid"]
    assert "maintenance action_time must equal recommended_maintenance_time=t+7" in invalid["violations"]


def test_late_feedback_enters_reflection_as_timing_miss() -> None:
    case = _case()
    action = _maintenance("t+20")
    feedback = maintenance_feedback(case, action, 0.25, failure_cycle=115)
    rule = feedback_to_rule(feedback, case, action)

    assert rule is not None
    assert rule["feedback_label"] == "missed_HPC_maintenance"
    assert rule["missed_maintenance_cause"] == "maintenance_scheduled_at_or_after_failure"
    assert rule["then_adjust_threshold"] == "unchanged"
    assert rule["recommended_time_rule"] == "maintenance_at_peak_score_cycle_not_horizon_end"


def test_monitoring_miss_updates_action_policy(tmp_path) -> None:
    case = _case()
    feedback = breakdown_feedback(
        case,
        {"action_type": "schedule_monitoring", "action_time": "t+20"},
        failure_cycle=115,
        action_history=[{"action_type": "schedule_monitoring", "action_time": "t+20"}],
    )
    policy_path = tmp_path / "llm_policy_tool.json"
    tool = LLMPolicyUpdateTool(policy_path)
    result = tool.predict(
        feedback=feedback,
        case=case,
        action={"action_type": "schedule_monitoring", "action_time": "t+20"},
        current_policy=initial_policy(),
        model="unused",
        ollama_url="unused",
        temperature=0,
        timeout=1,
        num_predict=1,
        format_json=True,
        dry_run=True,
    )

    updated = result["updated_policy"]
    assert "peak_threshold" not in updated
    assert updated["action_escalation_policy"] == (
        "maintenance_when_risk_activated_and_component_supported"
    )
    assert updated["missed_cause_counts"] == {"monitoring_without_maintenance": 1}
    assert result["action_policy_update"] == "escalate_when_component_supported"


def test_lhi_gate_miss_does_not_update_downstream_policy(tmp_path) -> None:
    case = _case()
    policy_path = tmp_path / "llm_policy_tool.json"
    result = LLMPolicyUpdateTool(policy_path).predict(
        feedback={
            "feedback_label": "missed_HPC_maintenance",
            "missed_maintenance_cause": "lhi_gate_not_triggered_before_failure",
        },
        case=case,
        action={"action_type": "continue_normal_operation", "action_time": None},
        current_policy=initial_policy(),
        model="unused",
        ollama_url="unused",
        temperature=0,
        timeout=1,
        num_predict=1,
        format_json=True,
        dry_run=True,
    )
    assert not result["update_policy"]
    assert result["action_policy_update"] == "unchanged"
    assert result["timing_policy_update"] == "unchanged"
    assert "peak_threshold" not in result["updated_policy"]


def test_adaptive_timing_and_monitoring_policy_change_executable_times() -> None:
    case = _case(peak_cycle="t+7")
    policy = initial_policy()
    assert recommended_maintenance_time(case, policy) == "t+7"
    assert recommended_monitoring_time(case, policy) == "t+20"

    policy["peak_offset_level"] = "small"
    policy["monitoring_interval"] = 10
    assert recommended_maintenance_time(case, policy) == "t+2"
    assert recommended_monitoring_time(case, policy) == "t+10"

    policy["peak_offset_level"] = "median"
    assert recommended_maintenance_time(case, policy) == "t+1"
    policy["peak_offset_level"] = "large"
    assert recommended_maintenance_time(case, policy) == "t+1"

    rules = {
        "allowed_actions": [
            "continue_normal_operation",
            "schedule_monitoring",
            "schedule_HPC_maintenance",
        ],
        "disallowed_actions": ["schedule_fan_maintenance"],
    }
    assert validate_action(
        _maintenance("t+1"), case, rules, [], llm_policy=policy
    )["valid"]
    invalid = validate_action(
        _maintenance("t+2"), case, rules, [], llm_policy=policy
    )
    assert not invalid["valid"]


def test_evaluation_tool_compares_windows_and_applies_supported_timing_patch() -> None:
    feedback = []
    for idx in range(1, 41):
        recent = idx > 20
        is_late = recent and idx <= 25
        feedback.append(
            {
                "case_id": f"ForecastCase_FD001_Engine{idx}_Cycle100",
                "feedback_label": (
                    "missed_HPC_maintenance" if is_late else "correct_maintenance"
                ),
                "missed_maintenance_cause": (
                    "maintenance_scheduled_at_or_after_failure" if is_late else None
                ),
                "signed_cycle_margin": -1 if is_late else 10,
            }
        )
    report = EvaluationTool(20).evaluate(
        fd="FD001",
        feedback_logs=feedback,
        action_hypotheses=[],
        current_policy=initial_policy(),
        lhi_trigger=1.0,
    )
    assert report["previous_window"]["correct_maintenance_rate"] == 1.0
    assert report["recent_window"]["correct_maintenance_rate"] == 0.75
    assert report["score_status"] == "degrading"
    assert report["score_change_engine_count"] == -5
    assert report["evidence_strength"] == "moderate"
    assert report["timing_error_balance"]["dominant_observed_timing_direction"] == (
        "increase_offset_schedule_earlier"
    )

    result = validate_and_apply_evaluation(
        decision={
            "policy_patch": {"peak_offset_level": "small"},
            "reason": "Repeated late maintenance",
            "confidence": 0.9,
        },
        report=report,
        current_policy=initial_policy(),
    )
    assert result["applied"]
    assert result["updated_policy"]["peak_offset_level"] == "small"
    assert result["updated_policy"]["effective_from_engine"] == 41


def test_first_evaluation_patch_is_controlled_by_llm() -> None:
    feedback = [
        {
            "case_id": f"ForecastCase_FD001_Engine{idx}_Cycle100",
            "feedback_label": "missed_HPC_maintenance",
            "missed_maintenance_cause": "maintenance_scheduled_at_or_after_failure",
        }
        for idx in range(1, 21)
    ]
    report = EvaluationTool(20).evaluate(
        fd="FD001",
        feedback_logs=feedback,
        action_hypotheses=[],
        current_policy=initial_policy(),
        lhi_trigger=1.0,
    )
    result = validate_and_apply_evaluation(
        decision={
            "policy_patch": {"peak_offset_level": "small"}
        },
        report=report,
        current_policy=initial_policy(),
    )
    assert result["applied"]
    assert result["violations"] == []
    assert result["updated_policy"]["peak_offset_level"] == "small"
    assert result["updated_policy"]["effective_from_engine"] == 21


def test_evaluation_llm_can_apply_multiple_fields_without_support_or_cooldown() -> None:
    policy = initial_policy()
    policy["policy_revision"] = 1
    policy["effective_from_engine"] = 40
    report = {
        "checkpoint_id": "FD001_engine_40",
        "engines_completed": 40,
        "window_size": 20,
        "first_evaluation": False,
        "score_status": "stable",
        "recent_window": {
            "correct_maintenance_rate": 1.0,
            "too_early": 0,
            "missed_cause_counts": {},
        },
        "previous_window": {"correct_maintenance_rate": 1.0},
    }
    result = validate_and_apply_evaluation(
        decision={
            "policy_patch": {
                "peak_offset_level": "small",
                "monitoring_interval": 5,
            },
            "reason": "LLM decision",
            "confidence": 0.8,
        },
        report=report,
        current_policy=policy,
        minimum_support=999,
    )
    assert result["applied"]
    assert result["violations"] == []
    assert result["decision_authority"] == "evaluation_agent"
    assert result["updated_policy"]["peak_offset_level"] == "small"
    assert result["updated_policy"]["monitoring_interval"] == 5


def test_evaluation_allows_direct_non_adjacent_peak_offset_jump() -> None:
    policy = initial_policy()
    report = {"checkpoint_id": "FD001_engine_40", "engines_completed": 40}
    result = validate_and_apply_evaluation(
        decision={"policy_patch": {"peak_offset_level": "median"}},
        report=report,
        current_policy=policy,
    )
    assert result["applied"]
    assert result["valid"]
    assert result["violations"] == []
    assert result["updated_policy"]["peak_offset_level"] == "median"


def test_evaluation_reports_actual_policy_exposure() -> None:
    feedback = [
        {
            "case_id": f"ForecastCase_FD001_Engine{idx}_Cycle100",
            "feedback_label": "correct_maintenance",
        }
        for idx in range(1, 21)
    ]
    actions = [
        {
            "case_id": f"ForecastCase_FD001_Engine{idx}_Cycle100",
            "action": {"action_type": "schedule_HPC_maintenance", "action_time": "t+18"},
            "context": {
                "llm_policy": {"peak_offset_level": "small", "policy_revision": 1},
                "maintenance_timing_profile": {
                    "peak_score_cycle": "t+20",
                    "recommended_maintenance_time": "t+18",
                },
            },
        }
        for idx in range(1, 21)
    ]
    report = EvaluationTool(20).evaluate(
        fd="FD001",
        feedback_logs=feedback,
        action_hypotheses=actions,
        current_policy=initial_policy(),
        lhi_trigger=1.0,
    )
    assert report["policy_exposure"]["recent_window"]["peak_offset_level_counts"] == {
        "small": 20
    }
    assert report["action_statistics"]["maintenance_recommended_time_match_rate"] == 1.0


def test_evaluation_agent_allows_weak_direct_jump_when_direction_matches() -> None:
    report = {
        "first_evaluation": False,
        "score_status": "degrading",
        "evidence_strength": "weak",
        "current_policy": {"peak_offset_level": "small"},
        "timing_error_balance": {
            "dominant_observed_timing_direction": "increase_offset_schedule_earlier",
            "required_primary_driver_for_timing_update": "late_timing",
        },
    }
    violations = _decision_rule_violations(
        {
            "primary_driver": "late_timing",
            "policy_patch": {"peak_offset_level": "large"},
        },
        report,
    )
    assert violations == []


def test_evaluation_agent_rejects_wrong_timing_direction() -> None:
    report = {
        "current_policy": {"peak_offset_level": "median"},
        "timing_error_balance": {
            "dominant_observed_timing_direction": "increase_offset_schedule_earlier",
            "required_primary_driver_for_timing_update": "late_timing",
        },
    }
    violations = _decision_rule_violations(
        {
            "primary_driver": "late_timing",
            "policy_patch": {"peak_offset_level": "small"},
        },
        report,
    )
    assert any("requires a larger offset" in item for item in violations)

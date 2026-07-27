from __future__ import annotations

from src.RAG_cmapss.kg_prompt_ablation import (
    NO_KG_SYSTEM_PROMPT,
    build_no_kg_prompt,
    neutral_validation,
    sanitize_risk_tool,
    summarize_ablation,
)


def _case() -> dict:
    return {
        "case_id": "ForecastCase_FD001_Engine1_Cycle125",
        "forecast_horizon": {"start": 1, "end": 20},
        "risk_statistics": {"peak_score": 0.5, "peak_score_cycle": "t+20"},
        "trend_statistics": {"slope": 0.1, "duration_above_unit_q95": 20},
        "sensor_evidence_statistics": {
            "dominant_top_sensors": ["S11", "S8"],
            "sensor_presence_ratio": {"S11": 1.0},
            "sensor_mean_rank": {"S11": 1.0},
            "sensor_pattern_stability": 0.9,
            "hpc_sensor_presence_ratio": 1.0,
            "fan_sensor_presence_ratio": 0.5,
            "conflict_sensor_presence_ratio": 0.5,
        },
    }


def test_no_kg_prompt_excludes_graph_derived_context() -> None:
    prompt = build_no_kg_prompt(
        _case(),
        {"risk_level": "high_persistent", "peak_score": 0.5},
        {
            "tool_name": "RiskTool",
            "maintenance_risk_score": 0.5,
            "predicted_risk_stage": "maintenance_window",
            "top_features": [{"feature": "HPC_related_degradation"}],
        },
        {"theta_low": 0.1, "theta_conf": 0.3},
    )
    forbidden = [
        "Top graph evidence paths",
        "Retrieved action paths",
        "component_gate",
        "hpc_sensor_presence_ratio",
        "fan_sensor_presence_ratio",
        "conflict_sensor_presence_ratio",
        "supports_hypothesis",
        "alias_of",
    ]
    assert all(token not in prompt for token in forbidden)
    assert '"dominant_top_sensors"' in prompt
    assert '"S11"' in prompt
    assert "schedule_maintenance" in NO_KG_SYSTEM_PROMPT
    assert "schedule_HPC_maintenance" not in NO_KG_SYSTEM_PROMPT
    assert "HPC_related_degradation" not in prompt


def test_risk_tool_sanitizer_drops_feature_and_component_payloads() -> None:
    sanitized = sanitize_risk_tool(
        {
            "tool_name": "RiskTool",
            "predicted_risk_stage": "late_or_missed",
            "top_features": ["component"],
            "recommended_action": "schedule_HPC_maintenance",
        }
    )
    assert sanitized == {
        "tool_name": "RiskTool",
        "predicted_risk_stage": "late_or_missed",
    }


def test_neutral_validation_does_not_apply_fd_policy() -> None:
    action = {
        "action_type": "schedule_fan_maintenance",
        "action_time": "t+20",
        "confidence": 0.5,
        "evidence_paths": [],
    }
    assert neutral_validation(action, _case())["valid"]


def test_summary_compares_action_types() -> None:
    records = [
        {
            "baseline_action": {"action_type": "schedule_HPC_maintenance"},
            "ablation_action": {"action_type": "schedule_monitoring"},
            "neutral_validation": {"valid": True},
            "production_validation": {"valid": True},
        }
    ]
    summary = summarize_ablation(records)
    assert summary["action_type_changed_rate"] == 1.0
    assert summary["baseline_maintenance_to_monitoring_count"] == 1

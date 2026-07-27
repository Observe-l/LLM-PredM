from __future__ import annotations

from src.RAG_cmapss.action_validator import validate_action
from src.RAG_cmapss.react_agent import prepare_no_kg_context
from src.RAG_cmapss.reflection_memory import reflection_features


def _case() -> dict:
    return {
        "case_id": "ForecastCase_FD001_Engine1_Cycle125",
        "dataset_subset": "FD001",
        "forecast_horizon": {"start": 1, "end": 20},
        "forecast_summary": {
            "peak_score": 0.8,
            "peak_score_cycle": "t+20",
            "dominant_top_sensors": ["S9", "S11"],
        },
        "risk_statistics": {"peak_score": 0.8, "peak_score_cycle": "t+20"},
        "trend_statistics": {},
        "multi_score_statistics": {},
        "sensor_evidence_statistics": {
            "dominant_top_sensors": ["S9", "S11"],
            "hpc_sensor_presence_ratio": 1.0,
            "fan_sensor_presence_ratio": 0.0,
        },
        "key_cycles": [],
    }


def test_no_kg_context_does_not_retrieve_component_evidence() -> None:
    context = prepare_no_kg_context(_case())
    assert context["sensor_paths"] == []
    assert context["action_paths"] == []
    assert context["component_evidence_statistics"] == {}
    assert context["component_gate"] == {}
    assert context["dataset_rules"]["allowed_actions"] == [
        "continue_normal_operation",
        "schedule_monitoring",
        "schedule_maintenance",
    ]


def test_generic_maintenance_is_valid_in_no_kg_context() -> None:
    context = prepare_no_kg_context(_case())
    action = {
        "action_type": "schedule_maintenance",
        "action_time": "t+20",
    }
    assert validate_action(
        action,
        context["case"],
        context["dataset_rules"],
        context["sensor_paths"],
    )["valid"]


def test_no_kg_reflection_neutralizes_component_fields() -> None:
    features = reflection_features(
        _case(),
        {
            "action_type": "schedule_maintenance",
            "action_time": "t+20",
        },
        component_stats={"hpc_path_score": 0.99, "dominant_component": "HPC"},
        component_aware=False,
    )
    assert features["component_evidence_strength"] == "not_available"
    assert features["hpc_path_score"] == ""
    assert features["dominant_component"] == ""
    assert features["hpc_sensor_presence_ratio"] == ""

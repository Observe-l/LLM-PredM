from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .lightgbm_features import extract_lightgbm_features, feature_frame, top_feature_names


DEFAULT_UPDATE_MODEL_DIR = Path("models")


class LightGBMUpdateTool:
    """Suggest rule adaptation operators after feedback."""

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_UPDATE_MODEL_DIR
        self.threshold_payload = _load_payload(self.model_dir / "lightgbm_update_threshold.pkl")
        self.timing_payload = _load_payload(self.model_dir / "lightgbm_update_timing.pkl")

    def predict(
        self,
        case: dict[str, Any],
        action: dict[str, Any],
        feedback: dict[str, Any],
        context: dict[str, Any],
        lightgbm_risk: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        features = extract_lightgbm_features(case, context, action=action, feedback=feedback)
        threshold, threshold_conf, threshold_source = self._predict_label(
            self.threshold_payload,
            features,
            fallback=_fallback_threshold_update(feedback),
        )
        timing, timing_conf, timing_source = self._predict_label(
            self.timing_payload,
            features,
            fallback=_fallback_timing_update(feedback),
        )
        confidence = min(threshold_conf, timing_conf)
        return {
            "tool_name": "LightGBMUpdateTool",
            "model_dir": str(self.model_dir),
            "threshold_update": threshold,
            "timing_update": timing,
            "component_preference_update": _fallback_component_update(feedback),
            "revise_action_type": _fallback_revise_action(feedback, action),
            "update_strength": update_strength(confidence),
            "confidence": round(float(confidence), 4),
            "model_sources": {
                "threshold": threshold_source,
                "timing": timing_source,
            },
            "risk_score": (lightgbm_risk or {}).get("maintenance_risk_score"),
            "top_features": top_feature_names(features),
        }

    def _predict_label(
        self,
        payload: dict[str, Any] | None,
        features: dict[str, Any],
        fallback: str,
    ) -> tuple[str, float, str]:
        if payload is None:
            return fallback, 0.55, "fallback_feedback_mapping"
        model = payload["model"]
        columns = payload.get("feature_columns")
        frame = feature_frame([features])
        if columns:
            frame = frame.reindex(columns=columns, fill_value=0.0)
        probabilities = model.predict_proba(frame)
        classes = list(getattr(model, "classes_", payload.get("classes", [])))
        if not classes:
            return str(model.predict(frame)[0]), 0.5, "lightgbm_model"
        best_idx = max(range(len(classes)), key=lambda idx: float(probabilities[0][idx]))
        return str(classes[best_idx]), float(probabilities[0][best_idx]), "lightgbm_model"


def update_strength(confidence: float) -> str:
    if confidence < 0.6:
        return "small"
    if confidence < 0.8:
        return "medium"
    return "large"


def _fallback_threshold_update(feedback: dict[str, Any]) -> str:
    label = str(feedback.get("feedback_label", ""))
    if label in {"too_early", "over_maintenance"}:
        return "higher"
    if label in {"missed_HPC_maintenance", "missed_fan_maintenance", "missed_maintenance_unknown"}:
        if str(feedback.get("missed_maintenance_cause", "")) in {
            "maintenance_scheduled_at_or_after_failure",
            "monitoring_without_maintenance",
            "continued_operation_without_maintenance",
            "lhi_gate_not_triggered_before_failure",
        }:
            return "unchanged"
        if str(feedback.get("missed_maintenance_cause", "")) in {
            "monitoring_due_to_policy_gate",
            "continued_operation_due_to_policy_gate",
        }:
            return "lower"
        return "unchanged"
    return "unchanged"


def _fallback_timing_update(feedback: dict[str, Any]) -> str:
    label = str(feedback.get("feedback_label", ""))
    if label in {"too_early", "over_maintenance"}:
        return "delay"
    if label in {"missed_HPC_maintenance", "missed_fan_maintenance", "missed_maintenance_unknown"}:
        return "earlier"
    return "keep"


def _fallback_component_update(feedback: dict[str, Any]) -> str:
    label = str(feedback.get("feedback_label", ""))
    if label == "missed_HPC_maintenance":
        return "HPC"
    if label == "missed_fan_maintenance":
        return "FAN"
    return "unchanged"


def _fallback_revise_action(feedback: dict[str, Any], action: dict[str, Any]) -> str:
    label = str(feedback.get("feedback_label", ""))
    if label == "missed_HPC_maintenance":
        return "schedule_HPC_maintenance"
    if label == "missed_fan_maintenance":
        return "schedule_fan_maintenance"
    if label in {"too_early", "over_maintenance"}:
        return "schedule_monitoring"
    return str(action.get("action_type", "schedule_monitoring"))


def _load_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "model" in payload:
        return payload
    return {"model": payload}

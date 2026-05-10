from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .lightgbm_features import RISK_FEATURE_COLUMNS, extract_lightgbm_features, feature_frame, top_feature_names


DEFAULT_RISK_MODEL_PATH = Path("models/lightgbm_risk.pkl")


class LightGBMRiskTool:
    """Learned risk perception tool.

    In hybrid mode this tool is used only when a LightGBM risk model exists.
    The neutral fallback below is kept only for infrastructure failures.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        theta_low: float = 0.4,
        theta_conf: float = 0.3,
        disable_model: bool = False,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_RISK_MODEL_PATH
        self.theta_low = float(theta_low)
        self.theta_conf = float(theta_conf)
        self.payload = None if disable_model else _load_payload(self.model_path)

    @property
    def has_model(self) -> bool:
        return self.payload is not None

    def predict(self, case: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        features = extract_lightgbm_features(case, context)
        if self.payload is not None:
            score, source = self._predict_model(features)
        else:
            score, source = _zero_shot_score(context)
            suggested_theta = _policy_theta_low(context)
            if suggested_theta is not None:
                self.theta_low = suggested_theta
        confidence = round(2.0 * abs(float(score) - 0.5), 4)
        stage = risk_stage(score)
        if score < self.theta_low:
            decision = "monitor_without_llm"
        elif confidence < self.theta_conf:
            decision = "activate_llm_agent_uncertain"
        else:
            decision = "activate_llm_agent"
        return {
            "tool_name": "LightGBMRiskTool",
            "model_path": str(self.model_path),
            "model_source": source,
            "feature_schema": self.payload.get("feature_schema") if self.payload else None,
            "maintenance_risk_score": round(float(score), 6),
            "predicted_risk_stage": stage,
            "confidence": confidence,
            "risk_decision": decision,
            "theta_low": self.theta_low,
            "theta_conf": self.theta_conf,
            "top_features": top_feature_names(features),
            "zero_shot_policy": context.get("llm_risk_policy") if self.payload is None else None,
        }

    def _predict_model(self, features: dict[str, Any]) -> tuple[float, str]:
        assert self.payload is not None
        model = self.payload["model"]
        columns = self.payload.get("feature_columns")
        frame = feature_frame([features], columns=columns or RISK_FEATURE_COLUMNS)
        if columns:
            frame = frame.reindex(columns=columns, fill_value=0.0)
        probabilities = model.predict_proba(frame)
        classes = list(getattr(model, "classes_", self.payload.get("classes", [0, 1])))
        if 1 in classes:
            score = float(probabilities[0][classes.index(1)])
        else:
            score = float(max(probabilities[0]))
        return score, "lightgbm_model"


def risk_stage(score: float) -> str:
    score = float(score)
    if score < 0.30:
        return "normal"
    if score < 0.60:
        return "early_warning"
    if score < 0.85:
        return "maintenance_window"
    return "late_or_missed"


def _zero_shot_score(context: dict[str, Any]) -> tuple[float, str]:
    policy = context.get("llm_risk_policy") or {}
    try:
        score = float(policy.get("maintenance_risk_score"))
    except (TypeError, ValueError):
        score = 0.5
        source = "emergency_neutral_zero_shot_score"
    else:
        source = str(policy.get("source") or "cold_start_zero_shot_policy")
    return min(max(score, 0.02), 0.98), source


def _policy_theta_low(context: dict[str, Any]) -> float | None:
    policy = context.get("llm_risk_policy") or {}
    threshold_policy = policy.get("risk_threshold_policy") or {}
    try:
        return min(max(float(threshold_policy.get("theta_low")), 0.1), 0.8)
    except (TypeError, ValueError):
        return None


def _load_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "model" in payload:
        return payload
    return {"model": payload}

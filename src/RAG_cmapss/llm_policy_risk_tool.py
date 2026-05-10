from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lightgbm_risk_tool import risk_stage


DEFAULT_LLM_POLICY_PATH = Path("models/llm_policy_tool.json")


class PolicyValidationError(ValueError):
    pass


class LLMPolicyRiskTool:
    """Experiment-local peak-score risk policy tool.

    The tool is deterministic and does not call an LLM. In zero-shot runs it
    starts from a simple peak_score boundary, then LLMPolicyUpdateTool updates
    that boundary from this experiment's feedback only.
    """

    def __init__(
        self,
        policy_path: str | Path | None = None,
        theta_low: float = 0.4,
        theta_conf: float = 0.3,
    ):
        self.policy_path = Path(policy_path) if policy_path else DEFAULT_LLM_POLICY_PATH
        self.default_theta_low = float(theta_low)
        self.default_theta_conf = float(theta_conf)
        self.policy = load_policy(self.policy_path)

    @property
    def exists(self) -> bool:
        return self.policy is not None

    def save(self, policy: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_policy(policy)
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False))
        self.policy = normalized
        return normalized

    def ensure(self, policy: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.policy is not None:
            return self.policy
        return self.save(policy or initial_policy(theta_low=self.default_theta_low, theta_conf=self.default_theta_conf))

    def predict(self, case: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        policy = self.ensure(context.get("llm_policy"))
        peak_score = _case_peak_score(case, context)
        threshold = float(policy["peak_threshold"])
        score = min(max(float(peak_score), 0.0), 0.98)
        confidence = round(2.0 * abs(score - 0.5), 4)
        if peak_score < threshold:
            decision = "monitor_without_llm"
            stage = "normal"
        elif confidence < float(policy.get("theta_conf", self.default_theta_conf)):
            decision = "activate_llm_agent_uncertain"
            stage = risk_stage(score)
        else:
            decision = "activate_llm_agent"
            stage = "late_or_missed"
        return {
            "tool_name": "LLMPolicyRiskTool",
            "model_path": str(self.policy_path),
            "model_source": "llm_policy_tool",
            "maintenance_risk_score": round(float(score), 6),
            "predicted_risk_stage": stage,
            "confidence": confidence,
            "risk_decision": decision,
            "theta_low": threshold,
            "theta_conf": float(policy.get("theta_conf", self.default_theta_conf)),
            "top_features": ["peak_score"],
            "llm_policy_tool": policy_summary(policy),
        }


def initial_policy(
    *,
    peak_threshold: float = 0.5,
    theta_low: float = 0.4,
    theta_conf: float = 0.3,
) -> dict[str, Any]:
    threshold = round(max(0.0, float(peak_threshold)), 6)
    return validate_policy(
        {
            "tool_name": "LLMPolicyRiskTool",
            "version": 2,
            "source": "initial_peak_threshold_policy",
            "policy_type": "online_peak_score_boundary",
            "peak_threshold": threshold,
            "theta_low": threshold if theta_low is None else threshold,
            "theta_conf": float(theta_conf),
            "positive_peak_min": None,
            "early_peak_max": None,
            "correct_anchor_count": 0,
            "missed_anchor_count": 0,
            "early_anchor_count": 0,
            "updates": [],
            "reason": "Zero-shot online peak-score boundary policy.",
        }
    )


def load_policy(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return validate_policy(parsed)


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise PolicyValidationError("policy must be a JSON object")
    peak_threshold = _num(policy.get("peak_threshold"), default=None)
    if peak_threshold is None:
        raise PolicyValidationError("policy.peak_threshold must be numeric")
    sanitized = {
        "tool_name": "LLMPolicyRiskTool",
        "version": int(policy.get("version", 2) or 2),
        "source": str(policy.get("source", "llm_policy_tool")),
        "policy_type": "online_peak_score_boundary",
        "peak_threshold": round(max(0.0, float(peak_threshold)), 6),
        "theta_low": round(max(0.0, float(peak_threshold)), 6),
        "theta_conf": _bounded(policy.get("theta_conf"), 0.3, 0.0, 1.0),
        "positive_peak_min": _optional_float(policy.get("positive_peak_min")),
        "early_peak_max": _optional_float(policy.get("early_peak_max")),
        "correct_anchor_count": int(policy.get("correct_anchor_count", 0) or 0),
        "missed_anchor_count": int(policy.get("missed_anchor_count", 0) or 0),
        "early_anchor_count": int(policy.get("early_anchor_count", 0) or 0),
        "updates": list(policy.get("updates", []))[-200:] if isinstance(policy.get("updates"), list) else [],
        "reason": str(policy.get("reason", ""))[:600],
    }
    return sanitized


def policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": policy.get("source"),
        "policy_type": policy.get("policy_type"),
        "peak_threshold": policy.get("peak_threshold"),
        "theta_low": policy.get("theta_low"),
        "theta_conf": policy.get("theta_conf"),
        "reason": policy.get("reason"),
    }


def _case_peak_score(case: dict[str, Any], context: dict[str, Any]) -> float:
    risk = case.get("risk_statistics", {})
    value = risk.get("peak_score")
    if value in {None, ""}:
        value = case.get("forecast_summary", {}).get("peak_score")
    if value in {None, ""}:
        value = context.get("risk_gate", {}).get("peak_score")
    return float(_num(value, 0.0) or 0.0)


def _optional_float(value: Any) -> float | None:
    number = _num(value, default=None)
    return round(float(number), 6) if number is not None else None


def _bounded(value: Any, default: float, low: float, high: float) -> float:
    number = _num(value, default)
    return round(min(max(float(number), low), high), 6)


def _num(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

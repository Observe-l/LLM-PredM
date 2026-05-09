from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lightgbm_features import extract_lightgbm_features, top_feature_names
from .lightgbm_risk_tool import risk_stage


DEFAULT_LLM_POLICY_PATH = Path("models/llm_policy_tool.json")


DEFAULT_POLICY: dict[str, Any] = {
    "tool_name": "LLMPolicyRiskTool",
    "version": 1,
    "source": "default_peak_score_policy",
    "policy_family": "peak_score_weighted_policy",
    "bias": 0.03,
    "weights": {
        "peak_score": 0.60,
        "peak_minus_unit_q95": 0.15,
        "duration_above_unit_q95": 0.10,
        "monotonicity": 0.05,
        "component_gate_supported": 0.05,
        "hpc_path_score": 0.025,
        "fan_path_score": 0.025,
    },
    "normalizers": {
        "peak_score": {"kind": "cap", "scale": 1.5},
        "peak_minus_unit_q95": {"kind": "positive_cap", "scale": 0.5},
        "duration_above_unit_q95": {"kind": "positive_cap", "scale": 20.0},
        "monotonicity": {"kind": "cap", "scale": 1.0},
        "component_gate_supported": {"kind": "binary"},
        "hpc_path_score": {"kind": "cap", "scale": 1.0},
        "fan_path_score": {"kind": "cap", "scale": 1.0},
    },
    "theta_low": 0.40,
    "theta_conf": 0.30,
    "maintenance_window_threshold": 0.60,
    "score_formula": (
        "bias + 0.60*norm(peak_score) + 0.15*norm(peak_minus_unit_q95) + "
        "0.10*norm(duration_above_unit_q95) + 0.05*norm(monotonicity) + "
        "0.05*component_gate_supported + 0.025*norm(hpc_path_score) + 0.025*norm(fan_path_score)"
    ),
    "reason": "Default cold-start policy with peak_score as the dominant damage-risk anchor.",
}


class LLMPolicyRiskTool:
    """Persistent deterministic risk policy designed by an LLM.

    The LLM writes or updates the policy parameters. The tool itself computes
    risk scores without calling the LLM, matching the LightGBMRiskTool contract.
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
        normalized = validate_policy(policy, fallback=DEFAULT_POLICY)
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False))
        self.policy = normalized
        return normalized

    def predict(self, case: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        policy = self.policy or DEFAULT_POLICY
        features = extract_lightgbm_features(case, context)
        score, calculation = score_with_policy(features, policy)
        theta_low = _bounded_float(policy.get("theta_low"), self.default_theta_low, 0.1, 0.8)
        theta_conf = _bounded_float(policy.get("theta_conf"), self.default_theta_conf, 0.0, 1.0)
        confidence = round(2.0 * abs(float(score) - 0.5), 4)
        stage = risk_stage(score)
        if score < theta_low:
            decision = "monitor_without_llm"
        elif confidence < theta_conf:
            decision = "activate_llm_agent_uncertain"
        else:
            decision = "activate_llm_agent"
        return {
            "tool_name": "LLMPolicyRiskTool",
            "model_path": str(self.policy_path),
            "model_source": "llm_policy_tool" if self.exists else "default_llm_policy_tool",
            "maintenance_risk_score": round(float(score), 6),
            "predicted_risk_stage": stage,
            "confidence": confidence,
            "risk_decision": decision,
            "theta_low": theta_low,
            "theta_conf": theta_conf,
            "top_features": top_feature_names(features),
            "llm_policy_tool": policy_summary(policy),
            "score_components": calculation["score_components"],
            "raw_score": calculation["raw_score"],
        }


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
    return validate_policy(parsed, fallback=DEFAULT_POLICY)


def validate_policy(policy: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = fallback or DEFAULT_POLICY
    candidate = {**fallback, **(policy or {})}
    weights = candidate.get("weights")
    if not isinstance(weights, dict):
        weights = dict(fallback["weights"])
    clean_weights = {str(k): max(0.0, float(v)) for k, v in weights.items() if _is_number(v)}
    if "peak_score" not in clean_weights:
        clean_weights["peak_score"] = float(fallback["weights"]["peak_score"])
    peak_weight = float(clean_weights["peak_score"])
    max_aux = max([v for k, v in clean_weights.items() if k != "peak_score"] or [0.0])
    if peak_weight < max_aux:
        clean_weights["peak_score"] = max_aux
    if clean_weights["peak_score"] < 0.45:
        clean_weights["peak_score"] = 0.45
    total = sum(clean_weights.values())
    if total > 1.5:
        clean_weights = {key: value / total for key, value in clean_weights.items()}
        if clean_weights.get("peak_score", 0.0) < 0.45:
            clean_weights["peak_score"] = 0.45
    candidate["weights"] = clean_weights
    normalizers = candidate.get("normalizers")
    if not isinstance(normalizers, dict):
        normalizers = dict(fallback["normalizers"])
    for feature in clean_weights:
        normalizers.setdefault(feature, fallback["normalizers"].get(feature, {"kind": "cap", "scale": 1.0}))
    candidate["normalizers"] = normalizers
    candidate["bias"] = _bounded_float(candidate.get("bias"), fallback.get("bias", 0.03), 0.0, 0.4)
    candidate["theta_low"] = _bounded_float(candidate.get("theta_low"), fallback.get("theta_low", 0.4), 0.1, 0.8)
    candidate["theta_conf"] = _bounded_float(candidate.get("theta_conf"), fallback.get("theta_conf", 0.3), 0.0, 1.0)
    candidate["maintenance_window_threshold"] = _bounded_float(
        candidate.get("maintenance_window_threshold"),
        fallback.get("maintenance_window_threshold", 0.6),
        0.3,
        0.9,
    )
    candidate["tool_name"] = "LLMPolicyRiskTool"
    candidate["version"] = int(candidate.get("version", 1) or 1)
    formula = str(candidate.get("score_formula", ""))
    if "peak_score" not in formula:
        candidate["score_formula"] = (
            "bias + peak_score-dominant weighted normalized features; "
            f"weights={json.dumps(clean_weights, sort_keys=True)}"
        )
    return candidate


def score_with_policy(features: dict[str, Any], policy: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    weights = policy.get("weights") or {}
    normalizers = policy.get("normalizers") or {}
    raw_score = _bounded_float(policy.get("bias"), 0.03, 0.0, 0.4)
    components: list[dict[str, Any]] = [{"name": "bias", "value": raw_score, "normalized": 1.0, "contribution": raw_score}]
    for name, weight in weights.items():
        observed = _num(features.get(name))
        normalized = normalize_feature(observed, normalizers.get(name, {}))
        contribution = float(weight) * normalized
        raw_score += contribution
        components.append(
            {
                "name": name,
                "value": round(observed, 6),
                "normalized": round(normalized, 6),
                "weight": round(float(weight), 6),
                "contribution": round(contribution, 6),
            }
        )
    score = round(min(max(raw_score, 0.02), 0.98), 6)
    return score, {
        "raw_score": round(float(score), 6),
        "score_components": components,
    }


def normalize_feature(value: float, spec: Any) -> float:
    if not isinstance(spec, dict):
        spec = {"kind": "cap", "scale": 1.0}
    kind = str(spec.get("kind", "cap"))
    scale = max(_num(spec.get("scale"), 1.0), 1e-9)
    if kind == "binary":
        return 1.0 if value > 0 else 0.0
    if kind == "positive_cap":
        return min(max(value, 0.0) / scale, 1.0)
    if kind == "signed_cap":
        return min(max((value / scale + 1.0) / 2.0, 0.0), 1.0)
    return min(max(value, 0.0) / scale, 1.0)


def policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": policy.get("source"),
        "policy_family": policy.get("policy_family"),
        "score_formula": policy.get("score_formula"),
        "weights": policy.get("weights"),
        "normalizers": policy.get("normalizers"),
        "theta_low": policy.get("theta_low"),
        "theta_conf": policy.get("theta_conf"),
        "maintenance_window_threshold": policy.get("maintenance_window_threshold"),
        "reason": policy.get("reason"),
        "last_update": policy.get("last_update"),
    }


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    number = _num(value, default)
    return round(min(max(number, low), high), 6)


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

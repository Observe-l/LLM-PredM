from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .action_validator import validate_action
from .action_validator import STRONG_FAN_SENSORS, STRONG_HPC_SENSORS
from .agentic_controller import initial_risk_policy, llm_design_policy_tool
from .graph_retriever import (
    build_component_evidence_statistics,
    get_dataset_rules,
    infer_candidate_action,
    retrieve_action_paths,
    retrieve_sensor_paths,
)
from .evidence_gates import build_component_gate, build_risk_gate
from .kg_store import KGStore
from .lightgbm_risk_tool import LightGBMRiskTool
from .llm_policy_risk_tool import LLMPolicyRiskTool
from .ollama_client import extract_json, ollama_chat
from .prompt_builder import SYSTEM_PROMPT, build_prompt


def normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    case = dict(case)
    horizon = case.get("forecast_horizon")
    if isinstance(horizon, str):
        case["forecast_horizon"] = {"start": 1, "end": _parse_horizon_end(horizon), "text": horizon}
    elif isinstance(horizon, dict):
        case["forecast_horizon"] = {
            "start": int(horizon.get("start", 1)),
            "end": int(horizon.get("end", _parse_horizon_end(str(horizon.get("text", "t+1 to t+50"))))),
            "text": str(horizon.get("text", f"t+{horizon.get('start', 1)} to t+{horizon.get('end', 50)}")),
        }
    else:
        case["forecast_horizon"] = {"start": 1, "end": 50, "text": "t+1 to t+50"}
    return case


def prepare_context(
    case: dict[str, Any],
    kg_dir: str,
    kg_store: KGStore,
    risk_threshold_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case = normalize_case(case)
    dataset_subset = str(case["dataset_subset"])
    forecast_summary = case["forecast_summary"]
    sensors = [str(s) for s in forecast_summary.get("dominant_top_sensors", [])]

    dataset_rules = get_dataset_rules(kg_store, dataset_subset)
    sensor_paths = retrieve_sensor_paths(
        kg_store,
        sensors,
        forecast_summary=forecast_summary,
        dataset_rules=dataset_rules,
    )
    component_evidence_statistics = build_component_evidence_statistics(
        sensor_paths,
        sensor_evidence_statistics=case.get("sensor_evidence_statistics", {}),
    )
    hypotheses = sorted({p["hypothesis"] for p in sensor_paths})
    action_paths = retrieve_action_paths(kg_store, hypotheses)
    candidate_action = infer_candidate_action(dataset_rules, sensor_paths, forecast_summary)
    reflection_rules: list[dict[str, Any]] = []
    risk_gate = build_risk_gate(case, reflection_rules=[], threshold_overrides=risk_threshold_overrides)
    component_gate = build_component_gate(component_evidence_statistics, dataset_rules)
    return {
        "case": case,
        "dataset_rules": dataset_rules,
        "sensor_paths": sensor_paths,
        "action_paths": action_paths,
        "candidate_action": candidate_action,
        "reflection_rules": reflection_rules,
        "component_evidence_statistics": component_evidence_statistics,
        "risk_gate": risk_gate,
        "component_gate": component_gate,
        "reflection_memory_usage": "training_only",
        "risk_threshold_overrides": risk_threshold_overrides or {},
    }


def run_agent(
    case: dict[str, Any],
    kg_dir: str,
    kg_store: KGStore,
    model: str,
    ollama_url: str,
    temperature: float = 0.1,
    timeout: int = 180,
    num_predict: int = 512,
    format_json: bool = True,
    dry_run: bool = False,
    risk_model_path: str | None = None,
    risk_theta_low: float = 0.4,
    risk_theta_conf: float = 0.3,
    risk_threshold_overrides: dict[str, Any] | None = None,
    risk_policy_mode: str = "hybrid",
    llm_policy_tool_path: str | None = None,
) -> dict[str, Any]:
    context = prepare_context(case, kg_dir, kg_store, risk_threshold_overrides=risk_threshold_overrides)
    risk_policy_mode = str(risk_policy_mode or "hybrid")
    risk_tool = LightGBMRiskTool(
        model_path=risk_model_path,
        theta_low=risk_theta_low,
        theta_conf=risk_theta_conf,
        disable_model=risk_policy_mode == "llm_only",
    )
    controller_raw_outputs = []
    if risk_policy_mode == "llm_only" or (risk_policy_mode == "hybrid" and not risk_tool.has_model):
        llm_policy_tool = LLMPolicyRiskTool(
            policy_path=llm_policy_tool_path,
            theta_low=risk_theta_low,
            theta_conf=risk_theta_conf,
        )
        if not llm_policy_tool.exists:
            designed_policy = llm_design_policy_tool(
                case=context["case"],
                context=context,
                model=model,
                ollama_url=ollama_url,
                temperature=temperature,
                timeout=timeout,
                num_predict=min(max(num_predict, 1024), 1536),
                format_json=format_json,
                dry_run=dry_run,
                reflection_rules_path=Path(kg_dir) / "reflection_rules.csv",
            )
            llm_policy_tool.save(designed_policy)
            if designed_policy.get("raw_output"):
                controller_raw_outputs.append(
                    {
                        "stage": "llm_policy_tool_design",
                        "format_json": format_json,
                        "num_predict": min(max(num_predict, 1024), 1536),
                        "messages": [
                            {"role": "system", "content": "Return only valid JSON."},
                            {"role": "user", "content": designed_policy.get("prompt", "")},
                        ],
                        "raw_output": designed_policy.get("raw_output", ""),
                        "parse_ok": True,
                    }
                )
        context["lightgbm_risk"] = llm_policy_tool.predict(context["case"], context)
        context["llm_risk_policy"] = {
            **context["lightgbm_risk"].get("llm_policy_tool", {}),
            "source": context["lightgbm_risk"].get("model_source"),
            "maintenance_risk_score": context["lightgbm_risk"].get("maintenance_risk_score"),
            "score_components": context["lightgbm_risk"].get("score_components", []),
            "risk_threshold_policy": {
                "theta_low": context["lightgbm_risk"].get("theta_low"),
                "theta_conf": context["lightgbm_risk"].get("theta_conf"),
                "maintenance_window_threshold": context["lightgbm_risk"]
                .get("llm_policy_tool", {})
                .get("maintenance_window_threshold"),
            },
        }
    elif not risk_tool.has_model:
        context["llm_risk_policy"] = initial_risk_policy(
            case=context["case"],
            context=context,
            source="initial_risk_policy",
        )
        context["lightgbm_risk"] = risk_tool.predict(context["case"], context)
    else:
        context["lightgbm_risk"] = risk_tool.predict(context["case"], context)
    llm_calls = len(controller_raw_outputs)
    llm_fallback_used = False
    llm_errors: list[str] = []
    if dry_run:
        action = rule_based_action(context)
        raw_outputs: list[dict[str, Any] | str] = [*controller_raw_outputs, "<dry_run rule_based_action>"]
    elif not _should_activate_llm(context):
        action = low_risk_action(context)
        raw_outputs = [*controller_raw_outputs, "<lightgbm_low_risk_monitoring>"]
    else:
        prompt = build_prompt(
            case=context["case"],
            dataset_rules=context["dataset_rules"],
            sensor_paths=context["sensor_paths"],
            action_paths=context["action_paths"],
            component_evidence_statistics=context["component_evidence_statistics"],
            risk_gate=context["risk_gate"],
            component_gate=context["component_gate"],
            lightgbm_risk=context["lightgbm_risk"],
        )
        raw_outputs = list(controller_raw_outputs)
        action = None
        for attempt in _llm_attempts(prompt, context, num_predict=num_predict, format_json=format_json):
            try:
                raw = ollama_chat(
                    attempt["messages"],
                    model=model,
                    url=ollama_url,
                    temperature=temperature,
                    timeout=timeout,
                    num_predict=attempt["num_predict"],
                    format_json=attempt["format_json"],
                    think=False,
                )
                llm_calls += 1
                entry: dict[str, Any] = {
                    "stage": attempt["stage"],
                    "format_json": attempt["format_json"],
                    "num_predict": attempt["num_predict"],
                    "messages": attempt["messages"],
                    "raw_output": raw,
                }
                try:
                    action = extract_json(raw)
                    entry["parse_ok"] = True
                    raw_outputs.append(entry)
                    break
                except Exception as exc:
                    entry["parse_ok"] = False
                    entry["error"] = str(exc)
                    raw_outputs.append(entry)
                    llm_errors.append(f"{attempt['stage']}: {exc}")
            except Exception as exc:
                raw_outputs.append(
                    {
                        "stage": attempt["stage"],
                        "format_json": attempt["format_json"],
                        "num_predict": attempt["num_predict"],
                        "messages": attempt["messages"],
                        "raw_output": "",
                        "parse_ok": False,
                        "error": str(exc),
                    }
                )
                llm_errors.append(f"{attempt['stage']}: {exc}")
                # Network/time-out failures usually will not improve with a prompt retry.
                # Fall back immediately so a long Ollama call does not block the whole simulation repeatedly.
                break

        if action is None:
            llm_fallback_used = True
            action = rule_based_action(context)
            action["reason"] = (
                "LLM did not return parseable JSON after retry; used deterministic KG rule fallback. "
                + action.get("reason", "")
            )

    validation = validate_action(
        action,
        context["case"],
        context["dataset_rules"],
        context["sensor_paths"],
        risk_gate=context["risk_gate"],
        lightgbm_risk=context["lightgbm_risk"],
    )

    if not validation["valid"] and not dry_run and not llm_fallback_used:
        local_action = _local_validation_repair(action, validation, context)
        if local_action is not None:
            action = local_action
            action["local_validation_repair_used"] = True
            validation = validate_action(
                action,
                context["case"],
                context["dataset_rules"],
                context["sensor_paths"],
                risk_gate=context["risk_gate"],
                lightgbm_risk=context["lightgbm_risk"],
            )

    if not validation["valid"] and not dry_run and not llm_fallback_used:
        repair_prompt = f"""The previous action is invalid.

Previous action:
{json.dumps(action, indent=2)}

Validation errors:
{json.dumps(validation, indent=2)}

Revise the action using the same forecast case, dataset rules, and graph evidence.
Return only valid JSON.
"""
        try:
            raw2 = ollama_chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt + "\n\n" + repair_prompt}],
                model=model,
                url=ollama_url,
                temperature=temperature,
                timeout=timeout,
                num_predict=num_predict,
                format_json=format_json,
                think=False,
            )
            llm_calls += 1
            repair_entry: dict[str, Any] = {
                "stage": "repair",
                "format_json": format_json,
                "num_predict": num_predict,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt + "\n\n" + repair_prompt},
                ],
                "raw_output": raw2,
            }
            try:
                repaired_action = extract_json(raw2)
                repair_entry["parse_ok"] = True
                action = repaired_action
                validation = validate_action(
                    action,
                    context["case"],
                    context["dataset_rules"],
                    context["sensor_paths"],
                    risk_gate=context["risk_gate"],
                    lightgbm_risk=context["lightgbm_risk"],
                )
                if not validation["valid"]:
                    repair_entry["post_repair_validation"] = validation
                    llm_errors.append(f"repair_invalid: {validation['violations']}")
                    llm_fallback_used = True
                    action = rule_based_action(context)
                    action["reason"] = "Repair output was still invalid; used deterministic KG rule fallback."
                    validation = validate_action(
                        action,
                        context["case"],
                        context["dataset_rules"],
                        context["sensor_paths"],
                        risk_gate=context["risk_gate"],
                        lightgbm_risk=context["lightgbm_risk"],
                    )
            except Exception as exc:
                repair_entry["parse_ok"] = False
                repair_entry["error"] = str(exc)
                llm_errors.append(f"repair: {exc}")
                llm_fallback_used = True
                action = rule_based_action(context)
                action["reason"] = "Repair output was not parseable; used deterministic KG rule fallback."
                validation = validate_action(
                    action,
                    context["case"],
                    context["dataset_rules"],
                    context["sensor_paths"],
                    risk_gate=context["risk_gate"],
                    lightgbm_risk=context["lightgbm_risk"],
                )
            raw_outputs.append(repair_entry)
        except Exception as exc:
            llm_errors.append(f"repair: {exc}")
            raw_outputs.append({"stage": "repair", "raw_output": "", "parse_ok": False, "error": str(exc)})
            llm_fallback_used = True
            action = rule_based_action(context)
            action["reason"] = "Repair call failed; used deterministic KG rule fallback."
            validation = validate_action(
                action,
                context["case"],
                context["dataset_rules"],
                context["sensor_paths"],
                risk_gate=context["risk_gate"],
                lightgbm_risk=context["lightgbm_risk"],
            )

    if not validation["valid"]:
        llm_errors.append(f"final_invalid: {validation['violations']}")
        llm_fallback_used = True
        action = rule_based_action(context)
        action["reason"] = "Final action was invalid; used deterministic KG rule fallback."
        validation = validate_action(
            action,
            context["case"],
            context["dataset_rules"],
            context["sensor_paths"],
            risk_gate=context["risk_gate"],
            lightgbm_risk=context["lightgbm_risk"],
        )

    action["validation"] = validation
    action["validation_status"] = "valid" if validation["valid"] else "invalid"
    if llm_fallback_used:
        action["llm_fallback_used"] = True
    if llm_errors:
        action["llm_errors"] = llm_errors
    return {
        "case_id": context["case"].get("case_id"),
        "action": action,
        "context": {
            "dataset_rules": context["dataset_rules"],
            "candidate_action": context["candidate_action"],
            "sensor_paths": context["sensor_paths"],
            "action_paths": context["action_paths"],
            "reflection_rules": context["reflection_rules"],
            "component_evidence_statistics": context["component_evidence_statistics"],
            "risk_gate": context["risk_gate"],
            "component_gate": context["component_gate"],
            "reflection_memory_usage": context["reflection_memory_usage"],
            "risk_threshold_overrides": context.get("risk_threshold_overrides", {}),
            "llm_risk_policy": context.get("llm_risk_policy"),
            "lightgbm_risk": context["lightgbm_risk"],
        },
        "llm_calls": llm_calls,
        "raw_outputs": raw_outputs,
    }


def _llm_attempts(
    prompt: str,
    context: dict[str, Any],
    num_predict: int,
    format_json: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "stage": "initial",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "format_json": format_json,
            "num_predict": num_predict,
        },
        {
            "stage": "minimal_retry",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _minimal_json_prompt(context)},
            ],
            "format_json": True,
            "num_predict": 384,
        },
        {
            "stage": "plain_retry",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _minimal_json_prompt(context)},
            ],
            "format_json": False,
            "num_predict": 384,
        },
    ]


def _hard_gate_blocks_llm(context: dict[str, Any]) -> bool:
    risk_gate = context.get("risk_gate", {})
    if not risk_gate.get("maintenance_candidate", False):
        return True
    return False


def _should_activate_llm(context: dict[str, Any]) -> bool:
    risk = context.get("lightgbm_risk", {})
    risk_gate = context.get("risk_gate", {})
    return bool(
        float(risk.get("maintenance_risk_score", 0.0)) >= float(risk.get("theta_low", 0.4))
        or risk_gate.get("statistical_candidate", False)
    )


def _local_validation_repair(
    action: dict[str, Any],
    validation: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any] | None:
    violations = list(validation.get("violations", []))
    if action.get("action_type") == "schedule_monitoring" and violations == [
        "schedule_monitoring must have non-null action_time"
    ]:
        repaired = dict(action)
        repaired["action_time"] = rule_based_action(context)["action_time"]
        repaired["reason"] = "Filled missing monitoring time from KG rule fallback."
        return repaired

    hard_gate_violations = {
        "maintenance action lacks strong statistical risk-gate support",
        "schedule_HPC_maintenance requires strong HPC sensor evidence from S7/S11/S3/S9/S14",
        "schedule_fan_maintenance requires strong Fan sensor evidence from S8/S13/S15",
    }
    if action.get("action_type") in {"schedule_HPC_maintenance", "schedule_fan_maintenance"} and any(
        item in hard_gate_violations for item in violations
    ):
        repaired = rule_based_action(context)
        repaired["reason"] = "Maintenance violated hard validation gates; used deterministic KG fallback."
        return repaired
    return None


def _minimal_json_prompt(context: dict[str, Any]) -> str:
    case = context["case"]
    summary = case["forecast_summary"]
    payload = {
        "case_id": case.get("case_id"),
        "forecast_horizon": case.get("forecast_horizon"),
        "forecast_summary": {
            "peak_score": summary.get("peak_score"),
            "peak_score_cycle": summary.get("peak_score_cycle"),
            "score_trend": summary.get("score_trend"),
            "first_warning_crossing_cycle": summary.get("first_warning_crossing_cycle"),
            "dominant_top_sensors": summary.get("dominant_top_sensors"),
        },
        "risk_gate": context.get("risk_gate"),
        "component_gate": context.get("component_gate"),
        "lightgbm_risk": context.get("lightgbm_risk"),
        "dataset_rules": context["dataset_rules"],
        "top_evidence_paths": [p["path_text"] for p in context["sensor_paths"][:5]],
        "action_paths": [p["path_text"] for p in context["action_paths"][:4]],
        "reflection_memory_usage": context.get("reflection_memory_usage"),
    }
    return (
        "Return one JSON object only. No prose. No markdown. "
        "Choose action_type from continue_normal_operation, schedule_monitoring, "
        "schedule_fan_maintenance, schedule_HPC_maintenance. "
        "Use action_time as null or a string like \"t+20\". "
        "Use evidence_paths as short IDs only, e.g. [\"E1\", \"E2\"]. "
        "Keep reason under 30 words.\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nRequired keys: action_type, action_time, risk_hypothesis, degradation_hypothesis, "
        "confidence, evidence_paths, reason, validation_status."
    )


def rule_based_action(context: dict[str, Any]) -> dict[str, Any]:
    case = context["case"]
    summary = case["forecast_summary"]
    dataset_rules = context["dataset_rules"]
    sensor_paths = context["sensor_paths"]
    path_hypotheses = {p["hypothesis"] for p in sensor_paths}
    disallowed = set(dataset_rules.get("disallowed_actions", []))
    risk_gate = context.get("risk_gate", {})
    component_gate = context.get("component_gate", {})
    learned_risk = context.get("lightgbm_risk", {})

    if summary.get("first_critical_crossing_cycle"):
        risk = "critical_risk_hypothesis"
    elif summary.get("first_warning_crossing_cycle") or summary.get("score_trend") == "increasing":
        risk = "warning_risk_hypothesis"
    else:
        risk = "low_risk_hypothesis"

    action_type = "continue_normal_operation"
    degradation = "low_risk_hypothesis"
    strong_hpc = _has_strong_evidence(sensor_paths, "HPC_related_degradation", STRONG_HPC_SENSORS)
    strong_fan = _has_strong_evidence(sensor_paths, "Fan_related_degradation", STRONG_FAN_SENSORS)

    learned_support = _learned_risk_supports_maintenance(learned_risk)
    if not learned_support and not risk_gate.get("maintenance_candidate", False):
        action_type = "schedule_monitoring"
        degradation = "uncertain_component_degradation"
    elif component_gate.get("component_supported", False):
        suggested = component_gate.get("suggested_component_action")
        if suggested == "schedule_HPC_maintenance" and strong_hpc and suggested not in disallowed:
            action_type = "schedule_HPC_maintenance"
            degradation = "HPC_related_degradation"
        elif suggested == "schedule_fan_maintenance" and strong_fan and suggested not in disallowed:
            action_type = "schedule_fan_maintenance"
            degradation = "Fan_related_degradation"
        else:
            action_type = "schedule_monitoring"
            degradation = "uncertain_component_degradation"
    elif risk != "low_risk_hypothesis" or "uncertain_component_degradation" in path_hypotheses:
        action_type = "schedule_monitoring"
        degradation = "uncertain_component_degradation"

    if action_type == "continue_normal_operation":
        action_time = None
    elif action_type == "schedule_monitoring":
        horizon = case.get("forecast_horizon", {})
        action_time = f"t+{int(horizon.get('end', 20))}"
    else:
        action_time = _maintenance_action_time(context)

    return {
        "action_type": action_type,
        "action_time": action_time,
        "risk_hypothesis": risk,
        "degradation_hypothesis": degradation,
        "confidence": _confidence(sensor_paths, degradation),
        "evidence_paths": [p["path_text"] for p in sensor_paths[:5]],
        "reason": "Rule-based dry run using KG-retrieved dataset rules, sensor paths, and action constraints.",
        "validation_status": "pending",
    }


def low_risk_action(context: dict[str, Any]) -> dict[str, Any]:
    action = rule_based_action(context)
    if action["action_type"] in {"schedule_HPC_maintenance", "schedule_fan_maintenance"}:
        action["action_type"] = "schedule_monitoring"
        action["degradation_hypothesis"] = "uncertain_component_degradation"
        horizon = context["case"].get("forecast_horizon", {})
        action["action_time"] = f"t+{int(horizon.get('end', 20))}"
    action["lightgbm_low_risk_gate_used"] = True
    action["reason"] = (
        "LightGBMRiskTool judged this case below LLM activation threshold; scheduled conservative monitoring."
    )
    return action


def _confidence(sensor_paths: list[dict[str, Any]], hypothesis: str) -> float:
    scores = [float(p.get("score", 0.0)) for p in sensor_paths if p.get("hypothesis") == hypothesis]
    if not scores:
        return 0.5
    return round(min(max(sum(scores[:3]) / min(len(scores), 3), 0.0), 1.0), 3)


def _maintenance_action_time(context: dict[str, Any]) -> str:
    case = context["case"]
    summary = case["forecast_summary"]
    peak_time = (
        summary.get("peak_score_cycle")
        or summary.get("first_critical_crossing_cycle")
        or summary.get("first_persistent_pattern_cycle")
        or "t+1"
    )
    return str(peak_time)


def _learned_risk_supports_maintenance(lightgbm_risk: dict[str, Any]) -> bool:
    score = _to_float(lightgbm_risk.get("maintenance_risk_score")) or 0.0
    theta = _to_float(lightgbm_risk.get("theta_low")) or 0.4
    decision = str(lightgbm_risk.get("risk_decision", ""))
    stage = str(lightgbm_risk.get("predicted_risk_stage", ""))
    return (
        score >= max(theta, 0.60)
        and decision in {"activate_llm_agent", "activate_llm_agent_uncertain"}
        and stage in {"maintenance_window", "late_or_missed"}
    )


def _offset_t_plus(value: Any, case: dict[str, Any], offset: int) -> str | None:
    number = _parse_t_plus_value(value)
    if number is None:
        return None
    horizon = case.get("forecast_horizon", {})
    start = int(horizon.get("start", 1))
    end = int(horizon.get("end", number))
    return f"t+{min(max(number + offset, start), end)}"


def _parse_t_plus_value(value: Any) -> int | None:
    if not isinstance(value, str) or not value.startswith("t+"):
        return None
    try:
        return int(value[2:])
    except ValueError:
        return None


def _has_strong_evidence(sensor_paths: list[dict[str, Any]], hypothesis: str, sensors: set[str]) -> bool:
    return any(
        path.get("hypothesis") == hypothesis and str(path.get("sensor", "")).upper() in sensors
        for path in sensor_paths
    )


def _parse_horizon_end(text: str) -> int:
    import re

    matches = re.findall(r"t\+(\d+)", text)
    return int(matches[-1]) if matches else 50


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

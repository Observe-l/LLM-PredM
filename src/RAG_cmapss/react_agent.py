from __future__ import annotations

import json
from typing import Any

from .action_validator import validate_action
from .graph_retriever import (
    build_component_evidence_statistics,
    get_dataset_rules,
    infer_candidate_action,
    retrieve_action_paths,
    retrieve_sensor_paths,
)
from .evidence_gates import (
    build_risk_gate,
)
from .kg_store import KGStore
from .lightgbm_risk_tool import LightGBMRiskTool
from .llm_policy_risk_tool import LLMPolicyRiskTool
from .ollama_client import extract_json, ollama_chat
from .kg_prompt_ablation import NO_KG_SYSTEM_PROMPT, build_no_kg_prompt
from .prompt_builder import (
    CURRENT_LHI_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_current_lhi_prompt,
    build_prompt,
)
from .timing_policy import (
    maintenance_timing_profile,
    recommended_maintenance_time,
    recommended_monitoring_time,
)


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
    llm_policy: dict[str, Any] | None = None,
    mixed_fleet: bool = False,
    component_consensus: dict[str, Any] | None = None,
    prior_monitoring_count: int = 0,
) -> dict[str, Any]:
    case = normalize_case(case)
    dataset_subset = str(case["dataset_subset"])
    forecast_summary = case["forecast_summary"]
    sensors = [str(s) for s in forecast_summary.get("dominant_top_sensors", [])]

    dataset_rules = get_dataset_rules(kg_store, dataset_subset, mixed_fleet=mixed_fleet)
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
    component_gate = {
        "decision_authority": "llm_evidence_only",
        "component_gate_applied": False,
    }
    context = {
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
        "prior_monitoring_count": int(prior_monitoring_count or 0),
    }
    if llm_policy:
        context["llm_policy"] = llm_policy
    return context


def prepare_no_kg_context(
    case: dict[str, Any],
    risk_threshold_overrides: dict[str, Any] | None = None,
    llm_policy: dict[str, Any] | None = None,
    mixed_fleet: bool = False,
    component_consensus: dict[str, Any] | None = None,
    prior_monitoring_count: int = 0,
) -> dict[str, Any]:
    case = normalize_case(case)
    risk_gate = build_risk_gate(case, reflection_rules=[], threshold_overrides=risk_threshold_overrides)
    context = {
        "case": case,
        "dataset_rules": {
            "dataset_subset": case.get("dataset_subset"),
            "allowed_hypotheses": ["unspecified_component_degradation"],
            "allowed_actions": [
                "continue_normal_operation",
                "schedule_monitoring",
                "schedule_maintenance",
            ],
            "disallowed_actions": [
                "schedule_fan_maintenance",
                "schedule_HPC_maintenance",
            ],
            "mixed_fleet": mixed_fleet,
        },
        "sensor_paths": [],
        "action_paths": [],
        "candidate_action": None,
        "reflection_rules": [],
        "component_evidence_statistics": {},
        "risk_gate": risk_gate,
        "component_gate": {},
        "reflection_memory_usage": "training_only_non_component",
        "risk_threshold_overrides": risk_threshold_overrides or {},
        "prior_monitoring_count": int(prior_monitoring_count or 0),
    }
    if llm_policy:
        context["llm_policy"] = llm_policy
    return context


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
    llm_policy: dict[str, Any] | None = None,
    prompt_variant: str = "kg",
    mixed_fleet: bool = False,
    component_consensus: dict[str, Any] | None = None,
    prior_monitoring_count: int = 0,
    decision_mode: str = "forecast_window",
) -> dict[str, Any]:
    if prompt_variant not in {"kg", "no_kg_evidence"}:
        raise ValueError(f"Unknown prompt_variant: {prompt_variant}")
    if decision_mode not in {"forecast_window", "current_lhi_only"}:
        raise ValueError(f"Unknown decision_mode: {decision_mode}")
    if prompt_variant == "no_kg_evidence":
        context = prepare_no_kg_context(
            case,
            risk_threshold_overrides=risk_threshold_overrides,
            llm_policy=llm_policy,
            mixed_fleet=mixed_fleet,
            component_consensus=component_consensus,
            prior_monitoring_count=prior_monitoring_count,
        )
    else:
        context = prepare_context(
            case,
            kg_dir,
            kg_store,
            risk_threshold_overrides=risk_threshold_overrides,
            llm_policy=llm_policy,
            mixed_fleet=mixed_fleet,
            component_consensus=component_consensus,
            prior_monitoring_count=prior_monitoring_count,
        )
    risk_policy_mode = str(risk_policy_mode or "hybrid")
    controller_raw_outputs = []
    risk_tool = None
    if risk_policy_mode == "hybrid":
        risk_tool = LightGBMRiskTool(
            model_path=risk_model_path,
            theta_low=risk_theta_low,
            theta_conf=risk_theta_conf,
        )
    if risk_policy_mode == "hybrid" and risk_tool is not None and not risk_tool.has_model:
        llm_policy_tool = LLMPolicyRiskTool(
            policy_path=llm_policy_tool_path,
            theta_conf=risk_theta_conf,
        )
        llm_policy_tool.ensure(context.get("llm_policy"))
        context["lightgbm_risk"] = llm_policy_tool.predict(context["case"], context)
        context["llm_risk_policy"] = context["lightgbm_risk"].get("llm_policy_tool", {})
    elif risk_policy_mode == "llm_only":
        llm_policy_tool = LLMPolicyRiskTool(
            policy_path=llm_policy_tool_path,
            theta_conf=risk_theta_conf,
        )
        llm_policy_tool.ensure(context.get("llm_policy"))
        context["lightgbm_risk"] = llm_policy_tool.predict(context["case"], context)
        context["llm_risk_policy"] = context["lightgbm_risk"].get("llm_policy_tool", {})
    else:
        assert risk_tool is not None
        context["lightgbm_risk"] = risk_tool.predict(context["case"], context)
    llm_calls = len(controller_raw_outputs)
    llm_fallback_used = False
    llm_errors: list[str] = []
    if dry_run:
        action = _fallback_action(context, prompt_variant, decision_mode)
        raw_outputs: list[dict[str, Any] | str] = [
            *controller_raw_outputs,
            f"<dry_run {prompt_variant} fallback_action>",
        ]
    elif not _should_activate_llm(context):
        action = low_risk_action(context, prompt_variant=prompt_variant, decision_mode=decision_mode)
        raw_outputs = [*controller_raw_outputs, "<lightgbm_low_risk_monitoring>"]
    else:
        if decision_mode == "current_lhi_only" and prompt_variant == "kg":
            prompt = build_current_lhi_prompt(
                case=context["case"],
                dataset_rules=context["dataset_rules"],
                sensor_paths=context["sensor_paths"],
                action_paths=context["action_paths"],
                component_evidence_statistics=context["component_evidence_statistics"],
                risk_gate=context["risk_gate"],
                lightgbm_risk=context["lightgbm_risk"],
                llm_policy=context.get("llm_policy"),
            )
            system_prompt = CURRENT_LHI_SYSTEM_PROMPT
        elif prompt_variant == "no_kg_evidence":
            prompt = build_no_kg_prompt(
                case=context["case"],
                risk_gate=context["risk_gate"],
                lightgbm_risk=context["lightgbm_risk"],
                llm_policy=context.get("llm_policy"),
            )
            system_prompt = NO_KG_SYSTEM_PROMPT
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
                llm_policy=context.get("llm_policy"),
            )
            system_prompt = SYSTEM_PROMPT
        raw_outputs = list(controller_raw_outputs)
        action = None
        for attempt in _llm_attempts(
            prompt,
            context,
            num_predict=num_predict,
            format_json=format_json,
            system_prompt=system_prompt,
            prompt_variant=prompt_variant,
            decision_mode=decision_mode,
        ):
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
            action = _fallback_action(context, prompt_variant, decision_mode)
            action["reason"] = (
                f"LLM did not return parseable JSON after retry; used {prompt_variant} fallback. "
                + action.get("reason", "")
            )

    validation = _validate_action_for_variant(action, context, prompt_variant, decision_mode)

    if not validation["valid"] and not dry_run and not llm_fallback_used:
        local_action = _local_validation_repair(action, validation, context, decision_mode)
        if local_action is not None:
            action = local_action
            action["local_validation_repair_used"] = True
            validation = _validate_action_for_variant(action, context, prompt_variant, decision_mode)

    if not validation["valid"] and not dry_run and not llm_fallback_used:
        evidence_instruction = (
            "Revise the action using only the same forecast, raw sensor-error, and risk-tool evidence."
            if prompt_variant == "no_kg_evidence"
            else "Revise the action using the same forecast case, dataset rules, and graph evidence."
        )
        if decision_mode == "current_lhi_only":
            evidence_instruction = "Revise the action using only the current LHI, current sensor contribution ranking, and supplied KG evidence. Do not introduce future values."
        timing_instruction = (
            "- For schedule_monitoring, action_time may be any t+1 through t+20.\n"
            "- For maintenance, action_time must be t+1."
            if decision_mode == "current_lhi_only"
            else (
                "- schedule_monitoring must use the adaptive recommended_monitoring_time.\n"
                f"- maintenance must use {recommended_maintenance_time(context['case'], context.get('llm_policy'))}."
            )
        )
        repair_prompt = f"""The previous action is invalid.

Previous action:
{json.dumps(action, indent=2)}

Validation errors:
{json.dumps(validation, indent=2)}

{evidence_instruction}
Return exactly one JSON object.
Mandatory output rules:
- continue_normal_operation must use "action_time": null.
{timing_instruction}
- Choose the maintenance component only from the supplied KG evidence paths.
- Do not use a numeric component ranking, component gate, FD identity, or fallback
  component rule. If the evidence is ambiguous, choose schedule_monitoring.
Return only valid JSON.
"""
        try:
            raw2 = ollama_chat(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt + "\n\n" + repair_prompt}],
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt + "\n\n" + repair_prompt},
                ],
                "raw_output": raw2,
            }
            try:
                repaired_action = extract_json(raw2)
                repair_entry["parse_ok"] = True
                action = repaired_action
                validation = _validate_action_for_variant(action, context, prompt_variant, decision_mode)
                if not validation["valid"]:
                    local_action = _local_validation_repair(action, validation, context, decision_mode)
                    if local_action is not None:
                        action = local_action
                        action["local_validation_repair_used"] = True
                        validation = _validate_action_for_variant(action, context, prompt_variant, decision_mode)
                    if not validation["valid"]:
                        repair_entry["post_repair_validation"] = validation
                        llm_errors.append(f"repair_invalid: {validation['violations']}")
                        llm_fallback_used = True
                        action = _fallback_action(context, prompt_variant, decision_mode)
                        action["reason"] = f"Repair output was still invalid; used {prompt_variant} fallback."
                        validation = _validate_action_for_variant(action, context, prompt_variant, decision_mode)
            except Exception as exc:
                repair_entry["parse_ok"] = False
                repair_entry["error"] = str(exc)
                llm_errors.append(f"repair: {exc}")
                llm_fallback_used = True
                action = _fallback_action(context, prompt_variant, decision_mode)
                action["reason"] = f"Repair output was not parseable; used {prompt_variant} fallback."
                validation = _validate_action_for_variant(action, context, prompt_variant, decision_mode)
            raw_outputs.append(repair_entry)
        except Exception as exc:
            llm_errors.append(f"repair: {exc}")
            raw_outputs.append({"stage": "repair", "raw_output": "", "parse_ok": False, "error": str(exc)})
            llm_fallback_used = True
            action = _fallback_action(context, prompt_variant, decision_mode)
            action["reason"] = f"Repair call failed; used {prompt_variant} fallback."
            validation = _validate_action_for_variant(action, context, prompt_variant, decision_mode)

    if not validation["valid"]:
        llm_errors.append(f"final_invalid: {validation['violations']}")
        llm_fallback_used = True
        action = _fallback_action(context, prompt_variant, decision_mode)
        action["reason"] = f"Final action was invalid; used {prompt_variant} fallback."
        validation = _validate_action_for_variant(action, context, prompt_variant, decision_mode)

    action["validation"] = validation
    action["validation_status"] = "valid" if validation["valid"] else "invalid"
    if llm_fallback_used:
        action["llm_fallback_used"] = True
    if llm_errors:
        action["llm_errors"] = llm_errors
    return {
        "case_id": context["case"].get("case_id"),
        "action": action,
        "prompt_variant": prompt_variant,
        "decision_mode": decision_mode,
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
            "llm_policy": context.get("llm_policy"),
            "llm_risk_policy": context.get("llm_risk_policy"),
            "maintenance_timing_profile": maintenance_timing_profile(
                context["case"], context.get("llm_policy")
            ),
            "lightgbm_risk": context["lightgbm_risk"],
            "prior_monitoring_count": context.get("prior_monitoring_count", 0),
            "decision_mode": decision_mode,
        },
        "llm_calls": llm_calls,
        "raw_outputs": raw_outputs,
    }


def _llm_attempts(
    prompt: str,
    context: dict[str, Any],
    num_predict: int,
    format_json: bool,
    system_prompt: str = SYSTEM_PROMPT,
    prompt_variant: str = "kg",
    decision_mode: str = "forecast_window",
) -> list[dict[str, Any]]:
    if decision_mode == "current_lhi_only" and prompt_variant == "kg":
        retry_prompt = build_current_lhi_prompt(
            case=context["case"],
            dataset_rules=context["dataset_rules"],
            sensor_paths=context["sensor_paths"],
            action_paths=context["action_paths"],
            component_evidence_statistics=context["component_evidence_statistics"],
            risk_gate=context["risk_gate"],
            lightgbm_risk=context.get("lightgbm_risk"),
            llm_policy=context.get("llm_policy"),
        )
    else:
        retry_prompt = (
            build_no_kg_prompt(
            case=context["case"],
            risk_gate=context["risk_gate"],
            lightgbm_risk=context.get("lightgbm_risk"),
            llm_policy=context.get("llm_policy"),
            )
            if prompt_variant == "no_kg_evidence"
            else _minimal_json_prompt(context)
        )
    return [
        {
            "stage": "initial",
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "format_json": format_json,
            "num_predict": num_predict,
        },
        {
            "stage": "minimal_retry",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_prompt},
            ],
            "format_json": True,
            "num_predict": 384,
        },
        {
            "stage": "plain_retry",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_prompt},
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
    score = float(risk.get("maintenance_risk_score", 0.0))
    decision = str(risk.get("risk_decision", ""))
    if risk.get("model_source") != "lightgbm_model":
        return decision in {"activate_llm_agent", "activate_llm_agent_uncertain"}
    learned_support = (
        risk.get("model_source") == "lightgbm_model"
        and score >= 0.60
        and _risk_evidence_allows_learned_support(risk_gate)
    )
    return bool(
        risk_gate.get("maintenance_candidate", False)
        or learned_support
    )


def _local_validation_repair(
    action: dict[str, Any],
    validation: dict[str, Any],
    context: dict[str, Any],
    decision_mode: str = "forecast_window",
) -> dict[str, Any] | None:
    violations = list(validation.get("violations", []))
    if action.get("action_type") in {
        "schedule_monitoring",
        "schedule_maintenance",
        "schedule_HPC_maintenance",
        "schedule_fan_maintenance",
    } and any(
        item in violations
        for item in [
            f"{action.get('action_type')} must have non-null action_time",
            "action_time is outside forecast_horizon",
            (
                "maintenance action_time must equal recommended_maintenance_time="
                f"{recommended_maintenance_time(context['case'], context.get('llm_policy'))}"
            ),
            (
                "schedule_monitoring action_time must equal recommended_monitoring_time="
                f"{recommended_monitoring_time(context['case'], context.get('llm_policy'))}"
            ),
            "current-only schedule_monitoring action_time must be in t+1..t+20",
            "current-only maintenance action_time must equal t+1",
        ]
    ):
        repaired = dict(action)
        if decision_mode == "current_lhi_only":
            repaired["action_time"] = (
                "t+20" if action.get("action_type") == "schedule_monitoring" else "t+1"
            )
        elif action.get("action_type") == "schedule_monitoring":
            repaired["action_time"] = recommended_monitoring_time(
                context["case"], context.get("llm_policy")
            )
        else:
            repaired["action_time"] = _maintenance_action_time(context)
        repaired["reason"] = "Repaired action_time format while preserving LLM action choice."
        return repaired

    if action.get("action_type") == "schedule_monitoring" and violations == [
        "schedule_monitoring must have non-null action_time"
    ]:
        repaired = dict(action)
        repaired["action_time"] = rule_based_action(context)["action_time"]
        repaired["reason"] = "Filled missing monitoring time from KG rule fallback."
        return repaired

    return None


def _validate_action_for_variant(
    action: dict[str, Any],
    context: dict[str, Any],
    prompt_variant: str,
    decision_mode: str = "forecast_window",
) -> dict[str, Any]:
    return validate_action(
        action,
        context["case"],
        context["dataset_rules"],
        context["sensor_paths"],
        risk_gate=context["risk_gate"],
        lightgbm_risk=context["lightgbm_risk"],
        component_gate=context.get("component_gate"),
        llm_policy=context.get("llm_policy"),
        current_lhi_only=decision_mode == "current_lhi_only",
    )


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
        "maintenance_timing_policy": maintenance_timing_profile(
            case, context.get("llm_policy")
        ),
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
        "Use action_time=null only for continue_normal_operation. "
        "For schedule_monitoring, action_time must equal "
        f"{recommended_monitoring_time(case, context.get('llm_policy'))}. "
        "For maintenance, action_time must equal "
        f"{recommended_maintenance_time(case, context.get('llm_policy'))}. "
        "Use evidence_paths as short IDs only, e.g. [\"E1\", \"E2\"]. "
        "Keep reason under 30 words.\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nRequired keys: action_type, action_time, risk_hypothesis, degradation_hypothesis, "
        "confidence, evidence_paths, reason, validation_status."
    )


def rule_based_action(context: dict[str, Any], decision_mode: str = "forecast_window") -> dict[str, Any]:
    case = context["case"]
    summary = case["forecast_summary"]
    dataset_rules = context["dataset_rules"]
    sensor_paths = context["sensor_paths"]
    path_hypotheses = {p["hypothesis"] for p in sensor_paths}
    disallowed = set(dataset_rules.get("disallowed_actions", []))
    risk_gate = context.get("risk_gate", {})
    learned_risk = context.get("lightgbm_risk", {})

    if summary.get("first_critical_crossing_cycle"):
        risk = "critical_risk_hypothesis"
    elif summary.get("first_warning_crossing_cycle") or summary.get("score_trend") == "increasing":
        risk = "warning_risk_hypothesis"
    else:
        risk = "low_risk_hypothesis"

    action_type = "continue_normal_operation"
    degradation = "low_risk_hypothesis"
    learned_support = _learned_risk_supports_maintenance(learned_risk) and _risk_evidence_allows_learned_support(risk_gate)
    if not (learned_support or risk_gate.get("maintenance_candidate", False)):
        action_type = "schedule_monitoring"
        degradation = "uncertain_component_degradation"
    else:
        # This is only a safe fallback when Ollama is unavailable. It must not
        # invent a component decision from path metadata; the normal path is
        # the LLM choosing from the evidence shown in the prompt.
        action_type = "schedule_monitoring"
        degradation = "uncertain_component_degradation"

    if action_type == "continue_normal_operation":
        action_time = None
    elif action_type == "schedule_monitoring":
        action_time = "t+20" if decision_mode == "current_lhi_only" else recommended_monitoring_time(case, context.get("llm_policy"))
    else:
        action_time = "t+1" if decision_mode == "current_lhi_only" else _maintenance_action_time(context)

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


def no_kg_fallback_action(context: dict[str, Any], decision_mode: str = "forecast_window") -> dict[str, Any]:
    case = context["case"]
    risk = context.get("lightgbm_risk", {})
    return {
        "action_type": "schedule_monitoring",
        "action_time": "t+20" if decision_mode == "current_lhi_only" else recommended_monitoring_time(case, context.get("llm_policy")),
        "risk_hypothesis": risk.get("predicted_risk_stage", "uncertain_risk"),
        "degradation_hypothesis": "unspecified_component_degradation",
        "confidence": float(risk.get("confidence", 0.5)),
        "evidence_paths": [],
        "reason": "No component evidence is available; scheduled conservative monitoring.",
        "validation_status": "pending",
    }


def _fallback_action(
    context: dict[str, Any],
    prompt_variant: str,
    decision_mode: str = "forecast_window",
) -> dict[str, Any]:
    if prompt_variant == "no_kg_evidence":
        return no_kg_fallback_action(context, decision_mode)
    return rule_based_action(context, decision_mode)


def low_risk_action(
    context: dict[str, Any],
    prompt_variant: str = "kg",
    decision_mode: str = "forecast_window",
) -> dict[str, Any]:
    if prompt_variant == "no_kg_evidence":
        action = no_kg_fallback_action(context, decision_mode)
        action["reason"] = (
            "Active risk tool judged this case below the maintenance-reasoning activation threshold."
        )
        return action
    action = rule_based_action(context, decision_mode)
    if action["action_type"] in {"schedule_HPC_maintenance", "schedule_fan_maintenance"}:
        action["action_type"] = "schedule_monitoring"
        action["degradation_hypothesis"] = "uncertain_component_degradation"
        action["action_time"] = (
            "t+20" if decision_mode == "current_lhi_only" else recommended_monitoring_time(
                context["case"], context.get("llm_policy")
            )
        )
        action["risk_tool_low_risk_gate_used"] = True
    action["reason"] = (
        "Active risk tool judged this case below LLM activation threshold; scheduled conservative monitoring."
    )
    return action


def _confidence(sensor_paths: list[dict[str, Any]], hypothesis: str) -> float:
    evidence_count = sum(1 for p in sensor_paths if p.get("hypothesis") == hypothesis)
    if not evidence_count:
        return 0.5
    return round(min(0.5 + 0.1 * min(evidence_count, 5), 1.0), 3)


def _maintenance_action_time(context: dict[str, Any]) -> str:
    return recommended_maintenance_time(context["case"], context.get("llm_policy"))


def _learned_risk_supports_maintenance(lightgbm_risk: dict[str, Any]) -> bool:
    score = _to_float(lightgbm_risk.get("maintenance_risk_score")) or 0.0
    theta = _to_float(lightgbm_risk.get("theta_low")) or 0.4
    decision = str(lightgbm_risk.get("risk_decision", ""))
    stage = str(lightgbm_risk.get("predicted_risk_stage", ""))
    return (
        score >= theta
        and decision in {"activate_llm_agent", "activate_llm_agent_uncertain"}
        and stage in {"maintenance_window", "late_or_missed"}
    )


def _risk_evidence_allows_learned_support(risk_gate: dict[str, Any]) -> bool:
    peak = _to_float(risk_gate.get("peak_score")) or 0.0
    return bool(
        risk_gate.get("maintenance_candidate", False)
        or risk_gate.get("statistical_candidate", False)
        or peak >= 0.50
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

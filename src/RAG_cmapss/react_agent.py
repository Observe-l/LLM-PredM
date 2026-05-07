from __future__ import annotations

import json
from typing import Any

from .action_validator import validate_action
from .action_validator import STRONG_FAN_SENSORS, STRONG_HPC_SENSORS
from .graph_retriever import (
    build_component_evidence_statistics,
    get_dataset_rules,
    infer_candidate_action,
    retrieve_action_paths,
    retrieve_sensor_paths,
)
from .evidence_gates import build_component_gate, build_reflection_gate, build_risk_gate
from .kg_store import KGStore
from .ollama_client import extract_json, ollama_chat
from .prompt_builder import SYSTEM_PROMPT, build_prompt
from .reflection_memory import retrieve_reflection_rules


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


def prepare_context(case: dict[str, Any], kg_dir: str, kg_store: KGStore) -> dict[str, Any]:
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
    reflection_rules = retrieve_multi_action_reflections(
        kg_dir=kg_dir,
        dataset_subset=dataset_subset,
        case=case,
        component_evidence_statistics=component_evidence_statistics,
        forecast_summary=forecast_summary,
        sensor_paths=sensor_paths,
        dataset_rules=dataset_rules,
        candidate_action=candidate_action,
    )
    risk_gate = build_risk_gate(case, reflection_rules=reflection_rules)
    component_gate = build_component_gate(component_evidence_statistics, dataset_rules)
    reflection_gate = build_reflection_gate(reflection_rules)
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
        "reflection_gate": reflection_gate,
    }


def retrieve_multi_action_reflections(
    kg_dir: str,
    dataset_subset: str,
    case: dict[str, Any],
    component_evidence_statistics: dict[str, Any],
    forecast_summary: dict[str, Any],
    sensor_paths: list[dict[str, Any]],
    dataset_rules: dict[str, Any],
    candidate_action: str,
    max_rules: int = 5,
) -> list[dict[str, Any]]:
    query_actions = _reflection_query_actions(candidate_action, sensor_paths, dataset_rules, forecast_summary)
    combined: dict[str, dict[str, Any]] = {}
    for query_action in query_actions:
        rows = retrieve_reflection_rules(
            kg_dir=kg_dir,
            dataset_subset=dataset_subset,
            forecast_summary=forecast_summary,
            candidate_action=query_action,
            max_rules=50,
            case=case,
            component_stats=component_evidence_statistics,
        )
        for row in rows:
            row = dict(row)
            row["reflection_query_action"] = query_action
            key = str(row.get("rule_id"))
            if key not in combined or float(row.get("retrieval_similarity", 0.0)) > float(
                combined[key].get("retrieval_similarity", 0.0)
            ):
                combined[key] = row
    return _label_balanced_reflections(list(combined.values()), max_rules=max_rules)


def _label_balanced_reflections(rows: list[dict[str, Any]], max_rules: int) -> list[dict[str, Any]]:
    preferred = ["correct_maintenance", "too_early", "missed_HPC_maintenance", "missed_fan_maintenance", "over_maintenance"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: float(item.get("retrieval_similarity", 0.0)), reverse=True):
        grouped.setdefault(str(row.get("feedback_label")), []).append(row)
    selected: list[dict[str, Any]] = []
    for label in preferred:
        selected.extend(grouped.get(label, [])[:2])
    if len(selected) < max_rules:
        selected_ids = {str(item.get("rule_id")) for item in selected}
        for row in sorted(rows, key=lambda item: float(item.get("retrieval_similarity", 0.0)), reverse=True):
            if str(row.get("rule_id")) not in selected_ids:
                selected.append(row)
            if len(selected) >= max_rules:
                break
    return selected[:max_rules]


def _reflection_query_actions(
    candidate_action: str,
    sensor_paths: list[dict[str, Any]],
    dataset_rules: dict[str, Any],
    forecast_summary: dict[str, Any],
) -> list[str]:
    actions = [candidate_action]
    allowed = set(dataset_rules.get("allowed_actions", []))
    if "schedule_HPC_maintenance" in allowed and _has_strong_evidence(
        sensor_paths, "HPC_related_degradation", STRONG_HPC_SENSORS
    ):
        actions.append("schedule_HPC_maintenance")
    if "schedule_fan_maintenance" in allowed and _has_strong_evidence(
        sensor_paths, "Fan_related_degradation", STRONG_FAN_SENSORS
    ):
        actions.append("schedule_fan_maintenance")
    return list(dict.fromkeys(actions))


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
) -> dict[str, Any]:
    context = prepare_context(case, kg_dir, kg_store)
    prompt = build_prompt(
        case=context["case"],
        dataset_rules=context["dataset_rules"],
        sensor_paths=context["sensor_paths"],
        action_paths=context["action_paths"],
        reflection_rules=context["reflection_rules"],
        component_evidence_statistics=context["component_evidence_statistics"],
        risk_gate=context["risk_gate"],
        component_gate=context["component_gate"],
        reflection_gate=context["reflection_gate"],
    )

    llm_calls = 0
    llm_fallback_used = False
    llm_errors: list[str] = []
    if dry_run:
        action = rule_based_action(context)
        raw_outputs: list[dict[str, Any] | str] = ["<dry_run rule_based_action>"]
    else:
        raw_outputs = []
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
            "reflection_gate": context["reflection_gate"],
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
        "reflection_gate": context.get("reflection_gate"),
        "dataset_rules": context["dataset_rules"],
        "top_evidence_paths": [p["path_text"] for p in context["sensor_paths"][:5]],
        "action_paths": [p["path_text"] for p in context["action_paths"][:4]],
        "reflection_rules": context["reflection_rules"][:3],
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

    if not risk_gate.get("maintenance_candidate", False):
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
        action_time = (
            summary.get("first_warning_crossing_cycle")
            or summary.get("first_persistent_pattern_cycle")
            or summary.get("peak_score_cycle")
            or "t+1"
        )
    else:
        action_time = (
            summary.get("peak_score_cycle")
            or summary.get("first_critical_crossing_cycle")
            or summary.get("first_persistent_pattern_cycle")
            or "t+1"
        )

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


def _confidence(sensor_paths: list[dict[str, Any]], hypothesis: str) -> float:
    scores = [float(p.get("score", 0.0)) for p in sensor_paths if p.get("hypothesis") == hypothesis]
    if not scores:
        return 0.5
    return round(min(max(sum(scores[:3]) / min(len(scores), 3), 0.0), 1.0), 3)


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

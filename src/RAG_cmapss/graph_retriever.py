from __future__ import annotations

from typing import Any

from .kg_store import KGStore


COMPONENT_HYPOTHESES = {"HPC_related_degradation", "Fan_related_degradation"}


def get_dataset_rules(
    kg: KGStore,
    dataset_subset: str,
    *,
    mixed_fleet: bool = False,
) -> dict[str, Any]:
    allowed_hypotheses = [e["tail"] for e in kg.outgoing(dataset_subset, "allows_hypothesis")]
    threshold_hypotheses = [e["tail"] for e in kg.outgoing(dataset_subset, "has_threshold_hypothesis")]
    policies = [e["tail"] for e in kg.outgoing(dataset_subset, "has_action_policy")]

    allowed_actions: list[str] = []
    disallowed_actions: list[str] = []
    for policy in policies:
        allowed_actions.extend(e["tail"] for e in kg.outgoing(policy, "allows_action_type"))
        disallowed_actions.extend(e["tail"] for e in kg.outgoing(policy, "disallows_action_type"))

    rules = {
        "dataset_subset": dataset_subset,
        "allowed_hypotheses": sorted(set(allowed_hypotheses)),
        "threshold_hypotheses": sorted(set(threshold_hypotheses)),
        "allowed_actions": sorted(set(allowed_actions) - set(disallowed_actions)),
        "disallowed_actions": sorted(set(disallowed_actions)),
        "policies": sorted(set(policies)),
    }
    if mixed_fleet:
        # In a mixed fleet, FD-level action policies must not suppress a
        # component before the per-engine evidence paths are inspected.  The
        # known FD component composition remains an evaluation constraint
        # (see joint_simulation.py), not an action-selection shortcut.
        rules.update(
            {
                "mixed_fleet": True,
                "action_constraint_scope": "per_engine_component_evidence",
                "allowed_hypotheses": sorted(
                    {
                        "HPC_related_degradation",
                        "Fan_related_degradation",
                        "uncertain_component_degradation",
                    }
                ),
                "allowed_actions": [
                    "continue_normal_operation",
                    "schedule_monitoring",
                    "schedule_HPC_maintenance",
                    "schedule_fan_maintenance",
                ],
                "disallowed_actions": [],
            }
        )
    return rules


def retrieve_sensor_paths(
    kg: KGStore,
    sensor_ids: list[str],
    forecast_summary: dict[str, Any] | None = None,
    dataset_rules: dict[str, Any] | None = None,
    max_paths: int = 24,
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    allowed_hypotheses = set((dataset_rules or {}).get("allowed_hypotheses", []))

    for sensor in sensor_ids:
        for e1 in kg.outgoing(sensor, "measures"):
            quantity = e1["tail"]
            base_path = [sensor, "measures", quantity]

            for e2 in kg.outgoing(quantity, "associated_with"):
                component = e2["tail"]
                for e3 in kg.outgoing(component, "supports_hypothesis"):
                    paths.append(
                        _make_sensor_path(
                            sensor=sensor,
                            hypothesis=e3["tail"],
                            component=component,
                            edges=[e1, e2, e3],
                            path=base_path + ["associated_with", component, "supports_hypothesis", e3["tail"]],
                            allowed_hypotheses=allowed_hypotheses,
                        )
                    )

            for e2 in kg.outgoing(quantity, "alias_of"):
                alias = e2["tail"]
                alias_path = base_path + ["alias_of", alias]
                for e3 in kg.outgoing(alias, "supports_working_characteristic"):
                    wc = e3["tail"]
                    for e4 in kg.outgoing(wc, "characterizes"):
                        component = e4["tail"]
                        for e5 in kg.outgoing(component, "supports_hypothesis"):
                            paths.append(
                                _make_sensor_path(
                                    sensor=sensor,
                                    hypothesis=e5["tail"],
                                    component=component,
                                    edges=[e1, e2, e3, e4, e5],
                                    path=alias_path
                                    + [
                                        "supports_working_characteristic",
                                        wc,
                                        "characterizes",
                                        component,
                                        "supports_hypothesis",
                                        e5["tail"],
                                    ],
                                    allowed_hypotheses=allowed_hypotheses,
                                )
                            )

            for e2 in kg.outgoing(sensor, "supports_hypothesis"):
                paths.append(
                    _make_sensor_path(
                        sensor=sensor,
                        hypothesis=e2["tail"],
                        component=None,
                        edges=[e2],
                        path=[sensor, "supports_hypothesis", e2["tail"]],
                        allowed_hypotheses=allowed_hypotheses,
                    )
                )

    unique: dict[str, dict[str, Any]] = {}
    for path in paths:
        key = path["path_text"]
        if key not in unique:
            unique[key] = path
    # Preserve the forecast sensor order and KG traversal order.  Evidence is
    # presented to the LLM as paths; no numeric component ranking is used to
    # select a component.
    return list(unique.values())[:max_paths]


def retrieve_action_paths(kg: KGStore, hypotheses: list[str]) -> list[dict[str, Any]]:
    action_paths: list[dict[str, Any]] = []
    for hyp in sorted(set(hypotheses)):
        for e in kg.outgoing(hyp, "suggests_action_type"):
            path = [hyp, "suggests_action_type", e["tail"]]
            action_paths.append(
                {
                    "hypothesis": hyp,
                    "action_type": e["tail"],
                    "path": path,
                    "path_text": " -> ".join(path),
                    "weight": e["weight"],
                }
            )
    return sorted(action_paths, key=lambda x: x["weight"], reverse=True)


def build_component_evidence_statistics(
    sensor_paths: list[dict[str, Any]],
    sensor_evidence_statistics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sensor_presence = (sensor_evidence_statistics or {}).get("sensor_presence_ratio", {})
    grouped: dict[str, dict[str, Any]] = {}
    for path in sensor_paths:
        hypothesis = str(path.get("hypothesis"))
        if hypothesis not in {
            "HPC_related_degradation",
            "Fan_related_degradation",
            "uncertain_component_degradation",
        }:
            continue
        item = grouped.setdefault(hypothesis, {"paths": [], "sensors": set()})
        item["paths"].append(str(path.get("path_text", "")))
        sensor = str(path.get("sensor", ""))
        if sensor:
            item["sensors"].add(sensor)

    result: dict[str, Any] = {}
    for hypothesis, item in grouped.items():
        result[hypothesis] = {
            "evidence_count": len(item["paths"]),
            "supporting_sensors": sorted(item["sensors"]),
            "evidence_paths": item["paths"][:12],
        }
    result["evidence_only"] = True
    result["component_hypotheses_present"] = sorted(
        hypothesis for hypothesis in result if hypothesis.endswith("_degradation")
    )
    return result


def infer_candidate_action(
    dataset_rules: dict[str, Any],
    sensor_paths: list[dict[str, Any]],
    forecast_summary: dict[str, Any],
) -> str:
    # Candidate actions are deliberately not inferred from path presence.
    # The LLM must compare the supplied evidence paths and choose the action.
    return "llm_evidence_decision"


def _make_sensor_path(
    sensor: str,
    hypothesis: str,
    component: str | None,
    edges: list[dict[str, Any]],
    path: list[str],
    allowed_hypotheses: set[str],
) -> dict[str, Any]:
    return {
        "sensor": sensor,
        "hypothesis": hypothesis,
        "component": component,
        "path": path,
        "path_text": " -> ".join(path),
    }

from __future__ import annotations

from collections import Counter
from typing import Any

from .kg_store import KGStore


COMPONENT_HYPOTHESES = {"HPC_related_degradation", "Fan_related_degradation"}


def get_dataset_rules(kg: KGStore, dataset_subset: str) -> dict[str, Any]:
    allowed_hypotheses = [e["tail"] for e in kg.outgoing(dataset_subset, "allows_hypothesis")]
    threshold_hypotheses = [e["tail"] for e in kg.outgoing(dataset_subset, "has_threshold_hypothesis")]
    policies = [e["tail"] for e in kg.outgoing(dataset_subset, "has_action_policy")]

    allowed_actions: list[str] = []
    disallowed_actions: list[str] = []
    for policy in policies:
        allowed_actions.extend(e["tail"] for e in kg.outgoing(policy, "allows_action_type"))
        disallowed_actions.extend(e["tail"] for e in kg.outgoing(policy, "disallows_action_type"))

    return {
        "dataset_subset": dataset_subset,
        "allowed_hypotheses": sorted(set(allowed_hypotheses)),
        "threshold_hypotheses": sorted(set(threshold_hypotheses)),
        "allowed_actions": sorted(set(allowed_actions) - set(disallowed_actions)),
        "disallowed_actions": sorted(set(disallowed_actions)),
        "policies": sorted(set(policies)),
    }


def retrieve_sensor_paths(
    kg: KGStore,
    sensor_ids: list[str],
    forecast_summary: dict[str, Any] | None = None,
    dataset_rules: dict[str, Any] | None = None,
    max_paths: int = 24,
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    contribution = _sensor_contribution_scores(sensor_ids)
    persistence_score = _persistence_score(forecast_summary or {})
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
                            contribution_score=contribution.get(sensor, 0.5),
                            persistence_score=persistence_score,
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
                                    contribution_score=contribution.get(sensor, 0.5),
                                    persistence_score=persistence_score,
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
                        contribution_score=contribution.get(sensor, 0.5),
                        persistence_score=persistence_score,
                        allowed_hypotheses=allowed_hypotheses,
                    )
                )

    unique: dict[str, dict[str, Any]] = {}
    for path in paths:
        key = path["path_text"]
        if key not in unique or path["score"] > unique[key]["score"]:
            unique[key] = path
    return sorted(unique.values(), key=lambda x: x["score"], reverse=True)[:max_paths]


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


def infer_candidate_action(
    dataset_rules: dict[str, Any],
    sensor_paths: list[dict[str, Any]],
    forecast_summary: dict[str, Any],
) -> str:
    disallowed = set(dataset_rules.get("disallowed_actions", []))
    dominant = forecast_summary.get("dominant_component_hypothesis")
    hypotheses = [p["hypothesis"] for p in sensor_paths]
    counts = Counter(h for h in hypotheses if h in COMPONENT_HYPOTHESES)

    if dominant == "HPC_related_degradation" or counts.get("HPC_related_degradation", 0) > 0:
        action = "schedule_HPC_maintenance"
    elif dominant == "Fan_related_degradation" or counts.get("Fan_related_degradation", 0) > 0:
        action = "schedule_fan_maintenance"
    else:
        action = "schedule_monitoring"
    return "schedule_monitoring" if action in disallowed else action


def _make_sensor_path(
    sensor: str,
    hypothesis: str,
    component: str | None,
    edges: list[dict[str, Any]],
    path: list[str],
    contribution_score: float,
    persistence_score: float,
    allowed_hypotheses: set[str],
) -> dict[str, Any]:
    edge_weight_mean = sum(float(e["weight"]) for e in edges) / max(len(edges), 1)
    dataset_policy_score = 1.0 if not allowed_hypotheses or hypothesis in allowed_hypotheses else -1.0
    score = (
        0.35 * contribution_score
        + 0.25 * edge_weight_mean
        + 0.20 * persistence_score
        + 0.20 * dataset_policy_score
    )
    return {
        "sensor": sensor,
        "hypothesis": hypothesis,
        "component": component,
        "path": path,
        "path_text": " -> ".join(path),
        "edge_weight_mean": round(edge_weight_mean, 4),
        "score": round(score, 4),
    }


def _sensor_contribution_scores(sensor_ids: list[str]) -> dict[str, float]:
    n = max(len(sensor_ids), 1)
    return {sensor: 1.0 - (rank / max(n, 1)) * 0.35 for rank, sensor in enumerate(sensor_ids)}


def _persistence_score(forecast_summary: dict[str, Any]) -> float:
    if forecast_summary.get("first_persistent_pattern_cycle") is not None:
        return 1.0
    if forecast_summary.get("score_trend") == "increasing":
        return 0.5
    return 0.2


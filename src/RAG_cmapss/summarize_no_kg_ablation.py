from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full online KG baseline and no-KG maintenance-timing experiments."
    )
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--no_kg_root", type=Path, required=True)
    parser.add_argument("--fds", nargs="+", default=["FD001", "FD002"])
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    return parser.parse_args()


def label_class(label: str) -> str:
    if label == "correct_maintenance":
        return "correct"
    if label in {"too_early", "over_maintenance"}:
        return "early"
    if label.startswith("missed_"):
        return "missed"
    return "other"


def summarize_arm(fd_dir: Path) -> dict[str, Any]:
    feedback = _read_json(fd_dir / "feedback_logs.json")
    engines = _read_json(fd_dir / "engine_summary.json")
    actions = _read_json(fd_dir / "action_hypotheses.json")
    run_summary = _read_json(fd_dir / "joint_simulation_summary.json")
    labels = Counter(str(item.get("feedback_label")) for item in feedback)
    classes = Counter(label_class(label) for label in labels.elements())
    n = len(engines)
    return {
        "engines": n,
        "feedback_records": len(feedback),
        "decision_points": len(actions),
        "label_counts": dict(sorted(labels.items())),
        "class_counts": {key: classes.get(key, 0) for key in ["correct", "early", "missed", "other"]},
        "class_rates": {
            key: round(classes.get(key, 0) / n, 6) if n else None
            for key in ["correct", "early", "missed", "other"]
        },
        "action_counts": run_summary.get("action_counts", {}),
        "terminal_counts": run_summary.get("terminal_counts", {}),
        "llm_fallback_count": sum(
            bool(item.get("action", {}).get("llm_fallback_used")) for item in actions
        ),
        "invalid_action_count": sum(
            item.get("action", {}).get("validation_status") != "valid" for item in actions
        ),
    }


def compare_fd(baseline_dir: Path, no_kg_dir: Path) -> dict[str, Any]:
    baseline = summarize_arm(baseline_dir)
    no_kg = summarize_arm(no_kg_dir)
    deltas = {
        key: round(
            100 * (float(no_kg["class_rates"][key]) - float(baseline["class_rates"][key])),
            3,
        )
        for key in ["correct", "early", "missed"]
    }
    return {
        "baseline_kg": baseline,
        "no_kg": no_kg,
        "no_kg_minus_kg_percentage_points": deltas,
        "qa": {
            "same_engine_count": baseline["engines"] == no_kg["engines"],
            "no_kg_one_feedback_per_engine": no_kg["engines"] == no_kg["feedback_records"],
            "no_kg_context_has_no_graph_evidence": _contexts_are_clean(no_kg_dir),
            "no_kg_reflection_component_fields_are_empty": _reflection_is_clean(no_kg_dir),
        },
    }


def main() -> None:
    args = parse_args()
    results = {
        fd: compare_fd(args.baseline_root / fd, args.no_kg_root / fd)
        for fd in args.fds
    }
    payload = {
        "metric_scope": (
            "Engine-level maintenance timing. no-KG uses schedule_maintenance without component identity."
        ),
        "maintenance_rule": "correct iff normalized remaining RUL < 0.25; otherwise early",
        "by_fd": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    with args.output_csv.open("w", newline="") as handle:
        fields = [
            "fd",
            "arm",
            "engines",
            "correct_count",
            "correct_rate",
            "early_count",
            "early_rate",
            "missed_count",
            "missed_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fd, comparison in results.items():
            for arm in ["baseline_kg", "no_kg"]:
                item = comparison[arm]
                writer.writerow(
                    {
                        "fd": fd,
                        "arm": arm,
                        "engines": item["engines"],
                        "correct_count": item["class_counts"]["correct"],
                        "correct_rate": item["class_rates"]["correct"],
                        "early_count": item["class_counts"]["early"],
                        "early_rate": item["class_rates"]["early"],
                        "missed_count": item["class_counts"]["missed"],
                        "missed_rate": item["class_rates"]["missed"],
                    }
                )
    print(json.dumps(payload, indent=2))


def _contexts_are_clean(fd_dir: Path) -> bool:
    actions = _read_json(fd_dir / "action_hypotheses.json")
    return all(
        not item.get("context", {}).get("sensor_paths")
        and not item.get("context", {}).get("action_paths")
        and not item.get("context", {}).get("component_evidence_statistics")
        and not item.get("context", {}).get("component_gate")
        for item in actions
    )


def _reflection_is_clean(fd_dir: Path) -> bool:
    path = fd_dir / "kg_memory" / "reflection_rules.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    component_fields = [
        "hpc_sensor_presence_ratio",
        "fan_sensor_presence_ratio",
        "conflict_sensor_presence_ratio",
    ]
    return all(
        row.get("component_evidence_strength") == "not_available"
        and all(not row.get(field) for field in component_fields)
        for row in rows
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()

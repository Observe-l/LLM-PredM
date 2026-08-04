from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .joint_simulation import MAINTENANCE_ACTIONS
from .timing_policy import recommended_maintenance_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit strict feedback and peak-grounded timing.")
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path)
    parser.add_argument("--fds", nargs="+", default=["FD002", "FD004"])
    return parser.parse_args()


def audit_run(run_dir: Path, fd: str) -> dict[str, Any]:
    feedback = _read_json(run_dir / "feedback_logs.json")
    actions = _read_json(run_dir / "action_hypotheses.json")
    cases = {
        str(item.get("case_id")): item
        for item in _read_json(run_dir / "forecast_cases.json")
    }
    engines = _read_json(run_dir / "engine_summary.json")
    updates = _read_json(run_dir / "llm_policy_update_logs.json")
    reflection_rows = _read_csv(run_dir / "kg_memory" / "reflection_rules.csv")
    policy = _read_json_object(run_dir / "models" / "llm_policy_tool.json")

    labels = Counter(str(row.get("feedback_label", "unknown")) for row in feedback)
    causes = Counter(
        str(row.get("missed_maintenance_cause"))
        for row in feedback
        if row.get("missed_maintenance_cause")
    )
    strict_recomputed = Counter(_strict_label(row) for row in feedback)
    inferred_causes = Counter(_inferred_missed_cause(row) for row in feedback)
    inferred_causes.pop("", None)

    maintenance_records: list[dict[str, Any]] = []
    validation_failures = 0
    kg_evidence_nonempty = 0
    local_timing_repairs = 0
    for record in actions:
        action = record.get("action", {})
        if not action.get("validation", {}).get("valid", False):
            validation_failures += 1
        if record.get("context", {}).get("sensor_paths"):
            kg_evidence_nonempty += 1
        if action.get("local_validation_repair_used"):
            local_timing_repairs += 1
        if str(action.get("action_type")) not in MAINTENANCE_ACTIONS:
            continue
        case = cases.get(str(record.get("case_id")), {})
        recommended = recommended_maintenance_time(case)
        peak = (
            case.get("risk_statistics", {}).get("peak_score_cycle")
            or case.get("forecast_summary", {}).get("peak_score_cycle")
        )
        maintenance_records.append(
            {
                "case_id": record.get("case_id"),
                "action_time": action.get("action_time"),
                "recommended_time": recommended,
                "peak_score_cycle": peak,
            }
        )

    timing_mismatches = [
        row for row in maintenance_records if row["action_time"] != row["recommended_time"]
    ]
    t20_nonpeak = [
        row
        for row in maintenance_records
        if row["action_time"] == "t+20" and row["peak_score_cycle"] != "t+20"
    ]
    missed_reflections = [
        row for row in reflection_rows if str(row.get("feedback_label", "")).startswith("missed_")
    ]
    missed_reflections_without_cause = [
        row for row in missed_reflections if not str(row.get("missed_maintenance_cause", "")).strip()
    ]
    update_labels = Counter(str(row.get("feedback_label", "")) for row in updates)
    update_results = [
        row.get("llm_policy_update_result", {})
        for row in updates
        if isinstance(row.get("llm_policy_update_result"), dict)
    ]

    total = len(feedback)
    return {
        "fd": fd,
        "run_dir": str(run_dir),
        "engines": len(engines),
        "feedback_rows": total,
        "reflection_rows": len(reflection_rows),
        "llm_decision_points": len(actions),
        "feedback_label_counts": dict(labels),
        "feedback_label_rates": {
            label: count / total if total else 0.0 for label, count in labels.items()
        },
        "strict_recomputed_counts": dict(strict_recomputed),
        "stored_labels_match_strict_recomputation": labels == strict_recomputed,
        "missed_cause_counts": dict(causes),
        "strict_inferred_missed_cause_counts": dict(inferred_causes),
        "missed_reflection_rows": len(missed_reflections),
        "missed_reflections_without_cause": len(missed_reflections_without_cause),
        "policy_update_rows": len(updates),
        "expected_noncorrect_updates": total - labels.get("correct_maintenance", 0),
        "policy_update_label_counts": dict(update_labels),
        "action_policy_updates": dict(
            Counter(str(row.get("action_policy_update", "")) for row in update_results)
        ),
        "timing_policy_updates": dict(
            Counter(str(row.get("timing_policy_update", "")) for row in update_results)
        ),
        "maintenance_actions": len(maintenance_records),
        "maintenance_timing_matches": len(maintenance_records) - len(timing_mismatches),
        "maintenance_timing_mismatches": len(timing_mismatches),
        "maintenance_at_t20": sum(
            row["action_time"] == "t+20" for row in maintenance_records
        ),
        "risk_peak_at_t20": sum(
            row["peak_score_cycle"] == "t+20" for row in maintenance_records
        ),
        "t20_maintenance_when_peak_not_t20": len(t20_nonpeak),
        "local_validation_repairs": local_timing_repairs,
        "validation_failures": validation_failures,
        "kg_evidence_coverage_rate": (
            kg_evidence_nonempty / len(actions) if actions else 0.0
        ),
        "final_action_escalation_policy": policy.get("action_escalation_policy"),
        "final_maintenance_timing_policy": policy.get("maintenance_timing_policy"),
        "final_peak_offset_level": policy.get("peak_offset_level"),
        "policy_missed_cause_counts": policy.get("missed_cause_counts", {}),
        "timing_mismatch_cases": timing_mismatches,
        "t20_nonpeak_cases": t20_nonpeak,
    }


def _strict_label(row: dict[str, Any]) -> str:
    label = str(row.get("feedback_label", "unknown"))
    if str(row.get("feedback_type", "")) == "maintenance_execution":
        action_cycle = row.get("action_abs_cycle")
        failure_cycle = row.get("failure_cycle")
        if action_cycle is not None and failure_cycle is not None:
            if int(action_cycle) >= int(failure_cycle):
                return (
                    label
                    if label.startswith("missed_")
                    else _component_missed_label(label)
                )
    return label


def _inferred_missed_cause(row: dict[str, Any]) -> str:
    explicit = str(row.get("missed_maintenance_cause") or "")
    if explicit:
        return explicit
    if str(row.get("feedback_type", "")) == "maintenance_execution":
        action_cycle = row.get("action_abs_cycle")
        failure_cycle = row.get("failure_cycle")
        if action_cycle is not None and failure_cycle is not None:
            if int(action_cycle) >= int(failure_cycle):
                return "maintenance_scheduled_at_or_after_failure"
    if str(row.get("feedback_type", "")) == "breakdown":
        return "legacy_unclassified_breakdown"
    return ""


def _component_missed_label(label: str) -> str:
    if "fan" in label.lower():
        return "missed_fan_maintenance"
    return "missed_HPC_maintenance"


def _read_json(path: Path) -> list[dict[str, Any]]:
    parsed = json.loads(path.read_text())
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array: {path}")
    return parsed


def _read_json_object(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text())
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return parsed


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    result = {
        "strict_experiment": {
            fd: audit_run(args.experiment_dir / fd, fd) for fd in args.fds
        }
    }
    if args.baseline_dir:
        result["baseline_experiment"] = {
            fd: audit_run(args.baseline_dir / fd, fd) for fd in args.fds
        }

    output_json = args.experiment_dir / "strict_experiment_audit.json"
    output_csv = args.experiment_dir / "strict_experiment_summary.csv"
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    rows = []
    for arm, arm_result in result.items():
        for fd, audit in arm_result.items():
            labels = audit["strict_recomputed_counts"]
            causes = audit["strict_inferred_missed_cause_counts"]
            rows.append(
                {
                    "arm": arm,
                    "fd": fd,
                    "engines": audit["engines"],
                    "correct": labels.get("correct_maintenance", 0),
                    "too_early": labels.get("too_early", 0),
                    "missed": sum(
                        count for label, count in labels.items() if label.startswith("missed_")
                    ),
                    "late_timing_missed": causes.get(
                        "maintenance_scheduled_at_or_after_failure", 0
                    ),
                    "monitoring_missed": causes.get("monitoring_without_maintenance", 0),
                    "policy_gate_monitoring_missed": causes.get(
                        "monitoring_due_to_policy_gate", 0
                    ),
                    "lhi_gate_not_triggered_missed": causes.get(
                        "lhi_gate_not_triggered_before_failure", 0
                    ),
                    "timing_mismatches": audit["maintenance_timing_mismatches"],
                    "t20_nonpeak": audit["t20_maintenance_when_peak_not_t20"],
                    "policy_updates": audit["policy_update_rows"],
                    "stored_labels_match_strict": audit[
                        "stored_labels_match_strict_recomputation"
                    ],
                }
            )
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"audit_json": str(output_json), "summary_csv": str(output_csv)}, indent=2))


if __name__ == "__main__":
    main()

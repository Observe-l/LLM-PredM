from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "outputs/CMAPSS/RAG/history_condition_h20_kg_strict_peak_timing"
EVALUATOR_ROOT = ROOT / "outputs/CMAPSS/RAG/agentic_evaluation_w20"
OUTPUT_DIR = ROOT / "outputs/CMAPSS/RAG/evaluator_impact_analysis"
FDS = ("FD001", "FD002", "FD003", "FD004")
ENGINE_RE = re.compile(r"_Engine(\d+)_")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def engine_id(row: dict[str, Any]) -> int:
    match = ENGINE_RE.search(str(row.get("case_id", "")))
    if not match:
        raise ValueError(f"Cannot parse engine from {row.get('case_id')!r}")
    return int(match.group(1))


def broad_label(label: str) -> str:
    if label == "correct_maintenance":
        return "correct"
    if label in {"too_early", "over_maintenance"}:
        return "early"
    if label.startswith("missed_"):
        return "missed"
    return label


def outcome_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(broad_label(str(row.get("feedback_label", "unknown"))) for row in rows)
    causes = Counter(
        str(row.get("missed_maintenance_cause"))
        for row in rows
        if row.get("missed_maintenance_cause")
    )
    total = len(rows)
    return {
        "n": total,
        "correct": labels["correct"],
        "correct_rate": labels["correct"] / total if total else None,
        "early": labels["early"],
        "early_rate": labels["early"] / total if total else None,
        "missed": labels["missed"],
        "missed_rate": labels["missed"] / total if total else None,
        "missed_causes": dict(causes),
    }


def policy_timeline(summary: dict[str, Any], maximum_engine: int) -> dict[int, str]:
    policy = "none"
    updates = sorted(
        summary.get("outputs", {}).get("llm_policy", {}).get("evaluation_updates", []),
        key=lambda row: int(row.get("engines_completed", 0)),
    )
    by_effective = {
        int(row["engines_completed"]) + 1: row.get("policy_patch", {})
        for row in updates
    }
    timeline: dict[int, str] = {}
    for unit in range(1, maximum_engine + 1):
        patch = by_effective.get(unit, {})
        if "peak_offset_level" in patch:
            policy = str(patch["peak_offset_level"])
        elif "maintenance_timing_policy" in patch:
            # Backward-compatible reading of pre-v6 experiment logs.
            policy = str(patch["maintenance_timing_policy"])
        timeline[unit] = policy
    return timeline


def action_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_json(path)
    return {str(row.get("case_id")): row for row in rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {
        "metric_definition": (
            "correct_maintenance_rate = engines whose terminal feedback_label is "
            "correct_maintenance / all engines with terminal feedback"
        ),
        "fd_comparison": [],
        "transition_matrices": {},
        "evaluator_policy_segments": {},
        "evaluator_windows": {},
        "action_timing": {},
        "configuration_audit": {},
        "baseline_window_variability": {},
        "paired_correctness_tests": {},
    }
    window_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    for fd in FDS:
        base_dir = BASELINE_ROOT / fd
        eval_dir = EVALUATOR_ROOT / fd
        base_summary = load_json(base_dir / "joint_simulation_summary.json")
        eval_summary = load_json(eval_dir / "joint_simulation_summary.json")
        base_feedback = load_json(base_dir / "feedback_logs.json")
        eval_feedback = load_json(eval_dir / "feedback_logs.json")
        base_by_engine = {engine_id(row): row for row in base_feedback}
        eval_by_engine = {engine_id(row): row for row in eval_feedback}
        if set(base_by_engine) != set(eval_by_engine):
            raise ValueError(f"Engine populations differ for {fd}")

        base_stats = outcome_stats(base_feedback)
        eval_stats = outcome_stats(eval_feedback)
        aggregate["fd_comparison"].append(
            {
                "fd": fd,
                "n": base_stats["n"],
                "baseline": base_stats,
                "evaluator": eval_stats,
                "correct_rate_delta": eval_stats["correct_rate"] - base_stats["correct_rate"],
                "correct_count_delta": eval_stats["correct"] - base_stats["correct"],
                "early_count_delta": eval_stats["early"] - base_stats["early"],
                "missed_count_delta": eval_stats["missed"] - base_stats["missed"],
            }
        )

        transitions = Counter(
            (
                broad_label(str(base_by_engine[unit].get("feedback_label"))),
                broad_label(str(eval_by_engine[unit].get("feedback_label"))),
            )
            for unit in sorted(base_by_engine)
        )
        aggregate["transition_matrices"][fd] = {
            f"{source}->{target}": count
            for (source, target), count in sorted(transitions.items())
        }
        gains = sum(
            count for (source, target), count in transitions.items()
            if source != "correct" and target == "correct"
        )
        losses = sum(
            count for (source, target), count in transitions.items()
            if source == "correct" and target != "correct"
        )
        aggregate["paired_correctness_tests"][fd] = {
            "gains": gains,
            "losses": losses,
            "discordant_pairs": gains + losses,
            "exact_two_sided_p": exact_binomial_two_sided(gains, losses),
        }

        maximum_engine = max(eval_by_engine)
        timeline = policy_timeline(eval_summary, maximum_engine)
        by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for unit, feedback in eval_by_engine.items():
            by_policy[timeline[unit]].append(feedback)
        aggregate["evaluator_policy_segments"][fd] = {}
        for policy, rows in by_policy.items():
            stats = outcome_stats(rows)
            aggregate["evaluator_policy_segments"][fd][policy] = stats
            segment_rows.append({"fd": fd, "policy": policy, **stats})

        checkpoints = load_json(eval_dir / "evaluation_logs.json")
        aggregate["evaluator_windows"][fd] = []
        baseline_window_rates: list[float] = []
        for checkpoint in checkpoints:
            report = checkpoint["evaluation_report"]
            completed = int(report["engines_completed"])
            start = completed - int(report["window_size"]) + 1
            policy = timeline[completed]
            baseline_window = outcome_stats(
                [base_by_engine[unit] for unit in range(start, completed + 1)]
            )
            evaluator_window = outcome_stats(
                [eval_by_engine[unit] for unit in range(start, completed + 1)]
            )
            item = {
                "fd": fd,
                "start_engine": start,
                "end_engine": completed,
                "policy": policy,
                "baseline_correct_rate": baseline_window["correct_rate"],
                "evaluator_correct_rate": evaluator_window["correct_rate"],
                "delta": evaluator_window["correct_rate"] - baseline_window["correct_rate"],
                "evaluator_correct": evaluator_window["correct"],
                "evaluator_early": evaluator_window["early"],
                "evaluator_missed": evaluator_window["missed"],
                "agent_patch": checkpoint["evaluation_agent_decision"].get("policy_patch", {}),
                "patch_applied": checkpoint["validation"].get("applied", False),
                "patch_violations": checkpoint["validation"].get("violations", []),
            }
            aggregate["evaluator_windows"][fd].append(item)
            window_rows.append(item)
            baseline_window_rates.append(float(baseline_window["correct_rate"]))

        aggregate["baseline_window_variability"][fd] = {
            "window_size": 20,
            "window_count": len(baseline_window_rates),
            "mean_correct_rate": statistics.mean(baseline_window_rates),
            "sample_sd_correct_rate": (
                statistics.stdev(baseline_window_rates)
                if len(baseline_window_rates) > 1
                else 0.0
            ),
            "min_correct_rate": min(baseline_window_rates),
            "max_correct_rate": max(baseline_window_rates),
        }

        actions = action_index(eval_dir / "action_hypotheses.json")
        timing_by_policy: dict[str, Counter[str]] = defaultdict(Counter)
        for feedback in eval_feedback:
            case_id = str(feedback.get("case_id"))
            action_row = actions.get(case_id)
            if action_row is None:
                continue
            unit = engine_id(feedback)
            action_time = str(action_row.get("action", {}).get("action_time"))
            timing_by_policy[timeline[unit]][action_time] += 1
        aggregate["action_timing"][fd] = {
            policy: dict(counts) for policy, counts in timing_by_policy.items()
        }

        aggregate["configuration_audit"][fd] = {
            "baseline_lhi_dir": base_summary.get("lhi_dir"),
            "evaluator_lhi_dir": eval_summary.get("lhi_dir"),
            "baseline_policy_version": base_summary.get("outputs", {}).get("llm_policy", {}).get("version"),
            "evaluator_policy_version": eval_summary.get("outputs", {}).get("llm_policy", {}).get("version"),
            "baseline_policy_type": base_summary.get("outputs", {}).get("llm_policy", {}).get("policy_type"),
            "evaluator_policy_type": eval_summary.get("outputs", {}).get("llm_policy", {}).get("policy_type"),
            "baseline_final_timing_policy": base_summary.get("outputs", {}).get("llm_policy", {}).get("maintenance_timing_policy"),
            "evaluator_final_timing_policy": eval_summary.get("outputs", {}).get("llm_policy", {}).get("maintenance_timing_policy"),
            "baseline_final_peak_offset_level": base_summary.get("outputs", {}).get("llm_policy", {}).get("peak_offset_level", "none"),
            "evaluator_final_peak_offset_level": eval_summary.get("outputs", {}).get("llm_policy", {}).get("peak_offset_level", "none"),
            "baseline_policy_update_count": len(base_summary.get("outputs", {}).get("llm_policy", {}).get("updates", [])),
            "evaluator_policy_update_count": len(eval_summary.get("outputs", {}).get("llm_policy", {}).get("evaluation_updates", [])),
            "baseline_evaluator_mode": base_summary.get("periodic_evaluation"),
            "evaluator_mode": eval_summary.get("periodic_evaluation"),
        }

    for arm, root in (("baseline", BASELINE_ROOT), ("evaluator", EVALUATOR_ROOT)):
        all_rows = []
        for fd in FDS:
            all_rows.extend(load_json(root / fd / "feedback_logs.json"))
        aggregate[f"overall_{arm}"] = outcome_stats(all_rows)
    aggregate["overall_correct_rate_delta"] = (
        aggregate["overall_evaluator"]["correct_rate"]
        - aggregate["overall_baseline"]["correct_rate"]
    )
    total_gains = sum(row["gains"] for row in aggregate["paired_correctness_tests"].values())
    total_losses = sum(row["losses"] for row in aggregate["paired_correctness_tests"].values())
    aggregate["overall_paired_correctness_test"] = {
        "gains": total_gains,
        "losses": total_losses,
        "discordant_pairs": total_gains + total_losses,
        "exact_two_sided_p": exact_binomial_two_sided(total_gains, total_losses),
    }

    clean_fds = {"FD002", "FD004"}
    clean_baseline_rows = []
    clean_evaluator_rows = []
    for fd in sorted(clean_fds):
        clean_baseline_rows.extend(load_json(BASELINE_ROOT / fd / "feedback_logs.json"))
        clean_evaluator_rows.extend(load_json(EVALUATOR_ROOT / fd / "feedback_logs.json"))
    clean_gains = sum(
        aggregate["paired_correctness_tests"][fd]["gains"] for fd in clean_fds
    )
    clean_losses = sum(
        aggregate["paired_correctness_tests"][fd]["losses"] for fd in clean_fds
    )
    aggregate["clean_v5_subset"] = {
        "fds": sorted(clean_fds),
        "reason": (
            "Both arms use policy schema v5 and the same LHI source; the no-evaluator arm "
            "keeps the policy frozen while the evaluator arm uses structural-only validation."
        ),
        "baseline": outcome_stats(clean_baseline_rows),
        "evaluator": outcome_stats(clean_evaluator_rows),
        "correct_rate_delta": (
            outcome_stats(clean_evaluator_rows)["correct_rate"]
            - outcome_stats(clean_baseline_rows)["correct_rate"]
        ),
        "paired_correctness_test": {
            "gains": clean_gains,
            "losses": clean_losses,
            "discordant_pairs": clean_gains + clean_losses,
            "exact_two_sided_p": exact_binomial_two_sided(clean_gains, clean_losses),
        },
    }

    (OUTPUT_DIR / "diagnostic_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n"
    )
    write_csv(OUTPUT_DIR / "window_comparison.csv", window_rows)
    write_csv(OUTPUT_DIR / "policy_segments.csv", segment_rows)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def exact_binomial_two_sided(gains: int, losses: int) -> float:
    total = gains + losses
    if total == 0:
        return 1.0
    observed = min(gains, losses)
    lower_tail = sum(
        _comb(total, value) for value in range(observed + 1)
    ) / (2**total)
    return min(1.0, 2 * lower_tail)


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for value in range(1, k + 1):
        result = result * (n - k + value) // value
    return result


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "outputs/CMAPSS/RAG/evaluator_impact_analysis"
SUMMARY_PATH = ANALYSIS_DIR / "diagnostic_summary.json"


def rate(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    data = json.loads(SUMMARY_PATH.read_text())
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    fd_rows = []
    fd_rate_rows = []
    for row in data["fd_comparison"]:
        base = row["baseline"]
        evaluated = row["evaluator"]
        fd_rows.append(
            {
                "fd": row["fd"],
                "n": row["n"],
                "baseline_correct_rate": base["correct_rate"],
                "evaluator_correct_rate": evaluated["correct_rate"],
                "delta_pp": 100 * row["correct_rate_delta"],
                "baseline_early": base["early"],
                "evaluator_early": evaluated["early"],
                "baseline_missed": base["missed"],
                "evaluator_missed": evaluated["missed"],
            }
        )
        for arm, stats in (("No evaluator", base), ("Evaluator", evaluated)):
            fd_rate_rows.append(
                {
                    "fd": row["fd"],
                    "arm": arm,
                    "correct_rate": stats["correct_rate"],
                    "correct": stats["correct"],
                    "early": stats["early"],
                    "missed": stats["missed"],
                    "n": stats["n"],
                }
            )

    window_long: dict[str, list[dict[str, Any]]] = {"FD002": [], "FD004": []}
    for fd in window_long:
        for row in data["evaluator_windows"][fd]:
            for arm, field in (
                ("No evaluator", "baseline_correct_rate"),
                ("Evaluator", "evaluator_correct_rate"),
            ):
                window_long[fd].append(
                    {
                        "checkpoint": row["end_engine"],
                        "arm": arm,
                        "correct_rate": row[field],
                        "policy": row["policy"],
                        "evaluator_early": row["evaluator_early"],
                        "evaluator_missed": row["evaluator_missed"],
                        "patch": json.dumps(row["agent_patch"], ensure_ascii=False),
                    }
                )

    policy_exposure = build_policy_exposure(data)
    key_switches = build_key_switches(data)
    source = {
        "id": "diagnostic-data",
        "label": "Evaluator impact diagnostic dataset",
        "path": "outputs/CMAPSS/RAG/evaluator_impact_analysis/diagnostic_summary.json",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_json_auto('outputs/CMAPSS/RAG/evaluator_impact_analysis/diagnostic_summary.json')",
            "description": "Paired engine-level comparison and 20-engine policy-window reconstruction.",
            "tables_used": [
                f"outputs/CMAPSS/RAG/{arm}/{fd}/{name}.json"
                for arm in (
                    "history_condition_h20_kg_strict_peak_timing",
                    "agentic_evaluation_w20",
                )
                for fd in ("FD001", "FD002", "FD003", "FD004")
                for name in ("feedback_logs", "joint_simulation_summary")
            ],
            "filters": [
                "All terminal engine feedback rows",
                "Same FD and engine ID paired across arms",
                "Evaluator windows use 20 completed engines",
            ],
            "metric_definitions": [
                "correct_maintenance_rate = correct_maintenance terminal outcomes / all engines with terminal feedback",
                "early = too_early or over_maintenance terminal outcomes",
                "missed = feedback labels beginning with missed_",
            ],
        },
    }
    code_source = {
        "id": "analysis-code",
        "label": "Reproducible diagnostic transformation",
        "path": "scripts/analyze_evaluator_impact.py",
    }

    clean = data["clean_v5_subset"]
    title = "Why periodic evaluation underperformed"
    datasets = {
        "fd-summary": fd_rows,
        "fd-rates": fd_rate_rows,
        "fd2-windows": window_long["FD002"],
        "fd4-windows": window_long["FD004"],
        "policy-exposure": policy_exposure,
        "key-switches": key_switches,
    }
    for dataset_id, rows in datasets.items():
        write_csv(ANALYSIS_DIR / f"{dataset_id}.csv", rows)
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}", "layout": "full"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "diagnostic-data",
            "body": (
                "## Technical summary\n\n"
                f"Across 709 paired engines, `correct_maintenance` fell from "
                f"**{rate(data['overall_baseline']['correct_rate'])}** to "
                f"**{rate(data['overall_evaluator']['correct_rate'])}** "
                f"({100 * data['overall_correct_rate_delta']:+.2f} percentage points). "
                f"The evaluator reduced missed maintenance from "
                f"{data['overall_baseline']['missed']} to {data['overall_evaluator']['missed']}, "
                f"but early maintenance rose from {data['overall_baseline']['early']} to "
                f"{data['overall_evaluator']['early']}; the extra early interventions more than "
                "consumed the gains.\n\n"
                f"The clean same-schema comparison is FD002+FD004: v5 without evaluator achieved "
                f"**{rate(clean['baseline']['correct_rate'])}** versus "
                f"**{rate(clean['evaluator']['correct_rate'])}** with evaluator "
                f"({100 * clean['correct_rate_delta']:+.2f} points). In paired engines the "
                f"evaluator repaired {clean['paired_correctness_test']['gains']} incorrect cases "
                f"but damaged {clean['paired_correctness_test']['losses']} correct cases.\n\n"
                "The regression is concentrated in FD002 and FD004. The LLM repeatedly changed "
                "the global timing anchor from the forecast peak to the first critical/persistent "
                "cycle, which usually changed actions from `t+20` to `t+1`. FD001 benefited from "
                "that shift, FD003 was unchanged, and the multi-condition datasets were harmed."
            ),
            "layout": "full",
        },
        {
            "id": "fd-result",
            "type": "markdown",
            "sourceId": "diagnostic-data",
            "body": (
                "## FD002 and FD004 account for the observed regression\n\n"
                f"FD001 improved by {100 * data['fd_comparison'][0]['correct_rate_delta']:.2f} points "
                f"and FD003 did not change. FD002 fell by "
                f"{abs(100 * data['fd_comparison'][1]['correct_rate_delta']):.2f} points, while FD004 "
                f"fell by {abs(100 * data['fd_comparison'][3]['correct_rate_delta']):.2f} points. "
                "Because FD002 and FD004 contain "
                "509 of the 709 engines, their losses dominate the weighted total."
            ),
            "layout": "full",
        },
        {"id": "fd-chart-block", "type": "chart", "chartId": "fd-rate-chart", "layout": "full"},
        {"id": "fd-table-block", "type": "table", "tableId": "fd-summary-table", "layout": "full"},
        {
            "id": "timing-driver",
            "type": "markdown",
            "sourceId": "diagnostic-data",
            "body": (
                "## A global `t+1` rule converted correct cases into early maintenance\n\n"
                "Under aggressive timing-policy exposure, FD002 corrected 16 previously missed "
                "engines but converted 28 previously correct engines to early maintenance. FD004 "
                "corrected only 1 missed engine while converting 30 correct engines to early. "
                "By contrast, FD001 converted 13 misses to correct outcomes without creating an "
                "early case. The same policy therefore has sharply different effects across FD "
                "distributions and should not be treated as a universal timing rule."
            ),
            "layout": "full",
        },
        {"id": "policy-table-block", "type": "table", "tableId": "policy-exposure-table", "layout": "full"},
        {
            "id": "fd2-window-note",
            "type": "markdown",
            "sourceId": "diagnostic-data",
            "body": (
                "## FD002: the controller became progressively more aggressive\n\n"
                "After the first switch at engine 60, early outcomes accumulated. At engine 200, "
                "the recent window contained 4 early cases, 0 late cases, and 1 LHI-gate miss, "
                "yet the LLM diagnosed `late_timing` and escalated from `first_critical_or_peak` "
                "to `first_persistent_or_peak`. That semantic error was accepted because the "
                "current validator checks structure only."
            ),
            "layout": "full",
        },
        {"id": "fd2-chart-block", "type": "chart", "chartId": "fd2-window-chart", "layout": "full"},
        {
            "id": "fd4-window-note",
            "type": "markdown",
            "sourceId": "diagnostic-data",
            "body": (
                "## FD004: three reactive switches explain almost the entire loss\n\n"
                "The evaluator switched early after engines 40, 140, and 200. The following "
                "20-engine windows produced 12, 10, and 9 early interventions, with evaluator "
                "correct rates of 40%, 45%, and 50% versus paired no-evaluator rates of 100%, "
                "90%, and 90%. Across those 60 engines, the evaluator achieved 27 correct outcomes "
                "versus 56 without it."
            ),
            "layout": "full",
        },
        {"id": "fd4-chart-block", "type": "chart", "chartId": "fd4-window-chart", "layout": "full"},
        {"id": "switch-table-block", "type": "table", "tableId": "switch-table", "layout": "full"},
        {
            "id": "control-loop",
            "type": "markdown",
            "sourceId": "diagnostic-data",
            "body": (
                "## The failure is a control-loop design problem\n\n"
                "Three mechanisms interact. First, at n=20 a 5-point decline is exactly one engine, "
                "yet the tool labels it `degrading`; the no-evaluator 20-engine rates already have "
                "a 9.4-point sample standard deviation in both FD002 and FD004. The evaluator therefore "
                "reacts to ordinary cohort variation despite strongly overlapping Wilson intervals. Second, "
                "the available policies are extremes: peak timing is usually `t+20`, while first "
                "critical/persistent timing is usually `t+1`. Third, evaluation occurs only every "
                "20 engines, so a harmful switch remains active for a full cohort before it can be "
                "reversed. The result is reactive oscillation between late and early errors rather "
                "than stable improvement."
            ),
            "layout": "full",
        },
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "diagnostic-data",
            "body": (
                "## Scope and metric definition\n\n"
                "The unit of analysis is one completed engine. `correct_maintenance_rate` is the "
                "number of terminal feedback rows labeled `correct_maintenance` divided by all "
                "terminal engine feedback rows. Both arms contain the same engine IDs within each "
                "FD, so outcome transitions were paired by FD and engine. The weighted overall "
                "comparison uses all 709 engines."
            ),
            "layout": "full",
        },
        {
            "id": "method",
            "type": "markdown",
            "sourceId": "analysis-code",
            "body": (
                "## Paired reconstruction supports the timing-policy diagnosis\n\n"
                "The analysis rebuilt every 20-engine evaluator window, reconstructed the policy "
                "effective for each engine from `evaluation_updates`, and joined terminal feedback "
                f"to action-policy context. Across all FDs, "
                f"{data['overall_paired_correctness_test']['gains']} engines changed from incorrect to "
                f"correct and {data['overall_paired_correctness_test']['losses']} changed from correct "
                f"to incorrect (exact paired two-sided binomial p = "
                f"{data['overall_paired_correctness_test']['exact_two_sided_p']:.4g}). In the clean "
                f"FD002+FD004 subset the corresponding counts are "
                f"{clean['paired_correctness_test']['gains']} versus "
                f"{clean['paired_correctness_test']['losses']} (p = "
                f"{clean['paired_correctness_test']['exact_two_sided_p']:.3g}). More importantly, "
                "outcomes match the no-evaluator arm before "
                "the first policy switch and in most later peak-policy windows, while divergence "
                "appears immediately in aggressive-policy windows."
            ),
            "layout": "full",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## FD002/FD004 are clean; the four-FD aggregate is still mixed\n\n"
                "The newly completed FD002/FD004 no-evaluator runs and their evaluator runs both use "
                "policy schema v5 and the same non-leaking LHI inputs, so they provide the strongest "
                "evaluator-only comparison. FD001/FD003 are not yet identical-policy controls: their "
                "no-evaluator summaries are v4 and contain per-feedback policy updates, while the "
                "evaluator runs are v5 and were produced with the earlier minimum-support validator. "
                "All four no-evaluator runs finish on `peak_score_cycle`, but that does not make their "
                "full policy/update configurations identical. The overall 709-engine result is useful "
                "descriptively; causal weight should be placed on FD002+FD004."
            ),
            "layout": "full",
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "body": (
                "## Recommended next experiment\n\n"
                "1. Keep the W=20 evaluation call, but make `no_change` the default when Wilson "
                "intervals overlap or the movement is only one engine. Record policy exposure for "
                "both the recent and previous windows so the LLM does not compare unlike regimes.\n"
                "2. Before asking the LLM to choose, replay every allowed candidate timing policy on "
                "completed engines and provide a counterfactual table of correct, early, and missed "
                "counts. This uses only completed-engine feedback and does not leak future data into "
                "the original forecast.\n"
                "3. Replace the categorical `t+20`→`t+1` jump with bounded peak offsets, for example "
                "peak, peak−3, and peak−5 cycles, or clamp first-critical timing so it cannot advance "
                "more than a fixed number of cycles. This is a timing policy, not a new threshold.\n"
                "4. Preserve LLM decision authority but require an explicit predicted net change: "
                "a patch is rational only when expected newly-correct cases exceed expected newly-early "
                "cases. Add a five-engine post-change canary/rollback check while retaining the regular "
                "20-engine cadence.\n"
                "5. Rerun FD001/FD003 as frozen v5 controls and structural-only evaluator arms before "
                "using the four-FD weighted result in the paper."
            ),
            "layout": "full",
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "The next design choice is whether early and missed maintenance should have equal "
                "cost. If they do not, `correct_maintenance_rate` alone is insufficient for policy "
                "selection; the evaluator should optimize an explicit cost-weighted utility and "
                "report both guardrail rates."
            ),
            "layout": "full",
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical diagnosis of the periodic evaluator's impact on CMAPSS maintenance outcomes.",
            "generatedAt": generated_at,
            "sources": [source, code_source],
            "blocks": blocks,
            "charts": [
                grouped_rate_chart("fd-rate-chart", "Correct maintenance rate by FD", "fd-rates", "fd", dataset_source(source, "fd-rates")),
                window_chart("fd2-window-chart", "FD002 correct maintenance rate by 20-engine window", "fd2-windows", dataset_source(source, "fd2-windows")),
                window_chart("fd4-window-chart", "FD004 correct maintenance rate by 20-engine window", "fd4-windows", dataset_source(source, "fd4-windows")),
            ],
            "tables": [
                {
                    "id": "fd-summary-table",
                    "title": "FD-level outcome comparison",
                    "subtitle": "Same engine population in both arms; delta is percentage points",
                    "dataset": "fd-summary",
                    "source": dataset_source(source, "fd-summary"),
                    "layout": "full",
                    "density": "spacious",
                    "defaultSort": {"field": "fd", "direction": "asc"},
                    "columns": [
                        {"field": "fd", "label": "FD", "type": "text"},
                        {"field": "n", "label": "Engines", "format": "number"},
                        {"field": "baseline_correct_rate", "label": "No evaluator", "format": "percent"},
                        {"field": "evaluator_correct_rate", "label": "Evaluator", "format": "percent"},
                        {"field": "delta_pp", "label": "Delta, pp", "format": "number", "movement": True},
                        {"field": "baseline_early", "label": "Early, baseline", "format": "number"},
                        {"field": "evaluator_early", "label": "Early, evaluator", "format": "number"},
                        {"field": "baseline_missed", "label": "Missed, baseline", "format": "number"},
                        {"field": "evaluator_missed", "label": "Missed, evaluator", "format": "number"},
                    ],
                },
                {
                    "id": "policy-exposure-table",
                    "title": "Paired outcomes during aggressive timing-policy exposure",
                    "subtitle": "Only engines for which evaluator timing was first-critical or first-persistent",
                    "dataset": "policy-exposure",
                    "source": dataset_source(source, "policy-exposure"),
                    "layout": "full",
                    "density": "spacious",
                    "defaultSort": {"field": "fd", "direction": "asc"},
                    "columns": [
                        {"field": "fd", "label": "FD", "type": "text"},
                        {"field": "n", "label": "Engines", "format": "number"},
                        {"field": "baseline_correct_rate", "label": "No evaluator", "format": "percent"},
                        {"field": "evaluator_correct_rate", "label": "Evaluator", "format": "percent"},
                        {"field": "miss_to_correct", "label": "Miss → correct", "format": "number"},
                        {"field": "correct_to_early", "label": "Correct → early", "format": "number"},
                        {"field": "evaluator_early", "label": "Evaluator early", "format": "number"},
                    ],
                },
                {
                    "id": "switch-table",
                    "title": "High-impact policy switches",
                    "subtitle": "Trigger evidence at checkpoint and outcomes in the following 20-engine window",
                    "dataset": "key-switches",
                    "source": dataset_source(source, "key-switches"),
                    "layout": "full",
                    "density": "spacious",
                    "defaultSort": {"field": "checkpoint", "direction": "asc"},
                    "columns": [
                        {"field": "fd", "label": "FD", "type": "text"},
                        {"field": "checkpoint", "label": "Checkpoint", "format": "number"},
                        {"field": "trigger_late", "label": "Late in trigger window", "format": "number"},
                        {"field": "patch", "label": "Applied timing patch", "type": "text"},
                        {"field": "next_baseline_rate", "label": "Next baseline", "format": "percent"},
                        {"field": "next_evaluator_rate", "label": "Next evaluator", "format": "percent"},
                        {"field": "next_early", "label": "Next early", "format": "number"},
                    ],
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": [source, code_source],
    }
    (ANALYSIS_DIR / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")


def dataset_source(source: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    result = deepcopy(source)
    result["id"] = f"diagnostic-data-{dataset_id}"
    result["path"] = f"outputs/CMAPSS/RAG/evaluator_impact_analysis/{dataset_id}.csv"
    result["query"]["sql"] = (
        "SELECT * FROM read_csv_auto("
        f"'outputs/CMAPSS/RAG/evaluator_impact_analysis/{dataset_id}.csv')"
    )
    result["query"]["tables_used"] = [result["path"]]
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_rate_chart(
    chart_id: str, title: str, dataset: str, x_field: str, source: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": chart_id,
        "title": title,
        "subtitle": "Terminal correct_maintenance outcomes divided by engines in each FD",
        "type": "bar",
        "dataset": dataset,
        "source": source,
        "layout": "full",
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": x_field, "type": "nominal", "label": "Dataset"},
            "y": {"field": "correct_rate", "type": "quantitative", "aggregate": "none", "format": "percent", "label": "Correct maintenance rate"},
            "color": {"field": "arm", "type": "nominal", "label": "Experiment arm"},
            "tooltip": [
                {"field": "correct", "type": "quantitative", "label": "Correct"},
                {"field": "early", "type": "quantitative", "label": "Early"},
                {"field": "missed", "type": "quantitative", "label": "Missed"},
                {"field": "n", "type": "quantitative", "label": "Engines"},
            ],
        },
    }


def window_chart(
    chart_id: str, title: str, dataset: str, source: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": chart_id,
        "title": title,
        "subtitle": "Paired 20-engine windows; evaluator policy is retained in source rows",
        "type": "line",
        "dataset": dataset,
        "source": source,
        "layout": "full",
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": "checkpoint", "type": "quantitative", "aggregate": "none", "label": "Completed engines"},
            "y": {"field": "correct_rate", "type": "quantitative", "aggregate": "none", "format": "percent", "label": "Correct maintenance rate"},
            "color": {"field": "arm", "type": "nominal", "label": "Experiment arm"},
            "tooltip": [
                {"field": "policy", "type": "text", "label": "Evaluator timing policy"},
                {"field": "evaluator_early", "type": "quantitative", "label": "Evaluator early"},
                {"field": "evaluator_missed", "type": "quantitative", "label": "Evaluator missed"},
            ],
        },
    }


def build_policy_exposure(data: dict[str, Any]) -> list[dict[str, Any]]:
    ranges = {
        "FD001": [(41, 100)],
        "FD002": [(61, 120), (181, 260)],
        "FD004": [(41, 60), (141, 160), (201, 220)],
    }
    roots = {
        "baseline": ROOT / "outputs/CMAPSS/RAG/history_condition_h20_kg_strict_peak_timing",
        "evaluator": ROOT / "outputs/CMAPSS/RAG/agentic_evaluation_w20",
    }
    import re
    from collections import Counter

    engine_re = re.compile(r"_Engine(\d+)_")
    result = []
    for fd, intervals in ranges.items():
        selected = {unit for start, end in intervals for unit in range(start, end + 1)}
        arm_rows = {}
        for arm, root in roots.items():
            rows = json.loads((root / fd / "feedback_logs.json").read_text())
            arm_rows[arm] = {
                int(engine_re.search(row["case_id"]).group(1)): broad(row["feedback_label"])
                for row in rows
            }
        transitions = Counter((arm_rows["baseline"][u], arm_rows["evaluator"][u]) for u in selected)
        base_correct = sum(arm_rows["baseline"][u] == "correct" for u in selected)
        eval_correct = sum(arm_rows["evaluator"][u] == "correct" for u in selected)
        result.append(
            {
                "fd": fd,
                "n": len(selected),
                "baseline_correct_rate": base_correct / len(selected),
                "evaluator_correct_rate": eval_correct / len(selected),
                "miss_to_correct": transitions[("missed", "correct")],
                "correct_to_early": transitions[("correct", "early")],
                "evaluator_early": sum(arm_rows["evaluator"][u] == "early" for u in selected),
            }
        )
    return result


def broad(label: str) -> str:
    if label == "correct_maintenance":
        return "correct"
    if label in {"too_early", "over_maintenance"}:
        return "early"
    return "missed" if label.startswith("missed_") else label


def build_key_switches(data: dict[str, Any]) -> list[dict[str, Any]]:
    keys = {"FD002": [60, 180, 200], "FD004": [40, 140, 200]}
    rows = []
    for fd, checkpoints in keys.items():
        windows = {row["end_engine"]: row for row in data["evaluator_windows"][fd]}
        log_rows = json.loads((ROOT / f"outputs/CMAPSS/RAG/agentic_evaluation_w20/{fd}/evaluation_logs.json").read_text())
        logs = {int(row["evaluation_report"]["engines_completed"]): row for row in log_rows}
        for checkpoint in checkpoints:
            next_window = windows.get(checkpoint + 20)
            if next_window is None:
                continue
            log = logs[checkpoint]
            patch = log["validation"].get("applied_patch", {})
            rows.append(
                {
                    "fd": fd,
                    "checkpoint": checkpoint,
                    "trigger_late": log["evaluation_report"]["timing_statistics"]["late_maintenance_count"],
                    "patch": str(patch.get("maintenance_timing_policy", "none")),
                    "next_baseline_rate": next_window["baseline_correct_rate"],
                    "next_evaluator_rate": next_window["evaluator_correct_rate"],
                    "next_early": next_window["evaluator_early"],
                }
            )
    return rows


if __name__ == "__main__":
    main()

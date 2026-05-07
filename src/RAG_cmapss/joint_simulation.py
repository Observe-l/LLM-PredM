from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_KG_DIR, DEFAULT_MODEL_NAME, DEFAULT_OLLAMA_URL, DEFAULT_OUTPUT_DIR
from .kg_store import KGStore
from .lhi_case_adapter import (
    GROUP_COLS,
    build_forecast_case,
    case_peak_lhi,
    iter_lhi_windows,
    load_lhi_frames,
    load_threshold_config,
)
from .logging_utils import action_decision_record, append_recent_ollama_records, write_json
from .react_agent import run_agent
from .reflection_memory import append_reflection_rule, feedback_to_rule, initialize_reflection_file


MAINTENANCE_ACTIONS = {"schedule_HPC_maintenance", "schedule_fan_maintenance"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint simulation from Layer-2 LHI evidence to Layer-3 KG RAG maintenance decisions."
    )
    parser.add_argument("--lhi_dir", type=Path, default=Path("outputs/CMAPSS/cluster_20/lhi"))
    parser.add_argument(
        "--load_top_drift_detail",
        action="store_true",
        help=(
            "Load the large top_drift_sensors.csv for detailed per-sensor values. "
            "By default joint simulation uses compact top_drift_sensors strings from lhi_scores.csv."
        ),
    )
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--eval_split", choices=["train", "test"], default="train")
    parser.add_argument("--kg_dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR / "joint_simulation")
    parser.add_argument("--import_reflection_rules", type=Path, default=None)
    parser.add_argument(
        "--use_seed_reflection",
        action="store_true",
        help="Initialize this experiment with KG seed reflection rules. Default starts with an empty memory.",
    )
    parser.add_argument("--fds", nargs="+", default=None)
    parser.add_argument("--score_col", default="lhi_rmse_roll_mean")
    parser.add_argument("--raw_score_col", default="d_rmse")
    parser.add_argument("--lhi_col", default="lhi_rmse_roll_mean")
    parser.add_argument("--lhi_trigger", type=float, default=1.0)
    parser.add_argument(
        "--maintenance_rul_threshold",
        type=float,
        default=0.25,
        help="Engineer simulator marks maintenance as reasonable when normalized remaining RUL is below this value.",
    )
    parser.add_argument("--threshold_config", type=Path, default=None)
    parser.add_argument("--default_interval", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ollama_num_predict", type=int, default=512)
    parser.add_argument(
        "--disable_ollama_json_format",
        action="store_true",
        help="Disable Ollama format=json. By default Ollama is asked to return a JSON object.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--save_recent_ollama_outputs",
        type=int,
        default=20,
        help="Save the latest K Ollama prompt/output audit records to recent_ollama_outputs.json. Use 0 to disable.",
    )
    parser.add_argument("--max_engines", type=int, default=None)
    parser.add_argument("--max_llm_calls", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_experiment_outputs(args.output_dir)
    case_dir = args.output_dir / "forecast_windows"
    case_dir.mkdir(parents=True, exist_ok=True)
    experiment_kg_dir = prepare_experiment_kg(
        source_kg_dir=args.kg_dir,
        output_dir=args.output_dir,
        import_reflection_rules=args.import_reflection_rules,
        use_seed_reflection=args.use_seed_reflection,
    )

    scores, top_drift = load_lhi_frames(args.lhi_dir, load_top_drift_detail=args.load_top_drift_detail)
    threshold_config = load_threshold_config(args.threshold_config)
    kg = KGStore(experiment_kg_dir)

    action_log_path = args.output_dir / "action_hypotheses.json"
    feedback_log_path = args.output_dir / "feedback_logs.json"
    cases_path = args.output_dir / "forecast_cases.json"
    engine_summary_path = args.output_dir / "engine_summary.json"
    engine_summary_csv_path = args.output_dir / "engine_summary.csv"
    recent_ollama_outputs_path = args.output_dir / "recent_ollama_outputs.json"
    recent_ollama_outputs: deque[dict[str, Any]] = deque(maxlen=max(args.save_recent_ollama_outputs, 0))

    engine_count = 0
    llm_call_count = 0
    action_counts: dict[str, int] = {}
    engine_summaries: list[dict[str, Any]] = []
    action_hypotheses: list[dict[str, Any]] = []
    feedback_logs: list[dict[str, Any]] = []
    forecast_cases: list[dict[str, Any]] = []
    persist_progress(
        feedback_log_path=feedback_log_path,
        feedback_logs=feedback_logs,
        engine_summary_path=engine_summary_path,
        engine_summary_csv_path=engine_summary_csv_path,
        engine_summaries=engine_summaries,
        recent_ollama_outputs_path=recent_ollama_outputs_path,
        recent_ollama_outputs=recent_ollama_outputs,
        save_recent_ollama_outputs=args.save_recent_ollama_outputs,
    )

    for (fd_name, unit_id), engine_windows in group_engine_windows(scores, args.fds):
        engine_count += 1
        if args.max_engines is not None and engine_count > args.max_engines:
            break

        state = EngineState(fd=str(fd_name), unit_id=int(unit_id), next_due_cutoff=None)
        last_case: dict[str, Any] | None = None
        last_action: dict[str, Any] | None = None
        last_component_stats: dict[str, Any] | None = None
        terminal_reason = "breakdown"

        for window in engine_windows:
            cutoff = int(window.iloc[0]["cutoff_cycle"])
            if state.next_due_cutoff is not None and cutoff < state.next_due_cutoff:
                continue
            peak_lhi = case_peak_lhi(window, args.lhi_col)
            if not pd.notna(peak_lhi) or peak_lhi <= args.lhi_trigger:
                continue
            if args.max_llm_calls is not None and llm_call_count >= args.max_llm_calls:
                terminal_reason = "max_llm_calls"
                break

            case = build_forecast_case(
                window=window,
                top_drift=top_drift,
                score_col=args.score_col,
                raw_score_col=args.raw_score_col,
                lhi_col=args.lhi_col,
                threshold_config=threshold_config,
                window_detail_dir=case_dir,
                engine_history=engine_history_frame(engine_windows, cutoff),
            )
            forecast_cases.append(case)

            started = time.time()
            result = run_agent(
                case=case,
                kg_dir=str(experiment_kg_dir),
                kg_store=kg,
                model=args.model,
                ollama_url=args.ollama_url,
                temperature=args.temperature,
                timeout=args.timeout,
                num_predict=args.ollama_num_predict,
                format_json=not args.disable_ollama_json_format,
                dry_run=args.dry_run,
            )
            result["latency_sec"] = round(time.time() - started, 3)
            result["lhi_gate"] = {"column": args.lhi_col, "peak_lhi": peak_lhi, "trigger": args.lhi_trigger}
            action_hypotheses.append(action_decision_record(result))
            append_recent_ollama_records(
                buffer=recent_ollama_outputs,
                max_items=args.save_recent_ollama_outputs,
                result=result,
                case=case,
            )
            if args.save_recent_ollama_outputs > 0:
                write_json(recent_ollama_outputs_path, list(recent_ollama_outputs))

            action = result["action"]
            action_type = str(action.get("action_type"))
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            llm_call_count += int(result.get("llm_calls", 0)) if not args.dry_run else 1
            last_case = case
            last_action = action
            last_component_stats = result.get("context", {}).get("component_evidence_statistics")

            if action_type in MAINTENANCE_ACTIONS:
                feedback = maintenance_feedback(
                    case=case,
                    action=action,
                    rul_threshold=args.maintenance_rul_threshold,
                    failure_cycle=engine_failure_cycle(engine_windows),
                )
                append_feedback_and_rule(
                    feedback,
                    case,
                    action,
                    feedback_logs,
                    experiment_kg_dir,
                    component_stats=result.get("context", {}).get("component_evidence_statistics"),
                )
                write_json(feedback_log_path, feedback_logs)
                terminal_reason = "maintenance_action"
                break

            if action_type == "schedule_monitoring":
                state.next_due_cutoff = cutoff + action_relative_cycle(action, default=args.default_interval)
            else:
                state.next_due_cutoff = cutoff + args.default_interval

        else:
            terminal_reason = "breakdown"

        if terminal_reason == "breakdown":
            if last_case is None:
                last_window = engine_windows[-1]
                last_case = build_forecast_case(
                    window=last_window,
                    top_drift=top_drift,
                    score_col=args.score_col,
                    raw_score_col=args.raw_score_col,
                    lhi_col=args.lhi_col,
                    threshold_config=threshold_config,
                    window_detail_dir=case_dir,
                    engine_history=engine_history_frame(engine_windows, int(last_window.iloc[0]["cutoff_cycle"])),
                )
                last_action = fallback_no_maintenance_action(last_case)
                last_component_stats = None
            assert last_action is not None
            feedback = breakdown_feedback(last_case, last_action, engine_failure_cycle(engine_windows))
            append_feedback_and_rule(
                feedback,
                last_case,
                last_action,
                feedback_logs,
                experiment_kg_dir,
                component_stats=last_component_stats,
            )
            write_json(feedback_log_path, feedback_logs)

        engine_summaries.append(
            {
                "fd": fd_name,
                "unit_id": int(unit_id),
                "terminal_reason": terminal_reason,
                "last_case_id": last_case.get("case_id") if last_case else None,
                "last_action_type": last_action.get("action_type") if last_action else None,
            }
        )
        write_json(engine_summary_path, engine_summaries)
        write_csv(engine_summary_csv_path, engine_summaries)

    summary = {
        "lhi_dir": str(args.lhi_dir),
        "experiment_kg_dir": str(experiment_kg_dir),
        "reflection_rules_path": str(experiment_kg_dir / "reflection_rules.csv"),
        "engines_processed": len(engine_summaries),
        "llm_decision_points": sum(action_counts.values()),
        "action_counts": action_counts,
        "terminal_counts": count_values(item["terminal_reason"] for item in engine_summaries),
        "dry_run": bool(args.dry_run),
        "outputs": {
            "action_hypotheses": str(action_log_path),
            "forecast_cases": str(cases_path),
            "feedback_logs": str(feedback_log_path),
            "engine_summary": str(engine_summary_path),
            "recent_ollama_outputs": str(recent_ollama_outputs_path) if args.save_recent_ollama_outputs > 0 else None,
        },
        "recent_ollama_outputs_path": str(recent_ollama_outputs_path) if args.save_recent_ollama_outputs > 0 else None,
        "engineer_feedback": {
            "source": str(args.data_dir),
            "eval_split": args.eval_split,
            "maintenance_rul_threshold": args.maintenance_rul_threshold,
        },
    }
    write_json(args.output_dir / "joint_simulation_summary.json", summary)
    write_json(action_log_path, action_hypotheses)
    write_json(cases_path, forecast_cases)
    write_json(feedback_log_path, feedback_logs)
    write_json(engine_summary_path, engine_summaries)
    write_csv(engine_summary_csv_path, engine_summaries)
    if args.save_recent_ollama_outputs > 0:
        write_json(recent_ollama_outputs_path, list(recent_ollama_outputs))
    print(json.dumps(summary, indent=2))


class EngineState:
    def __init__(self, fd: str, unit_id: int, next_due_cutoff: int | None):
        self.fd = fd
        self.unit_id = unit_id
        self.next_due_cutoff = next_due_cutoff


def group_engine_windows(scores: pd.DataFrame, fds: list[str] | None):
    current_key: tuple[str, int] | None = None
    current_windows: list[pd.DataFrame] = []
    for _key, window in iter_lhi_windows(scores, fds=fds):
        first = window.iloc[0]
        engine_key = (str(first["fd"]), int(first["unit_id"]))
        if current_key is None:
            current_key = engine_key
        if engine_key != current_key:
            yield current_key, current_windows
            current_key = engine_key
            current_windows = []
        current_windows.append(window)
    if current_key is not None:
        yield current_key, current_windows


def engine_history_frame(engine_windows: list[pd.DataFrame], cutoff_cycle: int) -> pd.DataFrame:
    history = [window for window in engine_windows if int(window.iloc[0]["cutoff_cycle"]) < int(cutoff_cycle)]
    if not history:
        return pd.DataFrame()
    return pd.concat(history, ignore_index=True)


def engine_failure_cycle(engine_windows: list[pd.DataFrame]) -> int:
    return max(int(window.iloc[0]["cutoff_cycle"]) for window in engine_windows) + 1


def prepare_experiment_kg(
    source_kg_dir: Path,
    output_dir: Path,
    import_reflection_rules: Path | None,
    use_seed_reflection: bool,
) -> Path:
    target = output_dir / "kg_memory"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_kg_dir, target)
    reflection_path = target / "reflection_rules.csv"
    if import_reflection_rules is not None:
        shutil.copyfile(import_reflection_rules, reflection_path)
    elif not use_seed_reflection:
        initialize_reflection_file(reflection_path)
    return target


def clean_experiment_outputs(output_dir: Path) -> None:
    for filename in (
        "action_hypotheses.jsonl",
        "action_hypotheses.json",
        "feedback_logs.jsonl",
        "feedback_logs.json",
        "forecast_cases.jsonl",
        "forecast_cases.json",
        "recent_ollama_outputs.jsonl",
        "recent_ollama_outputs.json",
        "joint_simulation_summary.json",
        "engine_summary.csv",
        "engine_summary.json",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()
    window_dir = output_dir / "forecast_windows"
    if window_dir.exists():
        shutil.rmtree(window_dir)


def persist_progress(
    feedback_log_path: Path,
    feedback_logs: list[dict[str, Any]],
    engine_summary_path: Path,
    engine_summary_csv_path: Path,
    engine_summaries: list[dict[str, Any]],
    recent_ollama_outputs_path: Path,
    recent_ollama_outputs: deque[dict[str, Any]],
    save_recent_ollama_outputs: int,
) -> None:
    write_json(feedback_log_path, feedback_logs)
    write_json(engine_summary_path, engine_summaries)
    write_csv(engine_summary_csv_path, engine_summaries)
    if save_recent_ollama_outputs > 0:
        write_json(recent_ollama_outputs_path, list(recent_ollama_outputs))


def maintenance_feedback(
    case: dict[str, Any],
    action: dict[str, Any],
    rul_threshold: float,
    failure_cycle: int,
) -> dict[str, Any]:
    fd_name = str(case["dataset_subset"])
    unit_id = int(case["unit_id"])
    action_abs_cycle = int(case["cutoff_cycle"]) + action_relative_cycle(action, default=0)
    remaining_rul = max(int(failure_cycle) - action_abs_cycle, 0)
    normalized_remaining_rul = (
        float(remaining_rul) / float(failure_cycle) if failure_cycle and failure_cycle > 0 else None
    )
    if normalized_remaining_rul is None:
        raise ValueError(
            f"Cannot compute maintenance feedback for {fd_name} unit {unit_id}: "
            "missing failure cycle from engine lifetimes and fallback."
        )
    is_reasonable = normalized_remaining_rul < float(rul_threshold)
    feedback_label = "correct_maintenance" if is_reasonable else "too_early"
    return {
        "feedback_id": f"Feedback_{case['case_id']}_maintenance",
        "feedback_type": "maintenance_execution",
        "case_id": case["case_id"],
        "action_id": f"ActionHypothesis_{case['case_id']}",
        "feedback_label": feedback_label,
        "maintenance_needed": bool(is_reasonable),
        "component_feedback": "correct" if is_reasonable else "unknown",
        "feedback_time": action.get("action_time"),
        "action_abs_cycle": action_abs_cycle,
        "failure_cycle": failure_cycle,
        "remaining_rul": remaining_rul,
        "normalized_remaining_rul": normalized_remaining_rul,
        "maintenance_rul_threshold": rul_threshold,
    }


def breakdown_feedback(
    case: dict[str, Any],
    action: dict[str, Any],
    failure_cycle: int,
) -> dict[str, Any]:
    likely_component = likely_component_from_case(case)
    fd_name = str(case["dataset_subset"])
    unit_id = int(case["unit_id"])
    label = {
        "HPC": "missed_HPC_maintenance",
        "FAN": "missed_fan_maintenance",
    }.get(likely_component, "missed_maintenance_unknown")
    return {
        "feedback_id": f"Feedback_{case['case_id']}_breakdown",
        "feedback_type": "breakdown",
        "case_id": case["case_id"],
        "action_id": f"ActionHypothesis_{case['case_id']}",
        "feedback_label": label,
        "breakdown_time": f"cycle_{int(failure_cycle)}",
        "failure_cycle": int(failure_cycle),
        "scheduled_action_time": action.get("action_time"),
        "timing_issue": "no_maintenance_scheduled"
        if action.get("action_type") in {"continue_normal_operation", "schedule_monitoring"}
        else "too_late",
        "likely_component": likely_component,
    }


def append_feedback_and_rule(
    feedback: dict[str, Any],
    case: dict[str, Any],
    action: dict[str, Any],
    feedback_logs: list[dict[str, Any]],
    experiment_kg_dir: Path,
    component_stats: dict[str, Any] | None = None,
) -> None:
    feedback_logs.append(feedback)
    rule = feedback_to_rule(feedback, case, action, component_stats=component_stats)
    if rule is not None:
        append_reflection_rule(experiment_kg_dir, rule)


def action_relative_cycle(action: dict[str, Any], default: int) -> int:
    value = action.get("action_time")
    if isinstance(value, str) and value.startswith("t+"):
        try:
            return max(1, int(value[2:]))
        except ValueError:
            return default
    return default


def fallback_no_maintenance_action(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": "continue_normal_operation",
        "action_time": None,
        "risk_hypothesis": "low_risk_hypothesis",
        "degradation_hypothesis": "uncertain_component_degradation",
        "confidence": 0.0,
        "evidence_paths": [],
        "reason": "No LHI-triggered LLM decision occurred before the engine reached the final available cycle.",
        "validation_status": "not_run",
    }


def likely_component_from_case(case: dict[str, Any]) -> str:
    sensors = [str(s).upper() for s in case.get("forecast_summary", {}).get("dominant_top_sensors", [])]
    if any(s in {"S7", "S11", "S3", "S9", "S14"} for s in sensors):
        return "HPC"
    if any(s in {"S8", "S13", "S15"} for s in sensors):
        return "FAN"
    return "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


if __name__ == "__main__":
    main()

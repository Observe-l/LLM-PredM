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

from .agentic_controller import llm_decide_adaptation
from .config import DEFAULT_KG_DIR, DEFAULT_MODEL_NAME, DEFAULT_OLLAMA_URL, DEFAULT_OUTPUT_DIR
from .kg_store import KGStore
from .lightgbm_train import train_lightgbm_models
from .lightgbm_update_tool import LightGBMUpdateTool
from .llm_policy_update_tool import LLMPolicyUpdateTool
from .llm_policy_risk_tool import initial_policy, load_policy
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
from .update_operator import append_update_operation, build_update_operation
from .update_validator import validate_update


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
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ollama_num_predict", type=int, default=512)
    parser.add_argument(
        "--disable_ollama_json_format",
        action="store_true",
        help="Disable Ollama format=json. By default Ollama is asked to return a JSON object.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--risk_model_path", type=Path, default=None)
    parser.add_argument("--risk_theta_low", type=float, default=0.4)
    parser.add_argument("--risk_theta_conf", type=float, default=0.3)
    parser.add_argument(
        "--risk_policy_mode",
        choices=["llm_only", "hybrid"],
        default="llm_only",
        help=(
            "Risk policy arm: llm_only uses only experiment-local LLMPolicyRiskTool and LLMPolicyUpdateTool; "
            "hybrid uses LightGBM when available and otherwise uses the LLM policy tools."
        ),
    )
    parser.add_argument("--update_model_dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--online_model_dir",
        type=Path,
        default=None,
        help="Directory for agentic online LightGBM retraining. Defaults to OUTPUT_DIR/models.",
    )
    parser.add_argument(
        "--online_train_min_rows",
        type=int,
        default=5,
        help="Minimum labeled reflection rows before online LightGBM training can succeed.",
    )
    parser.add_argument(
        "--hybrid_handoff_min_engines",
        type=int,
        default=5,
        help="Minimum number of engines before hybrid LightGBM handoff is attempted aggressively.",
    )
    parser.add_argument(
        "--llm_policy_initial_peak_threshold",
        type=float,
        default=0.5,
        help=(
            "Initial peak_score boundary for LLMPolicyRiskTool exploration. "
            "It is updated online only from this experiment's maintenance feedback."
        ),
    )
    parser.add_argument(
        "--disable_update_tool",
        action="store_true",
        help="Disable LightGBMUpdateTool feedback-time adaptation suggestions.",
    )
    parser.add_argument(
        "--disable_update_review_llm",
        action="store_true",
        help="Skip the LLM adaptation controller. Fallback policy then decides retraining/update.",
    )
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
    if args.risk_model_path is not None:
        raise SystemExit(
            "--risk_model_path is disabled for zero-shot online experiments; "
            "hybrid may only train LightGBM from this experiment's reflection history."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_experiment_outputs(args.output_dir)
    case_dir = args.output_dir / "forecast_windows"
    case_dir.mkdir(parents=True, exist_ok=True)
    experiment_kg_dir = prepare_experiment_kg(
        source_kg_dir=args.kg_dir,
        output_dir=args.output_dir,
    )

    scores, top_drift = load_lhi_frames(args.lhi_dir, load_top_drift_detail=args.load_top_drift_detail)
    threshold_config = load_threshold_config(args.threshold_config)
    kg = KGStore(experiment_kg_dir)
    online_model_dir = args.online_model_dir or (args.output_dir / "models")
    online_model_dir.mkdir(parents=True, exist_ok=True)
    active_risk_model_path: Path | None = None
    if args.risk_policy_mode == "hybrid" and active_risk_model_path is None and (online_model_dir / "lightgbm_risk.pkl").exists():
        active_risk_model_path = online_model_dir / "lightgbm_risk.pkl"
    adaptive_risk_thresholds: dict[str, dict[str, float]] = {}
    llm_policy_path = online_model_dir / "llm_policy_tool.json"

    action_log_path = args.output_dir / "action_hypotheses.json"
    feedback_log_path = args.output_dir / "feedback_logs.json"
    cases_path = args.output_dir / "forecast_cases.json"
    engine_summary_path = args.output_dir / "engine_summary.json"
    engine_summary_csv_path = args.output_dir / "engine_summary.csv"
    recent_ollama_outputs_path = args.output_dir / "recent_ollama_outputs.json"
    zero_shot_score_log_path = args.output_dir / "zero_shot_risk_scores.json"
    lightgbm_update_log_path = args.output_dir / "lightgbm_update_logs.json"
    llm_policy_update_log_path = args.output_dir / "llm_policy_update_logs.json"
    recent_ollama_outputs: deque[dict[str, Any]] = deque(maxlen=max(args.save_recent_ollama_outputs, 0))

    engine_count = 0
    llm_call_count = 0
    action_counts: dict[str, int] = {}
    engine_summaries: list[dict[str, Any]] = []
    action_hypotheses: list[dict[str, Any]] = []
    feedback_logs: list[dict[str, Any]] = []
    lightgbm_update_logs: list[dict[str, Any]] = []
    llm_policy_update_logs: list[dict[str, Any]] = []
    zero_shot_score_logs: list[dict[str, Any]] = []
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
        last_context: dict[str, Any] | None = None
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
                risk_model_path=str(active_risk_model_path) if active_risk_model_path else None,
                risk_theta_low=args.risk_theta_low,
                risk_theta_conf=args.risk_theta_conf,
                risk_threshold_overrides=adaptive_risk_thresholds.get(str(fd_name)),
                risk_policy_mode=args.risk_policy_mode,
                llm_policy_tool_path=str(llm_policy_path),
                llm_policy=current_llm_policy(
                    args=args,
                    policy_path=llm_policy_path,
                ),
            )
            result["latency_sec"] = round(time.time() - started, 3)
            result["lhi_gate"] = {"column": args.lhi_col, "peak_lhi": peak_lhi, "trigger": args.lhi_trigger}
            action_hypotheses.append(action_decision_record(result))
            record_zero_shot_score(zero_shot_score_logs, result, case)
            write_json(zero_shot_score_log_path, zero_shot_score_logs)
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
            last_context = result.get("context", {})
            last_component_stats = last_context.get("component_evidence_statistics")

            if action_type in MAINTENANCE_ACTIONS:
                feedback = maintenance_feedback(
                    case=case,
                    action=action,
                    rul_threshold=args.maintenance_rul_threshold,
                    failure_cycle=engine_failure_cycle(engine_windows),
                )
                reflection_rule = append_feedback_and_rule(
                    feedback,
                    case,
                    action,
                    feedback_logs,
                    experiment_kg_dir,
                    component_stats=result.get("context", {}).get("component_evidence_statistics"),
                )
                new_model_path = maybe_run_update_tool(
                    args=args,
                    feedback=feedback,
                    case=case,
                    action=action,
                    result=result,
                    reflection_rule=reflection_rule,
                    lightgbm_update_logs=lightgbm_update_logs,
                    llm_policy_update_logs=llm_policy_update_logs,
                    output_dir=args.output_dir,
                    experiment_kg_dir=experiment_kg_dir,
                    online_model_dir=online_model_dir,
                    active_risk_model_path=active_risk_model_path,
                    engine_index=engine_count,
                )
                if new_model_path is not None:
                    active_risk_model_path = new_model_path
                apply_latest_threshold_operation(lightgbm_update_logs, adaptive_risk_thresholds, str(fd_name))
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
                last_context = {}
            assert last_action is not None
            feedback = breakdown_feedback(last_case, last_action, engine_failure_cycle(engine_windows))
            reflection_rule = append_feedback_and_rule(
                feedback,
                last_case,
                last_action,
                feedback_logs,
                experiment_kg_dir,
                component_stats=last_component_stats,
            )
            new_model_path = maybe_run_update_tool(
                args=args,
                feedback=feedback,
                case=last_case,
                action=last_action,
                result={"context": last_context or {"component_evidence_statistics": last_component_stats or {}}},
                reflection_rule=reflection_rule,
                lightgbm_update_logs=lightgbm_update_logs,
                llm_policy_update_logs=llm_policy_update_logs,
                output_dir=args.output_dir,
                experiment_kg_dir=experiment_kg_dir,
                online_model_dir=online_model_dir,
                active_risk_model_path=active_risk_model_path,
                engine_index=engine_count,
            )
            if new_model_path is not None:
                active_risk_model_path = new_model_path
            apply_latest_threshold_operation(lightgbm_update_logs, adaptive_risk_thresholds, str(last_case.get("dataset_subset")))
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
        "risk_policy_mode": args.risk_policy_mode,
        "outputs": {
            "action_hypotheses": str(action_log_path),
            "forecast_cases": str(cases_path),
            "feedback_logs": str(feedback_log_path),
            "lightgbm_update_logs": str(lightgbm_update_log_path) if args.risk_policy_mode == "hybrid" else None,
            "llm_policy_update_logs": str(llm_policy_update_log_path),
            "engine_summary": str(engine_summary_path),
            "recent_ollama_outputs": str(recent_ollama_outputs_path) if args.save_recent_ollama_outputs > 0 else None,
            "zero_shot_risk_scores": str(zero_shot_score_log_path),
            "online_model_dir": str(online_model_dir),
            "llm_policy_tool_path": str(online_model_dir / "llm_policy_tool.json"),
            "active_risk_model_path": str(active_risk_model_path) if active_risk_model_path else None,
            "adaptive_risk_thresholds": adaptive_risk_thresholds,
            "llm_policy": load_policy(llm_policy_path),
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
    write_json(zero_shot_score_log_path, zero_shot_score_logs)
    write_json(cases_path, forecast_cases)
    write_json(feedback_log_path, feedback_logs)
    if args.risk_policy_mode == "hybrid":
        write_json(lightgbm_update_log_path, lightgbm_update_logs)
    write_json(llm_policy_update_log_path, llm_policy_update_logs)
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


def current_llm_policy(args: argparse.Namespace, policy_path: Path) -> dict[str, Any]:
    policy = load_policy(policy_path)
    if policy is not None:
        return policy
    return initial_policy(
        peak_threshold=float(args.llm_policy_initial_peak_threshold),
        theta_low=args.risk_theta_low,
        theta_conf=args.risk_theta_conf,
    )


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
) -> Path:
    target = output_dir / "kg_memory"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_kg_dir, target)
    reflection_path = target / "reflection_rules.csv"
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
        "lightgbm_update_logs.json",
        "llm_policy_update_logs.json",
        "lightgbm_update_operations.jsonl",
        "joint_simulation_summary.json",
        "engine_summary.csv",
        "engine_summary.json",
        "zero_shot_risk_scores.json",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()
    window_dir = output_dir / "forecast_windows"
    if window_dir.exists():
        shutil.rmtree(window_dir)
    model_dir = output_dir / "models"
    if model_dir.exists():
        shutil.rmtree(model_dir)


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


def record_zero_shot_score(
    zero_shot_score_logs: list[dict[str, Any]],
    result: dict[str, Any],
    case: dict[str, Any],
) -> None:
    context = result.get("context", {})
    policy = context.get("llm_risk_policy") or {}
    risk = context.get("lightgbm_risk", {})
    if not risk:
        return
    zero_shot_score_logs.append(
        {
            "case_id": result.get("case_id"),
            "dataset_subset": case.get("dataset_subset"),
            "unit_id": case.get("unit_id"),
            "cutoff_cycle": case.get("cutoff_cycle"),
            "model_source": risk.get("model_source"),
            "maintenance_risk_score": risk.get("maintenance_risk_score"),
            "predicted_risk_stage": risk.get("predicted_risk_stage"),
            "risk_decision": risk.get("risk_decision"),
            "theta_low": risk.get("theta_low"),
            "theta_conf": risk.get("theta_conf"),
            "tool_name": risk.get("tool_name"),
            "model_path": risk.get("model_path"),
            "top_features": risk.get("top_features"),
            "policy_source": policy.get("source"),
            "policy_type": policy.get("policy_type"),
            "peak_threshold": policy.get("peak_threshold"),
            "positive_peak_min": policy.get("positive_peak_min"),
            "early_peak_max": policy.get("early_peak_max"),
            "correct_anchor_count": policy.get("correct_anchor_count"),
            "missed_anchor_count": policy.get("missed_anchor_count"),
            "early_anchor_count": policy.get("early_anchor_count"),
            "policy_reason": policy.get("reason"),
            "action_type": result.get("action", {}).get("action_type"),
            "action_time": result.get("action", {}).get("action_time"),
        }
    )


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
) -> dict[str, Any] | None:
    feedback_logs.append(feedback)
    rule = feedback_to_rule(feedback, case, action, component_stats=component_stats)
    if rule is not None:
        append_reflection_rule(experiment_kg_dir, rule)
    return rule


def maybe_run_update_tool(
    args: argparse.Namespace,
    feedback: dict[str, Any],
    case: dict[str, Any],
    action: dict[str, Any],
    result: dict[str, Any],
    reflection_rule: dict[str, Any] | None,
    lightgbm_update_logs: list[dict[str, Any]],
    llm_policy_update_logs: list[dict[str, Any]],
    output_dir: Path,
    experiment_kg_dir: Path,
    online_model_dir: Path,
    active_risk_model_path: Path | None,
    engine_index: int,
) -> Path | None:
    context = result.get("context", {})
    handoff_training_result = None
    handoff_model_path = None
    if (
        args.risk_policy_mode == "hybrid"
        and active_risk_model_path is None
        and int(engine_index) >= int(args.hybrid_handoff_min_engines)
    ):
        handoff_training_result = maybe_train_online_lightgbm(
            reflection_rules_path=experiment_kg_dir / "reflection_rules.csv",
            online_model_dir=online_model_dir,
            min_rows=max(int(args.online_train_min_rows), int(args.hybrid_handoff_min_engines)),
        )
        risk_model = handoff_training_result.get("models", {}).get("risk", {}) if isinstance(handoff_training_result, dict) else {}
        if risk_model.get("path"):
            handoff_model_path = Path(risk_model["path"])

    if str(feedback.get("feedback_label", "")) == "correct_maintenance":
        if args.risk_policy_mode == "hybrid":
            lightgbm_record = {
                "case_id": case.get("case_id"),
                "feedback_label": feedback.get("feedback_label"),
                "llm_adaptation_decision": {
                    "retrain_lightgbm_risk": False,
                    "call_lightgbm_update_tool": False,
                    "reason": "Skipped adaptation: correct_maintenance feedback does not trigger LightGBM update review.",
                    "expected_update_focus": "none",
                    "confidence": 1.0,
                    "source": "skipped_correct_maintenance",
                },
                "training_result": handoff_training_result,
                "update_result": None,
                "validation": {"valid": True, "violations": []},
                "operation": None,
            }
            lightgbm_update_logs.append(lightgbm_record)
            write_json(output_dir / "lightgbm_update_logs.json", lightgbm_update_logs)
        return handoff_model_path
    if args.risk_policy_mode == "llm_only":
        llm_policy_update_result = maybe_update_llm_policy_tool(
            args=args,
            feedback=feedback,
            case=case,
            action=action,
            context=context,
            reflection_rule=reflection_rule,
            experiment_kg_dir=experiment_kg_dir,
            online_model_dir=online_model_dir,
        )
        llm_policy_update_logs.append(
            {
                "case_id": case.get("case_id"),
                "feedback_label": feedback.get("feedback_label"),
                "llm_policy_update_result": llm_policy_update_result,
                "validation": {"valid": True, "violations": []},
            }
        )
        write_json(output_dir / "llm_policy_update_logs.json", llm_policy_update_logs)
        return None
    review = review_update_with_llm(
        args=args,
        feedback=feedback,
        case=case,
        action=action,
        context=context,
        reflection_rule=reflection_rule,
        experiment_kg_dir=experiment_kg_dir,
        active_risk_model_path=active_risk_model_path,
    )
    llm_policy_update_result = maybe_update_llm_policy_tool(
        args=args,
        feedback=feedback,
        case=case,
        action=action,
        context=context,
        reflection_rule=reflection_rule,
        experiment_kg_dir=experiment_kg_dir,
        online_model_dir=online_model_dir,
    )
    if llm_policy_update_result is not None:
        llm_policy_update_logs.append(
            {
                "case_id": case.get("case_id"),
                "feedback_label": feedback.get("feedback_label"),
                "llm_policy_update_result": llm_policy_update_result,
            }
        )
        write_json(output_dir / "llm_policy_update_logs.json", llm_policy_update_logs)
    training_result = None
    new_model_path = handoff_model_path
    if review.get("retrain_lightgbm_risk", False):
        training_result = maybe_train_online_lightgbm(
            reflection_rules_path=experiment_kg_dir / "reflection_rules.csv",
            online_model_dir=online_model_dir,
            min_rows=max(int(args.online_train_min_rows), int(review.get("min_training_rows", args.online_train_min_rows))),
        )
        risk_model = training_result.get("models", {}).get("risk", {}) if isinstance(training_result, dict) else {}
        if risk_model.get("path"):
            new_model_path = Path(risk_model["path"])
    elif handoff_training_result is not None:
        training_result = handoff_training_result

    if args.disable_update_tool or not review.get("call_lightgbm_update_tool", False):
        lightgbm_update_logs.append(
            {
                "case_id": case.get("case_id"),
                "feedback_label": feedback.get("feedback_label"),
                "llm_adaptation_decision": review,
                "training_result": training_result,
                "update_result": None,
                "validation": {"valid": True, "violations": []},
                "operation": None,
            }
        )
        write_json(output_dir / "lightgbm_update_logs.json", lightgbm_update_logs)
        return new_model_path
    tool_model_dir = online_model_dir if _has_update_models(online_model_dir) else args.update_model_dir
    tool = LightGBMUpdateTool(model_dir=tool_model_dir)
    update_result = tool.predict(
        case=case,
        action=action,
        feedback=feedback,
        context=context,
        lightgbm_risk=context.get("lightgbm_risk"),
    )
    validation = validate_update(update_result, feedback, context=context)
    operation = build_update_operation(update_result) if validation["valid"] else None
    record = {
        "case_id": case.get("case_id"),
        "feedback_label": feedback.get("feedback_label"),
        "llm_adaptation_decision": review,
        "training_result": training_result,
        "update_result": update_result,
        "validation": validation,
        "operation": operation,
    }
    lightgbm_update_logs.append(record)
    write_json(output_dir / "lightgbm_update_logs.json", lightgbm_update_logs)
    if operation is not None:
        append_update_operation(output_dir / "lightgbm_update_operations.jsonl", record)
    return new_model_path


def maybe_update_llm_policy_tool(
    args: argparse.Namespace,
    feedback: dict[str, Any],
    case: dict[str, Any],
    action: dict[str, Any],
    context: dict[str, Any],
    reflection_rule: dict[str, Any] | None,
    experiment_kg_dir: Path,
    online_model_dir: Path,
) -> dict[str, Any] | None:
    if args.risk_policy_mode not in {"llm_only", "hybrid"}:
        return None
    if (
        args.risk_policy_mode == "hybrid"
        and str(context.get("lightgbm_risk", {}).get("model_source", "")) == "lightgbm_model"
    ):
        return None
    policy_path = online_model_dir / "llm_policy_tool.json"
    current_policy = load_policy(policy_path) or initial_policy(
        peak_threshold=float(args.llm_policy_initial_peak_threshold),
        theta_low=args.risk_theta_low,
        theta_conf=args.risk_theta_conf,
    )
    tool = LLMPolicyUpdateTool(policy_path=policy_path)
    result = tool.predict(
        current_policy=current_policy,
        feedback=feedback,
        case=case,
        action=action,
        reflection_rules_path=experiment_kg_dir / "reflection_rules.csv",
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        timeout=args.timeout,
        num_predict=args.ollama_num_predict,
        format_json=not args.disable_ollama_json_format,
        dry_run=args.dry_run,
        disable_llm=args.disable_update_review_llm,
    )
    result["policy_path"] = str(policy_path)
    return result


def maybe_train_online_lightgbm(
    reflection_rules_path: Path,
    online_model_dir: Path,
    min_rows: int,
) -> dict[str, Any]:
    try:
        summary = train_lightgbm_models(
            reflection_rules=reflection_rules_path,
            output_dir=online_model_dir,
            min_rows=min_rows,
        )
    except SystemExit as exc:
        return {"error": str(exc), "models": {}}
    except Exception as exc:
        return {"error": str(exc), "models": {}}
    summary["online_training"] = {
        "reflection_rules_path": str(reflection_rules_path),
        "online_model_dir": str(online_model_dir),
        "min_rows": int(min_rows),
    }
    return summary


def _has_update_models(model_dir: Path) -> bool:
    return (model_dir / "lightgbm_update_threshold.pkl").exists() or (
        model_dir / "lightgbm_update_timing.pkl"
    ).exists()


def apply_latest_threshold_operation(
    update_logs: list[dict[str, Any]],
    adaptive_risk_thresholds: dict[str, dict[str, float]],
    fd_name: str,
) -> None:
    if not update_logs:
        return
    operation = update_logs[-1].get("operation")
    if not operation:
        return
    delta = operation.get("threshold_delta") or {}
    if not delta:
        return
    current = adaptive_risk_thresholds.setdefault(
        fd_name,
        {"q95_excess_min": 0.35, "q99_excess_min": 0.25, "persistent_duration_min": 10.0},
    )
    current["q95_excess_min"] = max(0.0, float(current["q95_excess_min"]) + float(delta.get("q95_excess_min", 0.0)))
    current["q99_excess_min"] = max(0.0, float(current["q99_excess_min"]) + float(delta.get("q99_excess_min", 0.0)))
    current["persistent_duration_min"] = max(
        1.0,
        float(current["persistent_duration_min"]) + float(delta.get("persistent_duration_min", 0.0)),
    )


def review_update_with_llm(
    args: argparse.Namespace,
    feedback: dict[str, Any],
    case: dict[str, Any],
    action: dict[str, Any],
    context: dict[str, Any],
    reflection_rule: dict[str, Any] | None,
    experiment_kg_dir: Path,
    active_risk_model_path: Path | None,
) -> dict[str, Any]:
    return llm_decide_adaptation(
        feedback=feedback,
        case=case,
        action=action,
        context=context,
        reflection_rule=reflection_rule,
        reflection_rules_path=experiment_kg_dir / "reflection_rules.csv",
        active_risk_model_path=active_risk_model_path,
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        timeout=args.timeout,
        num_predict=384,
        format_json=not args.disable_ollama_json_format,
        dry_run=args.dry_run,
        disable_llm=args.disable_update_review_llm,
    )


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

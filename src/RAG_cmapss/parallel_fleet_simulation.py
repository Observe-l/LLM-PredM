"""Shared-clock, parallel-fleet PredM simulation.

Each engine owns its LHI windows and lifecycle state, while the fleet shares a
single adaptive policy, reflection memory, and evaluator.  The global cycle
loop deliberately performs terminal events first, applies one policy/evaluator
update for all events at that cycle, and only then lets still-active engines
make decisions under the updated policy.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_KG_DIR, DEFAULT_MODEL_NAME, DEFAULT_OLLAMA_URL
from .evaluation_agent import run_evaluation_agent
from .evaluation_tool import EvaluationTool, compact_evaluation_history
from .joint_simulation import (
    MAINTENANCE_ACTIONS,
    action_relative_cycle,
    append_feedback_and_rule,
    breakdown_feedback,
    clean_experiment_outputs,
    current_llm_policy,
    current_lhi_trigger,
    engine_history_frame,
    fallback_no_maintenance_action,
    maintenance_feedback,
    prepare_experiment_kg,
    summarize_feedback,
)
from .kg_store import KGStore
from .lhi_case_adapter import (
    build_forecast_case,
    build_current_lhi_case,
    case_peak_lhi,
    load_lhi_frames,
    load_threshold_config,
)
from .llm_policy_risk_tool import initial_policy, load_policy
from .llm_policy_update_tool import LLMPolicyUpdateTool
from .logging_utils import action_decision_record, write_json
from .react_agent import run_agent
from .reflection_memory import initialize_reflection_file


@dataclass
class FleetEngine:
    fd: str
    unit_id: int
    windows: dict[int, pd.DataFrame]
    ordered_cutoffs: list[int]
    failure_cycle: int
    next_observation_cutoff: int
    state: str = "active"
    pending_maintenance: dict[str, Any] | None = None
    last_case: dict[str, Any] | None = None
    last_action: dict[str, Any] | None = None
    last_context: dict[str, Any] = field(default_factory=dict)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    last_decision_threshold: float | None = None
    component_consensus: dict[str, Any] = field(default_factory=dict)
    terminal_feedback: dict[str, Any] | None = None
    terminal_cycle: int | None = None
    terminal_reflection_rule_id: str | None = None
    maintenance_cycle: int | None = None
    broken_cycle: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel fleet predictive-maintenance simulation.")
    parser.add_argument(
        "--lhi_dir",
        type=Path,
        default=None,
        help="Single LHI directory for a normal single-source fleet.",
    )
    parser.add_argument(
        "--lhi_dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Multiple LHI directories to concatenate for a mixed fleet.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fds", nargs="+", required=True)
    parser.add_argument(
        "--mixed_fleet",
        action="store_true",
        help="Mix multiple FD subsets in one shared-clock fleet and use per-engine component evidence.",
    )
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--eval_split", choices=["train", "test"], default="train")
    parser.add_argument("--kg_dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--score_col", default="lhi_rmse")
    parser.add_argument("--raw_score_col", default="d_rmse")
    parser.add_argument("--lhi_col", default="lhi_rmse")
    parser.add_argument("--lhi_trigger", type=float, default=0.25)
    parser.add_argument("--min_predm_cycle", type=int, default=None)
    parser.add_argument("--maintenance_rul_threshold", type=float, default=0.25)
    parser.add_argument("--risk_theta_conf", type=float, default=0.3)
    parser.add_argument("--threshold_config", type=Path, default=None)
    parser.add_argument("--default_interval", type=int, default=20)
    parser.add_argument(
        "--health_reference_cycles",
        type=int,
        default=50,
        help="Cycles reserved for the healthy reference; no LHI gate or PredM action before this cycle.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ollama_num_predict", type=int, default=512)
    parser.add_argument("--prompt_variant", choices=["kg", "no_kg_evidence"], default="kg")
    parser.add_argument(
        "--decision_mode",
        choices=["forecast_window", "current_lhi_only"],
        default="forecast_window",
        help="Use the existing forecast-window prompt or a current-LHI-only prompt with no future values.",
    )
    parser.add_argument("--disable_ollama_json_format", action="store_true")
    parser.add_argument("--disable_update_review_llm", action="store_true")
    parser.add_argument("--disable_periodic_evaluation", action="store_true")
    parser.add_argument(
        "--evaluation_minimum_support",
        type=int,
        default=10,
        help="Minimum number of completed engines before the shared evaluator can update policy.",
    )
    parser.add_argument("--save_recent_ollama_outputs", type=int, default=20)
    parser.add_argument("--max_engines", type=int, default=None)
    parser.add_argument(
        "--max_engines_per_fd",
        type=int,
        default=None,
        help="Optional smoke-test cap applied independently within each FD subset.",
    )
    parser.add_argument("--max_time", type=int, default=None)
    parser.add_argument("--max_llm_calls", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lhi_dir is None and not args.lhi_dirs:
        raise SystemExit("one of --lhi_dir or --lhi_dirs is required")
    if args.mixed_fleet and not args.lhi_dirs:
        raise SystemExit("--mixed_fleet requires --lhi_dirs so all LHI sources can be loaded")
    if args.health_reference_cycles < 0:
        raise SystemExit("--health_reference_cycles must be >= 0")
    if args.evaluation_minimum_support < 1:
        raise SystemExit("--evaluation_minimum_support must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_experiment_outputs(args.output_dir)
    # Do not leave a previous run's final summary visible while this run is
    # still in progress.
    stale_summary = args.output_dir / "parallel_fleet_summary.json"
    if stale_summary.exists():
        stale_summary.unlink()
    case_dir = args.output_dir / "forecast_windows"
    case_dir.mkdir(parents=True, exist_ok=True)
    experiment_kg_dir = prepare_experiment_kg(args.kg_dir, args.output_dir)

    scores, top_drift = load_fleet_lhi_frames(args, load_top_drift_detail=False)
    scores = scores[scores["fd"].isin(args.fds)].copy()
    threshold_config = load_threshold_config(args.threshold_config)
    kg = KGStore(experiment_kg_dir)
    online_model_dir = args.output_dir / "models"
    online_model_dir.mkdir(parents=True, exist_ok=True)
    policy_path = online_model_dir / "llm_policy_tool.json"

    paths = {
        "feedback": args.output_dir / "feedback_logs.json",
        "summary": args.output_dir / "engine_summary.json",
        "summary_csv": args.output_dir / "engine_summary.csv",
        "actions": args.output_dir / "action_hypotheses.json",
        "cases": args.output_dir / "forecast_cases.json",
        "evaluation": args.output_dir / "evaluation_logs.json",
        "policy_updates": args.output_dir / "llm_policy_update_logs.json",
        "recent": args.output_dir / "recent_ollama_outputs.json",
        "scores": args.output_dir / "zero_shot_risk_scores.json",
    }
    feedback_logs: list[dict[str, Any]] = []
    action_hypotheses: list[dict[str, Any]] = []
    forecast_cases: list[dict[str, Any]] = []
    evaluation_logs: list[dict[str, Any]] = []
    policy_update_logs: list[dict[str, Any]] = []
    zero_shot_logs: list[dict[str, Any]] = []
    recent_ollama_outputs: list[dict[str, Any]] = []

    fleet = build_fleet(
        scores,
        max_engines=args.max_engines,
        max_engines_per_fd=args.max_engines_per_fd,
    )
    raw_global_start = min(engine.next_observation_cutoff for engine in fleet.values()) if fleet else 0
    global_start = max(int(args.health_reference_cycles), int(raw_global_start))
    global_end = max(engine.failure_cycle for engine in fleet.values()) if fleet else 0
    llm_calls = 0
    start_time = time.time()

    for global_cycle in range(global_start, global_end + 1):
        if args.max_time is not None and time.time() - start_time >= args.max_time:
            break
        completed_feedback_items: list[dict[str, Any]] = []

        # First execute maintenance actions due at this shared fleet time.  An
        # action at or after failure is evaluated as a missed maintenance.
        for engine in fleet.values():
            if engine.state != "pending_maintenance" or not engine.pending_maintenance:
                continue
            due = int(engine.pending_maintenance["due_cycle"])
            if due > global_cycle:
                continue
            case = engine.pending_maintenance["case"]
            action = engine.pending_maintenance["action"]
            decision_threshold = engine.pending_maintenance.get("decision_threshold")
            if global_cycle >= engine.failure_cycle:
                feedback = maintenance_feedback(
                    case=case,
                    action=action,
                    rul_threshold=args.maintenance_rul_threshold,
                    failure_cycle=engine.failure_cycle,
                    component_aware=args.prompt_variant != "no_kg_evidence",
                    action_history=engine.action_history,
                )
                engine.state = "failed"
            else:
                feedback = maintenance_feedback(
                    case=case,
                    action=action,
                    rul_threshold=args.maintenance_rul_threshold,
                    failure_cycle=engine.failure_cycle,
                    component_aware=args.prompt_variant != "no_kg_evidence",
                    action_history=engine.action_history,
                )
                engine.state = "maintained"
            feedback["global_cycle"] = global_cycle
            feedback["execution_mode"] = "parallel_fleet"
            feedback["maintenance_cycle"] = global_cycle
            feedback["broken_cycle"] = None
            engine.terminal_feedback = feedback
            engine.terminal_cycle = global_cycle
            engine.maintenance_cycle = global_cycle
            rule = append_feedback_and_rule(
                feedback,
                case,
                action,
                feedback_logs,
                experiment_kg_dir,
                component_stats=(
                    None
                    if args.prompt_variant == "no_kg_evidence"
                    else engine.last_context.get("component_evidence_statistics")
                ),
                component_aware=args.prompt_variant != "no_kg_evidence",
            )
            engine.terminal_reflection_rule_id = (
                rule.get("rule_id") if isinstance(rule, dict) else None
            )
            completed_feedback_items.append(
                {
                    "feedback": feedback,
                    "case": case,
                    "action": action,
                    "reflection_rule": rule,
                    "decision_threshold": decision_threshold,
                }
            )
            link_terminal_event(zero_shot_logs, feedback, case, rule, global_cycle)
            engine.pending_maintenance = None

        # Engines that reach failure before a scheduled maintenance action are
        # terminal and contribute a breakdown reflection at this global time.
        for engine in fleet.values():
            if engine.state in {"maintained", "failed"}:
                continue
            if global_cycle < engine.failure_cycle:
                continue
            case = engine.last_case or build_last_case(engine, top_drift, args, case_dir, threshold_config)
            action = engine.last_action or fallback_no_maintenance_action(case)
            feedback = breakdown_feedback(
                case,
                action,
                engine.failure_cycle,
                component_aware=args.prompt_variant != "no_kg_evidence",
                action_history=engine.action_history,
                decision_context=engine.last_context,
            )
            feedback["global_cycle"] = global_cycle
            feedback["execution_mode"] = "parallel_fleet"
            feedback["maintenance_cycle"] = None
            feedback["broken_cycle"] = global_cycle
            engine.state = "failed"
            engine.terminal_feedback = feedback
            engine.terminal_cycle = global_cycle
            engine.broken_cycle = global_cycle
            rule = append_feedback_and_rule(
                feedback,
                case,
                action,
                feedback_logs,
                experiment_kg_dir,
                component_stats=engine.last_context.get("component_evidence_statistics"),
                component_aware=args.prompt_variant != "no_kg_evidence",
            )
            completed_feedback_items.append(
                {
                    "feedback": feedback,
                    "case": case,
                    "action": action,
                    "reflection_rule": rule,
                    "decision_threshold": engine.last_decision_threshold,
                }
            )
            engine.terminal_reflection_rule_id = (
                rule.get("rule_id") if isinstance(rule, dict) else None
            )
            link_terminal_event(zero_shot_logs, feedback, case, rule, global_cycle)

        # A reflection event updates theta once for all feedback created at the
        # same global time, then invokes evaluator once.
        if completed_feedback_items:
            update_result = update_theta_once(
                args=args,
                items=completed_feedback_items,
                feedback_logs=feedback_logs,
                experiment_kg_dir=experiment_kg_dir,
                online_model_dir=online_model_dir,
                policy_path=policy_path,
            )
            policy_update_logs.append(
                {
                    "global_cycle": global_cycle,
                    "feedback_count": len(completed_feedback_items),
                    "llm_policy_update_result": update_result,
                }
            )
            write_json(paths["policy_updates"], policy_update_logs)
            if (
                not args.disable_periodic_evaluation
                and len(feedback_logs) >= args.evaluation_minimum_support
            ):
                evaluation = evaluate_once(
                    args=args,
                    global_cycle=global_cycle,
                    feedback_logs=feedback_logs,
                    action_hypotheses=action_hypotheses,
                    evaluation_logs=evaluation_logs,
                    policy_path=policy_path,
                    event_size=len(completed_feedback_items),
                )
                evaluation_logs.append(evaluation)
                write_json(paths["evaluation"], evaluation_logs)
            if bool(update_result.get("update_threshold")):
                recheck_undecided_active_after_threshold_update(
                    fleet=fleet,
                    global_cycle=global_cycle,
                )

        # Re-evaluate all engines eligible at this shared time using the policy
        # that now includes any reflection update from this time.
        for engine in fleet.values():
            if engine.state != "active" or engine.next_observation_cutoff > global_cycle:
                continue
            cutoff = select_cutoff(engine, global_cycle)
            if cutoff is None:
                continue
            window = engine.windows[cutoff]
            engine.next_observation_cutoff = cutoff + 1
            trigger = current_lhi_trigger(policy_path, args.lhi_trigger)
            engine.last_decision_threshold = float(trigger)
            if args.decision_mode == "current_lhi_only":
                # The current observed cycle is normally stored in the
                # immediately preceding forecast window. Read only that one
                # small frame for the gate; build a full case only after the
                # current LHI crosses the gate.
                previous_cutoffs = [value for value in engine.ordered_cutoffs if value < cutoff]
                observed_window = engine.windows.get(cutoff - 1)
                if observed_window is None and previous_cutoffs:
                    observed_window = engine.windows[max(previous_cutoffs)]
                current_rows = (
                    observed_window[observed_window["cycle"].astype(int) == int(cutoff)]
                    if observed_window is not None
                    else pd.DataFrame()
                )
                if current_rows.empty or args.lhi_col not in current_rows:
                    engine.last_case = None
                    continue
                current_lhi = pd.to_numeric(current_rows[args.lhi_col], errors="coerce").dropna()
                peak_lhi = float(current_lhi.iloc[-1]) if not current_lhi.empty else float("nan")
                if not pd.notna(peak_lhi) or peak_lhi <= trigger:
                    engine.last_case = None
                    continue
                case = build_current_lhi_case(
                    window=window,
                    top_drift=top_drift,
                    score_col=args.score_col,
                    raw_score_col=args.raw_score_col,
                    lhi_col=args.lhi_col,
                    threshold_config=threshold_config,
                    current_cycle=cutoff,
                    window_detail_dir=case_dir,
                    engine_history=observed_window,
                )
            else:
                history = engine_history_frame(list(engine.windows.values()), cutoff)
                case = build_forecast_case(
                    window=window,
                    top_drift=top_drift,
                    score_col=args.score_col,
                    raw_score_col=args.raw_score_col,
                    lhi_col=args.lhi_col,
                    threshold_config=threshold_config,
                    window_detail_dir=case_dir,
                    engine_history=history,
                )
                peak_lhi = case_peak_lhi(window, args.lhi_col)
            if args.max_llm_calls is not None and llm_calls >= args.max_llm_calls:
                continue
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
                risk_model_path=None,
                risk_theta_low=0.4,
                risk_theta_conf=0.3,
                risk_threshold_overrides=None,
                risk_policy_mode="llm_only",
                llm_policy_tool_path=str(policy_path),
                llm_policy=current_llm_policy(args=args, policy_path=policy_path),
                prompt_variant=args.prompt_variant,
                mixed_fleet=args.mixed_fleet,
                component_consensus=engine.component_consensus,
                prior_monitoring_count=sum(
                    str(item.get("action_type")) == "schedule_monitoring"
                    for item in engine.action_history
                ),
                decision_mode=args.decision_mode,
            )
            llm_calls += int(result.get("llm_calls", 0)) if not args.dry_run else 1
            result["lhi_gate"] = {"column": args.lhi_col, "peak_lhi": peak_lhi, "trigger": trigger}
            result["global_cycle"] = global_cycle
            result["execution_mode"] = "parallel_fleet"
            action_hypotheses.append(action_decision_record(result))
            forecast_cases.append(case)
            zero_shot_logs.append({
                "event_type": "predm_decision",
                "case_id": case.get("case_id"),
                "global_cycle": global_cycle,
                "peak_lhi": peak_lhi,
                "trigger": trigger,
                "action_type": result.get("action", {}).get("action_type"),
                "action_time": result.get("action", {}).get("action_time"),
            })
            recent_ollama_outputs.append({"global_cycle": global_cycle, "case_id": case.get("case_id"), "result": result})
            if args.save_recent_ollama_outputs > 0:
                recent_ollama_outputs = recent_ollama_outputs[-args.save_recent_ollama_outputs :]
            engine.last_case = case
            engine.last_action = result["action"]
            engine.last_context = result.get("context", {})
            action_type = str(result["action"].get("action_type", ""))
            engine.action_history.append(
                {
                    "case_id": case.get("case_id"),
                    "cutoff_cycle": cutoff,
                    "action_type": action_type,
                    "action_time": result["action"].get("action_time"),
                    "decision_threshold": float(trigger),
                    "action_selection_source": "llm" if result.get("llm_calls", 0) else "fallback",
                }
            )
            if action_type in MAINTENANCE_ACTIONS:
                engine.state = "pending_maintenance"
                engine.pending_maintenance = {
                    "due_cycle": global_cycle + action_relative_cycle(result["action"], default=1),
                    "case": case,
                    "action": result["action"],
                    "decision_threshold": float(trigger),
                }
            else:
                interval = action_relative_cycle(result["action"], default=args.default_interval)
                engine.next_observation_cutoff = global_cycle + interval

        persist_parallel_progress(paths, feedback_logs, action_hypotheses, forecast_cases, evaluation_logs, policy_update_logs, zero_shot_logs, recent_ollama_outputs, fleet, args)
        if all(engine.state in {"maintained", "failed"} for engine in fleet.values()):
            break

    persist_parallel_progress(paths, feedback_logs, action_hypotheses, forecast_cases, evaluation_logs, policy_update_logs, zero_shot_logs, recent_ollama_outputs, fleet, args)
    engine_summaries = fleet_summary_rows(fleet)
    write_json(paths["summary"], engine_summaries)
    pd.DataFrame(engine_summaries).to_csv(paths["summary_csv"], index=False)
    summary = {
        "simulation_mode": "parallel_fleet",
        "decision_mode": args.decision_mode,
        "fds": args.fds,
        "global_cycle_start": global_start,
        "health_reference_cycles": args.health_reference_cycles,
        "global_cycle_end": global_end,
        "engines_processed": len(engine_summaries),
        "feedback_events": len(policy_update_logs),
        "llm_decision_points": len(action_hypotheses),
        "feedback_statistics": summarize_feedback(feedback_logs),
        "feedback_statistics_by_fd": summarize_feedback_by_fd(feedback_logs),
        "evaluation_count": len(evaluation_logs),
        "evaluation_minimum_support": args.evaluation_minimum_support,
        "lhi_trigger_initial": args.lhi_trigger,
        "max_engines": args.max_engines,
        "max_engines_per_fd": args.max_engines_per_fd,
        "max_time_seconds": args.max_time,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    write_json(args.output_dir / "parallel_fleet_summary.json", summary)
    print(json.dumps(summary, indent=2))


def load_fleet_lhi_frames(
    args: argparse.Namespace,
    *,
    load_top_drift_detail: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directories = list(args.lhi_dirs or ([args.lhi_dir] if args.lhi_dir is not None else []))
    frames: list[pd.DataFrame] = []
    top_frames: list[pd.DataFrame] = []
    for directory in directories:
        scores, top = load_lhi_frames(directory, load_top_drift_detail=load_top_drift_detail)
        source_fds = mixed_lhi_source_fds(directory, args)
        if source_fds is not None:
            scores = scores[scores["fd"].isin(source_fds)].copy()
            if not top.empty and "fd" in top.columns:
                top = top[top["fd"].isin(source_fds)].copy()
        frames.append(scores)
        top_frames.append(top)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    scores = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"],
        keep="first",
    )
    top_drift = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    return scores, top_drift


def mixed_lhi_source_fds(directory: Path, args: argparse.Namespace) -> set[str] | None:
    """Select the canonical source for each FD in the mixed CMAPSS run."""
    if not args.mixed_fleet:
        return None
    normalized = str(directory).replace("\\", "/")
    if normalized.endswith("/cluster_20/lhi_fix"):
        return {"FD001", "FD003"}
    if "history_condition_h20_fd002_fd004/lhi" in normalized:
        return {"FD002", "FD004"}
    return None


def summarize_feedback_by_fd(feedback_logs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in feedback_logs:
        fd = str(row.get("dataset_subset") or "unknown")
        grouped.setdefault(fd, []).append(row)
    return {fd: summarize_feedback(rows) for fd, rows in sorted(grouped.items())}


def build_fleet(
    scores: pd.DataFrame,
    max_engines: int | None = None,
    max_engines_per_fd: int | None = None,
) -> dict[tuple[str, int], FleetEngine]:
    """Build a deterministic state machine for each engine in the fleet."""
    fleet: dict[tuple[str, int], FleetEngine] = {}
    grouped = scores.groupby(["fd", "unit_id"], sort=False)
    keys = sorted(grouped.groups, key=lambda key: (str(key[0]), int(key[1])))
    if max_engines is not None:
        if int(max_engines) < 1:
            raise ValueError("max_engines must be >= 1 when provided")
        keys = keys[: int(max_engines)]
    if max_engines_per_fd is not None:
        if int(max_engines_per_fd) < 1:
            raise ValueError("max_engines_per_fd must be >= 1 when provided")
        counts: dict[str, int] = {}
        capped_keys: list[tuple[str, int]] = []
        for key in keys:
            fd = str(key[0])
            if counts.get(fd, 0) >= int(max_engines_per_fd):
                continue
            capped_keys.append(key)
            counts[fd] = counts.get(fd, 0) + 1
        keys = capped_keys
    for fd, unit_id in keys:
        group = grouped.get_group((fd, unit_id))
        windows = {
            int(cutoff): frame.sort_values("cycle").copy()
            for cutoff, frame in group.groupby("cutoff_cycle", sort=True)
        }
        cutoffs = sorted(windows)
        fleet[(str(fd), int(unit_id))] = FleetEngine(
            fd=str(fd),
            unit_id=int(unit_id),
            windows=windows,
            ordered_cutoffs=cutoffs,
            failure_cycle=max(cutoffs) + 1,
            next_observation_cutoff=min(cutoffs),
        )
    return fleet


def select_cutoff(engine: FleetEngine, global_cycle: int) -> int | None:
    available = [cutoff for cutoff in engine.ordered_cutoffs if engine.next_observation_cutoff <= cutoff <= global_cycle]
    return max(available) if available else None


def build_last_case(engine: FleetEngine, top_drift: pd.DataFrame, args: argparse.Namespace, case_dir: Path, threshold_config: dict[str, Any]) -> dict[str, Any]:
    cutoff = engine.ordered_cutoffs[-1]
    history = engine_history_frame(list(engine.windows.values()), cutoff)
    if args.decision_mode == "current_lhi_only":
        return build_current_lhi_case(
            window=engine.windows[cutoff],
            top_drift=top_drift,
            score_col=args.score_col,
            raw_score_col=args.raw_score_col,
            lhi_col=args.lhi_col,
            threshold_config=threshold_config,
            current_cycle=cutoff,
            window_detail_dir=case_dir,
            engine_history=history,
        )
    return build_forecast_case(
        window=engine.windows[cutoff],
        top_drift=top_drift,
        score_col=args.score_col,
        raw_score_col=args.raw_score_col,
        lhi_col=args.lhi_col,
        threshold_config=threshold_config,
        window_detail_dir=case_dir,
        engine_history=history,
    )


def update_theta_once(
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    feedback_logs: list[dict[str, Any]],
    experiment_kg_dir: Path,
    online_model_dir: Path,
    policy_path: Path,
) -> dict[str, Any]:
    tool = LLMPolicyUpdateTool(policy_path=policy_path)
    rows = [item.get("reflection_rule") or item.get("feedback", {}) for item in items]
    return tool.predict_batch(
        batch_items=items,
        current_policy=load_policy(policy_path) or initial_policy(peak_threshold=args.lhi_trigger),
        reflection_rules_path=experiment_kg_dir / "reflection_rules.csv",
        reflection_rows=rows,
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        timeout=args.timeout,
        num_predict=args.ollama_num_predict,
        format_json=not args.disable_ollama_json_format,
        dry_run=args.dry_run,
        disable_llm=args.disable_update_review_llm,
    )


def evaluate_once(
    *,
    args: argparse.Namespace,
    global_cycle: int,
    feedback_logs: list[dict[str, Any]],
    action_hypotheses: list[dict[str, Any]],
    evaluation_logs: list[dict[str, Any]],
    policy_path: Path,
    event_size: int,
) -> dict[str, Any]:
    policy = load_policy(policy_path) or initial_policy(peak_threshold=args.lhi_trigger)
    report = EvaluationTool(window_size=max(int(event_size), 1)).evaluate(
        fd="PARALLEL_FLEET",
        feedback_logs=feedback_logs,
        action_hypotheses=action_hypotheses,
        current_policy=policy,
        lhi_trigger=current_lhi_trigger(policy_path, args.lhi_trigger),
    )
    report["global_cycle"] = global_cycle
    report["evaluation_scope"] = "current_global_cycle_feedback_event"
    report["event_feedback_count"] = int(event_size)
    report["recent_evaluation_history"] = compact_evaluation_history(evaluation_logs, limit=4)
    decision = run_evaluation_agent(
        report,
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        timeout=args.timeout,
        num_predict=args.ollama_num_predict,
        format_json=True,
        dry_run=args.dry_run or args.disable_update_review_llm,
    )
    updated = dict(policy)
    patch = decision.get("policy_patch") if isinstance(decision.get("policy_patch"), dict) else {}
    for key in ("action_escalation_policy", "peak_offset_level", "monitoring_interval"):
        if key in patch:
            updated[key] = patch[key]
    if patch:
        updated["evaluation_updates"] = list(updated.get("evaluation_updates", [])) + [
            {"global_cycle": global_cycle, "policy_patch": patch, "source": decision.get("source")}
        ]
        policy_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
    return {
        "global_cycle": global_cycle,
        "checkpoint_id": report.get("checkpoint_id"),
        "evaluation_report": report,
        "evaluation_agent_decision": decision,
        "validation": {"applied": bool(patch), "updated_policy": updated},
    }


def fleet_summary_rows(
    fleet: dict[tuple[str, int], FleetEngine],
) -> list[dict[str, Any]]:
    """Return a live one-row-per-engine snapshot for progress and final logs."""
    rows: list[dict[str, Any]] = []
    for key in sorted(fleet, key=lambda item: (str(item[0]), int(item[1]))):
        engine = fleet[key]
        feedback = engine.terminal_feedback or {}
        rows.append(
            {
                "fd": engine.fd,
                "unit_id": engine.unit_id,
                "state": engine.state,
                "last_case_id": engine.last_case.get("case_id") if engine.last_case else None,
                "last_action_type": engine.last_action.get("action_type") if engine.last_action else None,
                "feedback_label": feedback.get("feedback_label"),
                "missed_maintenance_cause": feedback.get("missed_maintenance_cause"),
                "terminal_cycle": engine.terminal_cycle,
                "maintenance_cycle": engine.maintenance_cycle,
                "broken_cycle": engine.broken_cycle,
                "decision_count": len(engine.action_history),
                "feedback_id": feedback.get("feedback_id"),
                "reflection_rule_id": engine.terminal_reflection_rule_id,
            }
        )
    return rows


def recheck_undecided_active_after_threshold_update(
    *,
    fleet: dict[tuple[str, int], FleetEngine],
    global_cycle: int,
) -> None:
    """Recheck only active engines that have not committed an action.

    Once an engine has selected maintenance or monitoring, its decision is
    treated as committed and is not interrupted by later theta updates. This
    prevents repeated peak-time recalculation from postponing maintenance.
    Engines still waiting for their first action remain eligible for an
    immediate check at the current global cycle.
    """
    for engine in fleet.values():
        if engine.state != "active" or engine.last_action is not None:
            continue
        engine.next_observation_cutoff = min(
            int(engine.next_observation_cutoff), int(global_cycle)
        )


def link_terminal_event(
    zero_shot_logs: list[dict[str, Any]],
    feedback: dict[str, Any],
    case: dict[str, Any],
    reflection_rule: dict[str, Any] | None,
    global_cycle: int,
) -> None:
    """Link each reflection to its terminal case in the risk-score log.

    Decision points and terminal outcomes are both retained, but every
    reflection gets exactly one corresponding ``terminal_outcome`` record.
    This prevents the old apparent desynchronization where risk logs contained
    only LLM calls while reflection memory contained only terminal engines.
    """
    case_id = case.get("case_id")
    rule_id = reflection_rule.get("rule_id") if isinstance(reflection_rule, dict) else None
    matches = [row for row in reversed(zero_shot_logs) if row.get("case_id") == case_id]
    if matches:
        row = matches[0]
        row.update(
            {
                "feedback_id": feedback.get("feedback_id"),
                "feedback_label": feedback.get("feedback_label"),
                "reflection_rule_id": rule_id,
                "terminal_global_cycle": global_cycle,
                "maintenance_cycle": feedback.get("maintenance_cycle"),
                "broken_cycle": feedback.get("broken_cycle"),
            }
        )
        return
    zero_shot_logs.append(
        {
            "event_type": "terminal_outcome",
            "case_id": case_id,
            "dataset_subset": case.get("dataset_subset"),
            "unit_id": case.get("unit_id"),
            "cutoff_cycle": case.get("cutoff_cycle"),
            "global_cycle": global_cycle,
            "peak_lhi": (case.get("forecast_summary") or {}).get("peak_score"),
            "trigger": None,
            "action_type": None,
            "action_time": None,
            "feedback_id": feedback.get("feedback_id"),
            "feedback_label": feedback.get("feedback_label"),
            "reflection_rule_id": rule_id,
            "terminal_global_cycle": global_cycle,
            "maintenance_cycle": feedback.get("maintenance_cycle"),
            "broken_cycle": feedback.get("broken_cycle"),
        }
    )


def persist_parallel_progress(
    paths: dict[str, Path],
    feedback_logs: list[dict[str, Any]],
    action_hypotheses: list[dict[str, Any]],
    forecast_cases: list[dict[str, Any]],
    evaluation_logs: list[dict[str, Any]],
    policy_update_logs: list[dict[str, Any]],
    zero_shot_logs: list[dict[str, Any]],
    recent_ollama_outputs: list[dict[str, Any]],
    fleet: dict[tuple[str, int], FleetEngine],
    args: argparse.Namespace,
) -> None:
    engine_summaries = fleet_summary_rows(fleet)
    write_json(paths["feedback"], feedback_logs)
    write_json(paths["summary"], engine_summaries)
    pd.DataFrame(engine_summaries).to_csv(paths["summary_csv"], index=False)
    write_json(paths["actions"], action_hypotheses)
    write_json(paths["cases"], forecast_cases)
    write_json(paths["evaluation"], evaluation_logs)
    write_json(paths["policy_updates"], policy_update_logs)
    write_json(paths["scores"], zero_shot_logs)
    if args.save_recent_ollama_outputs > 0:
        write_json(paths["recent"], recent_ollama_outputs)


if __name__ == "__main__":
    main()

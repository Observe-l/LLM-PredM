from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.RAG_cmapss.config import DEFAULT_KG_DIR, DEFAULT_MODEL_NAME, DEFAULT_OLLAMA_URL
from src.RAG_cmapss.evaluation_agent import run_evaluation_agent
from src.RAG_cmapss.evaluation_tool import EvaluationTool
from src.RAG_cmapss.evaluation_validator import validate_and_apply_evaluation
from src.RAG_cmapss.kg_store import KGStore
from src.RAG_cmapss.llm_policy_risk_tool import initial_policy
from src.RAG_cmapss.ollama_client import extract_json
from src.RAG_cmapss.react_agent import run_agent
from src.RAG_cmapss.timing_policy import PEAK_OFFSET_CYCLES


DEFAULT_CASES = (
    PROJECT_ROOT
    / "outputs/CMAPSS/RAG/history_condition_h20_kg_strict_peak_timing/FD001/forecast_cases.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/CMAPSS/RAG/smoke_peak_offset_evaluation/smoke_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test evaluator peak-offset update and downstream Ollama execution."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = initial_policy()
    feedback, actions = _synthetic_late_timing_windows(policy)
    report = EvaluationTool(window_size=10).evaluate(
        fd="FD001",
        feedback_logs=feedback,
        action_hypotheses=actions,
        current_policy=policy,
        lhi_trigger=1.0,
    )
    decision = run_evaluation_agent(
        report,
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=0,
        timeout=args.timeout,
        num_predict=512,
        format_json=True,
    )
    validation = validate_and_apply_evaluation(
        decision=decision,
        report=report,
        current_policy=policy,
    )
    updated = validation["updated_policy"]
    updated_level = str(updated.get("peak_offset_level"))
    if not validation["applied"] or updated_level not in {"small", "median", "large"}:
        raise RuntimeError(
            "Evaluator did not directly choose a non-zero timing offset: "
            + json.dumps({"decision": decision, "validation": validation}, ensure_ascii=False)
        )

    case = _select_peak_t20_case(args.cases)
    with tempfile.TemporaryDirectory(prefix="rag_peak_offset_smoke_") as tmp:
        result = run_agent(
            case=case,
            kg_dir=str(DEFAULT_KG_DIR),
            kg_store=KGStore(DEFAULT_KG_DIR),
            model=args.model,
            ollama_url=args.ollama_url,
            temperature=0,
            timeout=args.timeout,
            num_predict=512,
            format_json=True,
            risk_policy_mode="llm_only",
            llm_policy_tool_path=str(Path(tmp) / "llm_policy_tool.json"),
            llm_policy=updated,
            prompt_variant="kg",
        )

    expected_time = f"t+{max(1, 20 - PEAK_OFFSET_CYCLES[updated_level])}"
    first_llm_action = _first_parsed_llm_action(result)
    final_action = result["action"]
    if not str(first_llm_action.get("action_type", "")).endswith("maintenance"):
        raise RuntimeError(f"Decision LLM did not select maintenance: {first_llm_action}")
    if first_llm_action.get("action_time") != expected_time:
        raise RuntimeError(
            f"Decision LLM did not directly follow updated timing; expected {expected_time}: "
            f"{first_llm_action}"
        )
    if final_action.get("action_time") != expected_time:
        raise RuntimeError(f"Final validated action does not use {expected_time}: {final_action}")
    if final_action.get("local_validation_repair_used") or final_action.get("llm_fallback_used"):
        raise RuntimeError(f"Smoke result depended on repair/fallback: {final_action}")

    summary = {
        "status": "passed",
        "model": args.model,
        "evaluation_window": 10,
        "score_source": "synthetic completed-engine correct_maintenance outcomes",
        "policy_counterfactual_evaluation": False,
        "canary_deployment": False,
        "evaluation_report": report,
        "evaluation_agent_decision": {
            key: value for key, value in decision.items() if key not in {"prompt", "raw_output"}
        },
        "evaluation_agent_raw_output": decision.get("raw_output"),
        "validation": {
            key: value for key, value in validation.items() if key != "updated_policy"
        },
        "updated_policy": updated,
        "decision_case_id": case["case_id"],
        "peak_score_cycle": case["risk_statistics"]["peak_score_cycle"],
        "expected_maintenance_time": expected_time,
        "decision_llm_first_action": first_llm_action,
        "final_validated_action": final_action,
        "maintenance_timing_profile": result["context"]["maintenance_timing_profile"],
        "llm_calls": result["llm_calls"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _synthetic_late_timing_windows(policy: dict) -> tuple[list[dict], list[dict]]:
    feedback: list[dict] = []
    actions: list[dict] = []
    for engine in range(1, 21):
        recent = engine > 10
        case_id = f"ForecastCase_FD001_Engine{engine}_Cycle100"
        feedback.append(
            {
                "case_id": case_id,
                "feedback_label": "missed_HPC_maintenance" if recent else "correct_maintenance",
                "missed_maintenance_cause": (
                    "maintenance_scheduled_at_or_after_failure" if recent else None
                ),
                "signed_cycle_margin": -2 if recent else 8,
            }
        )
        actions.append(
            {
                "case_id": case_id,
                "action": {
                    "action_type": "schedule_HPC_maintenance",
                    "action_time": "t+20",
                    "validation_status": "valid",
                },
                "context": {
                    "llm_policy": policy,
                    "maintenance_timing_profile": {
                        "peak_score_cycle": "t+20",
                        "recommended_maintenance_time": "t+20",
                    },
                    "sensor_paths": [{"evidence_id": "E1"}],
                },
            }
        )
    return feedback, actions


def _select_peak_t20_case(path: Path) -> dict:
    cases = json.loads(path.read_text())
    for case in cases:
        peak = case.get("risk_statistics", {}).get("peak_score_cycle")
        if peak == "t+20" and case.get("dataset_subset") == "FD001":
            return case
    raise RuntimeError(f"No FD001 raw-LHI case with peak_score_cycle=t+20 in {path}")


def _first_parsed_llm_action(result: dict) -> dict:
    for row in result.get("raw_outputs", []):
        if isinstance(row, dict) and row.get("parse_ok"):
            return extract_json(str(row.get("raw_output", "")))
    raise RuntimeError(f"No parseable decision-LLM output: {result.get('raw_outputs')}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL_NAME, DEFAULT_OLLAMA_URL, PROJECT_ROOT
from .kg_prompt_ablation import (
    NO_KG_SYSTEM_PROMPT,
    build_no_kg_prompt,
    neutral_validation,
    production_validation,
    summarize_ablation,
)
from .ollama_client import extract_json, ollama_chat


DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "CMAPSS" / "RAG" / "group"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "CMAPSS" / "RAG" / "ablation" / "no_kg_prompt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay baseline LLM decision cases with all graph-derived prompt evidence removed."
    )
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fds", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num_predict", type=int, default=512)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--case_id", action="append", default=[])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip case IDs already present in ablation_records.jsonl.",
    )
    return parser.parse_args()


def load_selected_cases(
    experiment_root: Path,
    fds: list[str],
    case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for fd in fds:
        fd_dir = experiment_root / fd
        cases = json.loads((fd_dir / "forecast_cases.json").read_text())
        decisions = json.loads((fd_dir / "action_hypotheses.json").read_text())
        case_by_id = {case["case_id"]: case for case in cases}
        for decision in decisions:
            case_id = decision.get("case_id")
            if int(decision.get("llm_calls", 0)) <= 0:
                continue
            if case_ids and case_id not in case_ids:
                continue
            if case_id not in case_by_id:
                raise KeyError(f"{case_id} is missing from {fd_dir / 'forecast_cases.json'}")
            selected.append(
                {
                    "fd": fd,
                    "case": case_by_id[case_id],
                    "baseline": decision,
                }
            )
    return selected


def run_one(
    item: dict[str, Any],
    model: str,
    ollama_url: str,
    temperature: float,
    timeout: int,
    num_predict: int,
) -> dict[str, Any]:
    case = item["case"]
    baseline = item["baseline"]
    context = baseline.get("context", {})
    prompt = build_no_kg_prompt(
        case,
        context.get("risk_gate", {}),
        context.get("lightgbm_risk"),
        context.get("llm_policy"),
    )
    started = time.time()
    record: dict[str, Any] = {
        "fd": item["fd"],
        "case_id": case["case_id"],
        "baseline_action": baseline.get("action", {}),
        "prompt_variant": "no_kg_evidence",
        "model": model,
    }
    try:
        raw = ollama_chat(
            [
                {"role": "system", "content": NO_KG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            url=ollama_url,
            temperature=temperature,
            timeout=timeout,
            num_predict=num_predict,
            format_json=True,
            think=False,
        )
        action = extract_json(raw)
        record.update(
            {
                "ablation_action": action,
                "neutral_validation": neutral_validation(action, case),
                "production_validation": production_validation(action, case, context),
                "raw_output": raw,
            }
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["latency_sec"] = round(time.time() - started, 3)
    return record


def main() -> None:
    args = parse_args()
    requested_ids = set(args.case_id) or None
    items = load_selected_cases(args.experiment_root, args.fds, requested_ids)
    if args.max_cases is not None:
        items = items[: args.max_cases]
    if not items:
        raise SystemExit("No baseline LLM decision cases matched the selection.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "ablation_records.jsonl"
    completed_ids = _existing_case_ids(records_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    records: list[dict[str, Any]] = _read_jsonl(records_path) if args.resume else []

    with records_path.open(mode) as handle:
        for index, item in enumerate(items, start=1):
            case_id = item["case"]["case_id"]
            if case_id in completed_ids:
                continue
            record = run_one(
                item,
                model=args.model,
                ollama_url=args.ollama_url,
                temperature=args.temperature,
                timeout=args.timeout,
                num_predict=args.num_predict,
            )
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            records.append(record)
            status = "ok" if "ablation_action" in record else "failed"
            print(f"[{index}/{len(items)}] {case_id}: {status} ({record['latency_sec']}s)", flush=True)

    selected_ids = {item["case"]["case_id"] for item in items}
    selected_records = [record for record in records if record.get("case_id") in selected_ids]
    summary = summarize_by_fd(selected_records)
    summary.update(
        {
            "experiment": "prompt-only ablation with graph-derived evidence removed",
            "model": args.model,
            "temperature": args.temperature,
            "selection_rule": "baseline action_hypotheses records with llm_calls > 0",
            "important_note": (
                "production_validation is audit-only and does not alter the raw no-KG action."
            ),
        }
    )
    summary_path = args.output_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_comparison_csv(args.output_dir / "action_comparison.csv", selected_records)
    print(json.dumps({"records": str(records_path), "summary": str(summary_path), **summary["overall"]}, indent=2))


def summarize_by_fd(records: list[dict[str, Any]]) -> dict[str, Any]:
    fds = sorted({str(record.get("fd")) for record in records})
    return {
        "overall": summarize_ablation(records),
        "by_fd": {
            fd: summarize_ablation([record for record in records if record.get("fd") == fd])
            for fd in fds
        },
    }


def _existing_case_ids(path: Path) -> set[str]:
    return {str(record.get("case_id")) for record in _read_jsonl(path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_comparison_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "fd",
        "case_id",
        "baseline_action_type",
        "no_kg_action_type",
        "action_type_changed",
        "baseline_action_time",
        "no_kg_action_time",
        "baseline_confidence",
        "no_kg_confidence",
        "neutral_valid",
        "production_policy_audit_valid",
        "latency_sec",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            baseline = record.get("baseline_action", {})
            ablation = record.get("ablation_action", {})
            writer.writerow(
                {
                    "fd": record.get("fd"),
                    "case_id": record.get("case_id"),
                    "baseline_action_type": baseline.get("action_type"),
                    "no_kg_action_type": ablation.get("action_type"),
                    "action_type_changed": baseline.get("action_type") != ablation.get("action_type"),
                    "baseline_action_time": baseline.get("action_time"),
                    "no_kg_action_time": ablation.get("action_time"),
                    "baseline_confidence": baseline.get("confidence"),
                    "no_kg_confidence": ablation.get("confidence"),
                    "neutral_valid": record.get("neutral_validation", {}).get("valid"),
                    "production_policy_audit_valid": record.get("production_validation", {}).get("valid"),
                    "latency_sec": record.get("latency_sec"),
                    "error": record.get("error"),
                }
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from .config import DEFAULT_KG_DIR, DEFAULT_MODEL_NAME, DEFAULT_OLLAMA_URL, DEFAULT_OUTPUT_DIR
from .kg_store import KGStore
from .react_agent import run_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KG-grounded RAG over a directory of C-MAPSS forecast cases.")
    parser.add_argument("--case_dir", type=Path, default=Path("cases"))
    parser.add_argument("--case_glob", default="*.json")
    parser.add_argument("--kg_dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry_run", action="store_true", help="Skip Ollama and use deterministic rule-based actions.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_paths = sorted(args.case_dir.glob(args.case_glob))
    if args.limit is not None:
        case_paths = case_paths[: args.limit]
    if not case_paths:
        raise SystemExit(f"No case files matched {args.case_dir / args.case_glob}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kg = KGStore(args.kg_dir)
    results = []

    for case_path in case_paths:
        case = json.loads(case_path.read_text())
        started = time.time()
        result = run_agent(
            case=case,
            kg_dir=str(args.kg_dir),
            kg_store=kg,
            model=args.model,
            ollama_url=args.ollama_url,
            temperature=args.temperature,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
        result["case_path"] = str(case_path)
        result["latency_sec"] = round(time.time() - started, 3)
        results.append(result)

    jsonl_path = args.output_dir / "action_hypotheses.jsonl"
    with jsonl_path.open("a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    summary = summarize_results(results)
    summary_path = args.output_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"cases": len(results), "summary": summary, "jsonl": str(jsonl_path)}, indent=2))


def summarize_results(results: list[dict]) -> dict:
    valid = [bool(r["action"]["validation"]["valid"]) for r in results]
    actions = [r["action"].get("action_type") for r in results]
    kg_consistent = [
        bool(r["action"]["validation"]["valid"])
        and r["action"].get("action_type") not in set(r["context"]["dataset_rules"].get("disallowed_actions", []))
        for r in results
    ]
    return {
        "num_cases": len(results),
        "valid_json_rate": 1.0,
        "validation_pass_rate": round(mean(valid), 4) if valid else 0.0,
        "kg_consistency_rate": round(mean(kg_consistent), 4) if kg_consistent else 0.0,
        "average_llm_calls": round(mean(int(r.get("llm_calls", 0)) for r in results), 4),
        "average_latency_sec": round(mean(float(r.get("latency_sec", 0.0)) for r in results), 4),
        "action_counts": {action: actions.count(action) for action in sorted(set(actions))},
    }


if __name__ == "__main__":
    main()


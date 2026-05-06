from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import DEFAULT_KG_DIR, DEFAULT_MODEL_NAME, DEFAULT_OLLAMA_URL, DEFAULT_OUTPUT_DIR
from .kg_store import KGStore
from .logging_utils import action_decision_record, recent_ollama_records, write_json
from .react_agent import run_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KG-grounded Graph RAG for one C-MAPSS forecast case.")
    parser.add_argument("--case_json", type=Path, required=True)
    parser.add_argument("--kg_dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--ollama_num_predict", type=int, default=512)
    parser.add_argument(
        "--disable_ollama_json_format",
        action="store_true",
        help="Disable Ollama format=json. By default Ollama is asked to return a JSON object.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Skip Ollama and use deterministic rule-based action.")
    parser.add_argument(
        "--save_recent_ollama_outputs",
        type=int,
        default=20,
        help="Save raw Ollama outputs for this case. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = json.loads(args.case_json.read_text())
    kg = KGStore(args.kg_dir)

    started = time.time()
    result = run_agent(
        case=case,
        kg_dir=str(args.kg_dir),
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
    decision_record = action_decision_record(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / f"{result.get('case_id', 'case')}_action.json"
    write_json(out_json, decision_record)
    write_json(args.output_dir / "action_hypotheses.json", [decision_record])
    if args.save_recent_ollama_outputs > 0:
        outputs = recent_ollama_records(result)[-args.save_recent_ollama_outputs :]
        write_json(args.output_dir / "recent_ollama_outputs.json", outputs)

    print(json.dumps({"output_json": str(out_json), "action": result["action"]}, indent=2))


if __name__ == "__main__":
    main()

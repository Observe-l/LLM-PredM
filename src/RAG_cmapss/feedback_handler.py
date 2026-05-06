from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_KG_DIR
from .reflection_memory import append_reflection_rule, feedback_to_rule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a KG reflection rule from structured feedback.")
    parser.add_argument("--case_json", type=Path, required=True)
    parser.add_argument("--action_json", type=Path, required=True)
    parser.add_argument("--feedback_json", type=Path, required=True)
    parser.add_argument("--kg_dir", type=Path, default=DEFAULT_KG_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = json.loads(args.case_json.read_text())
    action_payload = json.loads(args.action_json.read_text())
    action = action_payload.get("action", action_payload)
    feedback = json.loads(args.feedback_json.read_text())
    rule = feedback_to_rule(feedback, case, action)
    if rule is None:
        raise SystemExit(f"No reflection rule defined for feedback_label={feedback.get('feedback_label')!r}")
    path = append_reflection_rule(args.kg_dir, rule)
    print(json.dumps({"updated_reflection_rules": str(path), "rule": rule}, indent=2))


if __name__ == "__main__":
    main()


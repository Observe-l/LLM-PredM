from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from .ollama_client import extract_json


def write_json(path: Path, item: Any) -> None:
    path.write_text(json.dumps(item, indent=2, ensure_ascii=False))


def action_decision_record(result: dict[str, Any]) -> dict[str, Any]:
    """Keep decision logs focused; raw prompts belong in recent_ollama_outputs."""
    return {key: value for key, value in result.items() if key != "raw_outputs"}


def recent_ollama_records(result: dict[str, Any], case: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, raw in enumerate(result.get("raw_outputs", [])):
        if raw == "<dry_run rule_based_action>":
            continue

        if isinstance(raw, dict):
            raw_output = str(raw.get("raw_output", ""))
            stage = raw.get("stage", f"output_{idx}")
            is_repair = stage == "repair" or idx > 0
            messages = raw.get("messages")
            metadata = {k: v for k, v in raw.items() if k not in {"raw_output", "messages"}}
        else:
            raw_output = str(raw)
            stage = f"output_{idx}"
            is_repair = idx > 0
            messages = None
            metadata = {}

        record: dict[str, Any] = {
            "case_id": result.get("case_id"),
            "llm_output_index": idx,
            "stage": stage,
            "is_repair_output": is_repair,
            "prompt_messages": readable_messages(messages),
            "raw_output_lines": raw_output.splitlines(),
            "raw_output_json": parse_json_or_none(raw_output),
            "parsed_action": result.get("action"),
            "validation": result.get("action", {}).get("validation"),
            "metadata": metadata,
        }
        if case is not None:
            record.update(
                {
                    "dataset_subset": case.get("dataset_subset"),
                    "unit_id": case.get("unit_id"),
                    "cutoff_cycle": case.get("cutoff_cycle"),
                }
            )
        records.append(record)
    return records


def append_recent_ollama_records(
    buffer: deque[dict[str, Any]],
    max_items: int,
    result: dict[str, Any],
    case: dict[str, Any],
) -> None:
    if max_items <= 0:
        return
    for record in recent_ollama_records(result, case=case):
        buffer.append(record)


def readable_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    readable: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            readable.append({"role": None, "content_lines": str(message).splitlines()})
            continue
        content = str(message.get("content", ""))
        readable.append(
            {
                "role": message.get("role"),
                "content_line_count": len(content.splitlines()),
                "content_lines": content.splitlines(),
            }
        )
    return readable


def parse_json_or_none(text: str) -> dict[str, Any] | None:
    if not text.strip():
        return None
    try:
        return extract_json(text)
    except Exception:
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

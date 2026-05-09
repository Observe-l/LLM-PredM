from __future__ import annotations

import json
from pathlib import Path
from typing import Any


THRESHOLD_DELTAS = {
    "small": {"q95_excess_min": 0.03, "q99_excess_min": 0.02, "persistent_duration_min": 1},
    "medium": {"q95_excess_min": 0.05, "q99_excess_min": 0.03, "persistent_duration_min": 2},
    "large": {"q95_excess_min": 0.10, "q99_excess_min": 0.05, "persistent_duration_min": 4},
}


def build_update_operation(update_result: dict[str, Any]) -> dict[str, Any]:
    threshold_update = str(update_result.get("threshold_update", "unchanged"))
    strength = str(update_result.get("update_strength", "small"))
    deltas = dict(THRESHOLD_DELTAS.get(strength, THRESHOLD_DELTAS["small"]))
    sign = 1.0 if threshold_update == "higher" else -1.0 if threshold_update == "lower" else 0.0
    return {
        "operator": "threshold_timing_update",
        "threshold_update": threshold_update,
        "timing_update": update_result.get("timing_update", "keep"),
        "component_preference_update": update_result.get("component_preference_update", "unchanged"),
        "update_strength": strength,
        "threshold_delta": {
            "q95_excess_min": round(sign * deltas["q95_excess_min"], 6),
            "q99_excess_min": round(sign * deltas["q99_excess_min"], 6),
            "persistent_duration_min": int(sign * deltas["persistent_duration_min"]),
        },
        "note": (
            "This operation is logged as adaptation evidence for LightGBM retraining; "
            "reflection memory is not used directly as action evidence."
        ),
    }


def append_update_operation(path: str | Path, record: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return path

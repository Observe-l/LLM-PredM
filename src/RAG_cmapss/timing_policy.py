from __future__ import annotations

import re
from typing import Any


_T_PLUS = re.compile(r"t\+(\d+)")


def recommended_maintenance_time(case: dict[str, Any]) -> str:
    """Return the forecast-grounded maintenance cycle used by every decision path.

    The calibrated risk peak is the primary timing anchor. Other forecast-state
    crossings are used only when the peak cycle is unavailable or malformed.
    """

    horizon = case.get("forecast_horizon", {})
    start = int(horizon.get("start", 1))
    end = int(horizon.get("end", 20))
    risk = case.get("risk_statistics", {})
    summary = case.get("forecast_summary", {})
    candidates = [
        ("risk_statistics.peak_score_cycle", risk.get("peak_score_cycle")),
        ("forecast_summary.peak_score_cycle", summary.get("peak_score_cycle")),
        ("forecast_summary.first_critical_crossing_cycle", summary.get("first_critical_crossing_cycle")),
        ("forecast_summary.first_persistent_pattern_cycle", summary.get("first_persistent_pattern_cycle")),
        ("forecast_summary.first_warning_crossing_cycle", summary.get("first_warning_crossing_cycle")),
    ]
    for _, value in candidates:
        cycle = _relative_cycle(value)
        if cycle is not None and start <= cycle <= end:
            return f"t+{cycle}"
    return f"t+{start}"


def maintenance_timing_profile(case: dict[str, Any]) -> dict[str, Any]:
    risk = case.get("risk_statistics", {})
    summary = case.get("forecast_summary", {})
    recommended = recommended_maintenance_time(case)
    peak = risk.get("peak_score_cycle") or summary.get("peak_score_cycle")
    return {
        "recommended_maintenance_time": recommended,
        "primary_timing_anchor": "peak_score_cycle",
        "peak_score_cycle": peak,
        "first_critical_crossing_cycle": summary.get("first_critical_crossing_cycle"),
        "first_persistent_pattern_cycle": summary.get("first_persistent_pattern_cycle"),
        "rule": (
            "For a maintenance action, action_time must equal recommended_maintenance_time. "
            "The forecast-horizon end is a monitoring revisit time, not a maintenance default."
        ),
    }


def _relative_cycle(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = _T_PLUS.fullmatch(value.strip())
        if match:
            return int(match.group(1))
    return None

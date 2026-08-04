from __future__ import annotations

import re
from typing import Any


_T_PLUS = re.compile(r"t\+(\d+)")

PEAK_OFFSET_CYCLES = {
    "none": 0,
    "small": 5,
    "median": 10,
    "large": 15,
}


def recommended_maintenance_time(
    case: dict[str, Any],
    llm_policy: dict[str, Any] | None = None,
) -> str:
    """Return the forecast-grounded maintenance cycle used by every decision path.

    The calibrated risk peak is the primary timing anchor. Other forecast-state
    crossings are used only when the peak cycle is unavailable or malformed.
    """

    horizon = case.get("forecast_horizon", {})
    start = int(horizon.get("start", 1))
    end = int(horizon.get("end", 20))
    risk = case.get("risk_statistics", {})
    summary = case.get("forecast_summary", {})
    offset_level = peak_offset_level(llm_policy)
    requested_offset = PEAK_OFFSET_CYCLES[offset_level]
    peak_candidates = [
        ("risk_statistics.peak_score_cycle", risk.get("peak_score_cycle")),
        ("forecast_summary.peak_score_cycle", summary.get("peak_score_cycle")),
    ]
    for _, value in peak_candidates:
        cycle = _relative_cycle(value)
        if cycle is not None and start <= cycle <= end:
            adjusted = max(start, cycle - requested_offset)
            return f"t+{adjusted}"
    return f"t+{start}"


def recommended_monitoring_time(
    case: dict[str, Any],
    llm_policy: dict[str, Any] | None = None,
) -> str:
    horizon = case.get("forecast_horizon", {})
    start = int(horizon.get("start", 1))
    end = int(horizon.get("end", 20))
    interval = _safe_interval((llm_policy or {}).get("monitoring_interval", end), end)
    return f"t+{min(max(interval, start), end)}"


def maintenance_timing_profile(
    case: dict[str, Any],
    llm_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    risk = case.get("risk_statistics", {})
    summary = case.get("forecast_summary", {})
    recommended = recommended_maintenance_time(case, llm_policy)
    monitoring = recommended_monitoring_time(case, llm_policy)
    peak = risk.get("peak_score_cycle") or summary.get("peak_score_cycle")
    peak_cycle = _relative_cycle(peak)
    recommended_cycle = _relative_cycle(recommended)
    offset_level = peak_offset_level(llm_policy)
    requested_offset = PEAK_OFFSET_CYCLES[offset_level]
    effective_offset = (
        max(peak_cycle - recommended_cycle, 0)
        if peak_cycle is not None and recommended_cycle is not None
        else 0
    )
    return {
        "recommended_maintenance_time": recommended,
        "recommended_monitoring_time": monitoring,
        "primary_timing_anchor": "peak_score_cycle",
        "peak_score_cycle": peak,
        "peak_offset_level": offset_level,
        "requested_offset_cycles": requested_offset,
        "effective_offset_cycles": effective_offset,
        "offset_clamped": effective_offset < requested_offset,
        "rule": (
            "For a maintenance action, action_time must equal recommended_maintenance_time. "
            "For monitoring, action_time must equal recommended_monitoring_time."
        ),
    }


def peak_offset_level(llm_policy: dict[str, Any] | None = None) -> str:
    level = str((llm_policy or {}).get("peak_offset_level", "none"))
    return level if level in PEAK_OFFSET_CYCLES else "none"


def _relative_cycle(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = _T_PLUS.fullmatch(value.strip())
        if match:
            return int(match.group(1))
    return None


def _safe_interval(value: Any, default: int) -> int:
    try:
        interval = int(value)
    except (TypeError, ValueError):
        return int(default)
    return interval if interval in {5, 10, 20} else int(default)

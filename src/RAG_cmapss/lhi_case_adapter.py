from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HPC_EVIDENCE_SENSORS = {"S7", "S11", "S3", "S9", "S14"}
FAN_EVIDENCE_SENSORS = {"S8", "S13", "S15"}
CONFLICT_EVIDENCE_SENSORS = {"S4", "S12", "S20", "S21"}


DEFAULT_THRESHOLD_CONFIG = {
    "FD001": {"warning_score": None, "critical_score": None},
    "FD002": {"warning_score": None, "critical_score": None},
    "FD003": {"warning_score": None, "critical_score": None},
    "FD004": {"warning_score": None, "critical_score": None},
}


GROUP_COLS = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle"]


def load_threshold_config(path: str | Path | None) -> dict[str, dict[str, float | None]]:
    if path is None:
        return DEFAULT_THRESHOLD_CONFIG
    loaded = json.loads(Path(path).read_text())
    config = {fd: dict(DEFAULT_THRESHOLD_CONFIG.get(fd, {})) for fd in DEFAULT_THRESHOLD_CONFIG}
    for fd, values in loaded.items():
        config.setdefault(fd, {}).update(values)
    return config


LHI_SCORE_USECOLS = [
    "covariate_mode",
    "fd",
    "unit_id",
    "cutoff_cycle",
    "forecast_start_cycle",
    "cycle",
    "prediction_length",
    "d_mae",
    "d_rmse",
    "lhi_mae",
    "lhi_rmse",
    "top_drift_sensors",
    "top_drift_sensor_rmse_values",
    "top_drift_sensor_mae_values",
    "window_top_drift_sensors",
    "window_top_sensor_rmse_values",
    "d_mae_roll_mean",
    "d_rmse_roll_mean",
    "lhi_mae_roll_mean",
    "lhi_rmse_roll_mean",
]


def load_lhi_frames(lhi_dir: str | Path, load_top_drift_detail: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    lhi_dir = Path(lhi_dir)
    scores_path = lhi_dir / "lhi_scores.csv"
    top_path = lhi_dir / "top_drift_sensors.csv"
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    if load_top_drift_detail and not top_path.exists():
        raise FileNotFoundError(top_path)
    available_cols = list(pd.read_csv(scores_path, nrows=0).columns)
    if "window_top_drift_sensors" not in available_cols:
        warnings.warn(
            f"{scores_path} predates window-level sensor RMSE ranking; "
            "dominant_top_sensors will use the legacy per-cycle top-sensor aggregation.",
            UserWarning,
            stacklevel=2,
        )
    usecols = [col for col in LHI_SCORE_USECOLS if col in available_cols]
    scores = pd.read_csv(scores_path, usecols=usecols)
    if load_top_drift_detail:
        top = pd.read_csv(top_path)
    else:
        top = pd.DataFrame(
            columns=[
                *GROUP_COLS,
                "cycle",
                "top_drift_rank",
                "sensor",
                "sensor_d_rmse",
                "sensor_d_mae",
            ]
        )
    _normalize_sensor_columns(scores)
    _normalize_sensor_columns(top)
    return scores, top


def iter_lhi_windows(scores: pd.DataFrame, fds: list[str] | None = None):
    data = scores.copy()
    if fds:
        data = data[data["fd"].isin(fds)].copy()
    order = {fd: idx for idx, fd in enumerate(["FD001", "FD002", "FD003", "FD004"])}
    data["_fd_order"] = data["fd"].map(order).fillna(999)
    data = data.sort_values(["_fd_order", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle"])
    for key, group in data.groupby(GROUP_COLS, sort=False):
        yield key, group.drop(columns=["_fd_order"], errors="ignore").copy()


def build_forecast_case(
    window: pd.DataFrame,
    top_drift: pd.DataFrame,
    score_col: str,
    raw_score_col: str,
    lhi_col: str,
    threshold_config: dict[str, dict[str, float | None]],
    window_detail_dir: str | Path | None = None,
    engine_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if window.empty:
        raise ValueError("Cannot build ForecastCase from an empty LHI window.")
    missing = [col for col in GROUP_COLS + ["cycle", score_col] if col not in window.columns]
    if missing:
        raise ValueError(f"LHI window missing required columns: {missing}")

    window = window.sort_values(["cycle"]).copy()
    first = window.iloc[0]
    fd_name = str(first["fd"])
    unit_id = int(first["unit_id"])
    cutoff_cycle = int(first["cutoff_cycle"])
    forecast_start_cycle = int(first["forecast_start_cycle"])
    prediction_length = int(window.get("prediction_length", pd.Series([len(window)])).max())
    h_start = max(1, int(window["cycle"].min()) - cutoff_cycle)
    h_end = max(h_start, int(window["cycle"].max()) - cutoff_cycle)

    score_series = pd.to_numeric(window[score_col], errors="coerce")
    if score_series.notna().sum() == 0:
        raise ValueError(f"No finite score values in {score_col}.")
    peak_idx = score_series.idxmax()
    peak_row = window.loc[peak_idx]
    peak_abs_cycle = int(peak_row["cycle"])

    top_rows = _window_top_rows(top_drift, first)
    dominant_top_sensors = _dominant_top_sensors(top_rows, window)
    top_sensor_at_peak = _top_sensors_at_cycle(top_rows, peak_abs_cycle)
    if not top_sensor_at_peak:
        top_sensor_at_peak = _parse_sensor_list(str(peak_row.get("top_drift_sensors", "")))
    unit_past = _unit_past_statistics(engine_history, score_col, cutoff_cycle)
    if unit_past.get("unit_past_q95") is not None:
        unit_past["peak_minus_unit_q95"] = _finite_float(float(peak_row.get(score_col)) - float(unit_past["unit_past_q95"]))
    if unit_past.get("unit_past_q99") is not None:
        unit_past["peak_minus_unit_q99"] = _finite_float(float(peak_row.get(score_col)) - float(unit_past["unit_past_q99"]))

    thresholds = threshold_config.get(fd_name, {})
    warning_score = thresholds.get("warning_score")
    critical_score = thresholds.get("critical_score")
    if warning_score is None:
        warning_score = unit_past.get("unit_past_q95")
    if critical_score is None:
        critical_score = unit_past.get("unit_past_q99")

    warning_rel = _first_crossing(window, score_col, warning_score, cutoff_cycle)
    critical_rel = _first_crossing(window, score_col, critical_score, cutoff_cycle)
    persistent_duration = _persistent_duration(window, score_col, warning_score)
    first_persistent = warning_rel if persistent_duration > 1 else None
    trend_stats = _trend_statistics(window, score_col, cutoff_cycle, unit_past)
    multi_score_stats = _multi_score_statistics(window, score_col, raw_score_col, lhi_col)
    sensor_stats = _sensor_evidence_statistics(top_rows, window, top_sensor_at_peak)

    case_id = f"ForecastCase_{fd_name}_Engine{unit_id}_Cycle{cutoff_cycle}"
    detail_path = None
    if window_detail_dir is not None:
        detail_path = Path(window_detail_dir) / f"{case_id}_window.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(
            json.dumps(
                build_window_detail(case_id, window, top_rows, score_col, raw_score_col, lhi_col, cutoff_cycle),
                indent=2,
            )
        )

    return {
        "case_id": case_id,
        "dataset_subset": fd_name,
        "unit_id": unit_id,
        "cutoff_cycle": cutoff_cycle,
        "covariate_mode": str(first["covariate_mode"]),
        "forecast_start_cycle": forecast_start_cycle,
        "forecast_horizon": {"start": h_start, "end": h_end, "text": f"t+{h_start} to t+{h_end}"},
        "score_source": {
            "score_name": score_col,
            "raw_score_name": raw_score_col,
            "lhi_name": lhi_col,
            "warning_score": warning_score,
            "critical_score": critical_score,
        },
        "forecast_summary": {
            "summary_id": f"ForecastSummary_{fd_name}_Engine{unit_id}_Cycle{cutoff_cycle}",
            "current_score": _finite_float(window.iloc[0].get(score_col)),
            "peak_score": _finite_float(peak_row.get(score_col)),
            "peak_score_cycle": _relative_cycle(peak_abs_cycle, cutoff_cycle),
            "peak_score_abs_cycle": peak_abs_cycle,
            "score_trend": _score_trend(window, score_col),
            "first_warning_crossing_cycle": warning_rel,
            "first_critical_crossing_cycle": critical_rel,
            "persistent_high_risk_duration": persistent_duration,
            "first_persistent_pattern_cycle": first_persistent,
            "dominant_top_sensors": dominant_top_sensors,
            "top_sensor_at_peak_cycle": top_sensor_at_peak,
            "final_cycle_score": _finite_float(window.iloc[-1].get(score_col)),
            "preliminary_component_hint": None,
            "dominant_component_hypothesis": None,
        },
        "risk_statistics": {
            "score_col": score_col,
            "current_score": _finite_float(window.iloc[0].get(score_col)),
            "peak_score": _finite_float(peak_row.get(score_col)),
            "peak_score_cycle": _relative_cycle(peak_abs_cycle, cutoff_cycle),
            "final_score": _finite_float(window.iloc[-1].get(score_col)),
            "current_d_rmse": _finite_float(window.iloc[0].get(raw_score_col)),
            "peak_d_rmse": _finite_float(peak_row.get(raw_score_col)),
            "final_d_rmse": _finite_float(window.iloc[-1].get(raw_score_col)),
            **unit_past,
        },
        "trend_statistics": trend_stats,
        "multi_score_statistics": multi_score_stats,
        "sensor_evidence_statistics": sensor_stats,
        "key_cycles": _key_cycles(window, top_rows, score_col, cutoff_cycle, warning_rel, critical_rel, peak_abs_cycle),
        "forecast_window_detail_path": str(detail_path) if detail_path else None,
        "prediction_length": prediction_length,
    }


def build_current_lhi_case(
    *,
    window: pd.DataFrame,
    top_drift: pd.DataFrame,
    score_col: str,
    raw_score_col: str,
    lhi_col: str,
    threshold_config: dict[str, dict[str, float | None]],
    current_cycle: int,
    engine_history: pd.DataFrame | None = None,
    window_detail_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a case containing only the current LHI observation.

    This is deliberately separate from ``build_forecast_case`` so the normal
    forecast experiment remains unchanged.  The caller supplies the current
    cutoff cycle; the row is recovered from previously observed LHI windows,
    not from the future rows in the current forecast window.
    """
    if window.empty:
        raise ValueError("Cannot build a current-LHI case from an empty window.")
    source = engine_history if engine_history is not None else pd.DataFrame()
    candidates = source[source["cycle"].astype(int) == int(current_cycle)].copy() if not source.empty and "cycle" in source else pd.DataFrame()
    if candidates.empty:
        candidates = window[window["cycle"].astype(int) == int(current_cycle)].copy()
    if candidates.empty:
        raise ValueError(
            f"No observed LHI row found for current_cycle={current_cycle}; "
            "current-only decisions cannot use a future forecast row."
        )
    current = candidates.sort_values(["cutoff_cycle", "forecast_start_cycle"]).iloc[-1]
    fd_name = str(current["fd"])
    unit_id = int(current["unit_id"])
    # The source row normally belongs to the preceding forecast window
    # (e.g. observed cycle 121 is stored in the cutoff-120 window).  For a
    # current-only decision, the decision clock is the observed cycle itself.
    cutoff_cycle = int(current_cycle)
    current_lhi = _finite_float(current.get(lhi_col))
    current_raw = _finite_float(current.get(raw_score_col))
    current_mae = _finite_float(current.get("d_mae"))
    top_rows = _window_top_rows(top_drift, current)
    top_sensor_rows = _top_rows_for_cycle(top_rows, int(current_cycle))
    if not top_sensor_rows:
        rmse_values = _parse_sensor_value_map(str(current.get("top_drift_sensor_rmse_values", "")))
        mae_values = _parse_sensor_value_map(str(current.get("top_drift_sensor_mae_values", "")))
        top_sensor_rows = [
            {
                "sensor": sensor,
                "sensor_d_rmse": rmse_values.get(sensor),
                "sensor_d_mae": mae_values.get(sensor, rmse_values.get(sensor)),
            }
            for sensor in rmse_values
        ]
    top_sensor_rows = top_sensor_rows[:8]
    dominant_sensors = [str(item.get("sensor")) for item in top_sensor_rows if item.get("sensor")]
    sensor_presence = {sensor: 1.0 for sensor in dominant_sensors}
    detail = {
        "case_id": f"CurrentLHICase_{fd_name}_Engine{unit_id}_Cycle{current_cycle}",
        "current_observation": {
            "cycle": int(current_cycle),
            "lhi": current_lhi,
            "d_rmse": current_raw,
            "d_mae": current_mae,
            "sensor_contribution_ranking": top_sensor_rows,
        },
    }
    detail_path = None
    if window_detail_dir is not None:
        detail_path = Path(window_detail_dir) / f"{detail['case_id']}_current.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(json.dumps(detail, indent=2))

    thresholds = threshold_config.get(fd_name, {})
    return {
        "case_id": detail["case_id"],
        "dataset_subset": fd_name,
        "unit_id": unit_id,
        "cutoff_cycle": cutoff_cycle,
        "covariate_mode": str(current.get("covariate_mode", "unknown")),
        "forecast_start_cycle": None,
        "forecast_horizon": {
            "start": 1,
            "end": 20,
            "text": "t+1 to t+20 action-time range only; no future sensor or LHI values are provided",
        },
        "information_mode": "current_lhi_only",
        "score_source": {
            "score_name": score_col,
            "raw_score_name": raw_score_col,
            "lhi_name": lhi_col,
            "warning_score": thresholds.get("warning_score"),
            "critical_score": thresholds.get("critical_score"),
        },
        "current_observation": detail["current_observation"],
        "forecast_summary": {
            "summary_id": f"CurrentLHISummary_{fd_name}_Engine{unit_id}_Cycle{current_cycle}",
            "current_score": current_lhi,
            "peak_score": current_lhi,
            "peak_score_cycle": None,
            "peak_score_abs_cycle": int(current_cycle),
            "score_trend": "current_only",
            "first_warning_crossing_cycle": None,
            "first_critical_crossing_cycle": None,
            "persistent_high_risk_duration": None,
            "dominant_top_sensors": dominant_sensors,
            "top_sensor_at_peak_cycle": dominant_sensors,
            "final_cycle_score": current_lhi,
        },
        "risk_statistics": {
            "score_col": score_col,
            "current_score": current_lhi,
            "peak_score": current_lhi,
            "peak_score_cycle": None,
            "final_score": current_lhi,
            "current_d_rmse": current_raw,
            "peak_d_rmse": current_raw,
            "final_d_rmse": current_raw,
            "unit_past_count": 0,
            "unit_past_context_reliable": False,
            "unit_past_q95": None,
            "unit_past_q99": None,
            "peak_minus_unit_q95": None,
            "peak_minus_unit_q99": None,
        },
        "trend_statistics": {"mode": "current_only"},
        "multi_score_statistics": {
            "d_rmse_lhi_consistency": "not_computed_current_only",
        },
        "sensor_evidence_statistics": {
            "mode": "current_cycle_only",
            "dominant_top_sensors": dominant_sensors,
            "top_sensor_at_peak_cycle": dominant_sensors,
            "sensor_presence_ratio": sensor_presence,
            "sensor_mean_rank": {sensor: idx + 1 for idx, sensor in enumerate(dominant_sensors)},
            "sensor_pattern_stability": None,
            "current_sensor_contribution_ranking": top_sensor_rows,
        },
        "key_cycles": [{
            "cycle": int(current_cycle),
            "relative_cycle": "current",
            "score": current_lhi,
            "raw_score": current_raw,
            "lhi_score": current_lhi,
            "d_rmse": current_raw,
            "d_mae": current_mae,
            "top_drift_sensors": dominant_sensors,
            "top_drift_sensor_rmse_values": {
                str(item.get("sensor")): item.get("sensor_d_rmse") for item in top_sensor_rows
            },
        }],
        "forecast_window_detail_path": str(detail_path) if detail_path else None,
        "prediction_length": 0,
    }


def _unit_past_statistics(engine_history: pd.DataFrame | None, score_col: str, cutoff_cycle: int) -> dict[str, Any]:
    base = {
        "unit_past_count": 0,
        "unit_past_mean": None,
        "unit_past_std": None,
        "unit_past_q95": None,
        "unit_past_q99": None,
        "peak_minus_unit_q95": None,
        "peak_minus_unit_q99": None,
        "unit_past_context_reliable": False,
    }
    if engine_history is None or engine_history.empty or score_col not in engine_history:
        return base
    history = engine_history.copy()
    if "cutoff_cycle" in history:
        history = history[pd.to_numeric(history["cutoff_cycle"], errors="coerce") < cutoff_cycle]
    if "cycle" in history:
        history = history[pd.to_numeric(history["cycle"], errors="coerce") <= cutoff_cycle]
        history = history.sort_values(["cycle", "cutoff_cycle"]).drop_duplicates("cycle", keep="last")
    values = pd.to_numeric(history[score_col], errors="coerce").dropna()
    if values.empty:
        return base
    count = int(values.shape[0])
    reliable = count >= 10
    return {
        "unit_past_count": count,
        "unit_past_mean": _finite_float(values.mean()),
        "unit_past_std": _finite_float(values.std(ddof=0)),
        "unit_past_q95": _finite_float(values.quantile(0.95)),
        "unit_past_q99": _finite_float(values.quantile(0.99)),
        "peak_minus_unit_q95": None,
        "peak_minus_unit_q99": None,
        "unit_past_context_reliable": reliable,
    }


def _trend_statistics(window: pd.DataFrame, score_col: str, cutoff_cycle: int, unit_past: dict[str, Any]) -> dict[str, Any]:
    values = pd.to_numeric(window[score_col], errors="coerce")
    cycles = pd.to_numeric(window["cycle"], errors="coerce") - cutoff_cycle
    mask = values.notna() & cycles.notna()
    y = values[mask].to_numpy(dtype=float)
    x = cycles[mask].to_numpy(dtype=float)
    if len(y) == 0:
        return {}
    slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
    diffs = np.diff(y)
    q95 = unit_past.get("unit_past_q95")
    q99 = unit_past.get("unit_past_q99")
    return {
        "slope": _finite_float(slope),
        "delta_score": _finite_float(y[-1] - y[0]),
        "relative_increase": _finite_float((y[-1] - y[0]) / max(abs(y[0]), 1e-9)),
        "monotonicity": _finite_float(float((diffs >= 0).mean()) if len(diffs) else 1.0),
        "volatility": _finite_float(float(np.std(diffs)) if len(diffs) else 0.0),
        "duration_above_unit_q95": _duration_above_threshold(y, q95),
        "duration_above_unit_q99": _duration_above_threshold(y, q99),
        "area_above_unit_q95": _area_above_threshold(y, q95),
        "area_above_unit_q99": _area_above_threshold(y, q99),
        "first_unit_q95_crossing_cycle": _first_array_crossing(x, y, q95),
        "first_unit_q99_crossing_cycle": _first_array_crossing(x, y, q99),
    }


def _multi_score_statistics(window: pd.DataFrame, score_col: str, raw_score_col: str, lhi_col: str) -> dict[str, Any]:
    stats = {
        score_col: _series_stats(window, score_col),
        raw_score_col: _series_stats(window, raw_score_col),
        lhi_col: _series_stats(window, lhi_col),
    }
    score_slope = stats.get(score_col, {}).get("slope")
    raw_slope = stats.get(raw_score_col, {}).get("slope")
    if score_slope is None or raw_slope is None:
        consistency = "unknown"
    elif score_slope > 0 and raw_slope > 0:
        consistency = "consistent_increasing"
    elif score_slope < 0 and raw_slope < 0:
        consistency = "consistent_decreasing"
    else:
        consistency = "mixed"
    stats["d_rmse_lhi_consistency"] = consistency
    return stats


def _sensor_evidence_statistics(top_rows: pd.DataFrame, window: pd.DataFrame, top_sensor_at_peak: list[str]) -> dict[str, Any]:
    cycle_sets: list[list[str]] = []
    cycle_ranks: dict[str, list[int]] = defaultdict(list)
    for row in window.sort_values("cycle").itertuples(index=False):
        abs_cycle = int(getattr(row, "cycle"))
        sensors = _top_sensors_at_cycle(top_rows, abs_cycle)
        if not sensors:
            sensors = _parse_sensor_list(str(getattr(row, "top_drift_sensors", "")))
        cycle_sets.append(sensors)
        for rank, sensor in enumerate(sensors, start=1):
            cycle_ranks[sensor].append(rank)
    n = max(len(cycle_sets), 1)
    presence = {sensor: round(len(ranks) / n, 4) for sensor, ranks in sorted(cycle_ranks.items())}
    mean_rank = {sensor: round(float(np.mean(ranks)), 4) for sensor, ranks in sorted(cycle_ranks.items())}
    stability = _sensor_pattern_stability(cycle_sets)
    return {
        "dominant_top_sensors": [s for s, _ in sorted(presence.items(), key=lambda item: item[1], reverse=True)[:5]],
        "top_sensor_at_peak_cycle": top_sensor_at_peak,
        "sensor_presence_ratio": presence,
        "sensor_mean_rank": mean_rank,
        "sensor_pattern_stability": stability,
        "hpc_sensor_presence_ratio": _group_presence(presence, HPC_EVIDENCE_SENSORS),
        "fan_sensor_presence_ratio": _group_presence(presence, FAN_EVIDENCE_SENSORS),
        "conflict_sensor_presence_ratio": _group_presence(presence, CONFLICT_EVIDENCE_SENSORS),
    }


def _duration_above_threshold(values: np.ndarray, threshold: Any) -> int | None:
    threshold = _finite_float(threshold)
    if threshold is None:
        return None
    return int((values > threshold).sum())


def _area_above_threshold(values: np.ndarray, threshold: Any) -> float | None:
    threshold = _finite_float(threshold)
    if threshold is None:
        return None
    return _finite_float(np.maximum(values - threshold, 0.0).sum())


def _first_array_crossing(cycles: np.ndarray, values: np.ndarray, threshold: Any) -> str | None:
    threshold = _finite_float(threshold)
    if threshold is None:
        return None
    indices = np.where(values > threshold)[0]
    if len(indices) == 0:
        return None
    return f"t+{max(1, int(cycles[int(indices[0])]))}"


def _series_stats(window: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in window:
        return {"peak": None, "final": None, "slope": None}
    values = pd.to_numeric(window[col], errors="coerce")
    cycles = pd.to_numeric(window["cycle"], errors="coerce")
    mask = values.notna() & cycles.notna()
    if not mask.any():
        return {"peak": None, "final": None, "slope": None}
    y = values[mask].to_numpy(dtype=float)
    x = cycles[mask].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
    return {"peak": _finite_float(np.max(y)), "final": _finite_float(y[-1]), "slope": _finite_float(slope)}


def _sensor_pattern_stability(cycle_sets: list[list[str]]) -> float | None:
    if len(cycle_sets) < 2:
        return 1.0 if cycle_sets else None
    scores = []
    for left, right in zip(cycle_sets, cycle_sets[1:]):
        lset, rset = set(left), set(right)
        if not lset and not rset:
            scores.append(1.0)
        elif lset or rset:
            scores.append(len(lset & rset) / len(lset | rset))
    return _finite_float(float(np.mean(scores)) if scores else 1.0)


def _group_presence(presence: dict[str, float], sensors: set[str]) -> float:
    values = [presence.get(sensor, 0.0) for sensor in sensors]
    return round(max(values) if values else 0.0, 4)


def build_window_detail(
    case_id: str,
    window: pd.DataFrame,
    top_rows: pd.DataFrame,
    score_col: str,
    raw_score_col: str,
    lhi_col: str,
    cutoff_cycle: int,
) -> dict[str, Any]:
    rows = []
    for row in window.sort_values("cycle").itertuples(index=False):
        record = row._asdict()
        abs_cycle = int(record["cycle"])
        top_for_cycle = _top_rows_for_cycle(top_rows, abs_cycle)
        if not top_for_cycle:
            rmse_values = _parse_sensor_value_map(str(record.get("top_drift_sensor_rmse_values", "")))
            mae_values = _parse_sensor_value_map(str(record.get("top_drift_sensor_mae_values", "")))
            top_for_cycle = [
                {
                    "sensor": sensor,
                    "sensor_d_rmse": rmse_values.get(sensor),
                    "sensor_d_mae": mae_values.get(sensor),
                    "rank": rank,
                }
                for rank, sensor in enumerate(_parse_sensor_list(str(record.get("top_drift_sensors", ""))), start=1)
            ]
        rows.append(
            {
                "relative_cycle": _relative_cycle(abs_cycle, cutoff_cycle),
                "abs_cycle": abs_cycle,
                "score": _finite_float(record.get(score_col)),
                "raw_score": _finite_float(record.get(raw_score_col)),
                "lhi_score": _finite_float(record.get(lhi_col)),
                "d_rmse": _finite_float(record.get("d_rmse")),
                "d_rmse_roll_mean": _finite_float(record.get("d_rmse_roll_mean")),
                "lhi_rmse": _finite_float(record.get("lhi_rmse")),
                "lhi_rmse_roll_mean": _finite_float(record.get("lhi_rmse_roll_mean")),
                "top_drift_sensors": [r["sensor"] for r in top_for_cycle],
                "top_drift_sensor_rmse_values": {r["sensor"]: r["sensor_d_rmse"] for r in top_for_cycle},
            }
        )
    return {"case_id": case_id, "score_series": rows}


def case_peak_lhi(window: pd.DataFrame, lhi_gate_col: str) -> float:
    if lhi_gate_col not in window.columns:
        raise ValueError(f"Missing LHI gate column: {lhi_gate_col}")
    values = pd.to_numeric(window[lhi_gate_col], errors="coerce")
    return float(values.max()) if values.notna().any() else float("nan")


def _window_top_rows(top_drift: pd.DataFrame, first_row: pd.Series) -> pd.DataFrame:
    mask = np.ones(len(top_drift), dtype=bool)
    for col in GROUP_COLS:
        mask &= top_drift[col].to_numpy() == first_row[col]
    return top_drift.loc[mask].copy()


def _dominant_top_sensors(top_rows: pd.DataFrame, window: pd.DataFrame, limit: int = 3) -> list[str]:
    window_ranked_sensors = _window_ranked_sensors(window, limit)
    if window_ranked_sensors:
        return window_ranked_sensors

    # Backward compatibility for LHI outputs created before window-level sensor
    # RMSE was persisted. New LHI outputs always take the branch above.
    if top_rows.empty:
        counter: Counter[str] = Counter()
        for value in window.get("top_drift_sensors", []):
            counter.update(_parse_sensor_list(str(value)))
        return [s for s, _ in counter.most_common(limit)]

    scores: dict[str, float] = defaultdict(float)
    max_rmse = float(top_rows["sensor_d_rmse"].max()) if "sensor_d_rmse" in top_rows else 1.0
    max_rmse = max(max_rmse, 1e-12)
    for row in top_rows.itertuples(index=False):
        rank = float(getattr(row, "top_drift_rank", 99))
        rmse = float(getattr(row, "sensor_d_rmse", 0.0))
        scores[str(row.sensor)] += 0.5 + 0.3 * (1.0 / max(rank, 1.0)) + 0.2 * (rmse / max_rmse)
    return [sensor for sensor, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _window_ranked_sensors(window: pd.DataFrame, limit: int) -> list[str]:
    if "window_top_drift_sensors" not in window:
        return []

    rankings = {
        tuple(_parse_sensor_list(str(value)))
        for value in window["window_top_drift_sensors"]
        if _parse_sensor_list(str(value))
    }
    if not rankings:
        return []
    if len(rankings) != 1:
        raise ValueError("Forecast window contains inconsistent window_top_drift_sensors rankings.")
    return list(next(iter(rankings)))[:limit]


def _top_sensors_at_cycle(top_rows: pd.DataFrame, abs_cycle: int) -> list[str]:
    cycle_rows = _top_rows_for_cycle(top_rows, abs_cycle)
    return [r["sensor"] for r in cycle_rows]


def _top_rows_for_cycle(top_rows: pd.DataFrame, abs_cycle: int) -> list[dict[str, Any]]:
    if top_rows.empty:
        return []
    cycle_rows = top_rows[top_rows["cycle"] == abs_cycle].sort_values("top_drift_rank")
    return [
        {
            "sensor": str(row.sensor),
            "sensor_d_rmse": _finite_float(row.sensor_d_rmse),
            "sensor_d_mae": _finite_float(row.sensor_d_mae),
            "rank": int(row.top_drift_rank),
        }
        for row in cycle_rows.itertuples(index=False)
    ]


def _key_cycles(
    window: pd.DataFrame,
    top_rows: pd.DataFrame,
    score_col: str,
    cutoff_cycle: int,
    warning_rel: str | None,
    critical_rel: str | None,
    peak_abs_cycle: int,
) -> list[dict[str, Any]]:
    notes = {int(window.iloc[0]["cycle"]): "forecast_start", peak_abs_cycle: "peak_score"}
    for label, rel in [("first_warning_crossing", warning_rel), ("first_critical_crossing", critical_rel)]:
        if rel:
            notes[cutoff_cycle + int(rel[2:])] = label
    rows = []
    for abs_cycle, note in sorted(notes.items()):
        matched = window[window["cycle"] == abs_cycle]
        if matched.empty:
            continue
        row = matched.iloc[0]
        top_sensors = _top_sensors_at_cycle(top_rows, abs_cycle)
        if not top_sensors:
            top_sensors = _parse_sensor_list(str(row.get("top_drift_sensors", "")))
        rows.append(
            {
                "relative_cycle": _relative_cycle(abs_cycle, cutoff_cycle),
                "abs_cycle": abs_cycle,
                "score": _finite_float(row.get(score_col)),
                "top_sensors": top_sensors,
                "note": note,
            }
        )
    return rows


def _score_trend(window: pd.DataFrame, score_col: str) -> str:
    values = pd.to_numeric(window[score_col], errors="coerce")
    cycles = pd.to_numeric(window["cycle"], errors="coerce")
    mask = values.notna() & cycles.notna()
    if mask.sum() < 2:
        return "stable"
    x = cycles[mask].to_numpy(dtype=float)
    y = values[mask].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    delta = float(y[-1] - y[0])
    threshold = max(float(np.nanstd(y)) * 0.02, 1e-4)
    if slope > threshold and delta > 0:
        return "increasing"
    if slope < -threshold and delta < 0:
        return "decreasing"
    return "stable"


def _first_crossing(window: pd.DataFrame, score_col: str, threshold: float | None, cutoff_cycle: int) -> str | None:
    if threshold is None:
        return None
    values = pd.to_numeric(window[score_col], errors="coerce")
    crossed = window[values >= float(threshold)]
    if crossed.empty:
        return None
    return _relative_cycle(int(crossed.iloc[0]["cycle"]), cutoff_cycle)


def _persistent_duration(window: pd.DataFrame, score_col: str, threshold: float | None) -> int:
    if threshold is None:
        return 0
    values = pd.to_numeric(window[score_col], errors="coerce").to_numpy()
    crossed = values >= float(threshold)
    if not crossed.any():
        return 0
    first = int(np.argmax(crossed))
    duration = 0
    for flag in crossed[first:]:
        if not flag:
            break
        duration += 1
    return duration


def _relative_cycle(abs_cycle: int, cutoff_cycle: int) -> str:
    return f"t+{max(1, int(abs_cycle) - int(cutoff_cycle))}"


def _parse_sensor_list(value: str) -> list[str]:
    text = str(value).strip()
    # Mixed LHI sources may introduce a column that is absent from the legacy
    # source.  Pandas represents those cells as NaN; NaN is missing data, not
    # a sensor named "NAN".
    if text.lower() in {"", "nan", "none", "null"}:
        return []
    return [s.strip().upper() for s in text.split(",") if s.strip()]


def _parse_sensor_value_map(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split(";"):
        if ":" not in item:
            continue
        sensor, raw = item.split(":", 1)
        sensor = sensor.strip().upper()
        try:
            result[sensor] = float(raw)
        except ValueError:
            continue
    return result


def _normalize_sensor_columns(df: pd.DataFrame) -> None:
    for col in ["sensor", "top_drift_sensors"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number

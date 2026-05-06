from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_THRESHOLD_CONFIG = {
    "FD001": {"warning_score": 1.0, "critical_score": 2.0},
    "FD002": {"warning_score": 1.0, "critical_score": 2.0},
    "FD003": {"warning_score": 1.0, "critical_score": 2.0},
    "FD004": {"warning_score": 1.0, "critical_score": 2.0},
}


GROUP_COLS = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle"]


def load_threshold_config(path: str | Path | None) -> dict[str, dict[str, float]]:
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
    threshold_config: dict[str, dict[str, float]],
    window_detail_dir: str | Path | None = None,
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

    thresholds = threshold_config.get(fd_name, {})
    warning_score = thresholds.get("warning_score")
    critical_score = thresholds.get("critical_score")

    warning_rel = _first_crossing(window, score_col, warning_score, cutoff_cycle)
    critical_rel = _first_crossing(window, score_col, critical_score, cutoff_cycle)
    persistent_duration = _persistent_duration(window, score_col, warning_score)
    first_persistent = warning_rel if persistent_duration > 1 else None

    top_rows = _window_top_rows(top_drift, first)
    dominant_top_sensors = _dominant_top_sensors(top_rows, window)
    top_sensor_at_peak = _top_sensors_at_cycle(top_rows, peak_abs_cycle)
    if not top_sensor_at_peak:
        top_sensor_at_peak = _parse_sensor_list(str(peak_row.get("top_drift_sensors", "")))

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
        "key_cycles": _key_cycles(window, top_rows, score_col, cutoff_cycle, warning_rel, critical_rel, peak_abs_cycle),
        "forecast_window_detail_path": str(detail_path) if detail_path else None,
        "prediction_length": prediction_length,
    }


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
    return [s.strip().upper() for s in value.split(",") if s.strip()]


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

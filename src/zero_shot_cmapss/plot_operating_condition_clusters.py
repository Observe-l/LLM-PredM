from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


FD_NAMES = ("FD001", "FD002", "FD003", "FD004")
SETTING_COLUMNS = ["setting1", "setting2", "setting3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = ["unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]
DEFAULT_SENSORS = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Chronos-2 C-MAPSS forecasts after per-engine, per-window "
            "operating-condition clustering."
        )
    )
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--forecast_dir", type=Path, default=Path("outputs/stride_5_robust"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--covariate_modes",
        nargs="+",
        default=["known_future"],
        help="Forecast covariate modes to plot. Use past_only known_future to plot both.",
    )
    parser.add_argument("--fds", nargs="+", default=list(FD_NAMES), choices=list(FD_NAMES))
    parser.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS)
    parser.add_argument(
        "--plot_units",
        nargs="*",
        default=None,
        help="Optional units, e.g. FD001:1 FD004:3. Defaults to the first unit in each FD.",
    )
    parser.add_argument(
        "--cutoff_cycles",
        nargs="*",
        type=int,
        default=None,
        help="Optional cutoff cycles to plot. If omitted, representative early/middle/late cutoffs are used.",
    )
    parser.add_argument(
        "--max_windows_per_unit",
        type=int,
        default=3,
        help="Number of representative cutoff windows per plotted unit when --cutoff_cycles is omitted.",
    )
    parser.add_argument(
        "--write_labeled_csv",
        action="store_true",
        help="Also save the labeled forecast rows used by all selected plots.",
    )
    return parser.parse_args()


def load_cmapss_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


def load_eval_frames(data_dir: Path, eval_split: str) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for fd_name in FD_NAMES:
        frames[fd_name] = load_cmapss_file(data_dir / fd_name / f"{eval_split}_{fd_name}.txt")
    return frames


def parse_plot_units(items: Sequence[str] | None, forecasts: pd.DataFrame, fds: Sequence[str]) -> List[Tuple[str, int]]:
    if items:
        parsed = []
        for item in items:
            if ":" not in item:
                raise ValueError(f"--plot_units entries must look like FD004:3, got {item!r}")
            fd_name, unit_text = item.split(":", 1)
            if fd_name not in FD_NAMES:
                raise ValueError(f"Unknown FD in --plot_units: {fd_name!r}")
            parsed.append((fd_name, int(unit_text)))
        return parsed

    keys = (
        forecasts[forecasts["fd"].isin(fds)][["fd", "unit_id"]]
        .drop_duplicates()
        .sort_values(["fd", "unit_id"])
        .groupby("fd", as_index=False)
        .head(1)
    )
    return [(str(row.fd), int(row.unit_id)) for row in keys.itertuples(index=False)]


def chronos_style_operating_features(values: pd.DataFrame, history_values: pd.DataFrame) -> pd.DataFrame:
    mu = history_values.loc[:, SETTING_COLUMNS].mean()
    sigma = history_values.loc[:, SETTING_COLUMNS].std(ddof=0).replace(0.0, np.nan).fillna(1.0)
    scaled = (values.loc[:, SETTING_COLUMNS] - mu) / sigma
    return pd.DataFrame(
        np.arcsinh(scaled.to_numpy(dtype=np.float32)),
        columns=[f"{col}_chronos_scaled" for col in SETTING_COLUMNS],
        index=values.index,
    )


def make_condition_keys(frame: pd.DataFrame) -> pd.Series:
    # C-MAPSS operating settings are stored as noisy numeric readings around a
    # small set of nominal operating states. Canonicalizing the readings gives
    # Chronos-style ordinal categories without preselecting the number of states.
    canonical = pd.DataFrame(index=frame.index)
    canonical["setting1"] = np.rint(frame["setting1"]).astype(int)
    canonical["setting2"] = np.rint(frame["setting2"] * 100).astype(int)
    canonical["setting3"] = np.rint(frame["setting3"]).astype(int)
    return canonical.astype(str).agg("|".join, axis=1)


def label_window_conditions(
    unit_df: pd.DataFrame,
    cutoff_cycle: int,
    horizon_cycles: Sequence[int],
) -> pd.DataFrame:
    history = unit_df[unit_df["cycle"] <= cutoff_cycle].sort_values("cycle").copy()
    if history.empty:
        raise ValueError(f"No history found for cutoff_cycle={cutoff_cycle}")

    max_cycle = max([cutoff_cycle, *[int(cycle) for cycle in horizon_cycles]])
    timeline = unit_df[unit_df["cycle"] <= max_cycle].sort_values("cycle").copy()
    timeline = timeline[timeline["cycle"].isin(set(history["cycle"]).union(int(c) for c in horizon_cycles))].copy()

    labeled = timeline.loc[:, ["unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]].copy()
    features = chronos_style_operating_features(timeline, history)
    history_keys = make_condition_keys(history)
    condition_order = sorted(history_keys.unique())
    condition_to_id = {condition_key: idx for idx, condition_key in enumerate(condition_order)}
    labeled["op_condition_key"] = make_condition_keys(labeled)
    unseen_keys = sorted(set(labeled["op_condition_key"]) - set(condition_to_id))
    for condition_key in unseen_keys:
        condition_to_id[condition_key] = len(condition_to_id)
    labeled["op_condition"] = labeled["op_condition_key"].map(condition_to_id).astype(int)
    labeled["window_part"] = np.where(labeled["cycle"] <= cutoff_cycle, "history", "forecast_horizon")
    labeled["cutoff_cycle"] = int(cutoff_cycle)
    labeled["n_operating_conditions"] = int(len(condition_to_id))
    for col in features.columns:
        labeled[col] = features[col].to_numpy(dtype=np.float32)
    return labeled


def iter_representative_cutoffs(
    forecasts: pd.DataFrame,
    fd_name: str,
    unit_id: int,
    covariate_mode: str,
    explicit_cutoffs: Sequence[int] | None,
    max_windows: int,
) -> List[int]:
    unit_windows = (
        forecasts[
            (forecasts["fd"] == fd_name)
            & (forecasts["unit_id"] == unit_id)
            & (forecasts["covariate_mode"] == covariate_mode)
        ]["cutoff_cycle"]
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype=np.int64)
    )
    if explicit_cutoffs:
        available = set(int(x) for x in unit_windows)
        return [int(cutoff) for cutoff in explicit_cutoffs if int(cutoff) in available]
    if len(unit_windows) <= max_windows:
        return [int(x) for x in unit_windows]
    positions = np.linspace(0, len(unit_windows) - 1, max_windows).round().astype(int)
    return [int(unit_windows[pos]) for pos in np.unique(positions)]


def build_labeled_window(
    window_forecasts: pd.DataFrame,
    eval_frames: Dict[str, pd.DataFrame],
    fd_name: str,
    unit_id: int,
    covariate_mode: str,
    cutoff_cycle: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    window = window_forecasts[
        (window_forecasts["fd"] == fd_name)
        & (window_forecasts["unit_id"] == unit_id)
        & (window_forecasts["covariate_mode"] == covariate_mode)
        & (window_forecasts["cutoff_cycle"] == cutoff_cycle)
    ].copy()
    if window.empty:
        return pd.DataFrame(), pd.DataFrame()

    unit_df = (
        eval_frames[fd_name][eval_frames[fd_name]["unit_id"] == unit_id]
        .sort_values("cycle")
        .reset_index(drop=True)
    )
    horizon_cycles = sorted(int(cycle) for cycle in window["cycle"].unique())
    clustered_timeline = label_window_conditions(
        unit_df=unit_df,
        cutoff_cycle=cutoff_cycle,
        horizon_cycles=horizon_cycles,
    )

    horizon_labels = clustered_timeline.loc[
        clustered_timeline["window_part"] == "forecast_horizon",
        [
            "cycle",
            "op_condition",
            "op_condition_key",
            "n_operating_conditions",
            *[f"{col}_chronos_scaled" for col in SETTING_COLUMNS],
        ],
    ]
    labeled_forecasts = window.merge(horizon_labels, on="cycle", how="left", validate="many_to_one")
    return clustered_timeline, labeled_forecasts


def add_condition_runs(frame: pd.DataFrame, condition_col: str = "op_condition") -> pd.DataFrame:
    frame = frame.sort_values("cycle").copy()
    frame["condition_run"] = (frame[condition_col] != frame[condition_col].shift()).cumsum()
    return frame


def plot_sensor_window(
    timeline: pd.DataFrame,
    forecasts: pd.DataFrame,
    curve_forecasts: pd.DataFrame,
    sensor: str,
    covariate_mode: str,
    fd_name: str,
    unit_id: int,
    cutoff_cycle: int,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    sensor_forecasts = forecasts[forecasts["sensor"] == sensor].sort_values("cycle")
    if sensor_forecasts.empty:
        return

    conditions = sorted(int(x) for x in timeline["op_condition"].dropna().unique())
    n_conditions = len(conditions)
    fig, axes = plt.subplots(n_conditions, 1, figsize=(13, max(3.0, 2.6 * n_conditions)), sharex=True)
    if n_conditions == 1:
        axes = [axes]

    history_forecasts = curve_forecasts[
        (curve_forecasts["covariate_mode"] == covariate_mode)
        & (curve_forecasts["fd"] == fd_name)
        & (curve_forecasts["unit_id"] == unit_id)
        & (curve_forecasts["sensor"] == sensor)
        & (curve_forecasts["cycle"] <= cutoff_cycle)
    ][["cycle", "y_pred"]].copy()
    history_labels = timeline[timeline["window_part"] == "history"][["cycle", "op_condition"]]
    history_forecasts = history_forecasts.merge(history_labels, on="cycle", how="inner", validate="one_to_one")

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_conditions)))
    for ax, condition in zip(axes, conditions):
        color = colors[condition % len(colors)]
        condition_history = timeline[
            (timeline["op_condition"] == condition) & (timeline["window_part"] == "history")
        ][["cycle", sensor, "op_condition", "op_condition_key"]].rename(columns={sensor: "y_true"})
        condition_history_forecast = history_forecasts[history_forecasts["op_condition"] == condition]
        condition_horizon = sensor_forecasts[sensor_forecasts["op_condition"] == condition]
        condition_key = (
            str(condition_history["op_condition_key"].iloc[0])
            if not condition_history.empty
            else str(condition_horizon["op_condition_key"].iloc[0])
        )

        history_label_used = False
        for _, run_df in add_condition_runs(condition_history).groupby("condition_run", sort=True):
            ax.plot(
                run_df["cycle"],
                run_df["y_true"],
                color=color,
                linewidth=1.4,
                alpha=0.7,
                label="history truth" if not history_label_used else None,
            )
            history_label_used = True
        history_forecast_label_used = False
        for _, run_df in add_condition_runs(condition_history_forecast).groupby("condition_run", sort=True):
            ax.plot(
                run_df["cycle"],
                run_df["y_pred"],
                color="#2563eb",
                linewidth=1.2,
                alpha=0.75,
                linestyle="--",
                label="history forecast" if not history_forecast_label_used else None,
            )
            history_forecast_label_used = True
        if not condition_horizon.empty:
            ax.plot(
                condition_horizon["cycle"],
                condition_horizon["y_true"],
                color="#111827",
                linewidth=2.0,
                marker="o",
                label="horizon ground truth",
            )
            ax.plot(
                condition_horizon["cycle"],
                condition_horizon["y_pred"],
                color="#d97706",
                linewidth=2.0,
                marker="x",
                linestyle="--",
                label="horizon forecast",
            )

        ax.axvline(cutoff_cycle, color="#6b7280", linewidth=1.0, linestyle=":")
        ax.set_ylabel(f"cond {condition}\n{condition_key}")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), fontsize=8, loc="best")

    last_cycle = int(timeline["cycle"].max())
    forecast_start = int(sensor_forecasts["forecast_start_cycle"].iloc[0])
    fig.suptitle(
        (
            f"{covariate_mode} {fd_name} unit {unit_id} {sensor}: "
            f"operating conditions, cutoff={cutoff_cycle}, forecast={forecast_start}-{last_cycle}"
        )
    )
    fig.supxlabel("cycle")
    fig.supylabel("sensor reading")
    fig.tight_layout()

    plot_dir = output_dir / "plots" / covariate_mode / fd_name / f"unit{unit_id}" / f"cutoff_{cutoff_cycle}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"{sensor}_conditions.png", dpi=160)
    plt.close(fig)


def summarize_cluster_counts(timelines: Iterable[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for timeline in timelines:
        if timeline.empty:
            continue
        meta = timeline.iloc[0]
        for (part, condition), group in timeline.groupby(["window_part", "op_condition"], sort=True):
            rows.append(
                {
                    "unit_id": int(meta["unit_id"]),
                    "cutoff_cycle": int(meta["cutoff_cycle"]),
                    "window_part": part,
                    "op_condition": int(condition),
                    "op_condition_key": str(group["op_condition_key"].iloc[0]),
                    "count": int(len(group)),
                    "cycle_min": int(group["cycle"].min()),
                    "cycle_max": int(group["cycle"].max()),
                    "n_operating_conditions": int(meta["n_operating_conditions"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.forecast_dir / "operating_condition_clusters")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    run_config_path = args.forecast_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    eval_split = str(run_config.get("eval_split", "train"))

    window_forecasts = pd.read_csv(args.forecast_dir / "window_forecasts.csv")
    curve_forecasts = pd.read_csv(args.forecast_dir / "forecasts.csv")
    required = {"covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "cycle", "sensor", "y_true", "y_pred"}
    missing = required - set(window_forecasts.columns)
    if missing:
        raise ValueError(f"Missing required columns in window_forecasts.csv: {sorted(missing)}")
    curve_required = {"covariate_mode", "fd", "unit_id", "cycle", "sensor", "y_pred"}
    curve_missing = curve_required - set(curve_forecasts.columns)
    if curve_missing:
        raise ValueError(f"Missing required columns in forecasts.csv: {sorted(curve_missing)}")

    window_forecasts = window_forecasts[
        window_forecasts["fd"].isin(args.fds)
        & window_forecasts["covariate_mode"].isin(args.covariate_modes)
        & window_forecasts["sensor"].isin(args.sensors)
    ].copy()
    curve_forecasts = curve_forecasts[
        curve_forecasts["fd"].isin(args.fds)
        & curve_forecasts["covariate_mode"].isin(args.covariate_modes)
        & curve_forecasts["sensor"].isin(args.sensors)
    ].copy()
    eval_frames = load_eval_frames(args.data_dir, eval_split)
    units = parse_plot_units(args.plot_units, window_forecasts, args.fds)

    all_timelines: List[pd.DataFrame] = []
    all_labeled_forecasts: List[pd.DataFrame] = []
    for covariate_mode in args.covariate_modes:
        for fd_name, unit_id in units:
            if fd_name not in args.fds:
                continue
            cutoffs = iter_representative_cutoffs(
                forecasts=window_forecasts,
                fd_name=fd_name,
                unit_id=unit_id,
                covariate_mode=covariate_mode,
                explicit_cutoffs=args.cutoff_cycles,
                max_windows=args.max_windows_per_unit,
            )
            for cutoff_cycle in cutoffs:
                timeline, labeled_forecasts = build_labeled_window(
                    window_forecasts=window_forecasts,
                    eval_frames=eval_frames,
                    fd_name=fd_name,
                    unit_id=unit_id,
                    covariate_mode=covariate_mode,
                    cutoff_cycle=cutoff_cycle,
                )
                if timeline.empty or labeled_forecasts.empty:
                    continue
                timeline = timeline.assign(fd=fd_name, covariate_mode=covariate_mode)
                all_timelines.append(timeline)
                all_labeled_forecasts.append(labeled_forecasts)
                for sensor in args.sensors:
                    plot_sensor_window(
                        timeline=timeline,
                        forecasts=labeled_forecasts,
                        curve_forecasts=curve_forecasts,
                        sensor=sensor,
                        covariate_mode=covariate_mode,
                        fd_name=fd_name,
                        unit_id=unit_id,
                        cutoff_cycle=cutoff_cycle,
                        output_dir=output_dir,
                    )

    if all_timelines:
        timeline_df = pd.concat(all_timelines, ignore_index=True)
        summary = (
            timeline_df.groupby(
                ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "window_part", "op_condition", "op_condition_key"],
                sort=True,
            )
            .agg(
                count=("cycle", "size"),
                cycle_min=("cycle", "min"),
                cycle_max=("cycle", "max"),
                n_operating_conditions=("n_operating_conditions", "first"),
            )
            .reset_index()
        )
        summary.to_csv(output_dir / "cluster_counts.csv", index=False)
        timeline_df.to_csv(output_dir / "clustered_timelines.csv", index=False)

    if args.write_labeled_csv and all_labeled_forecasts:
        pd.concat(all_labeled_forecasts, ignore_index=True).to_csv(output_dir / "labeled_window_forecasts.csv", index=False)

    print(f"Saved operating-condition cluster plots to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

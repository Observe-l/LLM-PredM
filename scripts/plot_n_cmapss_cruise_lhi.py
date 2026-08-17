#!/usr/bin/env python3
"""Compute condition-matched LHI using only CruiseBench cruise samples.

This reuses the LHI aggregation from ``plot_n_cmapss_sample_lhi.py`` but
restricts both the health reference and target cycles to the accepted cruise
intervals in ``cruise_cycle_statistics.csv``.  K-means centers are fitted on
cruise W=(alt, Mach, TRA, T2) and are stored in raw units; the corresponding
cruise standardization is applied before nearest-center assignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from scripts.plot_n_cmapss_sample_lhi import (
        CONDITION_NAMES,
        SENSOR_NAMES,
        compute_sample_lhi,
        plot_cycle,
    )
except ModuleNotFoundError:  # direct execution: scripts/ is on sys.path
    from plot_n_cmapss_sample_lhi import (
        CONDITION_NAMES,
        SENSOR_NAMES,
        compute_sample_lhi,
        plot_cycle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/N-CMAPSS/N-CMAPSS_DS02-006.h5"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unit", type=int, default=5)
    parser.add_argument("--cruise-statistics", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude/cruise_cycle_statistics.csv"))
    parser.add_argument("--cluster-centers", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude/cluster_centers_k6.csv"))
    parser.add_argument("--standardization", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude/cruise_standardization.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_lhi_k6_DS02-006_dev_unit5_ref_cycles1-5"))
    parser.add_argument("--reference-end-cycle", type=int, default=5)
    parser.add_argument("--last-cycle-offset", type=int, default=0, help="Use last_cycle - offset as the second target cycle.")
    parser.add_argument("--ignore-operating-condition", action="store_true", help="Pool all cruise operating conditions into one LHI reference group.")
    parser.add_argument("--lhi-epsilon", type=float, default=1e-6)
    parser.add_argument("--range-epsilon", type=float, default=1e-12)
    return parser.parse_args()


def load_unit(path: Path, split: str, unit_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    with h5py.File(path, "r") as hdf:
        a_names = [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(hdf["A_var"]).reshape(-1)]
        w_names = [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(hdf["W_var"]).reshape(-1)]
        sensor_names = [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(hdf["X_s_var"]).reshape(-1)]
        unit_col = a_names.index("unit")
        cycle_col = a_names.index("cycle")
        units = np.asarray(hdf[f"A_{split}"][:, unit_col], dtype=np.int32)
        selected = np.flatnonzero(units == unit_id)
        if len(selected) == 0:
            raise ValueError(f"Unit {unit_id} not found; available={np.unique(units).tolist()}")
        cycles = np.asarray(hdf[f"A_{split}"][selected, cycle_col], dtype=np.int32)
        w = np.asarray(hdf[f"W_{split}"][selected], dtype=np.float64)
        sensors = np.asarray(hdf[f"X_s_{split}"][selected], dtype=np.float64)
    return cycles, w, sensors, w_names, sensor_names


def assign_cruise_clusters(w: np.ndarray, w_names: list[str], centers_path: Path, standardization_path: Path) -> tuple[np.ndarray, int]:
    centers_df = pd.read_csv(centers_path).sort_values("cluster")
    centers_raw = centers_df.loc[:, CONDITION_NAMES].to_numpy(dtype=float)
    scaling = pd.read_csv(standardization_path).set_index("feature").loc[CONDITION_NAMES]
    mean = scaling["mean"].to_numpy(dtype=float)
    scale = scaling["scale"].to_numpy(dtype=float)
    columns = [w_names.index(name) for name in CONDITION_NAMES]
    x = (w[:, columns] - mean) / scale
    centers = (centers_raw - mean) / scale
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(int), int(len(centers))


def cycle_local_samples(cycles: np.ndarray) -> np.ndarray:
    local = np.empty(len(cycles), dtype=np.int64)
    start = 0
    while start < len(cycles):
        stop = start + 1
        while stop < len(cycles) and cycles[stop] == cycles[start]:
            stop += 1
        local[start:stop] = np.arange(stop - start)
        start = stop
    return local


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervals = pd.read_csv(args.cruise_statistics)
    intervals = intervals[
        (intervals["dataset"] == "DS02-006")
        & (intervals["split"] == args.split)
        & (intervals["unit"] == args.unit)
    ].copy()
    if intervals.empty:
        raise ValueError("No cruise statistics found for the selected dataset/split/unit")
    accepted = intervals[intervals["cruise_status"] == "accepted"].copy()
    first_cycle = int(accepted["cycle"].min())
    last_cycle = int(accepted["cycle"].max())
    second_target = last_cycle - args.last_cycle_offset
    target_cycles = [args.reference_end_cycle + 1, second_target]
    target_cycles = list(dict.fromkeys(c for c in target_cycles if c in set(accepted["cycle"])))
    selected_cycles = list(range(first_cycle, args.reference_end_cycle + 1)) + target_cycles

    cycles, w, sensors, w_names, sensor_names = load_unit(args.data_file, args.split, args.unit)
    if sensor_names != SENSOR_NAMES:
        raise ValueError(f"Unexpected X_s variables: {sensor_names}")
    local = cycle_local_samples(cycles)
    interval_lookup = accepted.set_index("cycle")[["cruise_start_sample", "cruise_end_sample"]]
    wanted = np.isin(cycles, selected_cycles)
    cruise_start = np.full(len(cycles), -1.0)
    cruise_end = np.full(len(cycles), -1.0)
    for cycle in selected_cycles:
        if cycle not in interval_lookup.index:
            continue
        cruise_start[cycles == cycle] = interval_lookup.loc[cycle, "cruise_start_sample"]
        cruise_end[cycles == cycle] = interval_lookup.loc[cycle, "cruise_end_sample"]
    mask = wanted & (local >= cruise_start) & (local <= cruise_end)
    cycles = cycles[mask]
    w = w[mask]
    sensors = sensors[mask]
    if args.ignore_operating_condition:
        clusters = np.zeros(len(w), dtype=int)
        n_clusters = 1
    else:
        clusters, n_clusters = assign_cruise_clusters(w, w_names, args.cluster_centers, args.standardization)

    scores, sensor_scores, metadata = compute_sample_lhi(
        cycles=cycles,
        clusters=clusters,
        sensors=sensors,
        sensor_names=sensor_names,
        lhi_epsilon=args.lhi_epsilon,
        range_epsilon=args.range_epsilon,
        sensor_output_cycles=set(target_cycles),
        n_clusters=n_clusters,
        reference_end_cycle=args.reference_end_cycle,
    )
    metadata.update({
        "data_file": str(args.data_file),
        "split": args.split,
        "unit": int(args.unit),
        "cruise_only": True,
        "cruise_method": "CruiseBench common_altitude",
        "cruise_statistics": str(args.cruise_statistics),
        "cluster_centers": str(args.cluster_centers),
        "standardization": str(args.standardization),
        "reference": f"all cruise rows in cycles {first_cycle}..{args.reference_end_cycle}",
        "reference_cycles": list(range(first_cycle, args.reference_end_cycle + 1)),
        "target_cycles": target_cycles,
        "last_cycle": last_cycle,
        "n_rows_cruise_reference_and_targets": int(len(cycles)),
        "ignore_operating_condition": bool(args.ignore_operating_condition),
        "last_cycle_offset": int(args.last_cycle_offset),
    })
    scores.to_csv(args.output_dir / "cruise_lhi_scores.csv", index=False)
    sensor_scores.to_csv(args.output_dir / "cruise_sensor_lhi_scores.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    target_scores = scores[scores["cycle"].isin(target_cycles)]
    statistics = []
    for cycle, group in target_scores.groupby("cycle", sort=True):
        valid = group[group["valid_lhi"]]
        statistics.append({
            "cycle": int(cycle),
            "cruise_samples": int(len(group)),
            "valid_samples": int(len(valid)),
            "invalid_samples": int(len(group) - len(valid)),
            "valid_fraction": float(len(valid) / len(group)),
            "cluster_count": int(group["operating_condition_cluster"].nunique()),
            "lhi_rmse_mean": float(valid["lhi_rmse"].mean()) if len(valid) else np.nan,
            "lhi_rmse_median": float(valid["lhi_rmse"].median()) if len(valid) else np.nan,
            "lhi_rmse_std": float(valid["lhi_rmse"].std(ddof=0)) if len(valid) else np.nan,
            "lhi_rmse_p05": float(valid["lhi_rmse"].quantile(0.05)) if len(valid) else np.nan,
            "lhi_rmse_p95": float(valid["lhi_rmse"].quantile(0.95)) if len(valid) else np.nan,
        })
    statistics_df = pd.DataFrame(statistics)
    statistics_df.to_csv(args.output_dir / "cruise_lhi_statistics.csv", index=False)
    target_scores.groupby(["cycle", "operating_condition_cluster"], sort=True).agg(
        cruise_samples=("row_index", "size"),
        valid_samples=("valid_lhi", "sum"),
    ).reset_index().to_csv(args.output_dir / "condition_sample_counts_target_cycles.csv", index=False)
    for cycle in target_cycles:
        plot_cycle(
            scores,
            cycle,
            args.output_dir / f"cruise_lhi_vs_time_samples_cycle_{cycle:03d}.png",
            metadata,
            n_clusters,
        )
    print(statistics_df.to_string(index=False))
    print(f"Saved cruise-only LHI outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

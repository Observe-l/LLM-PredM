#!/usr/bin/env python3
"""Summarize all-cycle cluster assignments for one unit-specific cruise model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.plot_n_cmapss_cruise_lhi import cycle_local_samples, load_unit
except ModuleNotFoundError:
    from plot_n_cmapss_cruise_lhi import cycle_local_samples, load_unit


FEATURE_NAMES = ["alt", "Mach", "TRA", "T2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/N-CMAPSS/N-CMAPSS_DS02-006.h5"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unit", type=int, default=5)
    parser.add_argument("--cruise-statistics", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude/cruise_cycle_statistics.csv"))
    parser.add_argument("--centers", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_unit_cluster_DS02-006_dev_unit5/unit_cluster_centers_selected_k.csv"))
    parser.add_argument("--standardization", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_unit_cluster_DS02-006_dev_unit5/unit_cruise_standardization.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_unit_cluster_DS02-006_dev_unit5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    intervals = pd.read_csv(args.cruise_statistics)
    accepted = intervals[
        (intervals["dataset"] == "DS02-006")
        & (intervals["split"] == args.split)
        & (intervals["unit"] == args.unit)
        & (intervals["cruise_status"] == "accepted")
    ].copy()
    if accepted.empty:
        raise ValueError("No accepted cruise cycles found for the selected unit")

    cycles, w, _, w_names, _ = load_unit(args.data_file, args.split, args.unit)
    local = cycle_local_samples(cycles)
    starts = np.full(len(cycles), -1.0)
    ends = np.full(len(cycles), -1.0)
    lookup = accepted.set_index("cycle")
    for cycle in lookup.index.astype(int):
        mask = cycles == cycle
        starts[mask] = lookup.loc[cycle, "cruise_start_sample"]
        ends[mask] = lookup.loc[cycle, "cruise_end_sample"]
    mask = np.isfinite(starts) & (local >= starts) & (local <= ends)
    cycles = cycles[mask]
    local = local[mask]
    w = w[mask]

    columns = [w_names.index(name) for name in FEATURE_NAMES]
    x_raw = w[:, columns].astype(float)
    scaling = pd.read_csv(args.standardization).set_index("feature").loc[FEATURE_NAMES]
    mean = scaling["mean"].to_numpy(float)
    scale = scaling["scale"].to_numpy(float)
    centers = pd.read_csv(args.centers).sort_values("cluster")
    cluster_ids = centers["cluster"].astype(int).tolist()
    center_raw = centers[FEATURE_NAMES].to_numpy(float)
    x = (x_raw - mean) / scale
    center_x = (center_raw - mean) / scale
    distances = ((x[:, None, :] - center_x[None, :, :]) ** 2).sum(axis=2)
    labels = np.asarray(cluster_ids, dtype=int)[np.argmin(distances, axis=1)]

    frame = pd.DataFrame({"cycle": cycles, "cluster": labels})
    counts = pd.crosstab(frame["cycle"], frame["cluster"]).reindex(columns=cluster_ids, fill_value=0)
    counts.index.name = "cycle"
    count_cols = []
    fraction_cols = []
    for cluster in cluster_ids:
        count_col = f"cluster_{cluster}_samples"
        fraction_col = f"cluster_{cluster}_fraction"
        counts = counts.rename(columns={cluster: count_col})
        counts[fraction_col] = counts[count_col] / counts.sum(axis=1)
        count_cols.append(count_col)
        fraction_cols.append(fraction_col)
    counts["cruise_samples"] = counts[count_cols].sum(axis=1)
    counts["clusters_present"] = (counts[count_cols] > 0).sum(axis=1)
    counts = counts.reset_index()
    ordered = ["cycle", "cruise_samples", "clusters_present"] + count_cols + fraction_cols
    counts = counts[ordered]
    counts.to_csv(args.output_dir / "unit_cruise_cluster_counts_by_cycle.csv", index=False)
    counts[counts["cycle"].between(1, 10)].to_csv(args.output_dir / "unit_cruise_cluster_counts_cycles_1_10.csv", index=False)

    long = frame.groupby(["cycle", "cluster"], sort=True).size().rename("cruise_samples").reset_index()
    totals = long.groupby("cycle", as_index=False)["cruise_samples"].sum().rename(columns={"cruise_samples": "cycle_cruise_samples"})
    long = long.merge(totals, on="cycle", how="left")
    long["fraction"] = long["cruise_samples"] / long["cycle_cruise_samples"]
    long.to_csv(args.output_dir / "unit_cruise_cluster_counts_long.csv", index=False)

    overall = frame.groupby("cluster", sort=True).size().rename("cruise_samples").reindex(cluster_ids, fill_value=0).rename_axis("cluster").reset_index()
    overall["fraction"] = overall["cruise_samples"] / len(frame)
    overall["cycles_present"] = overall["cluster"].map(frame.groupby("cluster")["cycle"].nunique()).fillna(0).astype(int)
    overall.to_csv(args.output_dir / "unit_cruise_cluster_overall_summary.csv", index=False)

    print("Overall cluster summary:")
    print(overall.to_string(index=False))
    print("\nCycle 1-10 cluster summary:")
    print(counts[counts["cycle"].between(1, 10)].to_string(index=False))
    print("\nCycles by number of present clusters:")
    print(counts["clusters_present"].value_counts().sort_index().rename("cycles").to_string())
    print(f"\nSaved summaries to: {args.output_dir}")


if __name__ == "__main__":
    main()

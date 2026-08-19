#!/usr/bin/env python3
"""Plot cruise-stage LHI summaries versus cycle for one N-CMAPSS unit.

The reference is pooled across operating conditions: all accepted cruise rows
from cycles 1..N are used together to build one sensor mean, one sensor range,
and one drift baseline.  Each accepted cruise cycle is summarized by its
sample-level LHI_RMSE mean and maximum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.plot_n_cmapss_cruise_lhi import assign_cruise_clusters, cycle_local_samples, load_unit
    from scripts.plot_n_cmapss_sample_lhi import SENSOR_NAMES, compute_sample_lhi
except ModuleNotFoundError:  # direct execution: scripts/ is on sys.path
    from plot_n_cmapss_cruise_lhi import assign_cruise_clusters, cycle_local_samples, load_unit
    from plot_n_cmapss_sample_lhi import SENSOR_NAMES, compute_sample_lhi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/N-CMAPSS/N-CMAPSS_DS02-006.h5"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unit", type=int, default=5)
    parser.add_argument("--cruise-statistics", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude/cruise_cycle_statistics.csv"))
    parser.add_argument("--cluster-centers", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude/cluster_centers_selected_k.csv"))
    parser.add_argument("--standardization", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude/cruise_standardization.csv"))
    parser.add_argument("--ignore-operating-condition", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_lhi_k4_DS02-006_dev_unit5_ref_cycles1-10_vs_cycle"))
    parser.add_argument("--reference-end-cycle", type=int, default=10)
    parser.add_argument("--report-start-cycle", type=int, default=1, help="Only report and plot cycles >= this cycle.")
    parser.add_argument("--lhi-epsilon", type=float, default=1e-6)
    parser.add_argument("--range-epsilon", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervals = pd.read_csv(args.cruise_statistics)
    intervals = intervals[
        (intervals["dataset"] == "DS02-006")
        & (intervals["split"] == args.split)
        & (intervals["unit"] == args.unit)
        & (intervals["cruise_status"] == "accepted")
    ].copy()
    if intervals.empty:
        raise ValueError("No accepted cruise cycles found for the selected unit")

    cycles, w, sensors, w_names, sensor_names = load_unit(args.data_file, args.split, args.unit)
    if sensor_names != SENSOR_NAMES:
        raise ValueError(f"Unexpected X_s variables: {sensor_names}")
    local = cycle_local_samples(cycles)
    lookup = intervals.set_index("cycle")[["cruise_start_sample", "cruise_end_sample"]]
    start = np.full(len(cycles), -1.0)
    end = np.full(len(cycles), -1.0)
    for cycle in lookup.index:
        rows = cycles == int(cycle)
        start[rows] = lookup.loc[cycle, "cruise_start_sample"]
        end[rows] = lookup.loc[cycle, "cruise_end_sample"]
    cruise_mask = np.isfinite(start) & (local >= start) & (local <= end)
    cycles = cycles[cruise_mask]
    w = w[cruise_mask]
    sensors = sensors[cruise_mask]

    if args.ignore_operating_condition:
        clusters = np.zeros(len(cycles), dtype=int)
        n_clusters = 1
    else:
        clusters, n_clusters = assign_cruise_clusters(w, w_names, args.cluster_centers, args.standardization)
    scores, _, metadata = compute_sample_lhi(
        cycles=cycles,
        clusters=clusters,
        sensors=sensors,
        sensor_names=sensor_names,
        lhi_epsilon=args.lhi_epsilon,
        range_epsilon=args.range_epsilon,
        sensor_output_cycles=set(),
        n_clusters=n_clusters,
        reference_end_cycle=args.reference_end_cycle,
    )
    scores.to_csv(args.output_dir / "cruise_lhi_scores_all_cycles.csv", index=False)
    summary_all = scores.groupby("cycle", sort=True).agg(
        cruise_samples=("row_index", "size"),
        valid_samples=("valid_lhi", "sum"),
        mean_lhi_rmse=("lhi_rmse", "mean"),
        max_lhi_rmse=("lhi_rmse", "max"),
        median_lhi_rmse=("lhi_rmse", "median"),
        std_lhi_rmse=("lhi_rmse", lambda x: x.std(ddof=0)),
        mean_d_rmse=("d_rmse", "mean"),
    ).reset_index()
    summary_all["reference"] = summary_all["cycle"] <= args.reference_end_cycle
    summary = summary_all[summary_all["cycle"] >= args.report_start_cycle].copy()
    if summary.empty:
        raise ValueError(f"No cycles found at or after --report-start-cycle={args.report_start_cycle}")
    summary.to_csv(args.output_dir / "cruise_cycle_mean_lhi.csv", index=False)
    summary.to_csv(args.output_dir / "cruise_cycle_lhi_summary.csv", index=False)

    # Save the operating-condition composition used by the LHI calculation.
    # Include zero-count clusters so every cycle has the same columns.
    cluster_long = scores.groupby(["cycle", "operating_condition_cluster"], sort=True).agg(
        cruise_samples=("row_index", "size"),
        valid_lhi_samples=("valid_lhi", "sum"),
    )
    all_pairs = pd.MultiIndex.from_product(
        [sorted(scores["cycle"].unique()), range(n_clusters)],
        names=["cycle", "operating_condition_cluster"],
    )
    cluster_long = cluster_long.reindex(all_pairs, fill_value=0).reset_index()
    cycle_totals = cluster_long.groupby("cycle", as_index=False)["cruise_samples"].sum().rename(
        columns={"cruise_samples": "cycle_cruise_samples"}
    )
    cluster_long = cluster_long.merge(cycle_totals, on="cycle", how="left")
    cluster_long["fraction"] = cluster_long["cruise_samples"] / cluster_long["cycle_cruise_samples"]
    cluster_long.to_csv(args.output_dir / "cruise_cluster_counts_long.csv", index=False)

    count_wide = cluster_long.pivot(index="cycle", columns="operating_condition_cluster", values="cruise_samples").fillna(0).astype(int)
    count_wide.columns = [f"cluster_{int(value)}_samples" for value in count_wide.columns]
    fraction_wide = cluster_long.pivot(index="cycle", columns="operating_condition_cluster", values="fraction").fillna(0.0)
    fraction_wide.columns = [f"cluster_{int(value)}_fraction" for value in fraction_wide.columns]
    cycle_cluster_summary = pd.concat([count_wide, fraction_wide], axis=1).reset_index()
    cycle_cluster_summary.insert(1, "cruise_samples", cycle_cluster_summary.filter(like="_samples").sum(axis=1))
    cycle_cluster_summary.to_csv(args.output_dir / "cruise_cluster_counts_by_cycle.csv", index=False)

    overall_cluster = scores.groupby("operating_condition_cluster", sort=True).agg(
        cruise_samples=("row_index", "size"),
        valid_lhi_samples=("valid_lhi", "sum"),
    ).reindex(range(n_clusters), fill_value=0).reset_index()
    overall_cluster["fraction"] = overall_cluster["cruise_samples"] / len(scores)
    overall_cluster["valid_lhi_fraction"] = overall_cluster["valid_lhi_samples"] / overall_cluster["cruise_samples"].replace(0, np.nan)
    overall_cluster.to_csv(args.output_dir / "cruise_cluster_overall_summary.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=180)
    if summary["cycle"].min() <= args.reference_end_cycle:
        ax.axvspan(summary["cycle"].min() - 0.5, args.reference_end_cycle + 0.5, color="#dbeafe", alpha=0.8, label=f"health reference: cycles 1–{args.reference_end_cycle}")
    else:
        ax.axvline(args.report_start_cycle - 0.5, color="#6b7280", linestyle="--", linewidth=1.0, label=f"test start: cycle {args.report_start_cycle}")
    ax.axhline(0.0, color="#4b5563", linewidth=1.0)
    ax.plot(summary["cycle"], summary["mean_lhi_rmse"], color="#b45f06", marker="o", markersize=3.5, linewidth=1.4, label="mean LHI_RMSE per cycle")
    ax.fill_between(summary["cycle"], summary["mean_lhi_rmse"] - summary["std_lhi_rmse"], summary["mean_lhi_rmse"] + summary["std_lhi_rmse"], color="#b45f06", alpha=0.14, label="± within-cycle std")
    if args.ignore_operating_condition:
        title = "N-CMAPSS unit 5: cruise-only pooled LHI versus cycle"
        footer = "Reference pools all operating conditions; each cycle mean is computed from accepted cruise time samples."
    else:
        title = f"N-CMAPSS unit 5: cruise-only K={n_clusters} condition-matched LHI versus cycle"
        footer = "Each cycle mean is computed from accepted cruise time samples and its K-means condition-specific reference."
    ax.set_title(title)
    ax.set_xlabel("cycle")
    ax.set_ylabel("mean LHI_RMSE within cruise cycle")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, footer, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.output_dir / "cruise_mean_lhi_vs_cycle.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=180)
    if summary["cycle"].min() <= args.reference_end_cycle:
        ax.axvspan(summary["cycle"].min() - 0.5, args.reference_end_cycle + 0.5, color="#dbeafe", alpha=0.8, label=f"health reference: cycles 1–{args.reference_end_cycle}")
    else:
        ax.axvline(args.report_start_cycle - 0.5, color="#6b7280", linestyle="--", linewidth=1.0, label=f"test start: cycle {args.report_start_cycle}")
    ax.axhline(0.0, color="#4b5563", linewidth=1.0)
    ax.plot(summary["cycle"], summary["max_lhi_rmse"], color="#7a3e9d", marker="o", markersize=3.5, linewidth=1.4, label="maximum LHI_RMSE per cycle")
    if args.ignore_operating_condition:
        title = "N-CMAPSS unit 5: cruise-only pooled maximum LHI versus cycle"
        footer = "Maximum is taken over valid cruise time samples; reference pools all operating conditions."
    else:
        title = f"N-CMAPSS unit 5: cruise-only K={n_clusters} condition-matched maximum LHI versus cycle"
        footer = "Maximum is taken over valid cruise time samples and the K-means condition-specific reference."
    ax.set_title(title)
    ax.set_xlabel("cycle")
    ax.set_ylabel("maximum LHI_RMSE within cruise cycle")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, footer, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.output_dir / "cruise_max_lhi_vs_cycle.png", bbox_inches="tight")
    plt.close(fig)

    metadata.update({
        "data_file": str(args.data_file),
        "split": args.split,
        "unit": int(args.unit),
        "cruise_only": True,
        "ignore_operating_condition": bool(args.ignore_operating_condition),
        "cluster_centers": str(args.cluster_centers),
        "standardization": str(args.standardization),
        "n_clusters": int(n_clusters),
        "reference_cycles": list(range(1, args.reference_end_cycle + 1)),
        "report_start_cycle": int(args.report_start_cycle),
        "reported_cycles": [int(v) for v in summary["cycle"]],
        "n_cruise_rows": int(len(scores)),
        "summary_definition": "mean sample-level LHI_RMSE within each accepted cruise cycle",
        "maximum_summary_definition": "maximum valid sample-level LHI_RMSE within each accepted cruise cycle",
    })
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(summary.to_string(index=False))
    print(f"Saved cycle-level LHI outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

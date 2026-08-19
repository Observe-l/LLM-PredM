#!/usr/bin/env python3
"""Compute all-stage, condition-matched LHI versus cycle for one unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.plot_n_cmapss_cruise_lhi import assign_cruise_clusters, load_unit
    from scripts.plot_n_cmapss_sample_lhi import SENSOR_NAMES, compute_sample_lhi
except ModuleNotFoundError:
    from plot_n_cmapss_cruise_lhi import assign_cruise_clusters, load_unit
    from plot_n_cmapss_sample_lhi import SENSOR_NAMES, compute_sample_lhi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/N-CMAPSS/N-CMAPSS_DS02-006.h5"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unit", type=int, default=5)
    parser.add_argument("--cluster-centers", type=Path, default=Path("outputs/N-CMAPSS-figure/all_stage_health_ref_cycles1-20_all_oc/all_stage_oc_cluster_centers_k6.csv"))
    parser.add_argument("--standardization", type=Path, default=Path("outputs/N-CMAPSS-figure/all_stage_health_ref_cycles1-20_all_oc/all_stage_oc_standardization.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N-CMAPSS-figure/all_stage_lhi_health_ref_cycles1-20_K6_DS02-006_dev_unit5"))
    parser.add_argument("--reference-end-cycle", type=int, default=20)
    parser.add_argument("--lhi-epsilon", type=float, default=1e-6)
    parser.add_argument("--range-epsilon", type=float, default=1e-12)
    return parser.parse_args()


def plot_summary(summary: pd.DataFrame, args: argparse.Namespace, n_clusters: int, value_col: str, ylabel: str, filename: str, label: str, color: str, footer: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=180)
    ax.axvspan(summary["cycle"].min() - 0.5, args.reference_end_cycle + 0.5, color="#dbeafe", alpha=0.8, label=f"health reference: cycles 1–{args.reference_end_cycle}")
    ax.axhline(0.0, color="#4b5563", linewidth=1.0)
    ax.plot(summary["cycle"], summary[value_col], color=color, marker="o", markersize=3.5, linewidth=1.4, label=label)
    ax.set_title(f"N-CMAPSS DS02-006/{args.split} unit {args.unit}: all-stage K={n_clusters} LHI versus cycle")
    ax.set_xlabel("cycle")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, footer, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.output_dir / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cycles, w, sensors, w_names, sensor_names = load_unit(args.data_file, args.split, args.unit)
    if sensor_names != SENSOR_NAMES:
        raise ValueError(f"Unexpected X_s variables: {sensor_names}")
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
    scores.to_csv(args.output_dir / "all_stage_lhi_scores_all_cycles.csv", index=False)
    summary = scores.groupby("cycle", sort=True).agg(
        stage_samples=("row_index", "size"),
        valid_samples=("valid_lhi", "sum"),
        mean_lhi_rmse=("lhi_rmse", "mean"),
        max_lhi_rmse=("lhi_rmse", "max"),
        median_lhi_rmse=("lhi_rmse", "median"),
        std_lhi_rmse=("lhi_rmse", lambda x: x.std(ddof=0)),
        mean_d_rmse=("d_rmse", "mean"),
    ).reset_index()
    summary["reference"] = summary["cycle"] <= args.reference_end_cycle
    summary.to_csv(args.output_dir / "all_stage_cycle_lhi_summary.csv", index=False)

    footer = "All stages; cluster features: alt, Mach, TRA, T2; LHI uses condition-specific health-reference baselines."
    plot_summary(summary, args, n_clusters, "mean_lhi_rmse", "mean LHI_RMSE within cycle", "all_stage_mean_lhi_vs_cycle.png", "mean LHI_RMSE per cycle", "#b45f06", footer)
    plot_summary(summary, args, n_clusters, "max_lhi_rmse", "maximum LHI_RMSE within cycle", "all_stage_max_lhi_vs_cycle.png", "maximum LHI_RMSE per cycle", "#7a3e9d", footer)

    metadata.update({
        "data_file": str(args.data_file),
        "split": args.split,
        "unit": int(args.unit),
        "all_stage": True,
        "cruise_only": False,
        "cluster_centers": str(args.cluster_centers),
        "standardization": str(args.standardization),
        "cluster_features": ["alt", "Mach", "TRA", "T2"],
        "n_clusters": int(n_clusters),
        "reference_cycles": list(range(1, args.reference_end_cycle + 1)),
        "reported_cycles": [int(v) for v in summary["cycle"]],
        "n_rows": int(len(scores)),
        "mean_summary_definition": "mean valid sample-level LHI_RMSE within each cycle",
        "maximum_summary_definition": "maximum valid sample-level LHI_RMSE within each cycle",
    })
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(summary[["cycle", "stage_samples", "valid_samples", "mean_lhi_rmse", "max_lhi_rmse"]].to_string(index=False))
    print(f"Saved all-stage LHI outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

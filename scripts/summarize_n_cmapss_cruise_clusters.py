#!/usr/bin/env python3
"""Summarize K-means cruise-stage cluster assignments by cycle."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    required = {"cycle", "operating_condition_cluster"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    scores["cycle"] = scores["cycle"].astype(int)
    scores["operating_condition_cluster"] = scores["operating_condition_cluster"].astype(int)
    cluster_ids = list(range(args.n_clusters))

    counts = pd.crosstab(
        scores["cycle"], scores["operating_condition_cluster"]
    ).reindex(columns=cluster_ids, fill_value=0)
    counts.index.name = "cycle"
    counts = counts.rename(columns={k: f"cluster_{k}_samples" for k in cluster_ids})
    counts["cruise_samples"] = counts.sum(axis=1)
    count_cols = [f"cluster_{k}_samples" for k in cluster_ids]
    for k in cluster_ids:
        counts[f"cluster_{k}_fraction"] = counts[f"cluster_{k}_samples"] / counts["cruise_samples"]
    counts["clusters_present"] = (counts[count_cols] > 0).sum(axis=1)
    counts = counts.reset_index()
    ordered = ["cycle", "cruise_samples", "clusters_present"] + count_cols + [
        f"cluster_{k}_fraction" for k in cluster_ids
    ]
    counts[ordered].to_csv(args.output_dir / "cruise_cluster_counts_by_cycle.csv", index=False)

    long = (
        scores.groupby(["cycle", "operating_condition_cluster"], sort=True)
        .size()
        .rename("cruise_samples")
        .reset_index()
        .rename(columns={"operating_condition_cluster": "cluster"})
    )
    totals = long.groupby("cycle", as_index=False)["cruise_samples"].sum().rename(
        columns={"cruise_samples": "cycle_cruise_samples"}
    )
    long = long.merge(totals, on="cycle", how="left")
    long["fraction"] = long["cruise_samples"] / long["cycle_cruise_samples"]
    long.to_csv(args.output_dir / "cruise_cluster_counts_long.csv", index=False)

    overall = (
        scores.groupby("operating_condition_cluster", sort=True)
        .size()
        .rename("cruise_samples")
        .reindex(cluster_ids, fill_value=0)
        .rename_axis("cluster")
        .reset_index()
    )
    overall["fraction"] = overall["cruise_samples"] / len(scores)
    overall["cycles_present"] = overall["cluster"].map(
        scores.groupby("operating_condition_cluster")["cycle"].nunique()
    ).fillna(0).astype(int)
    overall.to_csv(args.output_dir / "cruise_cluster_overall_summary.csv", index=False)

    dominant = long.loc[long.groupby("cycle")["cruise_samples"].idxmax()].copy()
    dominant = dominant.rename(
        columns={
            "cluster": "dominant_cluster",
            "cruise_samples": "dominant_cluster_samples",
            "fraction": "dominant_cluster_fraction",
        }
    )[["cycle", "dominant_cluster", "dominant_cluster_samples", "dominant_cluster_fraction"]]
    cycle_summary = counts[["cycle", "cruise_samples", "clusters_present"]].merge(
        dominant, on="cycle", how="left"
    )
    cycle_summary.to_csv(args.output_dir / "cruise_cluster_cycle_summary.csv", index=False)

    print("Overall cluster summary:")
    print(overall.to_string(index=False))
    print("\nCycles by number of present clusters:")
    print(counts["clusters_present"].value_counts().sort_index().rename("cycles").to_string())
    print(f"\nSaved cluster summaries to: {args.output_dir}")


if __name__ == "__main__":
    main()

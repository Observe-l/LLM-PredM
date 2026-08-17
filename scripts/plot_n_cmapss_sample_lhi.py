#!/usr/bin/env python3
"""Plot sample-level, condition-matched N-CMAPSS LHI.

The implementation follows the condition-matched part of
``src.zero_shot_cmapss.lhi_indicator`` while retaining N-CMAPSS's native
within-cycle time samples:

1. assign every row to a previously fitted N-CMAPSS operating-condition
   K-means model;
2. use every row in cycle 1 as the health reference;
3. calculate a reference mean for every (operating-condition cluster, sensor);
4. calculate one sensor drift for every target time sample, then aggregate the
   14 physical X_s sensors into one D_MAE and D_RMSE for that sample;
5. calculate LHI as the log ratio to the cycle-1 baseline within that same
   operating-condition cluster.

The x-axis is the sample index within the selected cycle, not the cycle number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


SENSOR_NAMES = [
    "T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
    "Ps30", "P40", "P50", "Nf", "Nc", "Wf",
]
CONDITION_NAMES = ["alt", "Mach", "TRA", "T2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/N-CMAPSS/N-CMAPSS_DS02-006.h5"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unit", type=int, default=2)
    parser.add_argument("--cluster-centers", type=Path, default=Path("outputs/N-CMAPSS-figure/cluster_centers_selected_k_standardized.csv"))
    parser.add_argument("--standardization", type=Path, default=Path("outputs/N-CMAPSS-figure/standardization.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N-CMAPSS-figure/sample_lhi_DS02-006_dev_unit2"))
    parser.add_argument("--reference-end-cycle", type=int, default=1, help="Use cycles 1..N as the health reference.")
    parser.add_argument("--lhi-epsilon", type=float, default=1e-6)
    parser.add_argument("--range-epsilon", type=float, default=1e-12)
    return parser.parse_args()


def load_engine(path: Path, split: str, unit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    with h5py.File(path, "r") as hdf:
        a = np.asarray(hdf[f"A_{split}"][:, :2], dtype=np.int64)
        selected = np.flatnonzero(a[:, 0] == unit)
        if len(selected) == 0:
            available = np.unique(a[:, 0]).tolist()
            raise ValueError(f"Unit {unit} not found in {path.name} {split}; available={available}")
        w = np.asarray(hdf[f"W_{split}"][selected], dtype=np.float64)
        x_s = np.asarray(hdf[f"X_s_{split}"][selected], dtype=np.float64)
        w_names = [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(hdf["W_var"]).reshape(-1)]
        sensor_names = [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(hdf["X_s_var"]).reshape(-1)]
    return a[selected, 1], w, x_s, w_names, sensor_names


def assign_clusters(w: np.ndarray, w_names: list[str], centers_path: Path, standardization_path: Path) -> tuple[np.ndarray, int]:
    centers_df = pd.read_csv(centers_path).sort_values("cluster")
    centers = centers_df.drop(columns=["cluster"]).loc[:, CONDITION_NAMES].to_numpy(dtype=float)
    standardization = pd.read_csv(standardization_path).set_index("feature").loc[CONDITION_NAMES]
    mean = standardization["mean"].to_numpy(dtype=float)
    scale = standardization["scale"].to_numpy(dtype=float)
    columns = [w_names.index(name) for name in CONDITION_NAMES]
    standardized = (w[:, columns] - mean) / scale
    distances = ((standardized[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(int), int(len(centers))


def compute_sample_lhi(
    cycles: np.ndarray,
    clusters: np.ndarray,
    sensors: np.ndarray,
    sensor_names: list[str],
    lhi_epsilon: float,
    range_epsilon: float,
    sensor_output_cycles: set[int] | None = None,
    n_clusters: int | None = None,
    reference_end_cycle: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    reference_start_cycle = int(cycles.min())
    reference_mask = (cycles >= reference_start_cycle) & (cycles <= reference_end_cycle)
    if not np.any(reference_mask):
        raise ValueError(f"No rows were found in reference cycles {reference_start_cycle}..{reference_end_cycle}.")
    reference_values = sensors[reference_mask]
    reference_clusters = clusters[reference_mask]
    n_clusters = int(n_clusters or (max(clusters.max(), reference_clusters.max()) + 1))

    # The range is deliberately computed inside each (cycle-1, OC) group.
    # This is the strict condition-specific min-max normalization requested by
    # the experiment, rather than a single range shared by all OCs.
    condition_means: dict[int, np.ndarray] = {}
    condition_ranges: dict[int, np.ndarray] = {}
    condition_usable: dict[int, np.ndarray] = {}
    for cluster in np.unique(reference_clusters):
        cluster_values = reference_values[reference_clusters == cluster]
        if len(cluster_values) == 0:
            continue
        sensor_min = cluster_values.min(axis=0)
        sensor_max = cluster_values.max(axis=0)
        sensor_range = sensor_max - sensor_min
        usable = np.isfinite(sensor_range) & (sensor_range > range_epsilon)
        condition_means[int(cluster)] = cluster_values.mean(axis=0)
        condition_ranges[int(cluster)] = sensor_range
        condition_usable[int(cluster)] = usable

    missing = sorted(set(np.unique(clusters)) - set(condition_means))

    normalized_drift = np.full((len(cycles), sensors.shape[1]), np.nan, dtype=float)
    for cluster, mean in condition_means.items():
        mask = clusters == cluster
        usable = condition_usable[cluster]
        row_indices = np.flatnonzero(mask)
        normalized_drift[np.ix_(row_indices, usable)] = np.abs(
            (sensors[mask][:, usable] - mean[usable]) / condition_ranges[cluster][usable]
        )

    # Per-cluster health baselines B, computed from every cycle-1 sample.
    baseline_by_cluster: dict[int, tuple[float, float]] = {}
    for cluster in sorted(condition_means):
        mask = reference_mask & (clusters == cluster)
        d = normalized_drift[mask]
        baseline_by_cluster[cluster] = (
            float(np.nanmean(np.nanmean(d, axis=1))),
            float(np.sqrt(np.nanmean(np.nanmean(d ** 2, axis=1)))),
        )

    time_samples = np.empty(len(cycles), dtype=np.int64)
    for cycle in np.unique(cycles):
        cycle_indices = np.flatnonzero(cycles == cycle)
        time_samples[cycle_indices] = np.arange(1, len(cycle_indices) + 1)

    # Vectorized sample-level aggregation.  This matters for N-CMAPSS because
    # one engine can contain hundreds of thousands of time samples.
    cluster_baseline_mae = np.full(n_clusters, np.nan, dtype=float)
    cluster_baseline_rmse = np.full(n_clusters, np.nan, dtype=float)
    sensor_baseline = np.full((n_clusters, sensors.shape[1]), np.nan, dtype=float)
    for cluster, (b_mae, b_rmse) in baseline_by_cluster.items():
        cluster_baseline_mae[cluster] = b_mae
        cluster_baseline_rmse[cluster] = b_rmse
        ref_d = normalized_drift[reference_mask & (clusters == cluster)]
        sensor_baseline[cluster] = np.nanmean(ref_d, axis=0)

    valid_sensor_count = np.isfinite(normalized_drift).sum(axis=1)
    d_mae = np.divide(
        np.nansum(normalized_drift, axis=1),
        valid_sensor_count,
        out=np.full(len(cycles), np.nan, dtype=float),
        where=valid_sensor_count > 0,
    )
    d_rmse = np.sqrt(
        np.divide(
            np.nansum(normalized_drift ** 2, axis=1),
            valid_sensor_count,
            out=np.full(len(cycles), np.nan, dtype=float),
            where=valid_sensor_count > 0,
        )
    )
    b_mae = cluster_baseline_mae[clusters]
    b_rmse = cluster_baseline_rmse[clusters]
    scores = pd.DataFrame({
        "row_index": np.arange(len(cycles), dtype=np.int64),
        "cycle": cycles.astype(np.int64),
        "time_sample": time_samples,
        "operating_condition_cluster": clusters,
        "d_mae": d_mae,
        "d_rmse": d_rmse,
        "b_mae": b_mae,
        "b_rmse": b_rmse,
        "lhi_mae": np.log((d_mae + lhi_epsilon) / (b_mae + lhi_epsilon)),
        "lhi_rmse": np.log((d_rmse + lhi_epsilon) / (b_rmse + lhi_epsilon)),
    })
    scores["valid_lhi"] = np.isfinite(scores["lhi_rmse"])

    sensor_mask = np.ones(len(cycles), dtype=bool) if sensor_output_cycles is None else np.isin(cycles, list(sensor_output_cycles))
    sensor_rows_count = int(sensor_mask.sum())
    sensor_d = normalized_drift[sensor_mask]
    sensor_b = sensor_baseline[clusters[sensor_mask]]
    usable_names = np.asarray(sensor_names)
    sensor_d_flat = sensor_d.reshape(-1)
    sensor_b_flat = sensor_b.reshape(-1)
    valid_sensor_flat = np.isfinite(sensor_d_flat) & np.isfinite(sensor_b_flat)
    sensor_scores = pd.DataFrame({
        "row_index": np.repeat(np.flatnonzero(sensor_mask), len(usable_names)),
        "cycle": np.repeat(cycles[sensor_mask], len(usable_names)),
        "time_sample": np.repeat(time_samples[sensor_mask], len(usable_names)),
        "operating_condition_cluster": np.repeat(clusters[sensor_mask], len(usable_names)),
        "sensor": np.tile(usable_names, sensor_rows_count),
        "d": sensor_d_flat,
        "b": sensor_b_flat,
    })
    sensor_scores = sensor_scores.loc[valid_sensor_flat].reset_index(drop=True)
    sensor_scores["lhi"] = np.log((sensor_scores["d"] + lhi_epsilon) / (sensor_scores["b"] + lhi_epsilon))
    metadata = {
        "reference": f"all rows in cycles {reference_start_cycle}..{reference_end_cycle}",
        "reference_start_cycle": reference_start_cycle,
        "reference_end_cycle": reference_end_cycle,
        "sensor_block": "X_s physical sensor readings",
        "sensor_names": sensor_names,
        "usable_sensor_names": sensor_names,
        "condition_clusters": f"previously fitted N-CMAPSS K={n_clusters} K-means centers",
        "condition_mean": "mean sensor value within (cycle-1, operating-condition cluster)",
        "sensor_range": "min-max range within (cycle-1, operating-condition cluster), per sensor",
        "baseline": "mean cycle-1 drift within the same operating-condition cluster",
        "lhi_formula": "log((D + epsilon) / (B + epsilon))",
        "lhi_epsilon": lhi_epsilon,
        "range_epsilon": range_epsilon,
        "condition_cluster_counts_cycle1": {str(k): int(v) for k, v in zip(*np.unique(reference_clusters, return_counts=True))},
        "clusters_without_cycle1_reference": [int(value) for value in missing],
    }
    return scores, sensor_scores, metadata


def plot_cycle(scores: pd.DataFrame, cycle: int, output: Path, metadata: dict[str, object], n_clusters: int) -> None:
    frame = scores[scores["cycle"] == cycle].copy()
    if frame.empty:
        raise ValueError(f"No rows for cycle {cycle}")
    frame["time_sample"] = np.arange(1, len(frame) + 1)
    cmap = plt.get_cmap("turbo", n_clusters)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    x = frame["time_sample"].to_numpy()
    cluster = frame["operating_condition_cluster"].to_numpy()
    pooled = bool(metadata.get("ignore_operating_condition", False))
    drift_title = "pooled sample drift (no operating-condition grouping)" if pooled else "condition-matched sample drift"
    baseline_label = "pooled health-reference baseline" if pooled else "condition baseline B_RMSE"
    axes[0].plot(x, frame["d_rmse"], color="#2f5d8c", linewidth=0.9, label="D_RMSE")
    axes[0].plot(x, frame["b_rmse"], color="#6b7280", linewidth=1.2, linestyle="--", label=baseline_label)
    axes[0].set_ylabel("drift")
    axes[0].set_title(f"N-CMAPSS cycle {cycle}: {drift_title}")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(x, frame["lhi_rmse"], c=cluster, cmap=cmap, vmin=0, vmax=max(1, n_clusters - 1), s=5, alpha=0.72, linewidths=0)
    axes[1].plot(x, frame["lhi_rmse"], color="#b45f06", alpha=0.35, linewidth=0.65)
    axes[1].axhline(0.0, color="#4b5563", linewidth=1.0)
    axes[1].set_xlabel("time sample within cycle")
    axes[1].set_ylabel("LHI_RMSE")
    valid_count = int(frame["valid_lhi"].sum())
    axes[1].set_title(f"N-CMAPSS cycle {cycle}: sample-level LHI (valid={valid_count:,}/{len(frame):,})")
    axes[1].grid(alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(k), markersize=6, label=f"group {k}") for k in range(n_clusters)]
    axes[1].legend(handles=handles, title=("pooled reference" if pooled else f"K={n_clusters} condition"), ncol=4, loc="upper right", fontsize=7)
    reference_label = str(metadata.get("reference", "health reference"))
    fig.suptitle(f"{reference_label}; each point is one time sample", fontsize=14)
    fig.text(0.01, 0.01, "Positive LHI means greater condition-matched normalized drift than the health-reference baseline.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cycles, w, sensors, w_names, sensor_names = load_engine(args.data_file, args.split, args.unit)
    if sensor_names != SENSOR_NAMES:
        raise ValueError(f"Unexpected X_s variables: {sensor_names}")
    clusters, n_clusters = assign_clusters(w, w_names, args.cluster_centers, args.standardization)
    if args.reference_end_cycle < int(cycles.min()):
        raise ValueError("--reference-end-cycle must be >= the first observed cycle.")
    target_cycles = [args.reference_end_cycle + 1, int(cycles.max())]
    target_cycles = list(dict.fromkeys(cycle for cycle in target_cycles if cycle <= int(cycles.max())))
    scores, sensor_scores, metadata = compute_sample_lhi(
        cycles,
        clusters,
        sensors,
        sensor_names,
        args.lhi_epsilon,
        args.range_epsilon,
        sensor_output_cycles=set(target_cycles),
        n_clusters=n_clusters,
        reference_end_cycle=args.reference_end_cycle,
    )
    metadata.update({"data_file": str(args.data_file), "split": args.split, "unit": int(args.unit), "last_cycle": int(cycles.max()), "n_rows": int(len(cycles))})
    scores.to_csv(args.output_dir / "sample_lhi_scores.csv", index=False)
    sensor_scores.to_csv(args.output_dir / "sample_sensor_lhi_scores.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    target_scores = scores[scores["cycle"].isin(target_cycles)].copy()
    statistic_rows = []
    for cycle, group in target_scores.groupby("cycle", sort=True):
        valid = group[group["valid_lhi"]]
        statistic_rows.append({
            "cycle": int(cycle),
            "total_samples": int(len(group)),
            "valid_samples": int(len(valid)),
            "invalid_samples": int(len(group) - len(valid)),
            "valid_fraction": float(len(valid) / len(group)),
            "cluster_count_total": int(group["operating_condition_cluster"].nunique()),
            "cluster_count_valid": int(valid["operating_condition_cluster"].nunique()),
            "lhi_rmse_mean": float(valid["lhi_rmse"].mean()) if len(valid) else np.nan,
            "lhi_rmse_median": float(valid["lhi_rmse"].median()) if len(valid) else np.nan,
            "lhi_rmse_std": float(valid["lhi_rmse"].std(ddof=0)) if len(valid) else np.nan,
            "lhi_rmse_p05": float(valid["lhi_rmse"].quantile(0.05)) if len(valid) else np.nan,
            "lhi_rmse_p95": float(valid["lhi_rmse"].quantile(0.95)) if len(valid) else np.nan,
            "lhi_rmse_min": float(valid["lhi_rmse"].min()) if len(valid) else np.nan,
            "lhi_rmse_max": float(valid["lhi_rmse"].max()) if len(valid) else np.nan,
        })
    pd.DataFrame(statistic_rows).to_csv(args.output_dir / "sample_lhi_statistics.csv", index=False)
    target_scores.groupby(["cycle", "operating_condition_cluster"], sort=True).agg(
        total_samples=("row_index", "size"),
        valid_samples=("valid_lhi", "sum"),
    ).reset_index().to_csv(args.output_dir / "condition_sample_counts_target_cycles.csv", index=False)
    for cycle in target_cycles:
        plot_cycle(scores, cycle, args.output_dir / f"lhi_vs_time_samples_cycle_{cycle:03d}.png", metadata, n_clusters)
    print(pd.DataFrame(statistic_rows).to_string(index=False))
    print(f"Saved sample-level LHI outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

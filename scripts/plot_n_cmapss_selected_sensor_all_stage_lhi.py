#!/usr/bin/env python3
"""Plot all-stage N-CMAPSS LHI for a selected sensor subset and K values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.plot_n_cmapss_cruise_lhi import assign_cruise_clusters, load_unit
    from scripts.plot_n_cmapss_sample_lhi import compute_sample_lhi
except ModuleNotFoundError:
    from plot_n_cmapss_cruise_lhi import assign_cruise_clusters, load_unit
    from plot_n_cmapss_sample_lhi import compute_sample_lhi


SELECTED_SENSORS = ["W50", "SmFan", "SmLPC", "Wf", "T24", "T30", "T48", "T50"]
DEFAULT_CLUSTER_ROOT = Path("outputs/N-CMAPSS-figure/all_stage_health_ref_cycles1-20_all_oc")


def load_unit_with_static_and_virtual_sensors(path: Path, split: str, unit_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load A/W plus both X_s and X_v blocks from one N-CMAPSS unit."""
    with h5py.File(path, "r") as hdf:
        a_names = [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(hdf["A_var"]).reshape(-1)]
        w_names = [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(hdf["W_var"]).reshape(-1)]
        xs_names = [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(hdf["X_s_var"]).reshape(-1)]
        xv_names = [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(hdf["X_v_var"]).reshape(-1)]
        unit_col = a_names.index("unit")
        cycle_col = a_names.index("cycle")
        units = np.asarray(hdf[f"A_{split}"][:, unit_col], dtype=np.int32)
        selected = np.flatnonzero(units == unit_id)
        if len(selected) == 0:
            raise ValueError(f"Unit {unit_id} not found in {path.name} {split}")
        a = np.asarray(hdf[f"A_{split}"][selected, cycle_col], dtype=np.int32)
        w = np.asarray(hdf[f"W_{split}"][selected], dtype=np.float64)
        xs = np.asarray(hdf[f"X_s_{split}"][selected], dtype=np.float64)
        xv = np.asarray(hdf[f"X_v_{split}"][selected], dtype=np.float64)
    return a, w, np.concatenate([xs, xv], axis=1), w_names, xs_names + xv_names


def compute_health_variation_lhi(
    cycles: np.ndarray,
    clusters: np.ndarray,
    sensors: np.ndarray,
    reference_end_cycle: int,
    lhi_epsilon: float,
    range_epsilon: float,
    variation_epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], pd.DataFrame]:
    """Compute RMSE-LHI weighted by per-sensor health-reference variation.

    The health variation is the population variance of each sensor's absolute
    min-max-normalized reference values within its operating-condition cluster.
    """
    reference_mask = (cycles >= int(cycles.min())) & (cycles <= reference_end_cycle)
    n_clusters = int(clusters.max()) + 1
    normalized = np.full_like(sensors, np.nan, dtype=float)
    weighted_sq = np.full_like(sensors, np.nan, dtype=float)
    baseline = np.full(n_clusters, np.nan, dtype=float)
    variation_rows: list[dict[str, object]] = []

    for cluster in np.unique(clusters):
        cluster_reference = reference_mask & (clusters == cluster)
        if not np.any(cluster_reference):
            continue
        reference_values = sensors[cluster_reference]
        mean = np.mean(reference_values, axis=0)
        sensor_range = np.max(reference_values, axis=0) - np.min(reference_values, axis=0)
        usable_range = np.isfinite(sensor_range) & (sensor_range > range_epsilon)
        reference_normalized = np.full_like(reference_values, np.nan, dtype=float)
        reference_normalized[:, usable_range] = np.abs(
            (reference_values[:, usable_range] - mean[usable_range]) / sensor_range[usable_range]
        )
        health_variation = np.nanvar(reference_normalized, axis=0)
        usable = usable_range & np.isfinite(health_variation) & (health_variation > variation_epsilon)
        all_cluster = clusters == cluster
        # Assign by explicit row/column indices because boolean fancy indexing
        # returns a copy rather than a writable view.
        row_indices = np.flatnonzero(all_cluster)
        usable_indices = np.flatnonzero(usable)
        normalized[np.ix_(row_indices, usable_indices)] = np.abs(
            (sensors[np.ix_(row_indices, usable_indices)] - mean[usable_indices]) / sensor_range[usable_indices]
        )
        weighted_sq[np.ix_(row_indices, usable_indices)] = normalized[np.ix_(row_indices, usable_indices)] ** 2 / health_variation[usable_indices]
        reference_weighted = weighted_sq[np.ix_(np.flatnonzero(cluster_reference), usable_indices)]
        reference_d = np.sqrt(np.nanmean(reference_weighted, axis=1))
        baseline[int(cluster)] = float(np.nanmean(reference_d**2) ** 0.5)
        for sensor_index, sensor_name in enumerate(SELECTED_SENSORS):
            variation_rows.append({
                "cluster": int(cluster),
                "sensor": sensor_name,
                "reference_mean": float(mean[sensor_index]),
                "reference_minmax_range": float(sensor_range[sensor_index]),
                "health_variation_normalized_variance": float(health_variation[sensor_index]) if np.isfinite(health_variation[sensor_index]) else np.nan,
                "usable": bool(usable[sensor_index]),
                "reference_samples": int(np.sum(cluster_reference)),
            })

    valid_count = np.isfinite(weighted_sq).sum(axis=1)
    d_rmse_hv = np.sqrt(
        np.divide(
            np.nansum(weighted_sq, axis=1),
            valid_count,
            out=np.full(len(cycles), np.nan, dtype=float),
            where=valid_count > 0,
        )
    )
    b = baseline[clusters]
    lhi_rmse_hv = np.log((d_rmse_hv + lhi_epsilon) / (b + lhi_epsilon))
    metadata = {
        "health_variation_definition": "population variance of absolute min-max-normalized sensor values in cycles 1..reference_end_cycle within each operating-condition cluster",
        "new_drift_definition": "sqrt(mean(sensor_normalized_drift^2 / (health_variation + epsilon)))",
        "new_lhi_definition": "log((D_RMSE_HV + epsilon) / (B_RMSE_HV + epsilon))",
        "variation_epsilon": variation_epsilon,
        "baseline_rmse_hv_by_cluster": {str(i): float(value) for i, value in enumerate(baseline) if np.isfinite(value)},
    }
    return lhi_rmse_hv, d_rmse_hv, metadata, pd.DataFrame(variation_rows)


def plot_combined(summaries: dict[int, pd.DataFrame], output: Path, reference_end_cycle: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10.5), sharex=True, dpi=180)
    styles = {
        4: {"color": "#2457a6", "marker": "o"},
        6: {"color": "#b54708", "marker": "s"},
    }
    for ax, value_col, ylabel in [
        (axes[0], "mean_lhi_rmse", "mean LHI_RMSE within cycle"),
        (axes[1], "mean_lhi_mae", "mean LHI_MAE within cycle"),
        (axes[2], "mean_lhi_rmse_hv", "mean LHI_RMSE_HV within cycle"),
    ]:
        ax.axvspan(0.5, reference_end_cycle + 0.5, color="#dbeafe", alpha=0.8, label=f"health reference: cycles 1–{reference_end_cycle}")
        ax.axhline(0.0, color="#4b5563", linewidth=0.9)
        for k, summary in summaries.items():
            style = styles.get(k, {"color": "#333333", "marker": "o"})
            ax.plot(summary["cycle"], summary[value_col], linewidth=1.4, markersize=3.2, label=f"K={k}", **style)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("cycle")
    fig.suptitle("N-CMAPSS DS02-006 dev unit 5: selected-sensor all-stage LHI", fontsize=15)
    fig.text(
        0.01,
        0.01,
        "LHI sensors: W50, SmFan, SmLPC, Wf, T24, T30, T48, T50; cluster features: alt, Mach, TRA, T2; HV uses health-reference normalized variance.",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/N-CMAPSS/N-CMAPSS_DS02-006.h5"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unit", type=int, default=5)
    parser.add_argument("--cluster-root", type=Path, default=DEFAULT_CLUSTER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N-CMAPSS-figure/all_stage_selected_sensor_lhi_DS02-006_dev_unit5_ref_cycles1-20"))
    parser.add_argument("--reference-end-cycle", type=int, default=20)
    parser.add_argument("--ks", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--lhi-epsilon", type=float, default=1e-6)
    parser.add_argument("--range-epsilon", type=float, default=1e-12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cycles, w, sensors, w_names, sensor_names = load_unit_with_static_and_virtual_sensors(args.data_file, args.split, args.unit)
    missing_sensors = sorted(set(SELECTED_SENSORS) - set(sensor_names))
    if missing_sensors:
        raise ValueError(f"Missing requested sensors: {missing_sensors}; available={sensor_names}")
    sensor_indices = [sensor_names.index(name) for name in SELECTED_SENSORS]
    selected_sensors = sensors[:, sensor_indices]
    summaries: dict[int, pd.DataFrame] = {}
    run_metadata: dict[str, object] = {
        "data_file": str(args.data_file),
        "split": args.split,
        "unit": args.unit,
        "all_stage": True,
        "cruise_only": False,
        "reference_cycles": list(range(1, args.reference_end_cycle + 1)),
        "lhi_sensors": SELECTED_SENSORS,
        "sensor_blocks": {"X_s": [name for name in sensor_names if name in set(sensor_names[:14])], "X_v": [name for name in sensor_names if name in set(sensor_names[14:])]},
        "cluster_features": ["alt", "Mach", "TRA", "T2"],
        "cluster_scope": "global all-stage K-means fitted using all readable units and dev/test rows from cycles 1-20",
        "summary_definition": "mean sample-level LHI within each cycle",
        "ks": args.ks,
    }

    for k in args.ks:
        if k == 4:
            centers_path = args.cluster_root / "all_stage_oc_cluster_centers_selected_k.csv"
        else:
            centers_path = args.cluster_root / f"all_stage_oc_cluster_centers_k{k}.csv"
        standardization_path = args.cluster_root / "all_stage_oc_standardization.csv"
        if not centers_path.exists():
            raise FileNotFoundError(f"No cluster centers for K={k}: {centers_path}")
        clusters, n_clusters = assign_cruise_clusters(w, w_names, centers_path, standardization_path)
        scores, _, metadata = compute_sample_lhi(
            cycles=cycles,
            clusters=clusters,
            sensors=selected_sensors,
            sensor_names=SELECTED_SENSORS,
            lhi_epsilon=args.lhi_epsilon,
            range_epsilon=args.range_epsilon,
            sensor_output_cycles=set(),
            n_clusters=n_clusters,
            reference_end_cycle=args.reference_end_cycle,
        )
        lhi_rmse_hv, d_rmse_hv, hv_metadata, variation_table = compute_health_variation_lhi(
            cycles=cycles,
            clusters=clusters,
            sensors=selected_sensors,
            reference_end_cycle=args.reference_end_cycle,
            lhi_epsilon=args.lhi_epsilon,
            range_epsilon=args.range_epsilon,
        )
        scores["sensor_set"] = ",".join(SELECTED_SENSORS)
        scores["d_rmse_hv"] = d_rmse_hv
        scores["lhi_rmse_hv"] = lhi_rmse_hv
        scores.to_csv(args.output_dir / f"all_stage_selected_sensor_lhi_scores_K{k}.csv", index=False)
        variation_table.to_csv(args.output_dir / f"all_stage_selected_sensor_health_variation_K{k}.csv", index=False)
        summary = scores.groupby("cycle", sort=True).agg(
            stage_samples=("row_index", "size"),
            valid_samples=("valid_lhi", "sum"),
            mean_lhi_rmse=("lhi_rmse", "mean"),
            median_lhi_rmse=("lhi_rmse", "median"),
            max_lhi_rmse=("lhi_rmse", "max"),
            std_lhi_rmse=("lhi_rmse", lambda x: x.std(ddof=0)),
            mean_lhi_mae=("lhi_mae", "mean"),
            median_lhi_mae=("lhi_mae", "median"),
            max_lhi_mae=("lhi_mae", "max"),
            mean_d_rmse=("d_rmse", "mean"),
            mean_lhi_rmse_hv=("lhi_rmse_hv", "mean"),
            median_lhi_rmse_hv=("lhi_rmse_hv", "median"),
            max_lhi_rmse_hv=("lhi_rmse_hv", "max"),
            mean_d_rmse_hv=("d_rmse_hv", "mean"),
        ).reset_index()
        summary["reference"] = summary["cycle"] <= args.reference_end_cycle
        summary["cluster_count"] = scores.groupby("cycle")["operating_condition_cluster"].nunique().to_numpy()
        summary.to_csv(args.output_dir / f"all_stage_selected_sensor_cycle_lhi_summary_K{k}.csv", index=False)
        summaries[k] = summary
        run_metadata[f"K{k}"] = {
            "cluster_centers": str(centers_path),
            "standardization": str(standardization_path),
            "n_clusters": n_clusters,
            "n_rows": len(scores),
            "condition_cluster_counts_reference": metadata.get("condition_cluster_counts_cycle1"),
            "clusters_without_reference": metadata.get("clusters_without_cycle1_reference"),
            "health_variation": hv_metadata,
        }

    combined_path = args.output_dir / "all_stage_selected_sensor_lhi_vs_cycle_K4_K6.png"
    plot_combined(summaries, combined_path, args.reference_end_cycle)
    run_metadata["combined_plot"] = str(combined_path)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nCycle-level LHI summary:")
    for k, summary in summaries.items():
        print(f"\nK={k}")
        print(summary[["cycle", "stage_samples", "valid_samples", "mean_lhi_rmse", "mean_lhi_mae", "mean_lhi_rmse_hv", "cluster_count"]].to_string(index=False))
    print(f"\nSaved selected-sensor all-stage LHI outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

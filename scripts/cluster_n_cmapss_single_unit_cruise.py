#!/usr/bin/env python3
"""Select K and draw t-SNE for one N-CMAPSS unit's cruise samples."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from scripts.cluster_n_cmapss_cruise import choose_k, fit_kmeans, make_tsne, save_metric_plot, save_tsne_plot
    from scripts.plot_n_cmapss_cruise_lhi import cycle_local_samples, load_unit
except ModuleNotFoundError:
    from cluster_n_cmapss_cruise import choose_k, fit_kmeans, make_tsne, save_metric_plot, save_tsne_plot
    from plot_n_cmapss_cruise_lhi import cycle_local_samples, load_unit


FEATURE_NAMES = ["alt", "Mach", "TRA", "T2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/N-CMAPSS/N-CMAPSS_DS02-006.h5"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unit", type=int, default=5)
    parser.add_argument("--cruise-statistics", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude/cruise_cycle_statistics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N-CMAPSS-figure/cruise_unit_cluster_DS02-006_dev_unit5"))
    parser.add_argument("--fit-samples", type=int, default=240_000)
    parser.add_argument("--silhouette-samples", type=int, default=60_000)
    parser.add_argument("--tsne-samples", type=int, default=40_000)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tsne-max-iter", type=int, default=1_000)
    return parser.parse_args()


def stratified_indices(metadata: pd.DataFrame, target_size: int, random_state: int) -> np.ndarray:
    target_size = min(target_size, len(metadata))
    if target_size == len(metadata):
        return np.arange(len(metadata))
    rng = np.random.default_rng(random_state)
    groups = metadata.groupby("cycle", sort=False).indices
    per_group = max(1, target_size // len(groups))
    selected: list[np.ndarray] = []
    for indices in groups.values():
        indices = np.asarray(indices)
        selected.append(rng.choice(indices, size=min(len(indices), per_group), replace=False))
    chosen = np.concatenate(selected)
    if len(chosen) < target_size:
        remaining = np.setdiff1d(np.arange(len(metadata)), chosen, assume_unique=False)
        chosen = np.concatenate([chosen, rng.choice(remaining, size=target_size - len(chosen), replace=False)])
    rng.shuffle(chosen)
    return chosen


def scan_k(x_fit: np.ndarray, x_silhouette: np.ndarray, k_values: list[int], random_state: int) -> pd.DataFrame:
    rows = []
    for k in k_values:
        print(f"  evaluating K={k} ...", flush=True)
        model = fit_kmeans(x_fit, k, random_state)
        labels = model.predict(x_silhouette)
        rows.append({
            "k": k,
            "wcss": float(model.inertia_),
            "silhouette": float(silhouette_score(x_silhouette, labels)),
        })
    return pd.DataFrame(rows)


def save_unit_tsne_plot(embedding: np.ndarray, labels: np.ndarray, k: int, path: Path, seed: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 9), dpi=180)
    cmap = plt.get_cmap("turbo", k)
    for cluster in range(k):
        mask = labels == cluster
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.45,
                   color=cmap(cluster), linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i),
                       markersize=6, label=f"Cluster {i}") for i in range(k)]
    ax.legend(handles=handles, title="K-means cluster", ncol=min(4, k),
              loc="upper right", frameon=True)
    ax.set_title(f"N-CMAPSS DS02-006/dev unit 5 cruise states: t-SNE (selected K={k})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.text(0.01, 0.01,
            f"Cruise W variables: {', '.join(FEATURE_NAMES)}\nn={len(embedding):,}; seed={seed}",
            transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8", "pad": 4})
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
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

    cycles, w, _, w_names, _ = load_unit(args.data_file, args.split, args.unit)
    local = cycle_local_samples(cycles)
    starts = np.full(len(cycles), -1.0)
    ends = np.full(len(cycles), -1.0)
    lookup = intervals.set_index("cycle")
    for cycle in lookup.index.astype(int):
        mask = cycles == cycle
        starts[mask] = lookup.loc[cycle, "cruise_start_sample"]
        ends[mask] = lookup.loc[cycle, "cruise_end_sample"]
    cruise_mask = np.isfinite(starts) & (local >= starts) & (local <= ends)
    cycles = cycles[cruise_mask]
    w = w[cruise_mask]
    columns = [w_names.index(name) for name in FEATURE_NAMES]
    raw_x = w[:, columns].astype(np.float64)
    metadata = pd.DataFrame({"cycle": cycles, "cycle_local_sample": local[cruise_mask]})

    scaler = StandardScaler()
    x = scaler.fit_transform(raw_x).astype(np.float32)
    pd.DataFrame({"feature": FEATURE_NAMES, "mean": scaler.mean_, "scale": scaler.scale_}).to_csv(
        args.output_dir / "unit_cruise_standardization.csv", index=False
    )
    fit_indices = stratified_indices(metadata, args.fit_samples, args.random_state + 1)
    fit_metadata = metadata.iloc[fit_indices].reset_index(drop=True)
    x_fit = x[fit_indices]
    sil_indices = stratified_indices(fit_metadata, args.silhouette_samples, args.random_state + 2)
    x_silhouette = x_fit[sil_indices]

    metrics = scan_k(x_fit, x_silhouette, list(range(args.k_min, args.k_max + 1)), args.random_state)
    selected_k, selection_info = choose_k(metrics)
    metrics["selected_k"] = metrics["k"].eq(selected_k)
    metrics.to_csv(args.output_dir / "unit_k_selection_metrics.csv", index=False)
    save_metric_plot(metrics, selected_k, args.output_dir / "unit_k_selection_wcss_silhouette.png")

    selected = fit_kmeans(x_fit, selected_k, args.random_state)
    centers_raw = scaler.inverse_transform(selected.cluster_centers_)
    pd.DataFrame(centers_raw, columns=FEATURE_NAMES).assign(cluster=np.arange(selected_k)).to_csv(
        args.output_dir / "unit_cluster_centers_selected_k.csv", index=False
    )

    tsne_indices = stratified_indices(fit_metadata, args.tsne_samples, args.random_state + 3)
    x_tsne = x_fit[tsne_indices]
    tsne_metadata = fit_metadata.iloc[tsne_indices].reset_index(drop=True)
    print(f"Computing t-SNE for {len(x_tsne):,} sampled cruise rows ...", flush=True)
    embedding = make_tsne(x_tsne, args.random_state, args.tsne_max_iter)
    labels = selected.predict(x_tsne)
    points = tsne_metadata.copy()
    points["tsne_1"] = embedding[:, 0]
    points["tsne_2"] = embedding[:, 1]
    points["cluster"] = labels
    points.to_csv(args.output_dir / "unit_tsne_selected_k_points.csv", index=False)
    save_unit_tsne_plot(embedding, labels, selected_k, args.output_dir / "unit_tsne_selected_k.png", args.random_state)

    metadata_out = {
        "data_file": str(args.data_file),
        "dataset": "DS02-006",
        "split": args.split,
        "unit": args.unit,
        "cruise_method": "CruiseBench common_altitude; accepted intervals only",
        "cluster_features": FEATURE_NAMES,
        "clustering_scope": "only this unit's accepted cruise samples",
        "total_cycles": int(intervals["cycle"].nunique()),
        "cruise_samples": int(len(x)),
        "fit_samples": int(len(x_fit)),
        "silhouette_samples": int(len(x_silhouette)),
        "tsne_samples": int(len(x_tsne)),
        "k_range": [args.k_min, args.k_max],
        "selected_k": int(selected_k),
        "selection": selection_info,
        "random_state": args.random_state,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "unit_run_metadata.json").write_text(
        json.dumps(metadata_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Selected K={selected_k}; saved outputs to {args.output_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

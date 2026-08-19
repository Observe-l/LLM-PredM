#!/usr/bin/env python3
"""Re-label the existing all-stage t-SNE sample with a fixed K-means K."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    from scripts.cluster_n_cmapss_health_ref_cycles1_20_all_oc import fit_model, stratified_indices
except ModuleNotFoundError:
    from cluster_n_cmapss_health_ref_cycles1_20_all_oc import fit_model, stratified_indices


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N-CMAPSS-figure" / "all_stage_health_ref_cycles1-20_all_oc"
FEATURES = ["alt", "Mach", "TRA", "T2"]
SEED = 42
K = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True, help="Fixed K for K-means relabeling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k = int(args.k)
    if k < 2:
        raise ValueError("--k must be at least 2")
    with np.load(OUT / "all_stage_health_reference_features.npz") as data:
        raw = np.asarray(data["features"], dtype=np.float32)
    scaling = pd.read_csv(OUT / "all_stage_oc_standardization.csv").set_index("feature").loc[FEATURES]
    x = ((raw - scaling["mean"].to_numpy()) / scaling["scale"].to_numpy()).astype(np.float32)
    cycle_meta = pd.read_csv(OUT / "all_stage_health_reference_cycle_statistics.csv")
    groups = np.repeat(np.arange(len(cycle_meta), dtype=np.int32), cycle_meta["stage_samples"].to_numpy(dtype=np.int64))
    fit_idx = stratified_indices(groups, min(120_000, len(x)), SEED + 1)
    model = fit_model(x[fit_idx], k)

    centers = pd.DataFrame(model.cluster_centers_ * scaling["scale"].to_numpy() + scaling["mean"].to_numpy(), columns=FEATURES)
    centers.insert(0, "cluster", np.arange(k))
    centers.to_csv(OUT / f"all_stage_oc_cluster_centers_k{k}.csv", index=False)

    points = pd.read_csv(OUT / "all_stage_oc_tsne_selected_k_points.csv")
    point_x = ((points[FEATURES].to_numpy(dtype=np.float32) - scaling["mean"].to_numpy()) / scaling["scale"].to_numpy()).astype(np.float32)
    points["cluster"] = model.predict(point_x)
    points.to_csv(OUT / f"all_stage_oc_tsne_k{k}_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    cmap = plt.get_cmap("tab20", k)
    embedding = points[["tsne_1", "tsne_2"]].to_numpy()
    labels = points["cluster"].to_numpy()
    for cluster in range(k):
        mask = labels == cluster
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.45, color=cmap(cluster), linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i), markersize=6, label=f"Cluster {i}") for i in range(k)]
    ax.legend(handles=handles, title="K-means cluster", ncol=3, loc="upper right", frameon=True)
    ax.set_title(f"N-CMAPSS all-stage health reference t-SNE: K={k}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.01, 0.01, f"All stages; first 20 cycles per unit; features: alt, Mach, TRA, T2\nn={len(points):,}; seed={SEED}; same t-SNE coordinates as K=4 figure", transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom", bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.8", "pad": 4})
    fig.tight_layout()
    fig.savefig(OUT / f"all_stage_oc_tsne_k{k}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "k": k,
        "features": FEATURES,
        "fit_rows": int(len(fit_idx)),
        "tsne_rows": int(len(points)),
        "random_state": SEED,
        "tsne_coordinates_reused_from": str(OUT / "all_stage_oc_tsne_selected_k_points.csv"),
        "source_run_metadata": str(OUT / "all_stage_oc_run_metadata.json"),
    }
    (OUT / f"all_stage_oc_tsne_k{k}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

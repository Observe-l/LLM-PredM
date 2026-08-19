#!/usr/bin/env python3
"""Add a K=6 view to the DS02-006 cruise OC-only t-SNE run."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N-CMAPSS-figure" / "cruise_oc3_DS02-006"
INPUT = ROOT / "outputs" / "N-CMAPSS-figure" / "cruise_common_altitude" / "cruise_cluster_sample.csv"
FEATURES = ["Mach", "TRA", "T2"]
K = 6


def main() -> None:
    source = pd.read_csv(INPUT)
    source = source.loc[source["dataset"].eq("DS02-006")].reset_index(drop=True)
    standardization = pd.read_csv(OUT / "ds02_oc3_standardization.csv").set_index("feature")
    mean = standardization.loc[FEATURES, "mean"].to_numpy(dtype=float)
    scale = standardization.loc[FEATURES, "scale"].to_numpy(dtype=float)
    x = ((source[FEATURES].to_numpy(dtype=float) - mean) / scale).astype(np.float32)

    model = MiniBatchKMeans(
        n_clusters=K,
        init="k-means++",
        n_init=3,
        max_iter=100,
        batch_size=4096,
        init_size=min(len(x), max(3 * 4096, 3 * K)),
        random_state=42,
    )
    model.fit(x)

    centers = pd.DataFrame(model.cluster_centers_ * scale + mean, columns=FEATURES)
    centers.insert(0, "cluster", np.arange(K))
    centers.to_csv(OUT / "ds02_oc3_cluster_centers_k6.csv", index=False)

    points = pd.read_csv(OUT / "ds02_oc3_tsne_selected_k_points.csv")
    points_x = ((points[FEATURES].to_numpy(dtype=float) - mean) / scale).astype(np.float32)
    labels = model.predict(points_x)
    points["cluster"] = labels
    points.to_csv(OUT / "ds02_oc3_tsne_k6_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    cmap = plt.get_cmap("tab10", K)
    for cluster in range(K):
        mask = labels == cluster
        ax.scatter(points.loc[mask, "tsne_1"], points.loc[mask, "tsne_2"], s=5, alpha=0.45, color=cmap(cluster), linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i), markersize=6, label=f"Cluster {i}") for i in range(K)]
    ax.legend(handles=handles, title="K-means cluster", ncol=3, loc="upper right", frameon=True)
    ax.set_title("DS02-006 cruise OC-only t-SNE: K=6")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.01, 0.01, f"Features: Mach, TRA, T2\nn={len(points):,}; seed=42; altitude excluded", transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom", bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.8", "pad": 4})
    fig.tight_layout()
    fig.savefig(OUT / "ds02_oc3_tsne_k6.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "ds02_oc3_tsne_k6.png")


if __name__ == "__main__":
    main()

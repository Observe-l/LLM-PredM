#!/usr/bin/env python3
"""Create a K=4 t-SNE view using the existing cruise OC-only run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N-CMAPSS-figure" / "cruise_oc3"
INPUT_SAMPLE = ROOT / "outputs" / "N-CMAPSS-figure" / "cruise_common_altitude" / "cruise_cluster_sample.csv"
FEATURES = ["Mach", "TRA", "T2"]


def sample_indices(n: int, target: int, random_state: int) -> np.ndarray:
    if target >= n:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=target, replace=False))


def main() -> None:
    source = pd.read_csv(INPUT_SAMPLE)
    standardization = pd.read_csv(OUT / "cruise_oc3_standardization.csv").set_index("feature")
    mean = standardization.loc[FEATURES, "mean"].to_numpy(dtype=float)
    scale = standardization.loc[FEATURES, "scale"].to_numpy(dtype=float)
    x = ((source[FEATURES].to_numpy(dtype=float) - mean) / scale).astype(np.float32)

    fit_idx = sample_indices(len(source), 60_000, 43)
    model = MiniBatchKMeans(
        n_clusters=4,
        init="k-means++",
        n_init=1,
        max_iter=50,
        batch_size=4096,
        init_size=min(len(fit_idx), max(3 * 4096, 3 * 4)),
        random_state=42,
    )
    model.fit(x[fit_idx])

    centers = pd.DataFrame(model.cluster_centers_ * scale + mean, columns=FEATURES)
    centers.insert(0, "cluster", np.arange(4))
    centers.to_csv(OUT / "cruise_oc3_cluster_centers_k4.csv", index=False)

    tsne_points = pd.read_csv(OUT / "cruise_oc3_tsne_selected_k_points.csv")
    tsne_x = ((tsne_points[FEATURES].to_numpy(dtype=float) - mean) / scale).astype(np.float32)
    labels = model.predict(tsne_x)
    tsne_points["cluster"] = labels
    tsne_points.to_csv(OUT / "cruise_oc3_tsne_k4_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    cmap = plt.get_cmap("tab10", 4)
    for cluster in range(4):
        mask = labels == cluster
        ax.scatter(
            tsne_points.loc[mask, "tsne_1"],
            tsne_points.loc[mask, "tsne_2"],
            s=5,
            alpha=0.45,
            color=cmap(cluster),
            linewidths=0,
            rasterized=True,
        )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i), markersize=6, label=f"Cluster {i}")
        for i in range(4)
    ]
    ax.legend(handles=handles, title="K-means cluster", ncol=4, loc="upper right", frameon=True)
    ax.set_title("N-CMAPSS cruise OC-only t-SNE: K=4")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        0.01,
        f"Cruise operating-condition features: Mach, TRA, T2\nn={len(tsne_points):,}; seed=42; altitude excluded",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.8", "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(OUT / "cruise_oc3_tsne_k4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "cruise_oc3_tsne_k4.png")


if __name__ == "__main__":
    main()

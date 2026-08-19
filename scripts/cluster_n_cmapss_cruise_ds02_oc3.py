#!/usr/bin/env python3
"""Select K and visualize DS02-006 cruise operating states.

Only Mach, TRA and T2 are clustering features.  Altitude is retained in the
metadata only to document the cruise samples; it is not used by K-means.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "N-CMAPSS-figure" / "cruise_common_altitude" / "cruise_cluster_sample.csv"
OUT = ROOT / "outputs" / "N-CMAPSS-figure" / "cruise_oc3_DS02-006"
FEATURES = ["Mach", "TRA", "T2"]
DATASET = "DS02-006"
SEED = 42


def fit_model(x: np.ndarray, k: int) -> MiniBatchKMeans:
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=3,
        max_iter=100,
        batch_size=4096,
        init_size=min(len(x), max(3 * 4096, 3 * k)),
        random_state=SEED,
    )
    model.fit(x)
    return model


def exact_wcss(x: np.ndarray, model: MiniBatchKMeans) -> float:
    labels = model.predict(x)
    return float(np.sum((x - model.cluster_centers_[labels]) ** 2))


def choose_k(metrics: pd.DataFrame) -> tuple[int, dict[str, object]]:
    k = metrics["k"].to_numpy(dtype=float)
    wcss = metrics["wcss"].to_numpy(dtype=float)
    x = (k - k.min()) / (k.max() - k.min())
    y = (wcss - wcss.min()) / (wcss.max() - wcss.min())
    distance = np.abs(y - (1.0 - x)) / np.sqrt(2.0)
    elbow_k = int(k[int(np.argmax(distance))])
    best_silhouette_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    candidates = metrics.loc[(metrics["k"] >= elbow_k - 2) & (metrics["k"] <= elbow_k + 2)]
    selected_k = int(candidates.loc[candidates["silhouette"].idxmax(), "k"])
    return selected_k, {
        "elbow_k": elbow_k,
        "best_silhouette_k": best_silhouette_k,
        "candidate_ks": [int(v) for v in candidates["k"]],
        "selection_rule": "highest silhouette within WCSS-elbow K +/- 2",
    }


def plot_metrics(metrics: pd.DataFrame, selected_k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax in axes:
        ax.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].plot(metrics["k"], metrics["wcss"], marker="o", color="#2457a6", linewidth=2)
    axes[0].axvline(selected_k, color="#b54708", linestyle="--", label=f"selected K={selected_k}")
    axes[0].set_title("WCSS: DS02-006 cruise OC samples")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("WCSS, standardized features")
    axes[0].legend(frameon=False)
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", color="#7a3e9d", linewidth=2)
    axes[1].axvline(selected_k, color="#b54708", linestyle="--", label=f"selected K={selected_k}")
    axes[1].set_title("Silhouette: DS02-006 cruise OC samples")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Mean silhouette score")
    axes[1].legend(frameon=False)
    fig.suptitle("DS02-006 cruise operating-condition K selection", fontsize=15, y=1.02)
    fig.text(0.5, -0.02, "Features: Mach, TRA and T2; altitude excluded from clustering", ha="center", color="#667085")
    fig.tight_layout()
    fig.savefig(OUT / "ds02_oc3_k_selection_wcss_silhouette.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(embedding: np.ndarray, labels: np.ndarray, k: int, name: str, points: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5))
    cmap = plt.get_cmap("tab20", k)
    for cluster in range(k):
        mask = labels == cluster
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.45, color=cmap(cluster), linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i), markersize=6, label=f"Cluster {i}") for i in range(k)]
    ax.legend(handles=handles, title="K-means cluster", ncol=min(4, k), loc="upper right", frameon=True)
    ax.set_title(f"DS02-006 cruise OC-only t-SNE: K={k}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.01, 0.01, f"Features: Mach, TRA, T2\nn={len(points):,}; seed={SEED}; altitude excluded", transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom", bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.8", "pad": 4})
    fig.tight_layout()
    fig.savefig(OUT / f"ds02_oc3_tsne_{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(INPUT)
    source = full.loc[full["dataset"].eq(DATASET)].reset_index(drop=True)
    if source.empty:
        raise ValueError(f"No rows found for {DATASET}")

    raw = source[FEATURES].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    x = scaler.fit_transform(raw).astype(np.float32)
    pd.DataFrame({"feature": FEATURES, "mean": scaler.mean_, "scale": scaler.scale_}).to_csv(OUT / "ds02_oc3_standardization.csv", index=False)

    silhouette_n = min(20_000, len(x))
    rng = np.random.default_rng(SEED + 2)
    silhouette_idx = np.sort(rng.choice(len(x), size=silhouette_n, replace=False))
    x_silhouette = x[silhouette_idx]

    metrics_rows = []
    for k in range(2, 17):
        print(f"evaluating K={k} ...", flush=True)
        model = fit_model(x, k)
        labels = model.predict(x_silhouette)
        metrics_rows.append({"k": k, "wcss": exact_wcss(x, model), "silhouette": float(silhouette_score(x_silhouette, labels))})
    metrics = pd.DataFrame(metrics_rows)
    selected_k, selection = choose_k(metrics)
    metrics["selected_k"] = metrics["k"].eq(selected_k)
    metrics.to_csv(OUT / "ds02_oc3_k_selection_metrics.csv", index=False)
    plot_metrics(metrics, selected_k)

    models = {"selected_k": fit_model(x, selected_k), "k16": fit_model(x, 16)}
    assignments = source[["dataset", "split", "unit", "cycle", "flight_class", "alt", *FEATURES, "cycle_local_sample"]].copy()
    for name, model in models.items():
        centers = pd.DataFrame(model.cluster_centers_ * scaler.scale_ + scaler.mean_, columns=FEATURES)
        centers.insert(0, "cluster", np.arange(model.n_clusters))
        centers.to_csv(OUT / f"ds02_oc3_cluster_centers_{name}.csv", index=False)
        assignments[f"cluster_{name}"] = model.predict(x)
    assignments.to_csv(OUT / "ds02_oc3_cluster_assignments.csv", index=False)

    tsne_n = min(20_000, len(x))
    tsne_idx = np.sort(np.random.default_rng(SEED + 3).choice(len(x), size=tsne_n, replace=False))
    x_tsne = x[tsne_idx]
    points = source.iloc[tsne_idx].reset_index(drop=True)
    print(f"computing t-SNE for {len(x_tsne):,} points ...", flush=True)
    try:
        embedding = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", max_iter=1000, random_state=SEED).fit_transform(x_tsne)
    except TypeError:
        embedding = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", n_iter=1000, random_state=SEED).fit_transform(x_tsne)

    for name, model in models.items():
        labels = model.predict(x_tsne)
        tsne_points = points[["dataset", "split", "unit", "cycle", "flight_class", "alt", *FEATURES, "cycle_local_sample"]].copy()
        tsne_points["tsne_1"] = embedding[:, 0]
        tsne_points["tsne_2"] = embedding[:, 1]
        tsne_points["cluster"] = labels
        tsne_points.to_csv(OUT / f"ds02_oc3_tsne_{name}_points.csv", index=False)
        plot_tsne(embedding, labels, model.n_clusters, name, tsne_points)

    metadata = {
        "dataset": DATASET,
        "input_sample": str(INPUT),
        "cluster_features": FEATURES,
        "excluded_feature": "alt",
        "rows": int(len(source)),
        "splits": source["split"].value_counts().to_dict(),
        "units": sorted(source["unit"].unique().tolist()),
        "silhouette_rows": int(silhouette_n),
        "tsne_rows": int(tsne_n),
        "selected_k": selected_k,
        "selection": selection,
        "k16": 16,
        "random_state": SEED,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "ds02_oc3_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"selected_k": selected_k, "selection": selection, "output_dir": str(OUT)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

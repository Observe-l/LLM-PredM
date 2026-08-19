#!/usr/bin/env python3
"""Cluster N-CMAPSS cruise samples using the three non-altitude W variables.

Cruise intervals are taken from the previously generated CruiseBench-style
sample.  Altitude was used upstream to identify the cruise interval, but is
intentionally excluded here.  K-means uses only Mach, TRA and T2.
"""

from __future__ import annotations

import argparse
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


FEATURES = ["Mach", "TRA", "T2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-sample",
        type=Path,
        default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude/cruise_cluster_sample.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/N-CMAPSS-figure/cruise_oc3"),
    )
    parser.add_argument("--fit-samples", type=int, default=60_000)
    parser.add_argument("--silhouette-samples", type=int, default=30_000)
    parser.add_argument("--tsne-samples", type=int, default=20_000)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tsne-max-iter", type=int, default=1_000)
    return parser.parse_args()


def fit_kmeans(x: np.ndarray, k: int, random_state: int) -> MiniBatchKMeans:
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=1,
        max_iter=50,
        batch_size=4096,
        init_size=min(len(x), max(3 * 4096, 3 * k)),
        random_state=random_state,
    )
    model.fit(x)
    return model


def sample_indices(n: int, target: int, random_state: int) -> np.ndarray:
    target = min(n, target)
    if target == n:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=target, replace=False))


def scan_k(
    x_fit: np.ndarray,
    x_silhouette: np.ndarray,
    k_values: list[int],
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        print(f"  evaluating K={k} ...", flush=True)
        model = fit_kmeans(x_fit, k, random_state)
        labels = model.predict(x_silhouette)
        fit_labels = model.predict(x_fit)
        exact_wcss = float(np.sum((x_fit - model.cluster_centers_[fit_labels]) ** 2))
        rows.append(
            {
                "k": k,
                "wcss": exact_wcss,
                "silhouette": float(silhouette_score(x_silhouette, labels)),
            }
        )
    return pd.DataFrame(rows)


def choose_k(metrics: pd.DataFrame) -> tuple[int, dict[str, object]]:
    """Choose the best silhouette score near the WCSS elbow.

    The elbow is estimated by maximum distance from the normalized straight
    line joining the largest and smallest K values.  Among elbow +/- 2, the
    K with the highest silhouette score is selected.
    """
    if len(metrics) < 3:
        selected = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
        return selected, {"elbow_k": selected, "best_silhouette_k": selected, "candidate_ks": [selected]}

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


def save_metric_plot(metrics: pd.DataFrame, selected_k: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax in axes:
        ax.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].plot(metrics["k"], metrics["wcss"], marker="o", color="#2457a6", linewidth=2)
    axes[0].axvline(selected_k, color="#b54708", linestyle="--", linewidth=1.3, label=f"selected K={selected_k}")
    axes[0].set_title("WCSS on cruise OC samples")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("WCSS, standardized features")
    axes[0].legend(frameon=False)
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", color="#7a3e9d", linewidth=2)
    axes[1].axvline(selected_k, color="#b54708", linestyle="--", linewidth=1.3, label=f"selected K={selected_k}")
    axes[1].set_title("Silhouette on cruise OC samples")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Mean silhouette score")
    axes[1].legend(frameon=False)
    fig.suptitle("N-CMAPSS cruise operating-condition K selection", fontsize=15, y=1.02)
    fig.text(0.5, -0.02, "Features: Mach, TRA and T2; altitude excluded from K-means", ha="center", color="#667085")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_tsne_plot(embedding: np.ndarray, labels: np.ndarray, k: int, path: Path, title: str, n: int, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5))
    cmap = plt.get_cmap("tab20", k)
    for cluster in range(k):
        mask = labels == cluster
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.45, color=cmap(cluster), linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i), markersize=6, label=f"Cluster {i}") for i in range(k)]
    ax.legend(handles=handles, title="K-means cluster", ncol=min(4, k), loc="upper right", frameon=True)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        0.01,
        f"Cruise operating-condition features: Mach, TRA, T2\nn={n:,}; seed={seed}; altitude excluded",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.8", "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input_sample)
    missing = [feature for feature in FEATURES if feature not in source.columns]
    if missing:
        raise ValueError(f"Missing clustering features: {missing}")
    if source[FEATURES].isna().any().any():
        raise ValueError("NaN values found in clustering features")

    raw = source[FEATURES].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    x = scaler.fit_transform(raw).astype(np.float32)
    standardization = pd.DataFrame({"feature": FEATURES, "mean": scaler.mean_, "scale": scaler.scale_})
    standardization.to_csv(args.output_dir / "cruise_oc3_standardization.csv", index=False)

    fit_idx = sample_indices(len(source), args.fit_samples, args.random_state + 1)
    x_fit = x[fit_idx]
    silhouette_idx = sample_indices(len(x_fit), args.silhouette_samples, args.random_state + 2)
    x_silhouette = x_fit[silhouette_idx]
    metrics = scan_k(x_fit, x_silhouette, list(range(args.k_min, args.k_max + 1)), args.random_state)
    selected_k, selection = choose_k(metrics)
    metrics["selected_k"] = metrics["k"].eq(selected_k)
    metrics.to_csv(args.output_dir / "cruise_oc3_k_selection_metrics.csv", index=False)
    save_metric_plot(metrics, selected_k, args.output_dir / "cruise_oc3_k_selection_wcss_silhouette.png")

    models = {"k16": fit_kmeans(x_fit, 16, args.random_state), "selected_k": fit_kmeans(x_fit, selected_k, args.random_state)}
    assignments = source[[c for c in ["dataset", "split", "unit", "cycle", "flight_class", "cycle_local_sample"] if c in source.columns] + FEATURES].copy()
    for name, model in models.items():
        centers = scaler.inverse_transform(model.cluster_centers_)
        pd.DataFrame(centers, columns=FEATURES).assign(cluster=np.arange(model.n_clusters)).to_csv(args.output_dir / f"cruise_oc3_cluster_centers_{name}.csv", index=False)
        assignments[f"cluster_{name}"] = model.predict(x)
    assignments.to_csv(args.output_dir / "cruise_oc3_cluster_assignments.csv", index=False)

    tsne_idx = sample_indices(len(x_fit), args.tsne_samples, args.random_state + 3)
    x_tsne = x_fit[tsne_idx]
    source_tsne = source.iloc[fit_idx[tsne_idx]].reset_index(drop=True)
    print(f"Computing t-SNE for {len(x_tsne):,} sampled cruise rows ...", flush=True)
    try:
        embedding = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", max_iter=args.tsne_max_iter, random_state=args.random_state).fit_transform(x_tsne)
    except TypeError:
        embedding = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", n_iter=args.tsne_max_iter, random_state=args.random_state).fit_transform(x_tsne)

    for name, model in models.items():
        k = model.n_clusters
        labels = model.predict(x_tsne)
        points = source_tsne[[c for c in ["dataset", "split", "unit", "cycle", "flight_class", "alt", *FEATURES, "cycle_local_sample"] if c in source_tsne.columns]].copy()
        points["tsne_1"] = embedding[:, 0]
        points["tsne_2"] = embedding[:, 1]
        points["cluster"] = labels
        points.to_csv(args.output_dir / f"cruise_oc3_tsne_{name}_points.csv", index=False)
        save_tsne_plot(embedding, labels, k, args.output_dir / f"cruise_oc3_tsne_{name}.png", f"N-CMAPSS cruise OC-only t-SNE: K={k}", len(points), args.random_state)

    metadata = {
        "method": "K-means on previously extracted CruiseBench-style cruise samples",
        "input_sample": str(args.input_sample),
        "cluster_features": FEATURES,
        "excluded_feature": "alt",
        "cruise_detection_note": "Altitude was used upstream for cruise detection, but excluded from clustering and standardization here.",
        "input_rows": int(len(source)),
        "fit_rows": int(len(x_fit)),
        "silhouette_rows": int(len(x_silhouette)),
        "tsne_rows": int(len(x_tsne)),
        "k16": 16,
        "selected_k": selected_k,
        "selection": selection,
        "random_state": args.random_state,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "cruise_oc3_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"selected_k": selected_k, "selection": selection, "output_dir": str(args.output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

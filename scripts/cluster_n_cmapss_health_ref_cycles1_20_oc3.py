#!/usr/bin/env python3
"""Cluster cruise operating states from the first 20 cycles of every unit.

The cruise intervals come from the existing CruiseBench-style interval table.
For every readable N-CMAPSS dataset and both dev/test splits, all accepted
cruise rows from cycles 1--20 are loaded from the original HDF5 files.  Only
Mach, TRA and T2 are used for standardization and clustering; altitude is used
only by the upstream cruise detector.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "N-CMAPSS"
INTERVALS_PATH = ROOT / "outputs" / "N-CMAPSS-figure" / "cruise_common_altitude" / "cruise_cycle_statistics.csv"
OUT = ROOT / "outputs" / "N-CMAPSS-figure" / "cruise_health_ref_cycles1-20_oc3"
FEATURES = ["Mach", "TRA", "T2"]
SEED = 42

DATASET_NAMES = {
    "N-CMAPSS_DS01-005.h5": "DS01-005",
    "N-CMAPSS_DS02-006.h5": "DS02-006",
    "N-CMAPSS_DS03-012.h5": "DS03-012",
    "N-CMAPSS_DS04.h5": "DS04",
    "N-CMAPSS_DS05.h5": "DS05",
    "N-CMAPSS_DS06.h5": "DS06",
    "N-CMAPSS_DS07.h5": "DS07",
    "N-CMAPSS_DS08a-009.h5": "DS08a-009",
    "N-CMAPSS_DS08c-008.h5": "DS08c-008",
    "N-CMAPSS_DS08d-010.h5": "DS08d-010",
}


def decode_names(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(values).reshape(-1)]


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


def stratified_indices(groups: np.ndarray, target: int, seed: int) -> np.ndarray:
    target = min(target, len(groups))
    if target == len(groups):
        return np.arange(len(groups))
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    unique = np.unique(groups)
    per_group = max(1, target // len(unique))
    for group in unique:
        candidates = np.flatnonzero(groups == group)
        selected.append(rng.choice(candidates, size=min(len(candidates), per_group), replace=False))
    chosen = np.concatenate(selected)
    if len(chosen) < target:
        remaining = np.setdiff1d(np.arange(len(groups)), chosen, assume_unique=False)
        chosen = np.concatenate([chosen, rng.choice(remaining, size=target - len(chosen), replace=False)])
    rng.shuffle(chosen)
    return chosen


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
    axes[0].set_title("WCSS: first 20-cycle cruise reference")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("WCSS, standardized features")
    axes[0].legend(frameon=False)
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", color="#7a3e9d", linewidth=2)
    axes[1].axvline(selected_k, color="#b54708", linestyle="--", label=f"selected K={selected_k}")
    axes[1].set_title("Silhouette: first 20-cycle cruise reference")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Mean silhouette score")
    axes[1].legend(frameon=False)
    fig.suptitle("N-CMAPSS cruise operating-condition K selection", fontsize=15, y=1.02)
    fig.text(0.5, -0.02, "All units, dev/test; features: Mach, TRA and T2; altitude excluded", ha="center", color="#667085")
    fig.tight_layout()
    fig.savefig(OUT / "health_ref_oc3_k_selection_wcss_silhouette.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(embedding: np.ndarray, labels: np.ndarray, k: int, points: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5))
    cmap = plt.get_cmap("tab20", k)
    for cluster in range(k):
        mask = labels == cluster
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.45, color=cmap(cluster), linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i), markersize=6, label=f"Cluster {i}") for i in range(k)]
    ax.legend(handles=handles, title="K-means cluster", ncol=min(4, k), loc="upper right", frameon=True)
    ax.set_title(f"N-CMAPSS cruise health reference t-SNE: K={k}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.01, 0.01, f"First 20 cycles per unit; features: Mach, TRA, T2\nn={len(points):,}; seed={SEED}; altitude excluded", transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom", bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.8", "pad": 4})
    fig.tight_layout()
    fig.savefig(OUT / "health_ref_oc3_tsne_selected_k.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_reference_rows(intervals: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    accepted = intervals.loc[(intervals["cycle"] <= 20) & intervals["cruise_status"].eq("accepted")].copy()
    arrays: list[np.ndarray] = []
    group_arrays: list[np.ndarray] = []
    row_meta: list[dict[str, object]] = []
    missing_files: list[str] = []
    dataset_paths = {DATASET_NAMES.get(p.name, p.stem.replace("N-CMAPSS_", "")): p for p in DATA_DIR.glob("*.h5")}
    group_id = 0
    for (dataset, split), group in accepted.groupby(["dataset", "split"], sort=True):
        path = dataset_paths.get(dataset)
        if path is None:
            missing_files.append(f"{dataset}/{split}")
            continue
        print(f"Loading {dataset}/{split}: {len(group)} accepted cycles ...", flush=True)
        with h5py.File(path, "r") as hdf:
            w_names = decode_names(hdf["W_var"][()])
            feature_indices = [w_names.index(feature) for feature in FEATURES]
            for _, record in group.iterrows():
                start = int(record["hdf5_start_row"] + record["cruise_start_sample"])
                stop = int(record["hdf5_start_row"] + record["cruise_end_sample"] + 1)
                values = np.asarray(hdf[f"W_{split}"][start:stop, feature_indices], dtype=np.float32)
                arrays.append(values)
                group_arrays.append(np.full(len(values), group_id, dtype=np.int32))
                row_meta.append({
                    "dataset": dataset,
                    "split": split,
                    "unit": int(record["unit"]),
                    "cycle": int(record["cycle"]),
                    "cruise_samples": len(values),
                })
                group_id += 1
    if not arrays:
        raise ValueError("No accepted first-20-cycle cruise rows could be loaded")
    return np.vstack(arrays), np.concatenate(group_arrays), pd.DataFrame(row_meta), missing_files


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    intervals = pd.read_csv(INTERVALS_PATH)
    raw, groups, cycle_meta, missing = load_reference_rows(intervals)
    interval_datasets = set(intervals["dataset"].astype(str).unique())
    missing_interval_datasets = sorted(set(DATASET_NAMES.values()) - interval_datasets)
    cycle_meta.to_csv(OUT / "health_reference_cycle_statistics.csv", index=False)
    np.savez_compressed(OUT / "health_reference_cruise_features.npz", features=raw, feature_names=np.asarray(FEATURES))

    scaler = StandardScaler()
    x = scaler.fit_transform(raw).astype(np.float32)
    pd.DataFrame({"feature": FEATURES, "mean": scaler.mean_, "scale": scaler.scale_}).to_csv(OUT / "health_ref_oc3_standardization.csv", index=False)

    fit_idx = stratified_indices(groups, min(120_000, len(x)), SEED + 1)
    x_fit = x[fit_idx]
    silhouette_idx = stratified_indices(groups[fit_idx], min(30_000, len(x_fit)), SEED + 2)
    x_silhouette = x_fit[silhouette_idx]
    metrics_rows = []
    for k in range(2, 17):
        print(f"evaluating K={k} ...", flush=True)
        model = fit_model(x_fit, k)
        labels = model.predict(x_silhouette)
        metrics_rows.append({"k": k, "wcss": exact_wcss(x_fit, model), "silhouette": float(silhouette_score(x_silhouette, labels))})
        if k in {5, 16}:
            fixed_centers = pd.DataFrame(model.cluster_centers_ * scaler.scale_ + scaler.mean_, columns=FEATURES)
            fixed_centers.insert(0, "cluster", np.arange(k))
            fixed_centers.to_csv(OUT / f"health_ref_oc3_cluster_centers_k{k}.csv", index=False)
    metrics = pd.DataFrame(metrics_rows)
    selected_k, selection = choose_k(metrics)
    metrics["selected_k"] = metrics["k"].eq(selected_k)
    metrics.to_csv(OUT / "health_ref_oc3_k_selection_metrics.csv", index=False)
    plot_metrics(metrics, selected_k)

    model = fit_model(x_fit, selected_k)
    centers = pd.DataFrame(model.cluster_centers_ * scaler.scale_ + scaler.mean_, columns=FEATURES)
    centers.insert(0, "cluster", np.arange(selected_k))
    centers.to_csv(OUT / "health_ref_oc3_cluster_centers_selected_k.csv", index=False)
    fit_sample = cycle_meta.iloc[np.searchsorted(np.cumsum(cycle_meta["cruise_samples"]), fit_idx, side="right")].reset_index(drop=True)
    fit_sample[FEATURES] = raw[fit_idx]
    fit_sample["cluster"] = model.predict(x_fit)
    fit_sample.to_csv(OUT / "health_ref_oc3_fit_sample_assignments.csv", index=False)

    tsne_idx = stratified_indices(groups, min(20_000, len(x)), SEED + 3)
    x_tsne = x[tsne_idx]
    try:
        embedding = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", max_iter=1000, random_state=SEED).fit_transform(x_tsne)
    except TypeError:
        embedding = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", n_iter=1000, random_state=SEED).fit_transform(x_tsne)
    points = cycle_meta.iloc[np.searchsorted(np.cumsum(cycle_meta["cruise_samples"]), tsne_idx, side="right")].reset_index(drop=True)
    points[FEATURES] = raw[tsne_idx]
    points["tsne_1"] = embedding[:, 0]
    points["tsne_2"] = embedding[:, 1]
    points["cluster"] = model.predict(x_tsne)
    points.to_csv(OUT / "health_ref_oc3_tsne_selected_k_points.csv", index=False)
    plot_tsne(embedding, points["cluster"].to_numpy(), selected_k, points)

    metadata = {
        "method": "K-means on all accepted cruise rows from cycles 1-20 of every unit",
        "interval_source": str(INTERVALS_PATH),
        "cluster_features": FEATURES,
        "excluded_feature": "alt",
        "cruise_detection_note": "Altitude was used upstream for cruise detection, but excluded from standardization and clustering.",
        "raw_reference_rows": int(len(raw)),
        "accepted_reference_cycles": int(len(cycle_meta)),
        "reference_units": int(cycle_meta[["dataset", "split", "unit"]].drop_duplicates().shape[0]),
        "fit_rows": int(len(x_fit)),
        "silhouette_rows": int(len(x_silhouette)),
        "tsne_rows": int(len(x_tsne)),
        "selected_k": selected_k,
        "selection": selection,
        "unavailable_source_blocks": missing + [f"{dataset}: absent from cruise interval table (source may be unreadable)" for dataset in missing_interval_datasets],
        "random_state": SEED,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "health_ref_oc3_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"selected_k": selected_k, "raw_rows": len(raw), "selection": selection, "output_dir": str(OUT)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

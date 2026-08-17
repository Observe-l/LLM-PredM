#!/usr/bin/env python3
"""Cluster N-CMAPSS operating-condition variables and make t-SNE figures.

The N-CMAPSS HDF5 files expose four operating-condition variables in ``W``:
altitude, Mach, throttle resolver angle, and T2.  This script uses those
variables only; unit/cycle, health-state, sensor, and RUL fields are not used
as clustering features.

The source files are much larger than the number of points needed for a
useful clustering diagnostic.  Each readable file contributes a deterministic
random sample from both its development and test split.  Sampling is done
after opening each HDF5 file, so the full collection of readable datasets is
represented without loading the sensor arrays into memory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ["alt", "Mach", "TRA", "T2"]
SPLITS = ("dev", "test")
DEFAULT_K_RANGE = tuple(range(2, 25))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/N-CMAPSS"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/N-CMAPSS-figure"),
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=20_000,
        help="Maximum sampled W rows per file and per split (default: 20000).",
    )
    parser.add_argument(
        "--tsne-samples",
        type=int,
        default=40_000,
        help="Number of points used in each t-SNE plot (default: 40000).",
    )
    parser.add_argument(
        "--silhouette-samples",
        type=int,
        default=60_000,
        help="Number of points used to estimate each silhouette score.",
    )
    parser.add_argument(
        "--k-min", type=int, default=2, help="Smallest K in the diagnostic scan."
    )
    parser.add_argument(
        "--k-max", type=int, default=24, help="Largest K in the diagnostic scan."
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--tsne-max-iter",
        type=int,
        default=1000,
        help="Maximum t-SNE optimization iterations.",
    )
    return parser.parse_args()


def decode_names(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def sample_rows(
    dataset: h5py.Dataset,
    n_rows: int,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Read a deterministic random subset from a contiguous HDF5 dataset."""
    count = min(n_rows, sample_size)
    if count == n_rows:
        return np.asarray(dataset, dtype=np.float64)
    indices = np.sort(rng.choice(n_rows, size=count, replace=False))
    return np.asarray(dataset[indices], dtype=np.float64)


def load_condition_samples(
    data_dir: Path,
    samples_per_split: int,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Load W samples from every readable N-CMAPSS file and split."""
    rng = np.random.default_rng(random_state)
    samples: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    skipped: list[str] = []
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {data_dir}")

    for path in files:
        try:
            with h5py.File(path, "r") as hdf:
                if "W_var" not in hdf:
                    raise KeyError("missing W_var")
                names = decode_names(np.asarray(hdf["W_var"]).reshape(-1))
                if names != FEATURE_NAMES:
                    raise ValueError(f"W_var={names!r}, expected {FEATURE_NAMES!r}")

                for split in SPLITS:
                    key = f"W_{split}"
                    if key not in hdf:
                        raise KeyError(f"missing {key}")
                    n_rows = int(hdf[key].shape[0])
                    block = sample_rows(hdf[key], n_rows, samples_per_split, rng)
                    samples.append(block)
                    metadata.append(
                        pd.DataFrame(
                            {
                                "dataset": np.repeat(path.stem, len(block)),
                                "split": np.repeat(split, len(block)),
                                "row_count": np.repeat(n_rows, len(block)),
                                "sampled_count": np.repeat(len(block), len(block)),
                            },
                            index=np.arange(len(block)),
                        )
                    )
        except Exception as exc:  # keep the run auditable when one file is corrupt
            skipped.append(f"{path.name}: {type(exc).__name__}: {exc}")

    if not samples:
        raise RuntimeError("No readable N-CMAPSS HDF5 files were found")
    return np.vstack(samples), pd.concat(metadata, ignore_index=True), skipped


def stratified_indices(
    metadata: pd.DataFrame,
    target_size: int,
    random_state: int,
) -> np.ndarray:
    """Sample approximately equally from each dataset/split block."""
    target_size = min(target_size, len(metadata))
    if target_size == len(metadata):
        return np.arange(len(metadata))

    rng = np.random.default_rng(random_state)
    groups = metadata.groupby(["dataset", "split"], sort=False).indices
    per_group = max(1, target_size // len(groups))
    selected: list[np.ndarray] = []
    for indices in groups.values():
        indices = np.asarray(indices)
        count = min(len(indices), per_group)
        selected.append(rng.choice(indices, size=count, replace=False))
    selected_indices = np.concatenate(selected)
    if len(selected_indices) < target_size:
        remaining = np.setdiff1d(np.arange(len(metadata)), selected_indices, assume_unique=False)
        extra = rng.choice(remaining, size=target_size - len(selected_indices), replace=False)
        selected_indices = np.concatenate([selected_indices, extra])
    rng.shuffle(selected_indices)
    return selected_indices


def fit_kmeans(x: np.ndarray, k: int, random_state: int) -> KMeans:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
        algorithm="lloyd",
    )
    model.fit(x)
    return model


def scan_k(
    x: np.ndarray,
    k_values: tuple[int, ...],
    silhouette_x: np.ndarray,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        print(f"  evaluating K={k} ...", flush=True)
        model = fit_kmeans(x, k, random_state)
        labels = model.predict(silhouette_x)
        score = silhouette_score(silhouette_x, labels, metric="euclidean")
        rows.append({"k": k, "wcss": float(model.inertia_), "silhouette": float(score)})
    return pd.DataFrame(rows)


def choose_k(metrics: pd.DataFrame) -> tuple[int, dict[str, object]]:
    """Choose the best silhouette among Ks near the WCSS elbow.

    The elbow is estimated using the maximum perpendicular distance from the
    line joining the first and last normalized WCSS points.  To make the
    result stable on smooth, continuous condition spaces, the final choice is
    the highest-silhouette K among the elbow's immediate neighborhood
    (elbow-2 through elbow+2), while still reporting the raw best silhouette.
    """
    if len(metrics) < 3:
        best = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
        return best, {"elbow_k": best, "best_silhouette_k": best, "candidate_ks": [best]}

    k = metrics["k"].to_numpy(dtype=float)
    wcss = metrics["wcss"].to_numpy(dtype=float)
    x = (k - k.min()) / (k.max() - k.min())
    y = (wcss - wcss.min()) / (wcss.max() - wcss.min())
    line_y = 1.0 - x
    distance = np.abs(y - line_y) / np.sqrt(2.0)
    elbow_idx = int(np.argmax(distance))
    elbow_k = int(k[elbow_idx])
    best_silhouette_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    candidate_mask = (metrics["k"] >= elbow_k - 2) & (metrics["k"] <= elbow_k + 2)
    candidates = metrics.loc[candidate_mask]
    selected_k = int(candidates.loc[candidates["silhouette"].idxmax(), "k"])
    return selected_k, {
        "elbow_k": elbow_k,
        "best_silhouette_k": best_silhouette_k,
        "candidate_ks": [int(value) for value in candidates["k"]],
        "selection_rule": "highest silhouette within elbow K +/- 2",
    }


def save_tsne_plot(
    embedding: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    k: int,
    path: Path,
    title: str,
    random_state: int,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 9), dpi=180)
    cmap = plt.get_cmap("turbo", k)
    for cluster in range(k):
        mask = labels == cluster
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=5,
            alpha=0.48,
            color=cmap(cluster),
            linewidths=0,
            rasterized=True,
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=cmap(cluster),
            markersize=6,
            label=f"Cluster {cluster}",
        )
        for cluster in range(k)
    ]
    ax.legend(handles=handles, title="K-means cluster", ncol=min(4, k), loc="upper right", frameon=True)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.text(
        0.01,
        0.01,
        f"Standardized W variables: {', '.join(FEATURE_NAMES)}\n"
        f"n={len(embedding):,}; seed={random_state}; source blocks={metadata[['dataset', 'split']].drop_duplicates().shape[0]}",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8", "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(metrics: pd.DataFrame, selected_k: int, path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    axes[0].plot(metrics["k"], metrics["wcss"], marker="o", color="#2f5d8c", linewidth=2)
    axes[0].axvline(selected_k, color="#b45f06", linestyle="--", linewidth=1.5, label=f"selected K={selected_k}")
    axes[0].set_title("WCSS (within-cluster sum of squares)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("WCSS")
    axes[0].legend()
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", color="#8f3b76", linewidth=2)
    axes[1].axvline(selected_k, color="#b45f06", linestyle="--", linewidth=1.5, label=f"selected K={selected_k}")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("mean silhouette")
    axes[1].legend()
    fig.suptitle("N-CMAPSS operating-condition K selection", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_tsne(x: np.ndarray, random_state: int, max_iter: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        init="pca",
        learning_rate="auto",
        n_iter=max_iter,
        random_state=random_state,
        method="barnes_hut",
        verbose=1,
    )
    return tsne.fit_transform(x)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading sampled operating-condition rows ...", flush=True)
    raw_x, metadata, skipped = load_condition_samples(
        args.data_dir, args.samples_per_split, args.random_state
    )
    scaler = StandardScaler()
    x = scaler.fit_transform(raw_x).astype(np.float32)
    metadata.to_csv(args.output_dir / "sample_summary.csv", index=False)
    pd.DataFrame({"feature": FEATURE_NAMES, "mean": scaler.mean_, "scale": scaler.scale_}).to_csv(
        args.output_dir / "standardization.csv", index=False
    )

    fit_indices = stratified_indices(metadata, min(len(x), 240_000), args.random_state + 1)
    x_fit = x[fit_indices]
    fit_metadata = metadata.iloc[fit_indices].reset_index(drop=True)
    k16 = fit_kmeans(x_fit, 16, args.random_state)
    print("Fitted K=16; preparing t-SNE sample ...", flush=True)
    tsne_indices = stratified_indices(fit_metadata, args.tsne_samples, args.random_state + 2)
    x_tsne = x_fit[tsne_indices]
    tsne_metadata = fit_metadata.iloc[tsne_indices].reset_index(drop=True)
    embedding = make_tsne(x_tsne, args.random_state, args.tsne_max_iter)
    k16_labels = k16.predict(x_tsne)
    save_tsne_plot(
        embedding,
        k16_labels,
        tsne_metadata,
        16,
        args.output_dir / "tsne_k16.png",
        "N-CMAPSS operating conditions: t-SNE colored by K-means (K=16)",
        args.random_state,
    )
    pd.DataFrame(embedding, columns=["tsne_1", "tsne_2"]).assign(
        cluster=k16_labels,
        dataset=tsne_metadata["dataset"].to_numpy(),
        split=tsne_metadata["split"].to_numpy(),
    ).to_csv(args.output_dir / "tsne_k16_points.csv", index=False)
    pd.DataFrame(k16.cluster_centers_, columns=FEATURE_NAMES).assign(cluster=np.arange(16)).to_csv(
        args.output_dir / "cluster_centers_k16_standardized.csv", index=False
    )

    silhouette_indices = stratified_indices(metadata, args.silhouette_samples, args.random_state + 3)
    x_silhouette = x[silhouette_indices]
    k_values = tuple(range(args.k_min, args.k_max + 1))
    print(f"Scanning K={args.k_min}..{args.k_max} on {len(x_fit):,} fit rows ...", flush=True)
    metrics = scan_k(x_fit, k_values, x_silhouette, args.random_state)
    selected_k, selection_info = choose_k(metrics)
    metrics.to_csv(args.output_dir / "k_selection_metrics.csv", index=False)
    save_metric_plot(metrics, selected_k, args.output_dir / "k_selection_wcss_silhouette.png")

    selected_model = fit_kmeans(x_fit, selected_k, args.random_state)
    selected_labels = selected_model.predict(x_tsne)
    save_tsne_plot(
        embedding,
        selected_labels,
        tsne_metadata,
        selected_k,
        args.output_dir / "tsne_selected_k.png",
        f"N-CMAPSS operating conditions: t-SNE colored by K-means (selected K={selected_k})",
        args.random_state,
    )
    pd.DataFrame(embedding, columns=["tsne_1", "tsne_2"]).assign(
        cluster=selected_labels,
        dataset=tsne_metadata["dataset"].to_numpy(),
        split=tsne_metadata["split"].to_numpy(),
    ).to_csv(args.output_dir / "tsne_selected_k_points.csv", index=False)
    pd.DataFrame(selected_model.cluster_centers_, columns=FEATURE_NAMES).assign(
        cluster=np.arange(selected_k)
    ).to_csv(args.output_dir / "cluster_centers_selected_k_standardized.csv", index=False)

    metadata_summary = (
        metadata.groupby(["dataset", "split"], as_index=False)
        .agg(row_count=("row_count", "first"), sampled_count=("sampled_count", "sum"))
    )
    run_metadata = {
        "feature_definition": "N-CMAPSS W operating-condition variables only",
        "features": FEATURE_NAMES,
        "n_readable_files": int(metadata["dataset"].nunique()),
        "n_sampled_rows": int(len(x)),
        "n_fit_rows": int(len(x_fit)),
        "n_tsne_rows": int(len(x_tsne)),
        "n_silhouette_rows": int(len(x_silhouette)),
        "requested_k16": 16,
        "selected_k": selected_k,
        "selection": selection_info,
        "random_state": args.random_state,
        "samples_per_split": args.samples_per_split,
        "skipped_files": skipped,
        "elapsed_seconds": round(time.time() - started, 2),
        "source_summary": metadata_summary.to_dict(orient="records"),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(f"Selected K={selected_k}; best raw silhouette K={selection_info['best_silhouette_k']}; elbow K={selection_info['elbow_k']}")
    if skipped:
        print("Skipped unreadable files:")
        for item in skipped:
            print(f"  - {item}")
    print(f"Saved figures and tables to {args.output_dir}")


if __name__ == "__main__":
    main()

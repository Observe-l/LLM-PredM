#!/usr/bin/env python3
"""Extract CruiseBench-style cruise intervals and cluster cruise operating states.

The cruise detector follows the common-altitude method described in CruiseBench
and in the authors' released ``common_altitude.py`` implementation:

* 31-sample centered rolling-median smoothing of altitude;
* reject cycles with smoothed altitude span below 500 ft;
* search high-altitude 50-ft bins over 50 shifted origins;
* merge gaps up to 80 samples when the combined segment remains a plateau;
* accept only runs of at least 512 samples and raw-altitude std <= 50 ft;
* keep the longest accepted run and do not use a fallback interval.

Only W=(alt, Mach, TRA, T2) from accepted cruise rows is used for K-means.
Unit, cycle, flight class, health state, sensors, and RUL are not clustering
features.  The script writes per-cycle cruise intervals/statistics, sampled
cruise rows, K=16 and selected-K t-SNE figures, and WCSS/Silhouette metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
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
}


@dataclass(frozen=True)
class CommonAltitudeProfile:
    rolling_window: int = 31
    altitude_quantile: float = 0.50
    min_cycle_altitude_span: float = 500.0
    altitude_bin_width: float = 50.0
    bin_shifts: int = 50
    altitude_tolerance: float = 50.0
    top_levels: int = 3
    min_cruise_samples: int = 512
    max_cruise_altitude_std: float = 50.0
    min_altitude_span_fraction: float = 0.35
    max_gap_samples: int = 80


PROFILE = CommonAltitudeProfile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/N-CMAPSS"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/N-CMAPSS-figure/cruise_common_altitude"),
    )
    parser.add_argument("--cruise-samples-per-block", type=int, default=20_000)
    parser.add_argument("--fit-samples", type=int, default=240_000)
    parser.add_argument("--tsne-samples", type=int, default=40_000)
    parser.add_argument("--silhouette-samples", type=int, default=60_000)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tsne-max-iter", type=int, default=1_000)
    return parser.parse_args()


def decode_names(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in np.asarray(values).reshape(-1)]


def dataset_label(path: Path) -> str:
    return DATASET_NAMES.get(path.name, path.stem.replace("N-CMAPSS_", ""))


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return []
    padded = np.r_[False, values, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2], changes[1::2]))


def merge_short_same_level_gaps(
    mask: np.ndarray,
    smooth_altitude: np.ndarray,
    max_gap_samples: int,
    max_level_jump: float,
    max_segment_range: float,
) -> np.ndarray:
    """Merge short gaps only when the adjacent plateau remains narrow."""
    merged = np.asarray(mask, dtype=bool).copy()
    smooth_altitude = np.asarray(smooth_altitude)
    changed = True
    while changed:
        changed = False
        runs = true_runs(merged)
        for (left_start, left_stop), (right_start, right_stop) in zip(runs, runs[1:]):
            gap = right_start - left_stop
            if gap > max_gap_samples:
                continue
            left_level = smooth_altitude[left_stop - 1]
            right_level = smooth_altitude[right_start]
            combined = smooth_altitude[left_start:right_stop]
            if (
                abs(left_level - right_level) <= max_level_jump
                and combined.max() - combined.min() <= max_segment_range
            ):
                merged[left_stop:right_start] = True
                changed = True
                break
    return merged


def detect_common_altitude_interval(
    altitude: np.ndarray,
    profile: CommonAltitudeProfile = PROFILE,
) -> dict[str, float | int | str | None]:
    """Return one inclusive cruise interval and diagnostics for one cycle."""
    # CruiseBench loads altitude as float32 before applying the rolling
    # median.  Preserve that precision so borderline plateau decisions match
    # the released implementation.
    altitude = np.asarray(altitude, dtype=np.float32)
    if len(altitude) == 0 or not np.isfinite(altitude).all():
        return {
            "cruise_start_sample": None,
            "cruise_end_sample": None,
            "cruise_samples": 0,
            "smoothed_altitude_span": np.nan,
            "cruise_altitude_std": np.nan,
            "cruise_status": "invalid_altitude",
        }

    smooth = pd.Series(altitude).rolling(
        window=profile.rolling_window,
        center=True,
        min_periods=1,
    ).median().to_numpy()
    altitude_span = float(smooth.max() - smooth.min())
    base = {
        "cruise_start_sample": None,
        "cruise_end_sample": None,
        "cruise_samples": 0,
        "smoothed_altitude_span": altitude_span,
        "cruise_altitude_std": np.nan,
    }
    if altitude_span < profile.min_cycle_altitude_span:
        base["cruise_status"] = "altitude_span_below_500ft"
        return base

    floor = max(
        float(np.quantile(smooth, profile.altitude_quantile)),
        float(smooth.min() + profile.min_altitude_span_fraction * altitude_span),
    )
    valid_runs: list[tuple[int, int]] = []
    offsets = np.linspace(
        0.0,
        profile.altitude_bin_width,
        max(1, profile.bin_shifts),
        endpoint=False,
    )
    for offset in offsets:
        level_bins = (
            np.floor((smooth - offset) / profile.altitude_bin_width)
            * profile.altitude_bin_width
            + offset
            + profile.altitude_bin_width / 2.0
        )
        high_levels = level_bins[level_bins >= floor]
        if high_levels.size == 0:
            continue
        levels, counts = np.unique(high_levels, return_counts=True)
        top_levels = levels[np.argsort(counts)[::-1][: profile.top_levels]]
        for level in top_levels:
            candidate = np.abs(smooth - level) <= profile.altitude_tolerance
            candidate = merge_short_same_level_gaps(
                candidate,
                smooth,
                max_gap_samples=profile.max_gap_samples,
                max_level_jump=profile.altitude_tolerance,
                max_segment_range=2 * profile.altitude_tolerance,
            )
            for start, stop in true_runs(candidate):
                if stop - start < profile.min_cruise_samples:
                    continue
                segment_std = float(pd.Series(altitude[start:stop]).std(ddof=0))
                if segment_std > profile.max_cruise_altitude_std:
                    continue
                valid_runs.append((start, stop))

    if not valid_runs:
        base["cruise_status"] = "no_valid_cruise_run"
        return base
    start, stop = max(valid_runs, key=lambda run: run[1] - run[0])
    base.update(
        cruise_start_sample=int(start),
        cruise_end_sample=int(stop - 1),
        cruise_samples=int(stop - start),
        cruise_altitude_std=float(np.std(altitude[start:stop], ddof=0)),
        cruise_status="accepted",
    )
    return base


def cycle_boundaries(unit: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    if len(unit) == 0:
        return np.array([0], dtype=np.int64)
    changes = np.flatnonzero((unit[1:] != unit[:-1]) | (cycle[1:] != cycle[:-1])) + 1
    return np.r_[0, changes, len(unit)].astype(np.int64)


def extract_block(
    path: Path,
    split: str,
    profile: CommonAltitudeProfile,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Extract cycle intervals and a uniform sample of accepted cruise W rows."""
    with h5py.File(path, "r") as hdf:
        a_names = decode_names(hdf["A_var"][()])
        w_names = decode_names(hdf["W_var"][()])
        if w_names != FEATURE_NAMES:
            raise ValueError(f"{path.name}: W_var={w_names!r}, expected {FEATURE_NAMES!r}")
        unit = np.asarray(hdf[f"A_{split}"][:, a_names.index("unit")], dtype=np.int32)
        cycle = np.asarray(hdf[f"A_{split}"][:, a_names.index("cycle")], dtype=np.int32)
        fc = np.asarray(hdf[f"A_{split}"][:, a_names.index("Fc")], dtype=np.int8)
        altitude = np.asarray(hdf[f"W_{split}"][:, w_names.index("alt")], dtype=np.float32)
        boundaries = cycle_boundaries(unit, cycle)
        rows: list[dict[str, object]] = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            detected = detect_common_altitude_interval(altitude[start:stop], profile)
            detected.update(
                dataset=dataset_label(path),
                split=split,
                unit=int(unit[start]),
                cycle=int(cycle[start]),
                flight_class=int(fc[start]),
                total_samples=int(stop - start),
                hdf5_start_row=int(start),
                hdf5_end_row=int(stop - 1),
            )
            detected["cruise_fraction"] = float(detected["cruise_samples"] / (stop - start))
            rows.append(detected)
        intervals = pd.DataFrame(rows)
        accepted = intervals.loc[intervals["cruise_status"] == "accepted"].reset_index(drop=True)
        accepted_total = int(accepted["cruise_samples"].sum())
        take = min(sample_size, accepted_total)
        if take == 0:
            return intervals, pd.DataFrame(columns=["dataset", "split", "unit", "cycle", "flight_class", *FEATURE_NAMES]), accepted_total

        # Sample uniformly in the concatenation of all accepted cruise intervals,
        # then translate those positions to raw HDF5 row indices.
        global_positions = np.sort(rng.choice(accepted_total, size=take, replace=False))
        lengths = accepted["cruise_samples"].to_numpy(dtype=np.int64)
        cumulative = np.cumsum(lengths)
        interval_idx = np.searchsorted(cumulative, global_positions, side="right")
        previous = np.r_[0, cumulative[:-1]][interval_idx]
        local_cruise_position = global_positions - previous
        raw_rows = (
            accepted["hdf5_start_row"].to_numpy(dtype=np.int64)[interval_idx]
            + accepted["cruise_start_sample"].to_numpy(dtype=np.int64)[interval_idx]
            + local_cruise_position
        )
        order = np.argsort(raw_rows)
        raw_sorted = raw_rows[order]
        w_sorted = np.asarray(hdf[f"W_{split}"][raw_sorted], dtype=np.float64)
        w = np.empty_like(w_sorted)
        w[order] = w_sorted
        sampled = pd.DataFrame(w, columns=FEATURE_NAMES)
        sampled.insert(0, "flight_class", accepted["flight_class"].to_numpy(dtype=np.int8)[interval_idx])
        sampled.insert(0, "cycle", accepted["cycle"].to_numpy(dtype=np.int32)[interval_idx])
        sampled.insert(0, "unit", accepted["unit"].to_numpy(dtype=np.int32)[interval_idx])
        sampled.insert(0, "split", split)
        sampled.insert(0, "dataset", dataset_label(path))
        sampled["cycle_local_sample"] = (
            accepted["cruise_start_sample"].to_numpy(dtype=np.int64)[interval_idx] + local_cruise_position
        )
        return intervals, sampled, accepted_total


def stratified_indices(metadata: pd.DataFrame, target_size: int, random_state: int) -> np.ndarray:
    target_size = min(target_size, len(metadata))
    if target_size == len(metadata):
        return np.arange(len(metadata))
    rng = np.random.default_rng(random_state)
    groups = metadata.groupby(["dataset", "split"], sort=False).indices
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


def scan_k(x: np.ndarray, silhouette_x: np.ndarray, k_values: list[int], random_state: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        print(f"  evaluating K={k} ...", flush=True)
        model = fit_kmeans(x, k, random_state)
        labels = model.predict(silhouette_x)
        rows.append({"k": k, "wcss": float(model.inertia_), "silhouette": float(silhouette_score(silhouette_x, labels))})
    return pd.DataFrame(rows)


def choose_k(metrics: pd.DataFrame) -> tuple[int, dict[str, object]]:
    """Select highest silhouette in the elbow's +/-2 neighborhood."""
    if len(metrics) < 3:
        k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
        return k, {"elbow_k": k, "best_silhouette_k": k, "candidate_ks": [k]}
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
        "selection_rule": "highest silhouette within elbow K +/- 2",
    }


def make_tsne(x: np.ndarray, random_state: int, max_iter: int) -> np.ndarray:
    kwargs = dict(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=random_state, method="barnes_hut", verbose=1)
    try:
        return TSNE(max_iter=max_iter, **kwargs).fit_transform(x)
    except TypeError:  # compatibility with older scikit-learn
        return TSNE(n_iter=max_iter, **kwargs).fit_transform(x)


def save_tsne_plot(embedding: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame, k: int, path: Path, title: str, seed: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 9), dpi=180)
    cmap = plt.get_cmap("turbo", k)
    for cluster in range(k):
        mask = labels == cluster
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.45, color=cmap(cluster), linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i), markersize=6, label=f"Cluster {i}") for i in range(k)]
    ax.legend(handles=handles, title="K-means cluster", ncol=min(4, k), loc="upper right", frameon=True)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    blocks = metadata[["dataset", "split"]].drop_duplicates().shape[0]
    ax.text(0.01, 0.01, f"Cruise W variables: {', '.join(FEATURE_NAMES)}\nn={len(embedding):,}; seed={seed}; source blocks={blocks}", transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom", bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8", "pad": 4})
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(metrics: pd.DataFrame, selected_k: int, path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    axes[0].plot(metrics["k"], metrics["wcss"], marker="o", color="#2f5d8c", linewidth=2)
    axes[0].axvline(selected_k, color="#b45f06", linestyle="--", linewidth=1.5, label=f"selected K={selected_k}")
    axes[0].set_title("WCSS on cruise W samples")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("WCSS")
    axes[0].legend()
    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", color="#8f3b76", linewidth=2)
    axes[1].axvline(selected_k, color="#b45f06", linestyle="--", linewidth=1.5, label=f"selected K={selected_k}")
    axes[1].set_title("Silhouette score on cruise W samples")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("mean silhouette")
    axes[1].legend()
    fig.suptitle("N-CMAPSS cruise operating-condition K selection", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_length_plots(intervals: pd.DataFrame, output_dir: Path) -> None:
    accepted = intervals.loc[intervals["cruise_status"] == "accepted"].copy()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=180)
    axes[0].hist(accepted["cruise_samples"], bins=40, color="#2f5d8c", alpha=0.85)
    axes[0].set_title("Accepted cruise-length distribution")
    axes[0].set_xlabel("cruise samples per cycle")
    axes[0].set_ylabel("cycles")
    datasets = sorted(intervals["dataset"].unique())
    box_data = [accepted.loc[accepted["dataset"] == d, "cruise_samples"].to_numpy() for d in datasets]
    axes[1].boxplot(box_data, labels=datasets, showfliers=False)
    axes[1].set_title("Accepted cruise length by dataset")
    axes[1].set_xlabel("dataset")
    axes[1].set_ylabel("cruise samples per cycle")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "cruise_length_distribution.png", bbox_inches="tight")
    plt.close(fig)


def summarize_lengths(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def aggregate(group: pd.DataFrame) -> pd.Series:
        accepted = group.loc[group["cruise_status"] == "accepted"]
        lengths = accepted["cruise_samples"].to_numpy(dtype=float)
        return pd.Series({
            "total_cycles": len(group),
            "accepted_cycles": len(accepted),
            "acceptance_ratio": len(accepted) / len(group) if len(group) else np.nan,
            "total_source_samples": int(group["total_samples"].sum()),
            "accepted_cruise_samples": int(accepted["cruise_samples"].sum()),
            "source_sample_ratio": accepted["cruise_samples"].sum() / group["total_samples"].sum() if group["total_samples"].sum() else np.nan,
            "cruise_length_min": np.min(lengths) if len(lengths) else np.nan,
            "cruise_length_p05": np.quantile(lengths, 0.05) if len(lengths) else np.nan,
            "cruise_length_median": np.median(lengths) if len(lengths) else np.nan,
            "cruise_length_mean": np.mean(lengths) if len(lengths) else np.nan,
            "cruise_length_p95": np.quantile(lengths, 0.95) if len(lengths) else np.nan,
            "cruise_length_max": np.max(lengths) if len(lengths) else np.nan,
        })

    by_dataset = intervals.groupby(["dataset", "split"], sort=True).apply(aggregate, include_groups=False).reset_index()
    by_unit = intervals.groupby(["dataset", "split", "unit"], sort=True).apply(aggregate, include_groups=False).reset_index()
    return by_dataset, by_unit


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_state)
    all_intervals: list[pd.DataFrame] = []
    all_samples: list[pd.DataFrame] = []
    skipped: list[str] = []
    files = sorted(args.data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")

    for path in files:
        for split in SPLITS:
            print(f"Extracting {path.name} / {split} ...", flush=True)
            try:
                intervals, samples, accepted_total = extract_block(path, split, PROFILE, args.cruise_samples_per_block, rng)
                all_intervals.append(intervals)
                if len(samples):
                    all_samples.append(samples)
                print(f"  cycles={len(intervals):,}; accepted={int((intervals['cruise_status'] == 'accepted').sum()):,}; cruise_samples={accepted_total:,}", flush=True)
            except Exception as exc:
                message = f"{path.name}/{split}: {type(exc).__name__}: {exc}"
                skipped.append(message)
                print(f"  SKIPPED: {message}", flush=True)

    intervals = pd.concat(all_intervals, ignore_index=True)
    samples = pd.concat(all_samples, ignore_index=True)
    intervals.to_csv(args.output_dir / "cruise_cycle_statistics.csv", index=False)
    by_dataset, by_unit = summarize_lengths(intervals)
    by_dataset.to_csv(args.output_dir / "cruise_length_summary_by_dataset.csv", index=False)
    by_unit.to_csv(args.output_dir / "cruise_length_summary_by_unit.csv", index=False)
    save_length_plots(intervals, args.output_dir)

    raw_x = samples[FEATURE_NAMES].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    x = scaler.fit_transform(raw_x).astype(np.float32)
    samples.to_csv(args.output_dir / "cruise_cluster_sample.csv", index=False)
    pd.DataFrame({"feature": FEATURE_NAMES, "mean": scaler.mean_, "scale": scaler.scale_}).to_csv(args.output_dir / "cruise_standardization.csv", index=False)

    fit_indices = stratified_indices(samples, min(len(samples), args.fit_samples), args.random_state + 1)
    x_fit = x[fit_indices]
    fit_metadata = samples.iloc[fit_indices].reset_index(drop=True)
    silhouette_indices = stratified_indices(fit_metadata, min(len(x_fit), args.silhouette_samples), args.random_state + 2)
    silhouette_x = x_fit[silhouette_indices]
    metrics = scan_k(x_fit, silhouette_x, list(range(args.k_min, args.k_max + 1)), args.random_state)
    selected_k, selection_info = choose_k(metrics)
    metrics["selected_k"] = metrics["k"].eq(selected_k)
    metrics.to_csv(args.output_dir / "k_selection_metrics.csv", index=False)
    save_metric_plot(metrics, selected_k, args.output_dir / "k_selection_wcss_silhouette.png")

    k16 = fit_kmeans(x_fit, 16, args.random_state)
    selected = fit_kmeans(x_fit, selected_k, args.random_state)
    for name, model in [("k16", k16), ("selected_k", selected)]:
        centers = scaler.inverse_transform(model.cluster_centers_)
        pd.DataFrame(centers, columns=FEATURE_NAMES).assign(cluster=np.arange(model.n_clusters)).to_csv(args.output_dir / f"cluster_centers_{name}.csv", index=False)

    tsne_indices = stratified_indices(fit_metadata, min(len(x_fit), args.tsne_samples), args.random_state + 3)
    x_tsne = x_fit[tsne_indices]
    tsne_metadata = fit_metadata.iloc[tsne_indices].reset_index(drop=True)
    print(f"Computing t-SNE for {len(x_tsne):,} sampled cruise rows ...", flush=True)
    embedding = make_tsne(x_tsne, args.random_state, args.tsne_max_iter)
    labels16 = k16.predict(x_tsne)
    labels_selected = selected.predict(x_tsne)
    for name, labels, k, title in [
        ("k16", labels16, 16, "N-CMAPSS cruise operating states: t-SNE colored by K-means (K=16)"),
        ("selected_k", labels_selected, selected_k, f"N-CMAPSS cruise operating states: t-SNE colored by selected K={selected_k}"),
    ]:
        points = tsne_metadata[["dataset", "split", "unit", "cycle", "flight_class", *FEATURE_NAMES, "cycle_local_sample"]].copy()
        points["tsne_1"] = embedding[:, 0]
        points["tsne_2"] = embedding[:, 1]
        points["cluster"] = labels
        points.to_csv(args.output_dir / f"tsne_{name}_points.csv", index=False)
        save_tsne_plot(embedding, labels, tsne_metadata, k, args.output_dir / f"tsne_{name}.png", title, args.random_state)

    metadata = {
        "method": "CruiseBench common_altitude",
        "paper": "CruiseBench: A Real-Flight-Aligned N-CMAPSS Benchmark for Engine RUL Prediction",
        "source_code": "https://github.com/NostalgiaJohn/CruiseBench/blob/main/tools/cruise_mask/common_altitude.py",
        "profile": asdict(PROFILE),
        "cluster_features": FEATURE_NAMES,
        "files_discovered": [p.name for p in files],
        "readable_source_blocks": int(intervals[["dataset", "split"]].drop_duplicates().shape[0]),
        "skipped_source_blocks": skipped,
        "total_cycles": int(len(intervals)),
        "accepted_cruise_cycles": int((intervals["cruise_status"] == "accepted").sum()),
        "accepted_cruise_samples": int(intervals["cruise_samples"].sum()),
        "cluster_sample_rows": int(len(samples)),
        "fit_rows": int(len(x_fit)),
        "tsne_rows": int(len(x_tsne)),
        "k16": 16,
        "selected_k": selected_k,
        "selection": selection_info,
        "random_state": args.random_state,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Selected K={selected_k}; saved outputs to {args.output_dir}", flush=True)
    print(json.dumps(by_dataset.to_dict(orient="records"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute condition-matched log-ratio LHI from all-dataset N-CMAPSS HI.

For each dataset and operating-condition class, B is the median HI among
flights marked ``hs == 1``.  The normalized indicator is

    LHI = log((HI + epsilon) / (B + epsilon)).

This keeps the raw model HI and exposes the baseline used for every row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/N-CMAPSS/lstm_autoencoder_paper/DS02-006/all_datasets_all_engines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hi-column", choices=("hi_paper_9", "hi_all_13"), default="hi_paper_9")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--rolling-window", type=int, default=5)
    return parser.parse_args()


def compute_lhi(scores: pd.DataFrame, hi_column: str, epsilon: float, rolling_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    if rolling_window < 1:
        raise ValueError("--rolling-window must be >= 1")
    scores = scores.copy()
    healthy = scores[(scores["health_stage"] == 1) & np.isfinite(scores[hi_column])]
    group_cols = ["dataset", "flight_class"]
    baselines = (
        healthy.groupby(group_cols, as_index=False)
        .agg(
            B_hi=(hi_column, "median"),
            healthy_reference_flights=(hi_column, "size"),
        )
    )
    dataset_fallback = healthy.groupby("dataset")[hi_column].median().to_dict()
    global_fallback = float(healthy[hi_column].median())
    baseline_map = {(row.dataset, row.flight_class): float(row.B_hi) for row in baselines.itertuples(index=False)}
    values = []
    sources = []
    reference_counts = []
    for row in scores[group_cols].itertuples(index=False):
        key = (row.dataset, row.flight_class)
        if key in baseline_map:
            values.append(baseline_map[key])
            sources.append("dataset_flight_class_hs1_median")
            reference_counts.append(int(baselines.loc[(baselines.dataset == row.dataset) & (baselines.flight_class == row.flight_class), "healthy_reference_flights"].iloc[0]))
        elif row.dataset in dataset_fallback:
            values.append(float(dataset_fallback[row.dataset]))
            sources.append("dataset_hs1_median_fallback")
            reference_counts.append(int((healthy["dataset"] == row.dataset).sum()))
        else:
            values.append(global_fallback)
            sources.append("global_hs1_median_fallback")
            reference_counts.append(len(healthy))
    scores["B_hi"] = values
    scores["B_source"] = sources
    scores["B_healthy_reference_flights"] = reference_counts
    scores["lhi"] = np.log((scores[hi_column] + epsilon) / (scores["B_hi"] + epsilon))
    scores["lhi_ratio"] = (scores[hi_column] + epsilon) / (scores["B_hi"] + epsilon)
    scores = scores.sort_values(["dataset", "split", "unit", "flight"]).reset_index(drop=True)
    scores["lhi_roll_mean"] = (
        scores.groupby(["dataset", "split", "unit"], sort=False)["lhi"]
        .transform(lambda series: series.rolling(rolling_window, min_periods=1).mean())
    )
    baselines["B_source"] = "dataset_flight_class_hs1_median"
    return scores, baselines


def plot_lhi(scores: pd.DataFrame, output: Path, hi_column: str, rolling_window: int) -> None:
    import matplotlib.pyplot as plt

    datasets = sorted(scores["dataset"].unique())
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.8 * nrows), squeeze=False, constrained_layout=True)
    colors = {"dev": "#2563eb", "test": "#c27c00"}
    for axis, dataset in zip(axes.flat, datasets):
        subset_dataset = scores[scores["dataset"] == dataset]
        for split in ("dev", "test"):
            subset = subset_dataset[subset_dataset["split"] == split]
            for _, frame in subset.groupby("unit", sort=True):
                frame = frame.sort_values("flight")
                axis.plot(frame["flight"], frame["lhi_roll_mean"], color=colors[split], alpha=0.55, linewidth=0.9)
        axis.axhline(0.0, color="#374151", linewidth=0.8, linestyle="--")
        axis.set_title(f"{dataset}: dev {subset_dataset[subset_dataset.split == 'dev'].unit.nunique()} / test {subset_dataset[subset_dataset.split == 'test'].unit.nunique()} engines")
        axis.set_xlabel("Flight")
        axis.set_ylabel(f"LHI (rolling mean, w={rolling_window})")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(datasets):]:
        axis.axis("off")
    fig.suptitle(
        f"N-CMAPSS: condition-matched log-ratio LHI from {hi_column}\n"
        "B = median HI of hs=1 flights within dataset and flight class",
        fontsize=15,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_dataset(scores: pd.DataFrame, output: Path, hi_column: str, rolling_window: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    colors = {"dev": "#2563eb", "test": "#c27c00"}
    for axis, split in zip(axes, ("dev", "test")):
        subset = scores[scores["split"] == split]
        for unit, frame in subset.groupby("unit", sort=True):
            frame = frame.sort_values("flight")
            axis.plot(frame["flight"], frame["lhi_roll_mean"], color=colors[split], alpha=0.8, linewidth=1.1, label=f"engine {int(unit)}")
        axis.axhline(0.0, color="#374151", linewidth=0.8, linestyle="--", label="healthy baseline B")
        axis.set_title(f"{split}: {subset['unit'].nunique()} engines")
        axis.set_xlabel("Flight")
        axis.set_ylabel(f"LHI (rolling mean, w={rolling_window})")
        axis.grid(alpha=0.22)
        axis.legend(ncol=6, frameon=False, fontsize=8, loc="upper left")
    dataset = str(scores["dataset"].iloc[0])
    fig.suptitle(f"N-CMAPSS {dataset}: condition-matched LHI for all engines", fontsize=15)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "lhi"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.input_dir / "all_datasets_all_engine_hi.csv"
    scores = pd.read_csv(input_path)
    scores, baselines = compute_lhi(scores, args.hi_column, args.epsilon, args.rolling_window)
    scores.to_csv(output_dir / "lhi_all_datasets_all_engines.csv", index=False)
    baselines.to_csv(output_dir / "lhi_baselines.csv", index=False)
    plot_lhi(scores, output_dir / "all_datasets_lhi.png", args.hi_column, args.rolling_window)
    for dataset, frame in scores.groupby("dataset", sort=True):
        dataset_dir = output_dir / str(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dataset_dir / "lhi_all_engines.csv", index=False)
        plot_dataset(frame, dataset_dir / "all_engines_lhi.png", args.hi_column, args.rolling_window)
    (output_dir / "metadata.json").write_text(json.dumps({
        "input": str(input_path),
        "hi_column": args.hi_column,
        "baseline": "median HI among hs==1 flights grouped by dataset and flight_class",
        "lhi_formula": "log((HI + epsilon) / (B + epsilon))",
        "epsilon": args.epsilon,
        "rolling_window": args.rolling_window,
    }, indent=2))
    print(f"saved N-CMAPSS LHI outputs to {output_dir}")


if __name__ == "__main__":
    main()

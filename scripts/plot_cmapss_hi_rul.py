#!/usr/bin/env python3
"""Align C-MAPSS LSTM-AE health indicators with true remaining useful life.

For training engines, true RUL is computed from the complete trajectory.  For
test engines, C-MAPSS supplies the RUL at the final observed cycle; RUL at an
earlier observed cycle is that value plus the number of observed cycles since
the final row.  HI rows are taken from the paper-transfer LSTM-AE output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FD_NAMES = ("FD001", "FD002", "FD003", "FD004")
RAW_COLUMNS = ["unit_id", "cycle", "setting1", "setting2", "setting3", *[f"s{i}" for i in range(1, 22)]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--hi-dir", type=Path, default=Path("outputs/CMAPSS/lstm_autoencoder_paper_transfer"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fds", nargs="+", choices=FD_NAMES, default=list(FD_NAMES))
    parser.add_argument("--max-example-engines", type=int, default=4)
    return parser.parse_args()


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)


def train_rul(data: pd.DataFrame) -> pd.DataFrame:
    last_cycle = data.groupby("unit_id")["cycle"].transform("max")
    return pd.DataFrame({"unit_id": data["unit_id"], "cycle": data["cycle"], "true_rul": last_cycle - data["cycle"]})


def test_rul(data: pd.DataFrame, endpoint_rul: np.ndarray) -> pd.DataFrame:
    last_cycle = data.groupby("unit_id")["cycle"].transform("max")
    endpoint = pd.Series(endpoint_rul, index=np.arange(1, len(endpoint_rul) + 1), name="endpoint_rul")
    endpoint_for_row = data["unit_id"].map(endpoint)
    if endpoint_for_row.isna().any():
        missing = sorted(data.loc[endpoint_for_row.isna(), "unit_id"].unique())
        raise ValueError(f"RUL file has no value for test engines: {missing}")
    return pd.DataFrame(
        {
            "unit_id": data["unit_id"],
            "cycle": data["cycle"],
            "endpoint_rul": endpoint_for_row,
            "true_rul": endpoint_for_row + last_cycle - data["cycle"],
        }
    )


def align_fd(fd: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    fd_hi = args.hi_dir / fd
    train_hi = pd.read_csv(fd_hi / "hi_train.csv")
    test_hi = pd.read_csv(fd_hi / "hi_test.csv")
    train_raw = load_raw(args.data_dir / fd / f"train_{fd}.txt")
    test_raw = load_raw(args.data_dir / fd / f"test_{fd}.txt")
    endpoint_rul = pd.read_csv(args.data_dir / fd / f"RUL_{fd}.txt", header=None).iloc[:, 0].to_numpy(dtype=float)

    train_truth = train_rul(train_raw)
    test_truth = test_rul(test_raw, endpoint_rul)
    train_result = train_hi.merge(train_truth, on=["unit_id", "cycle"], how="left", validate="one_to_one")
    test_result = test_hi.merge(test_truth, on=["unit_id", "cycle"], how="left", validate="one_to_one")
    for name, result in (("train", train_result), ("test", test_result)):
        if result["true_rul"].isna().any():
            raise ValueError(f"{fd} {name}: some HI rows could not be aligned to true RUL")
        result.insert(0, "fd", fd)
        result["split"] = name
    return train_result, test_result


def plot_engine_overlays(test_scores: pd.DataFrame, output: Path, fd: str, max_engines: int) -> None:
    import matplotlib.pyplot as plt

    # Pick engines whose HI spans the largest range so the overlay is useful
    # for visual inspection; all engines remain in the CSV outputs.
    units = (
        test_scores.groupby("unit_id")["hi_0_1"]
        .max()
        .sort_values(ascending=False)
        .head(max_engines)
        .index.astype(int)
        .tolist()
    )
    ncols = 2
    nrows = int(np.ceil(len(units) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.6 * nrows), squeeze=False, constrained_layout=True)
    for axis, unit in zip(axes.flat, units):
        frame = test_scores[test_scores["unit_id"] == unit].sort_values("cycle")
        left = axis
        right = axis.twinx()
        left.plot(frame["cycle"], frame["hi_0_1"], color="#2563eb", linewidth=1.7, label="HI")
        right.plot(frame["cycle"], frame["true_rul"], color="#c27c00", linewidth=1.5, label="true RUL")
        left.set_title(f"{fd} test engine {int(unit)}")
        left.set_xlabel("Cycle")
        left.set_ylabel("HI (healthy-baseline normalized; unbounded)", color="#2563eb")
        right.set_ylabel("True RUL (cycles)", color="#c27c00")
        left.grid(alpha=0.22)
        left.spines["top"].set_visible(False)
        handles = [left.lines[0], right.lines[0]]
        left.legend(handles, ["HI", "true RUL"], loc="upper left", frameon=False)
    for axis in axes.flat[len(units):]:
        axis.axis("off")
    fig.suptitle(f"{fd}: HI and true RUL for the same test engines", fontsize=15)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_hi_rul_scatter(scores: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, fd in zip(axes.flat, FD_NAMES):
        frame = scores[scores["fd"] == fd]
        for split, color, marker in (("train", "#2563eb", "."), ("test", "#c27c00", ".")):
            subset = frame[frame["split"] == split]
            axis.scatter(subset["true_rul"], subset["hi_0_1"], s=3, alpha=0.18, color=color, marker=marker, label=split)
        axis.set_title(fd)
        axis.set_xlabel("True RUL (cycles)")
        axis.set_ylabel("HI (unbounded above)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("C-MAPSS: relationship between HI and true RUL", fontsize=15)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def correlation_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fd, split), frame in scores.groupby(["fd", "split"], sort=True):
        rows.append(
            {
                "fd": fd,
                "split": split,
                "rows": len(frame),
                "engines": frame["unit_id"].nunique(),
                "pearson_hi_vs_rul": frame["hi_0_1"].corr(frame["true_rul"], method="pearson"),
                "spearman_hi_vs_rul": frame["hi_0_1"].corr(frame["true_rul"], method="spearman"),
                "hi_min": frame["hi_0_1"].min(),
                "hi_max": frame["hi_0_1"].max(),
                "rul_min": frame["true_rul"].min(),
                "rul_max": frame["true_rul"].max(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.hi_dir / "hi_rul_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scores: list[pd.DataFrame] = []
    for fd in args.fds:
        train_scores, test_scores = align_fd(fd, args)
        fd_dir = output_dir / fd
        fd_dir.mkdir(parents=True, exist_ok=True)
        train_scores.to_csv(fd_dir / "hi_rul_train.csv", index=False)
        test_scores.to_csv(fd_dir / "hi_rul_test.csv", index=False)
        plot_engine_overlays(test_scores, fd_dir / "hi_true_rul_test_engines.png", fd, args.max_example_engines)
        all_scores.extend([train_scores, test_scores])

    combined = pd.concat(all_scores, ignore_index=True)
    combined.to_csv(output_dir / "hi_rul_all_fds.csv", index=False)
    correlation_summary(combined).to_csv(output_dir / "correlation_summary.csv", index=False)
    plot_hi_rul_scatter(combined, output_dir / "hi_vs_true_rul_scatter.png")
    print(f"saved HI/RUL alignment outputs to {output_dir}")


if __name__ == "__main__":
    main()

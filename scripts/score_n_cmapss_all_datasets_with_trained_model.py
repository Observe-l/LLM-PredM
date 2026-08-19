#!/usr/bin/env python3
"""Use the trained N-CMAPSS LSTM-AE to score every readable dataset file.

The model and normalization parameters are loaded from the paper-faithful
DS02-006 training output.  Every flight in every readable N-CMAPSS HDF5 file
is one model-scored sequence after the same one-minute aggregation used during
training.  The script then plots all engines, separated by dataset and split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

try:
    from src.unsupervised_lstm_ae.n_cmapss_paper import PaperConditionalLSTMAE
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.unsupervised_lstm_ae.n_cmapss_paper import PaperConditionalLSTMAE


DEFAULT_DATA_DIR = Path("dataset/N-CMAPSS")
DEFAULT_MODEL_DIR = Path("outputs/N-CMAPSS/lstm_autoencoder_paper/DS02-006")
MODEL_SENSORS = ("Wf", "Nf", "T24", "T30", "T48", "T50", "P2", "P50", "W21", "W50", "SmFan", "SmLPC", "SmHPC")
HI_SENSORS = ("W50", "SmFan", "SmLPC", "SmHPC", "Wf", "T24", "T30", "T48", "T50")
OPERATING_CONDITIONS = ("alt", "Mach", "TRA", "T2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--aggregate-seconds", type=int, default=60)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def decode_names(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def aggregate_minutes(values: np.ndarray, seconds_per_minute: int) -> np.ndarray:
    starts = np.arange(0, len(values), seconds_per_minute)
    sums = np.add.reduceat(values, starts, axis=0)
    counts = np.diff(np.append(starts, len(values)))
    return sums / counts[:, None]


def flight_boundaries(auxiliary: np.ndarray) -> list[tuple[int, int]]:
    if len(auxiliary) == 0:
        return []
    changed = np.concatenate([np.array([True]), np.any(auxiliary[1:, :2] != auxiliary[:-1, :2], axis=1)])
    starts = np.flatnonzero(changed)
    ends = np.append(starts[1:], len(auxiliary))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def load_model(model_dir: Path, device: torch.device) -> tuple[PaperConditionalLSTMAE, np.ndarray, np.ndarray, dict[str, object]]:
    metadata = json.loads((model_dir / "metadata.json").read_text())
    model = PaperConditionalLSTMAE(
        sensor_dim=len(MODEL_SENSORS), operating_condition_dim=len(OPERATING_CONDITIONS),
        hidden_size=4, attention_window=5, fully_connected_hidden=128,
    ).to(device)
    checkpoint = torch.load(model_dir / "paper_lstm_ae.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    minimum = np.asarray(metadata["scaler"]["minimum"], dtype=np.float32)
    maximum = np.asarray(metadata["scaler"]["maximum"], dtype=np.float32)
    return model, minimum, maximum, metadata


def normalize(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    scale = np.where(np.isfinite(maximum - minimum) & ((maximum - minimum) > 1e-12), maximum - minimum, 1.0)
    return (2.0 * (values - minimum) / scale - 1.0).astype(np.float32)


def score_file(
    path: Path, model: PaperConditionalLSTMAE, minimum: np.ndarray, maximum: np.ndarray,
    aggregate_seconds: int, device: torch.device,
) -> pd.DataFrame:
    dataset_name = path.stem.replace("N-CMAPSS_", "")
    rows: list[dict[str, object]] = []
    with h5py.File(path, "r") as hdf:
        sensor_names_s = decode_names(hdf["X_s_var"][:])
        sensor_names_v = decode_names(hdf["X_v_var"][:])
        condition_names = decode_names(hdf["W_var"][:])
        if condition_names != list(OPERATING_CONDITIONS):
            raise ValueError(f"{dataset_name}: unexpected operating conditions {condition_names}")
        sensor_sources: list[tuple[str, int]] = []
        for name in MODEL_SENSORS:
            if name in sensor_names_s:
                sensor_sources.append(("X_s", sensor_names_s.index(name)))
            elif name in sensor_names_v:
                sensor_sources.append(("X_v", sensor_names_v.index(name)))
            else:
                raise ValueError(f"{dataset_name}: missing model sensor {name}")

        for split in ("dev", "test"):
            auxiliary = hdf[f"A_{split}"][:]
            boundaries = flight_boundaries(auxiliary)
            print(f"{dataset_name} {split}: {len(boundaries)} flights", flush=True)
            for index, (start, end) in enumerate(boundaries, start=1):
                a = auxiliary[start:end]
                w = aggregate_minutes(hdf[f"W_{split}"][start:end], aggregate_seconds)
                x_s = hdf[f"X_s_{split}"][start:end]
                x_v = hdf[f"X_v_{split}"][start:end]
                sensor_columns = [x_s[:, i] if source == "X_s" else x_v[:, i] for source, i in sensor_sources]
                sensors = aggregate_minutes(np.column_stack(sensor_columns), aggregate_seconds)
                values = normalize(np.concatenate([sensors, w], axis=1), minimum, maximum)
                if len(values) < 2:
                    continue
                sensor_tensor = torch.from_numpy(values[:, :len(MODEL_SENSORS)]).unsqueeze(0).to(device)
                condition_tensor = torch.from_numpy(values[:, len(MODEL_SENSORS):]).unsqueeze(0).to(device)
                with torch.inference_mode():
                    prediction = model(sensor_tensor, condition_tensor, teacher_forcing=False)
                errors = torch.abs(prediction - sensor_tensor[:, 1:]).squeeze(0).cpu().numpy()
                losses = errors.mean(axis=0)
                row: dict[str, object] = {
                    "dataset": dataset_name,
                    "split": split,
                    "unit": int(a[0, 0]),
                    "flight": int(a[0, 1]),
                    "flight_class": int(a[0, 2]),
                    "health_stage": int(a[0, 3]),
                    "n_minutes": len(values),
                }
                for name, loss in zip(MODEL_SENSORS, losses):
                    row[f"loss_{name}"] = float(loss)
                row["hi_all_13"] = float(losses.sum())
                row["hi_paper_9"] = float(sum(losses[MODEL_SENSORS.index(name)] for name in HI_SENSORS))
                rows.append(row)
                if index % 50 == 0 or index == len(boundaries):
                    print(f"  scored {index}/{len(boundaries)}", flush=True)
    return pd.DataFrame(rows)


def plot_dataset(scores: pd.DataFrame, output: Path, hi_column: str = "hi_paper_9") -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    colors = {"dev": "#2563eb", "test": "#c27c00"}
    for axis, split in zip(axes, ("dev", "test")):
        subset = scores[scores["split"] == split]
        for unit, frame in subset.groupby("unit", sort=True):
            frame = frame.sort_values("flight")
            axis.plot(frame["flight"], frame[hi_column], color=colors[split], alpha=0.8, linewidth=1.1, marker="o", markersize=1.8, label=f"engine {int(unit)}")
            healthy = frame[frame["health_stage"] == 1]
            degraded = frame[frame["health_stage"] == 0]
            if not healthy.empty and not degraded.empty:
                axis.axvline(healthy["flight"].max() + 0.5, color="#6b7280", linestyle="--", linewidth=0.7, alpha=0.45)
        axis.set_title(f"{split}: {subset['unit'].nunique()} engines, {len(subset)} flights")
        axis.set_xlabel("Flight")
        axis.set_ylabel("Paper HI (sum of 9 losses)")
        axis.grid(alpha=0.22)
        axis.legend(ncol=6, frameon=False, fontsize=8, loc="upper left")
    dataset = str(scores["dataset"].iloc[0])
    fig.suptitle(f"N-CMAPSS {dataset}: HI for all engines", fontsize=15)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_overview(scores: pd.DataFrame, output: Path, hi_column: str = "hi_paper_9") -> None:
    import matplotlib.pyplot as plt

    datasets = sorted(scores["dataset"].unique())
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.7 * nrows), squeeze=False, constrained_layout=True)
    colors = {"dev": "#2563eb", "test": "#c27c00"}
    for axis, dataset in zip(axes.flat, datasets):
        subset_dataset = scores[scores["dataset"] == dataset]
        for split in ("dev", "test"):
            subset = subset_dataset[subset_dataset["split"] == split]
            for _, frame in subset.groupby("unit", sort=True):
                frame = frame.sort_values("flight")
                axis.plot(frame["flight"], frame[hi_column], color=colors[split], alpha=0.42, linewidth=0.8)
        axis.set_title(f"{dataset}: dev {subset_dataset[subset_dataset.split == 'dev'].unit.nunique()} / test {subset_dataset[subset_dataset.split == 'test'].unit.nunique()} engines")
        axis.set_xlabel("Flight")
        axis.set_ylabel("HI")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(datasets):]:
        axis.axis("off")
    fig.suptitle("N-CMAPSS: HI for every engine in every readable dataset", fontsize=16)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.aggregate_seconds < 1:
        raise ValueError("--aggregate-seconds must be positive")
    output_dir = args.output_dir or args.model_dir / "all_datasets_all_engines"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, minimum, maximum, model_metadata = load_model(args.model_dir, device)
    results: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(args.data_dir.glob("N-CMAPSS_*.h5")):
        try:
            scores = score_file(path, model, minimum, maximum, args.aggregate_seconds, device)
        except Exception as exc:
            skipped.append({"file": str(path), "error": str(exc)})
            print(f"SKIPPED {path.name}: {exc}", flush=True)
            continue
        if scores.empty:
            continue
        results.append(scores)
        dataset_dir = output_dir / str(scores["dataset"].iloc[0])
        dataset_dir.mkdir(parents=True, exist_ok=True)
        scores.to_csv(dataset_dir / "all_engine_hi.csv", index=False)
        plot_dataset(scores, dataset_dir / "all_engines_hi.png")
    if not results:
        raise RuntimeError("No readable N-CMAPSS datasets were scored")
    combined = pd.concat(results, ignore_index=True).sort_values(["dataset", "split", "unit", "flight"]).reset_index(drop=True)
    combined.to_csv(output_dir / "all_datasets_all_engine_hi.csv", index=False)
    plot_overview(combined, output_dir / "all_datasets_all_engine_hi.png")
    (output_dir / "skipped_datasets.json").write_text(json.dumps(skipped, indent=2))
    (output_dir / "metadata.json").write_text(json.dumps({
        "model_dir": str(args.model_dir),
        "model_sensors": list(MODEL_SENSORS),
        "hi_sensors": list(HI_SENSORS),
        "operating_conditions": list(OPERATING_CONDITIONS),
        "aggregation_seconds": args.aggregate_seconds,
        "scaling": model_metadata["scaler"],
        "readable_datasets": sorted(combined["dataset"].unique()),
        "skipped_datasets": skipped,
    }, indent=2))
    print(f"saved all-dataset HI outputs to {output_dir}")


if __name__ == "__main__":
    main()

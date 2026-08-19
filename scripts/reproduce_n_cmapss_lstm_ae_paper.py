#!/usr/bin/env python3
"""Reproduce the N-CMAPSS DS02 LSTM-AE health-indicator method.

This script follows de Pater & Mitici (2023), Sections 2--4:

* DS02-006, development engines 2/5/10/16/18/20 and test engines 11/14/15;
* the 13 correlation-selected sensors from Table 1 and four operating
  conditions (alt, Mach, TRA, T2);
* one-minute means, followed by train-set min-max normalization to [-1, 1];
* healthy flights only (``hs == 1``) and the paper's 60, 70, ... , n windows
  rolled with a five-minute stride;
* hidden size 4, local Luong attention D=5, three fully connected layers,
  Adam, learning rate 0.01, 100 epochs, and 90/10 validation split;
* flight-level HI as the sum of mean reconstruction losses for the nine
  sensors selected in the paper's Table 3.

The paper's sensor measurements are stored across X_s and X_v in the local
HDF5 file. No C-MAPSS cycle-level preprocessing is used here.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.unsupervised_lstm_ae.n_cmapss_paper import PaperConditionalLSTMAE
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.unsupervised_lstm_ae.n_cmapss_paper import PaperConditionalLSTMAE


PAPER_MODEL_SENSORS = (
    "Wf", "Nf", "T24", "T30", "T48", "T50", "P2", "P50", "W21", "W50", "SmFan", "SmLPC", "SmHPC"
)
PAPER_HI_SENSORS = ("W50", "SmFan", "SmLPC", "SmHPC", "Wf", "T24", "T30", "T48", "T50")
OPERATING_CONDITIONS = ("alt", "Mach", "TRA", "T2")


@dataclass
class Flight:
    unit: int
    flight: int
    flight_class: int
    health_stage: int
    sensors: np.ndarray
    operating_conditions: np.ndarray


@dataclass
class MinMaxParameters:
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, records: list[Flight]) -> "MinMaxParameters":
        sensors = np.concatenate([record.sensors for record in records], axis=0)
        conditions = np.concatenate([record.operating_conditions for record in records], axis=0)
        values = np.concatenate([sensors, conditions], axis=1)
        return cls(minimum=values.min(axis=0), maximum=values.max(axis=0))

    def transform(self, record: Flight) -> Flight:
        scale = self.maximum - self.minimum
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        sensor_dim = record.sensors.shape[1]
        values = np.concatenate([record.sensors, record.operating_conditions], axis=1)
        normalized = 2.0 * (values - self.minimum) / scale - 1.0
        return Flight(
            unit=record.unit,
            flight=record.flight,
            flight_class=record.flight_class,
            health_stage=record.health_stage,
            sensors=normalized[:, :sensor_dim].astype(np.float32),
            operating_conditions=normalized[:, sensor_dim:].astype(np.float32),
        )

    def to_json(self, sensor_names: tuple[str, ...], condition_names: tuple[str, ...]) -> dict[str, object]:
        return {
            "normalization": "2 * (x - train_min) / (train_max - train_min) - 1",
            "fit_scope": "all one-minute aggregated development-set rows, before healthy-window filtering",
            "sensor_names": list(sensor_names),
            "operating_condition_names": list(condition_names),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("dataset/N-CMAPSS/N-CMAPSS_DS02-006.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N-CMAPSS/lstm_autoencoder_paper/DS02-006"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--window-start", type=int, default=60)
    parser.add_argument("--window-step", type=int, default=10)
    parser.add_argument("--roll-stride-minutes", type=int, default=5)
    parser.add_argument("--aggregate-seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def decode_names(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def aggregate_minutes(values: np.ndarray, seconds_per_minute: int) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("Cannot aggregate an empty flight")
    starts = np.arange(0, len(values), seconds_per_minute)
    sums = np.add.reduceat(values, starts, axis=0)
    counts = np.diff(np.append(starts, len(values)))
    return sums / counts[:, None]


def flight_boundaries(auxiliary: np.ndarray) -> list[tuple[int, int]]:
    if len(auxiliary) == 0:
        return []
    changed = np.concatenate(
        [np.array([True]), np.any(auxiliary[1:, :2] != auxiliary[:-1, :2], axis=1)]
    )
    starts = np.flatnonzero(changed)
    ends = np.append(starts[1:], len(auxiliary))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def load_split(
    hdf: h5py.File,
    split: str,
    model_sensor_indices: dict[str, tuple[str, int]],
    seconds_per_minute: int,
) -> list[Flight]:
    auxiliary = hdf[f"A_{split}"][:]
    sensor_names_s = decode_names(hdf["X_s_var"][:])
    sensor_names_v = decode_names(hdf["X_v_var"][:])
    condition_names = decode_names(hdf["W_var"][:])
    s_lookup = {name: index for index, name in enumerate(sensor_names_s)}
    v_lookup = {name: index for index, name in enumerate(sensor_names_v)}
    if condition_names != list(OPERATING_CONDITIONS):
        raise ValueError(f"Unexpected operating conditions: {condition_names}")

    records: list[Flight] = []
    for start, end in flight_boundaries(auxiliary):
        a = auxiliary[start:end]
        w = aggregate_minutes(hdf[f"W_{split}"][start:end], seconds_per_minute)
        x_s = hdf[f"X_s_{split}"][start:end]
        x_v = hdf[f"X_v_{split}"][start:end]
        sensor_columns = []
        for name in PAPER_MODEL_SENSORS:
            source, index = model_sensor_indices[name]
            sensor_columns.append(x_s[:, index] if source == "X_s" else x_v[:, index])
        x = aggregate_minutes(np.column_stack(sensor_columns), seconds_per_minute)
        records.append(
            Flight(
                unit=int(a[0, 0]),
                flight=int(a[0, 1]),
                flight_class=int(a[0, 2]),
                health_stage=int(a[0, 3]),
                sensors=x,
                operating_conditions=w,
            )
        )
    return records


def load_dataset(path: Path, seconds_per_minute: int) -> tuple[list[Flight], list[Flight], MinMaxParameters]:
    with h5py.File(path, "r") as hdf:
        sensor_names_s = decode_names(hdf["X_s_var"][:])
        sensor_names_v = decode_names(hdf["X_v_var"][:])
        model_indices: dict[str, tuple[str, int]] = {}
        for name in PAPER_MODEL_SENSORS:
            if name in sensor_names_s:
                model_indices[name] = ("X_s", sensor_names_s.index(name))
            elif name in sensor_names_v:
                model_indices[name] = ("X_v", sensor_names_v.index(name))
            else:
                raise ValueError(f"Paper sensor {name} is not present in DS02-006")
        development = load_split(hdf, "dev", model_indices, seconds_per_minute)
        test = load_split(hdf, "test", model_indices, seconds_per_minute)
    scaler = MinMaxParameters.fit(development)
    development = [scaler.transform(record) for record in development]
    test = [scaler.transform(record) for record in test]
    return development, test, scaler


def window_lengths(n_steps: int, start: int, step: int) -> list[int]:
    lengths = list(range(start, n_steps - 9, step))
    if start <= n_steps and n_steps not in lengths:
        lengths.append(n_steps)
    return sorted(set(lengths))


def build_training_groups(
    records: list[Flight],
    window_start: int,
    window_step: int,
    roll_stride: int,
) -> dict[int, np.ndarray]:
    groups: dict[int, list[np.ndarray]] = {}
    for record in records:
        if record.health_stage != 1:
            continue
        n_steps = len(record.sensors)
        for length in window_lengths(n_steps, window_start, window_step):
            for start in range(0, n_steps - length + 1, roll_stride):
                sensor = record.sensors[start : start + length]
                condition = record.operating_conditions[start : start + length]
                groups.setdefault(length, []).append(np.concatenate([sensor, condition], axis=1))
    if not groups:
        raise ValueError("No healthy training windows were built")
    return {length: np.stack(windows).astype(np.float32) for length, windows in groups.items()}


def make_loaders(
    groups: dict[int, np.ndarray],
    validation_fraction: float,
    batch_size: int,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[DataLoader], list[DataLoader], int, int]:
    rng = np.random.default_rng(seed)
    train_loaders: list[DataLoader] = []
    validation_loaders: list[DataLoader] = []
    n_train = 0
    n_validation = 0
    for length in sorted(groups):
        values = groups[length]
        order = rng.permutation(len(values))
        n_val = max(1, int(round(len(values) * validation_fraction)))
        if n_val >= len(values):
            n_val = 1
        validation_values = torch.from_numpy(values[order[:n_val]])
        train_values = torch.from_numpy(values[order[n_val:]])
        train_dataset = TensorDataset(train_values[:, :, : len(PAPER_MODEL_SENSORS)], train_values[:, :, len(PAPER_MODEL_SENSORS) :])
        validation_dataset = TensorDataset(validation_values[:, :, : len(PAPER_MODEL_SENSORS)], validation_values[:, :, len(PAPER_MODEL_SENSORS) :])
        if len(train_values) > 0:
            train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda"))
            n_train += len(train_values)
        if len(validation_values) > 0:
            validation_loaders.append(DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda"))
            n_validation += len(validation_values)
    return train_loaders, validation_loaders, n_train, n_validation


def sequence_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Eq. (1): total absolute reconstruction error over t=2..n and sensors.
    return torch.abs(prediction - target[:, 1:]).sum(dim=(1, 2)).mean()


def train_model(
    groups: dict[int, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PaperConditionalLSTMAE, pd.DataFrame]:
    train_loaders, validation_loaders, n_train, n_validation = make_loaders(
        groups, args.validation_fraction, args.batch_size, args.seed, args.num_workers, device
    )
    model = PaperConditionalLSTMAE(
        sensor_dim=len(PAPER_MODEL_SENSORS),
        operating_condition_dim=len(OPERATING_CONDITIONS),
        hidden_size=4,
        attention_window=5,
        fully_connected_hidden=128,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=1, threshold=0.0, threshold_mode="abs"
    )
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for loader in train_loaders:
            for sensors, conditions in loader:
                sensors = sensors.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(sensors, conditions, teacher_forcing=True)
                loss = sequence_loss(prediction, sensors)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for loader in validation_loaders:
                for sensors, conditions in loader:
                    sensors = sensors.to(device, non_blocking=True)
                    conditions = conditions.to(device, non_blocking=True)
                    validation_losses.append(float(sequence_loss(model(sensors, conditions, True), sensors).cpu()))
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        scheduler.step(validation_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}/{args.epochs}: train={train_loss:.4f} val={validation_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6g}")
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    model.load_state_dict(best_state)
    history_frame = pd.DataFrame(history)
    history_frame.attrs["n_train_windows"] = n_train
    history_frame.attrs["n_validation_windows"] = n_validation
    return model, history_frame


def score_flights(records: list[Flight], model: PaperConditionalLSTMAE, device: torch.device, split: str) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for index, record in enumerate(records):
            sensors = torch.from_numpy(record.sensors).unsqueeze(0).to(device)
            conditions = torch.from_numpy(record.operating_conditions).unsqueeze(0).to(device)
            prediction = model(sensors, conditions, teacher_forcing=False)
            error = torch.abs(prediction - sensors[:, 1:]).squeeze(0).cpu().numpy()
            losses = error.mean(axis=0)
            row: dict[str, object] = {
                "split": split,
                "unit": record.unit,
                "flight": record.flight,
                "flight_class": record.flight_class,
                "health_stage": record.health_stage,
                "n_minutes": len(record.sensors),
            }
            for name, loss in zip(PAPER_MODEL_SENSORS, losses):
                row[f"loss_{name}"] = float(loss)
            row["hi_all_13"] = float(losses.sum())
            row["hi_paper_9"] = float(sum(losses[PAPER_MODEL_SENSORS.index(name)] for name in PAPER_HI_SENSORS))
            rows.append(row)
            if split == "test":
                print(f"scored test engine {record.unit}, flight {record.flight} ({index + 1}/{len(records)})", end="\r")
    if split == "test":
        print()
    return pd.DataFrame(rows).sort_values(["unit", "flight"]).reset_index(drop=True)


def plot_test_hi(scores: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for unit, frame in scores.groupby("unit", sort=True):
        axis.plot(frame["flight"], frame["hi_paper_9"], linewidth=1.8, label=f"engine {int(unit)}")
        healthy = frame[frame["health_stage"] == 1]
        if not healthy.empty:
            axis.axvline(healthy["flight"].max(), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axis.set_xlabel("Flight")
    axis.set_ylabel("Health indicator (sum of 9 sensor losses)")
    axis.set_title("N-CMAPSS DS02: paper-style LSTM-AE health indicator")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_sensor_losses(train_scores: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False, constrained_layout=True)
    for axis, sensor in zip(axes.flat, PAPER_HI_SENSORS):
        for unit, frame in train_scores.groupby("unit", sort=True):
            axis.plot(frame["flight"], frame[f"loss_{sensor}"], alpha=0.25, linewidth=0.6)
        axis.set_title(sensor)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Flight")
        axis.set_ylabel("mean abs. loss")
    fig.suptitle("Training reconstruction losses for the nine paper HI sensors")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.aggregate_seconds < 1 or args.roll_stride_minutes < 1:
        raise ValueError("aggregation and rolling strides must be positive")
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.data_file} and aggregating one-minute flight samples...")
    development, test, scaler = load_dataset(args.data_file, args.aggregate_seconds)
    print(
        f"development flights={len(development)}, healthy flights={sum(r.health_stage == 1 for r in development)}, "
        f"test flights={len(test)}, device={device}"
    )
    groups = build_training_groups(development, args.window_start, args.window_step, args.roll_stride_minutes)
    print(f"training window lengths={sorted(groups)}, windows={sum(len(x) for x in groups.values())}")
    model, history = train_model(groups, args, device)
    train_scores = score_flights(development, model, device, "development")
    test_scores = score_flights(test, model, device, "test")
    all_scores = pd.concat([train_scores, test_scores], ignore_index=True)
    all_scores.to_csv(args.output_dir / "flight_health_indicators.csv", index=False)
    train_scores.to_csv(args.output_dir / "development_flight_health_indicators.csv", index=False)
    test_scores.to_csv(args.output_dir / "test_flight_health_indicators.csv", index=False)
    history.to_csv(args.output_dir / "training_history.csv", index=False)
    plot_test_hi(test_scores, args.output_dir / "figure10_style_test_hi.png")
    plot_sensor_losses(train_scores, args.output_dir / "development_sensor_losses.png")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sensor_names": list(PAPER_MODEL_SENSORS),
            "hi_sensor_names": list(PAPER_HI_SENSORS),
            "operating_condition_names": list(OPERATING_CONDITIONS),
            "hidden_size": 4,
            "attention_window": 5,
            "fully_connected_hidden": 128,
            "aggregate_seconds": args.aggregate_seconds,
        },
        args.output_dir / "paper_lstm_ae.pt",
    )
    metadata = {
        "paper": "de Pater & Mitici (2023), Developing health indicators and RUL prognostics for systems with few failure instances and varying operating conditions using a LSTM autoencoder",
        "dataset": str(args.data_file),
        "development_units": sorted({record.unit for record in development}),
        "test_units": sorted({record.unit for record in test}),
        "development_flights": len(development),
        "healthy_development_flights": sum(record.health_stage == 1 for record in development),
        "test_flights": len(test),
        "sensor_names_table_1": list(PAPER_MODEL_SENSORS),
        "health_indicator_sensor_names_table_3": list(PAPER_HI_SENSORS),
        "operating_conditions": list(OPERATING_CONDITIONS),
        "training_window_lengths": sorted(groups),
        "training_window_count": int(sum(len(x) for x in groups.values())),
        "validation_fraction": args.validation_fraction,
        "model": {"hidden_size": 4, "attention_window_D": 5, "fc_layers": 3, "fc_hidden": 128},
        "optimization": {"optimizer": "Adam", "initial_learning_rate": args.learning_rate, "epochs": args.epochs, "lr_factor": 0.1, "lr_patience": 1},
        "flight_aggregation": "mean over consecutive 60-second blocks; final partial block retained",
        "hi_definition": "sum over Table 3 sensors of each sensor's mean absolute reconstruction error over decoder time steps 2..n",
        "scaler": scaler.to_json(PAPER_MODEL_SENSORS, OPERATING_CONDITIONS),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved paper-style N-CMAPSS outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

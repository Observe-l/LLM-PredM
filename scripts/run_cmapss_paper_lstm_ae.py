#!/usr/bin/env python3
"""Transfer the paper-style conditional LSTM-AE from N-CMAPSS to C-MAPSS.

The N-CMAPSS paper uses flight-level sequences and an ``hs`` flag.  C-MAPSS
has cycle-level engine trajectories and no health-stage flag, so this script
keeps the previous C-MAPSS convention: cycles 1--50 of every training engine
are treated as healthy.  The remaining details are transferred from the
paper: min-max scaling to [-1, 1], operating conditions in the recurrent
input, hidden size 4, local Luong attention D=5, teacher-forced training,
autoregressive scoring, and absolute reconstruction-loss HI.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.unsupervised_lstm_ae.n_cmapss_paper import PaperConditionalLSTMAE
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.unsupervised_lstm_ae.n_cmapss_paper import PaperConditionalLSTMAE


FD_NAMES = ("FD001", "FD002", "FD003", "FD004")
SETTING_NAMES = ("setting1", "setting2", "setting3")
ALL_SENSOR_NAMES = tuple(f"s{i}" for i in range(1, 22))
# This is the 14-sensor list used by the earlier C-MAPSS processing.
SELECTED_SENSORS = (
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14",
    "s15", "s17", "s20", "s21",
)
CSV_COLUMNS = ("unit_id", "cycle", *SETTING_NAMES, *ALL_SENSOR_NAMES)


@dataclass
class Engine:
    unit_id: int
    cycles: np.ndarray
    sensors: np.ndarray
    conditions: np.ndarray


@dataclass
class MinMaxScaler:
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, engines: list[Engine], healthy_cycles: int) -> "MinMaxScaler":
        values = np.concatenate(
            [
                np.concatenate([e.sensors[e.cycles <= healthy_cycles], e.conditions[e.cycles <= healthy_cycles]], axis=1)
                for e in engines
            ],
            axis=0,
        )
        return cls(values.min(axis=0), values.max(axis=0))

    def transform(self, engine: Engine) -> Engine:
        values = np.concatenate([engine.sensors, engine.conditions], axis=1)
        scale = self.maximum - self.minimum
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        normalized = 2.0 * (values - self.minimum) / scale - 1.0
        split = engine.sensors.shape[1]
        return Engine(
            unit_id=engine.unit_id,
            cycles=engine.cycles,
            sensors=normalized[:, :split].astype(np.float32),
            conditions=normalized[:, split:].astype(np.float32),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "method": "2 * (x - healthy_train_min) / (healthy_train_max - healthy_train_min) - 1",
            "fit_scope": "all training-engine rows with cycle <= healthy_cycles",
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/CMAPSS/lstm_autoencoder_paper_transfer"))
    parser.add_argument("--fds", nargs="+", choices=FD_NAMES, default=list(FD_NAMES))
    parser.add_argument("--healthy-cycles", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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


def load_cmapss_file(path: Path) -> list[Engine]:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=CSV_COLUMNS)
    engines: list[Engine] = []
    for unit_id, group in frame.groupby("unit_id", sort=True):
        group = group.sort_values("cycle")
        engines.append(
            Engine(
                unit_id=int(unit_id),
                cycles=group["cycle"].to_numpy(dtype=np.int64),
                sensors=group.loc[:, SELECTED_SENSORS].to_numpy(dtype=np.float32),
                conditions=group.loc[:, SETTING_NAMES].to_numpy(dtype=np.float32),
            )
        )
    return engines


def build_windows(
    engines: list[Engine],
    window_size: int,
    stride: int,
    healthy_only: bool = False,
    healthy_cycles: int = 50,
) -> np.ndarray:
    windows: list[np.ndarray] = []
    sensor_dim = len(SELECTED_SENSORS)
    for engine in engines:
        stop = len(engine.cycles) - window_size + 1
        if stop <= 0:
            continue
        for start in range(0, stop, stride):
            end = start + window_size
            if healthy_only and np.any(engine.cycles[start:end] > healthy_cycles):
                break
            windows.append(np.concatenate([engine.sensors[start:end], engine.conditions[start:end]], axis=1))
    if not windows:
        raise ValueError("No C-MAPSS windows were built")
    return np.stack(windows).astype(np.float32)


def make_loaders(
    values: np.ndarray,
    validation_fraction: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, int, int]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(values))
    n_val = max(1, int(round(len(values) * validation_fraction)))
    n_val = min(n_val, len(values) - 1)
    split = order[:n_val], order[n_val:]
    sensor_dim = len(SELECTED_SENSORS)
    def dataset(indices: np.ndarray) -> TensorDataset:
        subset = torch.from_numpy(values[indices])
        return TensorDataset(subset[:, :, :sensor_dim], subset[:, :, sensor_dim:])
    train = DataLoader(dataset(split[1]), batch_size=batch_size, shuffle=True, pin_memory=device.type == "cuda")
    validation = DataLoader(dataset(split[0]), batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    return train, validation, len(split[1]), len(split[0])


def sequence_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(prediction - target[:, 1:]).sum(dim=(1, 2)).mean()


def train_model(values: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[PaperConditionalLSTMAE, pd.DataFrame, int, int]:
    train_loader, validation_loader, n_train, n_validation = make_loaders(
        values, args.validation_fraction, args.batch_size, args.seed, device
    )
    model = PaperConditionalLSTMAE(
        sensor_dim=len(SELECTED_SENSORS), operating_condition_dim=len(SETTING_NAMES),
        hidden_size=4, attention_window=5, fully_connected_hidden=128,
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
        for sensors, conditions in train_loader:
            sensors, conditions = sensors.to(device, non_blocking=True), conditions.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model(sensors, conditions, teacher_forcing=True), sensors)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for sensors, conditions in validation_loader:
                sensors, conditions = sensors.to(device, non_blocking=True), conditions.to(device, non_blocking=True)
                validation_losses.append(float(sequence_loss(model(sensors, conditions, True), sensors).cpu()))
        train_loss, validation_loss = float(np.mean(train_losses)), float(np.mean(validation_losses))
        scheduler.step(validation_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}/{args.epochs}: train={train_loss:.4f} val={validation_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6g}")
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), n_train, n_validation


def score_engines(
    engines: list[Engine], model: PaperConditionalLSTMAE, device: torch.device,
    split: str, window_size: int, stride: int, batch_size: int,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    sensor_dim = len(SELECTED_SENSORS)
    with torch.no_grad():
        for engine_index, engine in enumerate(engines):
            # A few C-MAPSS test engines are shorter than the default training
            # window.  The paper architecture supports variable sequence
            # lengths, so retain them at their available length for scoring.
            effective_window = min(window_size, len(engine.cycles))
            if effective_window < 2:
                continue
            windows = build_windows([engine], effective_window, stride)
            sensors = torch.from_numpy(windows[:, :, :sensor_dim])
            conditions = torch.from_numpy(windows[:, :, sensor_dim:])
            unit_losses: list[np.ndarray] = []
            for start in range(0, len(windows), batch_size):
                sb = sensors[start:start + batch_size].to(device)
                cb = conditions[start:start + batch_size].to(device)
                prediction = model(sb, cb, teacher_forcing=False)
                error = torch.abs(prediction - sb[:, 1:]).mean(dim=1).cpu().numpy()
                unit_losses.append(error)
            losses = np.concatenate(unit_losses, axis=0)
            end_indices = np.arange(effective_window - 1, len(engine.cycles), stride)[:len(losses)]
            for loss, end_index in zip(losses, end_indices):
                row: dict[str, object] = {
                    "split": split, "unit_id": engine.unit_id,
                    "window_start_cycle": int(engine.cycles[end_index - effective_window + 1]),
                    "cycle": int(engine.cycles[end_index]), "window_size": effective_window,
                }
                for name, value in zip(SELECTED_SENSORS, loss):
                    row[f"loss_{name}"] = float(value)
                row["hi_sum_14"] = float(loss.sum())
                row["hi_mean_14"] = float(loss.mean())
                rows.append(row)
            print(f"scored {split} engine {engine.unit_id} ({engine_index + 1}/{len(engines)})", end="\r")
    print()
    return pd.DataFrame(rows).sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def add_health_normalization(
    train_scores: pd.DataFrame, test_scores: pd.DataFrame, healthy_cycles: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Retain raw paper HI and add an unbounded health-baseline calibration."""

    healthy = train_scores[train_scores["cycle"] <= healthy_cycles]["hi_sum_14"]
    baseline = float(healthy.median())
    upper = float(train_scores["hi_sum_14"].quantile(0.95))
    if not np.isfinite(upper) or upper <= baseline + 1e-12:
        upper = baseline + 1.0
    for scores in (train_scores, test_scores):
        scores["hi_relative_to_healthy"] = scores["hi_sum_14"] / max(baseline, 1e-12)
        # Keep the healthy baseline at 0, but do not saturate degradation at 1.
        # Thus a value of 2 means roughly twice the healthy-to-q95 excursion.
        scores["hi_0_1"] = np.maximum(
            (scores["hi_sum_14"] - baseline) / (upper - baseline), 0.0
        )
    return train_scores, test_scores, {
        "healthy_hi_sum_14_median": baseline,
        "train_hi_sum_14_q95": upper,
        "hi_0_1_definition": "max((hi_sum_14 - healthy_train_median) / (train_q95 - healthy_train_median), 0); no upper clipping",
    }


def plot_fd(train_scores: pd.DataFrame, test_scores: pd.DataFrame, path: Path, fd_name: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    for axis, scores, title in ((axes[0], train_scores, "training engines"), (axes[1], test_scores, "test engines")):
        for unit, frame in scores.groupby("unit_id", sort=True):
            axis.plot(frame["cycle"], frame["hi_0_1"], color="steelblue", alpha=0.28, linewidth=0.7)
        for unit, frame in scores.groupby("unit_id", sort=True):
            if unit in (1, 2, 3, 4):
                axis.plot(frame["cycle"], frame["hi_0_1"], linewidth=1.4, label=f"engine {int(unit)}")
        axis.set_title(f"{fd_name} — {title}")
        axis.set_ylabel("normalized HI (unbounded above)")
        axis.grid(alpha=0.25)
        if not scores.empty:
            axis.legend(ncol=4, fontsize=8)
    axes[1].set_xlabel("Cycle (window endpoint)")
    fig.suptitle("Paper-style conditional LSTM-AE health indicator on C-MAPSS", fontsize=15)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_all(scores: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fds = list(scores["fd"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for axis, fd_name in zip(axes.flat, fds):
        frame_fd = scores[scores["fd"] == fd_name]
        for (split, unit), frame in frame_fd.groupby(["split", "unit_id"], sort=True):
            axis.plot(frame["cycle"], frame["hi_0_1"], alpha=0.22 if split == "train" else 0.4, linewidth=0.65)
        axis.set_title(fd_name)
        axis.set_xlabel("Cycle")
        axis.set_ylabel("normalized HI (unbounded above)")
        axis.grid(alpha=0.2)
    fig.suptitle("C-MAPSS: N-CMAPSS paper-style LSTM-AE transfer", fontsize=15)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_fd(fd_name: str, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    train_raw = load_cmapss_file(args.data_dir / fd_name / f"train_{fd_name}.txt")
    test_raw = load_cmapss_file(args.data_dir / fd_name / f"test_{fd_name}.txt")
    scaler = MinMaxScaler.fit(train_raw, args.healthy_cycles)
    train = [scaler.transform(engine) for engine in train_raw]
    test = [scaler.transform(engine) for engine in test_raw]
    train_windows = build_windows(train, args.window_size, args.stride, True, args.healthy_cycles)
    print(f"{fd_name}: train engines={len(train)}, test engines={len(test)}, healthy windows={len(train_windows)}")
    model, history, n_train, n_validation = train_model(train_windows, args, device)
    fd_output = args.output_dir / fd_name
    fd_output.mkdir(parents=True, exist_ok=True)
    train_scores = score_engines(train, model, device, "train", args.window_size, args.stride, args.batch_size)
    test_scores = score_engines(test, model, device, "test", args.window_size, args.stride, args.batch_size)
    train_scores, test_scores, normalization = add_health_normalization(
        train_scores, test_scores, args.healthy_cycles
    )
    train_scores.to_csv(fd_output / "hi_train.csv", index=False)
    test_scores.to_csv(fd_output / "hi_test.csv", index=False)
    pd.concat([train_scores, test_scores], ignore_index=True).to_csv(fd_output / "hi_all.csv", index=False)
    history.to_csv(fd_output / "training_history.csv", index=False)
    plot_fd(train_scores, test_scores, fd_output / "hi_curves.png", fd_name)
    torch.save({"model_state_dict": model.state_dict(), "sensor_names": list(SELECTED_SENSORS), "operating_condition_names": list(SETTING_NAMES), "window_size": args.window_size}, fd_output / "paper_transfer_lstm_ae.pt")
    metadata = {
        "dataset": "C-MAPSS", "fd": fd_name, "data_dir": str(args.data_dir),
        "sensor_names": list(SELECTED_SENSORS), "operating_conditions": list(SETTING_NAMES),
        "healthy_definition": f"training cycles <= {args.healthy_cycles}",
        "window_size": args.window_size, "stride": args.stride,
        "training_windows": len(train_windows), "optimization_windows": {"train": n_train, "validation": n_validation},
        "model": {"hidden_size": 4, "attention_window_D": 5, "fc_layers": 3, "fc_hidden": 128},
        "optimization": {"optimizer": "Adam", "initial_learning_rate": args.learning_rate, "epochs": args.epochs, "lr_factor": 0.1, "lr_patience": 1},
        "hi_definition": "sum of the 14 per-sensor mean absolute autoregressive reconstruction losses over decoder steps 2..window_size",
        "health_indicator_visual_normalization": normalization,
        "scaler": scaler.to_json(),
    }
    (fd_output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    all_scores = pd.concat([train_scores.assign(fd=fd_name), test_scores.assign(fd=fd_name)], ignore_index=True)
    return all_scores


def main() -> None:
    args = parse_args()
    if args.healthy_cycles < args.window_size or args.window_size < 2 or args.stride < 1:
        raise ValueError("healthy-cycles must be >= window-size >= 2 and stride must be positive")
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}; sensors={','.join(SELECTED_SENSORS)}")
    results = [run_fd(fd_name, args, device) for fd_name in args.fds]
    # When a subset is rerun after an interrupted batch, include completed FD
    # outputs so the root-level summary and combined figure remain complete.
    completed = {str(frame["fd"].iloc[0]) for frame in results if not frame.empty}
    for fd_name in FD_NAMES:
        existing = args.output_dir / fd_name / "hi_all.csv"
        if fd_name not in completed and existing.exists():
            results.append(pd.read_csv(existing).assign(fd=fd_name))
    all_scores = pd.concat(results, ignore_index=True)
    all_scores.to_csv(args.output_dir / "hi_all_fds.csv", index=False)
    summary = all_scores.groupby(["fd", "split", "unit_id"], as_index=False).agg(
        first_cycle=("cycle", "min"), last_cycle=("cycle", "max"),
        first_hi=("hi_sum_14", "first"), last_hi=("hi_sum_14", "last"), max_hi=("hi_sum_14", "max"),
    )
    summary.to_csv(args.output_dir / "engine_summary.csv", index=False)
    plot_all(all_scores, args.output_dir / "hi_curves_all_fds.png")
    print(f"saved C-MAPSS paper-transfer outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

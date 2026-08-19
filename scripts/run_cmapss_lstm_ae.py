#!/usr/bin/env python3
"""Train conditional LSTM autoencoders and build C-MAPSS health indicators.

For each FD subset independently:

* the first ``--healthy-cycles`` cycles of every training engine are the only
  samples used to fit the scaler and train the autoencoder;
* all 3 operating conditions enter the encoder and decoder, while all 21
  sensors are reconstructed;
* a rolling-window reconstruction error is emitted for every cycle that has
  a complete window in train and test;
* train/test HI CSV files and publication-friendly PNG curves are saved.

Example:
    python scripts/run_cmapss_lstm_ae.py --fds FD001 FD002 FD003 FD004
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

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
except ImportError as exc:  # pragma: no cover - gives a useful CLI error
    raise SystemExit(
        "PyTorch is required. Activate an environment containing torch, pandas, and matplotlib."
    ) from exc

try:
    from src.unsupervised_lstm_ae.cmapss_lstm_ae import ConditionalLSTMAutoencoder
except ModuleNotFoundError:  # allow execution from outside the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.unsupervised_lstm_ae.cmapss_lstm_ae import ConditionalLSTMAutoencoder


FD_NAMES = ("FD001", "FD002", "FD003", "FD004")
OPERATION_COLUMNS = ("op_1", "op_2", "op_3")
SENSOR_COLUMNS = tuple(f"sensor_{i}" for i in range(1, 22))
ALL_COLUMNS = ("unit_id", "cycle", *OPERATION_COLUMNS, *SENSOR_COLUMNS)


@dataclass
class MinMaxScaler:
    """Min-max scaler fitted only on the healthy training rows."""

    minimum: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "MinMaxScaler":
        values = np.asarray(values, dtype=np.float64)
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        scale = maximum - minimum
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        return cls(minimum=minimum, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.minimum.astype(np.float32)) / self.scale.astype(np.float32)

    def to_json(self) -> dict[str, list[float]]:
        return {"minimum": self.minimum.tolist(), "scale": self.scale.tolist(), "range": "[0, 1]"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/CMAPSS/lstm_autoencoder"))
    parser.add_argument("--fds", nargs="+", default=list(FD_NAMES), choices=FD_NAMES)
    parser.add_argument("--healthy-cycles", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_cmapss(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLUMNS, engine="python")
    if frame.isna().any().any():
        raise ValueError(f"Missing value found in {path}")
    frame["unit_id"] = frame["unit_id"].astype(int)
    frame["cycle"] = frame["cycle"].astype(int)
    return frame


def make_training_windows(
    frame: pd.DataFrame,
    sensor_scaler: MinMaxScaler,
    condition_scaler: MinMaxScaler,
    window_size: int,
    healthy_cycles: int,
) -> tuple[np.ndarray, np.ndarray]:
    sensor_windows: list[np.ndarray] = []
    condition_windows: list[np.ndarray] = []
    for _, unit in frame.groupby("unit_id", sort=True):
        healthy = unit.sort_values("cycle")
        healthy = healthy[healthy["cycle"] <= healthy_cycles]
        sensors = sensor_scaler.transform(healthy[list(SENSOR_COLUMNS)].to_numpy())
        conditions = condition_scaler.transform(healthy[list(OPERATION_COLUMNS)].to_numpy())
        for start in range(max(0, len(healthy) - window_size + 1)):
            sensor_windows.append(sensors[start : start + window_size])
            condition_windows.append(conditions[start : start + window_size])
    if not sensor_windows:
        raise ValueError("No complete healthy windows were found; reduce --window-size")
    return np.stack(sensor_windows).astype(np.float32), np.stack(condition_windows).astype(np.float32)


def fit_scalers(frame: pd.DataFrame, healthy_cycles: int) -> tuple[MinMaxScaler, MinMaxScaler]:
    healthy = frame[frame["cycle"] <= healthy_cycles]
    if healthy.empty:
        raise ValueError("No healthy rows found")
    return (
        MinMaxScaler.fit(healthy[list(SENSOR_COLUMNS)].to_numpy()),
        MinMaxScaler.fit(healthy[list(OPERATION_COLUMNS)].to_numpy()),
    )


def reconstruction_loss(predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    return torch.abs(predicted - actual).mean()


def train_model(
    sensors: np.ndarray,
    conditions: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConditionalLSTMAutoencoder, pd.DataFrame]:
    dataset = TensorDataset(torch.from_numpy(sensors), torch.from_numpy(conditions))
    validation_size = max(1, int(round(len(dataset) * args.validation_fraction)))
    if validation_size >= len(dataset):
        validation_size = 1
    training_size = len(dataset) - validation_size
    split_generator = torch.Generator().manual_seed(args.seed)
    train_set, validation_set = random_split(
        dataset, [training_size, validation_size], generator=split_generator
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ConditionalLSTMAutoencoder(
        sensor_dim=len(SENSOR_COLUMNS),
        operating_condition_dim=len(OPERATION_COLUMNS),
        hidden_size=args.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_sensors, batch_conditions in train_loader:
            batch_sensors = batch_sensors.to(device)
            batch_conditions = batch_conditions.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_sensors, batch_conditions)
            loss = reconstruction_loss(prediction, batch_sensors)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for batch_sensors, batch_conditions in validation_loader:
                prediction = model(batch_sensors.to(device), batch_conditions.to(device))
                validation_losses.append(
                    float(reconstruction_loss(prediction, batch_sensors.to(device)).cpu())
                )
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_validation - 1e-7:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def score_frame(
    frame: pd.DataFrame,
    split: str,
    model: ConditionalLSTMAutoencoder,
    sensor_scaler: MinMaxScaler,
    condition_scaler: MinMaxScaler,
    window_size: int,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    model.eval()
    for unit_id, unit in frame.groupby("unit_id", sort=True):
        unit = unit.sort_values("cycle").reset_index(drop=True)
        sensor_values = sensor_scaler.transform(unit[list(SENSOR_COLUMNS)].to_numpy())
        condition_values = condition_scaler.transform(unit[list(OPERATION_COLUMNS)].to_numpy())
        if len(unit) < window_size:
            # Keep very short test engines in the output. The model still
            # receives its fixed-length input; the missing history is padded
            # with the first observed (healthy-looking) row, and only the
            # terminal cycle is reported for this exceptional case.
            pad = window_size - len(unit)
            sensor_windows = np.concatenate(
                [np.repeat(sensor_values[:1], pad, axis=0), sensor_values], axis=0
            )[None, ...].astype(np.float32)
            condition_windows = np.concatenate(
                [np.repeat(condition_values[:1], pad, axis=0), condition_values], axis=0
            )[None, ...].astype(np.float32)
            scored_rows = unit.iloc[[-1]].copy()
            window_starts = np.asarray([unit.iloc[0]["cycle"]])
        else:
            sensor_windows = np.stack(
                [sensor_values[start : start + window_size] for start in range(len(unit) - window_size + 1)]
            ).astype(np.float32)
            condition_windows = np.stack(
                [condition_values[start : start + window_size] for start in range(len(unit) - window_size + 1)]
            ).astype(np.float32)
            scored_rows = unit.iloc[window_size - 1 :].copy()
            window_starts = unit.iloc[: len(sensor_windows)]["cycle"].to_numpy()

        scores: list[np.ndarray] = []
        for start in range(0, len(sensor_windows), batch_size):
            sensor_batch = torch.from_numpy(sensor_windows[start : start + batch_size]).to(device)
            condition_batch = torch.from_numpy(condition_windows[start : start + batch_size]).to(device)
            with torch.no_grad():
                reconstructed = model(sensor_batch, condition_batch)
            # The final timestep is the current cycle of a rolling window.
            error = torch.abs(reconstructed[:, -1, :] - sensor_batch[:, -1, :]).cpu().numpy()
            scores.append(error)
        errors = np.concatenate(scores, axis=0)
        current_rows = scored_rows
        current_rows["split"] = split
        current_rows["hi_mae"] = errors.mean(axis=1)
        current_rows["hi_rmse"] = np.sqrt(np.square(errors).mean(axis=1))
        current_rows["window_start_cycle"] = window_starts
        current_rows["window_end_cycle"] = current_rows["cycle"].to_numpy()
        rows.append(
            current_rows[["split", "unit_id", "cycle", "window_start_cycle", "window_end_cycle", "hi_mae", "hi_rmse"]]
        )
    if not rows:
        raise ValueError(f"No scorable units in {split} split")
    return pd.concat(rows, ignore_index=True)


def add_health_normalization(scores: pd.DataFrame, healthy_cycles: int) -> tuple[pd.DataFrame, dict[str, float]]:
    healthy = scores[(scores["split"] == "train") & (scores["cycle"] <= healthy_cycles)]["hi_mae"]
    baseline = float(healthy.median()) if not healthy.empty else float(scores["hi_mae"].median())
    # Calibrate the display HI using train data only. The lower anchor is the
    # healthy reconstruction level; the upper anchor is a robust high-error
    # level from the training trajectories. This gives an online-safe [0, 1]
    # scale without using test lifetimes or test maxima.
    train_errors = scores.loc[scores["split"] == "train", "hi_mae"]
    upper = float(train_errors.quantile(0.95)) if not train_errors.empty else baseline + 1.0
    if not np.isfinite(baseline) or baseline < 1e-8:
        baseline = 1e-8
    if not np.isfinite(upper) or upper <= baseline + 1e-8:
        upper = baseline + 1.0
    scores = scores.copy()
    scores["hi_relative_to_healthy"] = scores["hi_mae"] / baseline
    scores["hi_0_1"] = np.clip((scores["hi_mae"] - baseline) / (upper - baseline), 0.0, 1.0)
    scores["hi_rolling_mean"] = scores.groupby(["split", "unit_id"], sort=False)["hi_relative_to_healthy"].transform(
        lambda values: values.rolling(5, min_periods=1, center=True).mean()
    )
    scores["hi_0_1_rolling_mean"] = scores.groupby(["split", "unit_id"], sort=False)["hi_0_1"].transform(
        lambda values: values.rolling(5, min_periods=1, center=True).mean()
    )
    return scores, {
        "healthy_hi_mae_median": baseline,
        "train_hi_mae_q95": upper,
        "hi_0_1_definition": "clip((hi_mae - healthy_train_median) / (train_hi_mae_q95 - healthy_train_median), 0, 1)",
    }


def attach_test_rul(scores: pd.DataFrame, rul_path: Path, expected_test_units: int) -> pd.DataFrame:
    rul_values = [int(value.strip()) for value in rul_path.read_text().splitlines() if value.strip()]
    if len(rul_values) != expected_test_units:
        raise ValueError(f"{rul_path} has {len(rul_values)} labels, expected {expected_test_units}")
    # RUL files are ordered by the original test unit id. Some very short
    # units can have no complete rolling window and therefore are absent from
    # scores; that should not change the RUL mapping for the other units.
    rul_map = dict(enumerate(rul_values, start=1))
    scores = scores.copy()
    scores["rul"] = np.nan
    test = scores["split"] == "test"
    max_cycles = scores.loc[test].groupby("unit_id")["cycle"].transform("max")
    scored_test_units = scores.loc[test, "unit_id"].astype(int)
    if not scored_test_units.empty and int(scored_test_units.max()) > expected_test_units:
        raise ValueError("A scored test unit id exceeds the number of RUL labels")
    scores.loc[test, "rul"] = (
        max_cycles.to_numpy()
        + scored_test_units.map(rul_map).to_numpy()
        - scores.loc[test, "cycle"].to_numpy()
    )
    return scores


def plot_fd_scores(scores: pd.DataFrame, fd: str, output_path: Path, healthy_cycles: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False, constrained_layout=True)
    for axis, split, title in zip(axes, ("train", "test"), ("Training engines", "Test engines")):
        subset = scores[scores["split"] == split]
        for unit_id, unit in subset.groupby("unit_id", sort=True):
            axis.plot(
                unit["cycle"], unit["hi_0_1"], color="#2563eb" if split == "train" else "#dc2626",
                alpha=0.04, linewidth=0.45,
            )
        # Emphasize a few deterministic units so the degradation shape remains readable.
        unit_ids = sorted(subset["unit_id"].unique())
        selected = unit_ids[:: max(1, len(unit_ids) // 4)][:4]
        for unit_id in selected:
            unit = subset[subset["unit_id"] == unit_id]
            axis.plot(unit["cycle"], unit["hi_0_1_rolling_mean"], linewidth=1.6, label=f"unit {unit_id}")
        axis.axvline(healthy_cycles, color="#111827", linestyle="--", linewidth=0.9, alpha=0.7)
        axis.set_title(title)
        axis.set_ylabel("normalized HI (0–1)")
        axis.grid(alpha=0.2)
        if selected:
            axis.legend(ncol=4, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Cycle")
    fig.suptitle(f"{fd}: conditional LSTM-autoencoder health indicator (all sensors + operating conditions)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_summary_plot(all_scores: dict[str, pd.DataFrame], output_path: Path, healthy_cycles: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)
    for row, fd in enumerate(FD_NAMES):
        if fd not in all_scores:
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            continue
        scores = all_scores[fd]
        for col, split in enumerate(("train", "test")):
            axis = axes[row, col]
            subset = scores[scores["split"] == split]
            for _, unit in subset.groupby("unit_id", sort=True):
                axis.plot(unit["cycle"], unit["hi_0_1"], alpha=0.035, linewidth=0.4)
            axis.axvline(healthy_cycles, color="#111827", linestyle="--", linewidth=0.8)
            axis.set_title(f"{fd} {split}")
            axis.set_ylabel("normalized HI (0–1)")
            axis.set_xlabel("cycle")
            axis.grid(alpha=0.2)
    fig.suptitle("C-MAPSS LSTM-autoencoder health indicators")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_fd(fd: str, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    fd_output = args.output_dir / fd
    fd_output.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and (fd_output / "hi_scores.csv").exists():
        return pd.read_csv(fd_output / "hi_scores.csv")

    train_path = args.data_dir / fd / f"train_{fd}.txt"
    test_path = args.data_dir / fd / f"test_{fd}.txt"
    train = load_cmapss(train_path)
    test = load_cmapss(test_path)
    sensor_scaler, condition_scaler = fit_scalers(train, args.healthy_cycles)
    healthy_sensors, healthy_conditions = make_training_windows(
        train, sensor_scaler, condition_scaler, args.window_size, args.healthy_cycles
    )
    model, history = train_model(healthy_sensors, healthy_conditions, args, device)
    train_scores = score_frame(
        train, "train", model, sensor_scaler, condition_scaler, args.window_size, device, args.batch_size
    )
    test_scores = score_frame(
        test, "test", model, sensor_scaler, condition_scaler, args.window_size, device, args.batch_size
    )
    scores, baseline = add_health_normalization(
        pd.concat([train_scores, test_scores], ignore_index=True), args.healthy_cycles
    )
    scores = attach_test_rul(
        scores,
        args.data_dir / fd / f"RUL_{fd}.txt",
        expected_test_units=int(test["unit_id"].nunique()),
    )
    scores.insert(0, "fd", fd)
    scores.to_csv(fd_output / "hi_scores.csv", index=False)
    scores[scores["split"] == "train"].to_csv(fd_output / "hi_train.csv", index=False)
    scores[scores["split"] == "test"].to_csv(fd_output / "hi_test.csv", index=False)
    history.to_csv(fd_output / "training_history.csv", index=False)
    (fd_output / "sensor_scaler.json").write_text(json.dumps(sensor_scaler.to_json(), indent=2))
    (fd_output / "operating_condition_scaler.json").write_text(json.dumps(condition_scaler.to_json(), indent=2))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sensor_columns": list(SENSOR_COLUMNS),
            "operating_condition_columns": list(OPERATION_COLUMNS),
            "hidden_size": args.hidden_size,
            "window_size": args.window_size,
            "healthy_cycles": args.healthy_cycles,
            "best_validation_loss": float(history["validation_loss"].min()),
        },
        fd_output / "model.pt",
    )
    metadata = {
        "fd": fd,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_units": int(train["unit_id"].nunique()),
        "test_units": int(test["unit_id"].nunique()),
        "sensor_columns": list(SENSOR_COLUMNS),
        "operating_condition_columns": list(OPERATION_COLUMNS),
        "healthy_cycles": args.healthy_cycles,
        "window_size": args.window_size,
        "training_windows": int(len(healthy_sensors)),
        "device": str(device),
        "hi_definition": "mean absolute reconstruction error over all 21 min-max-normalized sensors at the current rolling-window endpoint",
        "conditioning": "all 3 operating conditions are input to encoder and decoder and are not reconstructed",
        "input_normalization": "per-feature min-max to [0, 1], fitted on train cycles 1..healthy_cycles only",
        "short_unit_policy": "if a sequence is shorter than the rolling window, left-pad its first row and report its terminal cycle",
        "healthy_baseline": baseline,
    }
    (fd_output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    plot_fd_scores(scores, fd, fd_output / "hi_curves.png", args.healthy_cycles)
    print(
        f"{fd}: windows={len(healthy_sensors)}, train_scores={len(train_scores)}, "
        f"test_scores={len(test_scores)}, best_val={history['validation_loss'].min():.6f}"
    )
    return scores


def main() -> None:
    args = parse_args()
    if args.healthy_cycles < args.window_size:
        raise ValueError("--healthy-cycles must be >= --window-size")
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_scores = {fd: run_fd(fd, args, device) for fd in args.fds}
    save_summary_plot(all_scores, args.output_dir / "all_fd_hi_curves.png", args.healthy_cycles)
    summary_rows = []
    for fd, scores in all_scores.items():
        for split, subset in scores.groupby("split"):
            summary_rows.append(
                {
                    "fd": fd,
                    "split": split,
                    "units": int(subset["unit_id"].nunique()),
                    "scored_rows": int(len(subset)),
                    "hi_0_1_median": float(subset["hi_0_1"].median()),
                    "hi_0_1_max": float(subset["hi_0_1"].max()),
                }
            )
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.csv", index=False)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


CHANNELS = ["accHorizFrontal_C", "accHorizRear_A"]
CHANNEL_LABELS = {
    "accHorizFrontal_C": "accHorizFrontal_C",
    "accHorizRear_A": "accHorizRear_A",
}


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#344054")
    ax.xaxis.label.set_color("#344054")
    ax.yaxis.label.set_color("#344054")


def read_mat(path: Path) -> dict[str, np.ndarray]:
    raw = loadmat(path, squeeze_me=True)
    return {key: np.asarray(raw[key]).reshape(-1) for key in CHANNELS + ["measTime"]}


def main(experiment: str) -> None:
    if not re.fullmatch(r"B\d{2}", experiment):
        raise ValueError(f"Experiment must look like B01, B02, ...; got {experiment!r}")
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "dataset" / "PHME24" / experiment
    vib_dir = data_dir / "vibrationData"
    out_dir = root / "outputs" / "PHME24-figure" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_re = re.compile(rf"data_{re.escape(experiment)}_M(\d{{4}})\.mat$")

    temp = pd.read_csv(data_dir / f"{experiment}_meanTemperatures.csv")
    cond = pd.read_csv(data_dir / f"{experiment}_operatingConditions.csv")
    mat_files = sorted(vib_dir.glob(f"data_{experiment}_M*.mat"))

    if len(temp) != len(cond) or len(temp) != len(mat_files):
        raise ValueError(
            f"Measurement count mismatch: temperature={len(temp)}, "
            f"conditions={len(cond)}, mat={len(mat_files)}"
        )

    rows: list[dict[str, float | int | str]] = []
    for idx, path in enumerate(mat_files):
        match = mat_re.search(path.name)
        if not match:
            raise ValueError(f"Unexpected MAT filename: {path.name}")
        measurement = int(match.group(1))
        data = read_mat(path)
        row: dict[str, float | int | str] = {
            "measurement": measurement,
            "mat_file": path.name,
            "time": temp.iloc[idx]["Time"],
            "n_samples": len(data["measTime"]),
            "sampling_duration_s": float(data["measTime"][-1] - data["measTime"][0]),
        }
        for channel in CHANNELS:
            signal_v = data[channel]
            signal_g = signal_v * 10.0
            row[f"{channel}_mean_V"] = float(np.mean(signal_v))
            row[f"{channel}_std_V"] = float(np.std(signal_v))
            row[f"{channel}_rms_V"] = float(np.sqrt(np.mean(signal_v**2)))
            row[f"{channel}_rms_g"] = float(np.sqrt(np.mean(signal_g**2)))
            row[f"{channel}_peak_abs_V"] = float(np.max(np.abs(signal_v)))
        for column in cond.columns[1:]:
            row[column] = float(cond.iloc[idx][column])
        for column in temp.columns[1:]:
            row[column] = float(temp.iloc[idx][column])
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{experiment}_measurement_sensor_summary.csv", index=False)

    # Raw waveform view: all samples from three representative measurements.
    selected = [0, len(mat_files) // 2, len(mat_files) - 1]
    colors = {CHANNELS[0]: "#2457a6", CHANNELS[1]: "#c26a00"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=False, sharey="row")
    fig.suptitle(
        f"PHME24 {experiment} vibration sensor readings: representative measurements",
        fontsize=16,
        color="#1d2939",
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        "Each panel uses the complete 204,800-point waveform over approximately 1.6 s; values are stored in V",
        ha="center",
        fontsize=10,
        color="#667085",
    )
    for col, idx in enumerate(selected):
        data = read_mat(mat_files[idx])
        m = int(summary.iloc[idx]["measurement"])
        for row_idx, channel in enumerate(CHANNELS):
            ax = axes[row_idx, col]
            ax.plot(
                data["measTime"],
                data[channel],
                color=colors[channel],
                linewidth=0.35,
                rasterized=True,
            )
            ax.set_title(f"M{m:04d}  |  {channel}", fontsize=10, color="#1d2939")
            ax.set_xlabel("Measurement time [s]")
            ax.set_ylabel("Acceleration [V]")
            style_axes(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_dir / f"{experiment}_raw_sensor_readings_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Measurement-level sensor trend: RMS is the most compact honest overview of
    # 77 million raw samples, while retaining the raw waveform figure above.
    x = summary["measurement"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"PHME24 {experiment} sensor and experiment overview", fontsize=16, color="#1d2939", y=0.98)
    fig.text(
        0.5,
        0.945,
        f"{len(mat_files)} measurements; one point per MAT file. Vibration RMS is converted from stored V using 10 g/V.",
        ha="center",
        fontsize=10,
        color="#667085",
    )
    axes[0].plot(x, summary[f"{CHANNELS[0]}_rms_g"], color=colors[CHANNELS[0]], label=CHANNEL_LABELS[CHANNELS[0]], linewidth=1.4)
    axes[0].plot(x, summary[f"{CHANNELS[1]}_rms_g"], color=colors[CHANNELS[1]], label=CHANNEL_LABELS[CHANNELS[1]], linewidth=1.4)
    axes[0].set_ylabel("Vibration RMS [g]")
    axes[0].set_title("Measurement-level vibration intensity", loc="left", fontsize=11, color="#344054")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    style_axes(axes[0])

    axes[1].plot(x, cond["meanAbs_speed / rpm"], color="#7a3e9d", linewidth=1.2, label="meanAbs_speed")
    axes[1].plot(x, cond["meanAbs_statLoad / N"], color="#2d7d6f", linewidth=1.2, label="meanAbs_statLoad")
    axes[1].set_ylabel("Speed [rpm] / Load [N]")
    axes[1].set_title("Operating-condition measurements", loc="left", fontsize=11, color="#344054")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    style_axes(axes[1])

    axes[2].plot(x, temp["Mean Abs. Temp. T1 / °C"], color="#b54708", linewidth=1.2, label="T1")
    axes[2].plot(x, temp["Mean Abs. Temp. T2 / °C"], color="#d92d20", linewidth=1.2, label="T2")
    axes[2].plot(x, temp["Mean Room Temp. / °C"], color="#667085", linewidth=1.2, label="Room")
    axes[2].set_ylabel("Temperature [°C]")
    axes[2].set_xlabel("Measurement identifier")
    axes[2].set_title("Mean temperature measurements", loc="left", fontsize=11, color="#344054")
    axes[2].legend(frameon=False, ncol=3, loc="upper left")
    style_axes(axes[2])
    axes[2].set_xlim(x.min(), x.max())
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_dir / f"{experiment}_sensor_readings_over_measurements.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "dataset": "PHME24",
        "experiment": experiment,
        "measurement_count": len(mat_files),
        "mat_pattern": f"data_{experiment}_MXXXX.mat",
        "mat_variables": CHANNELS + ["measTime"],
        "vibration_channels": {
            channel: {
                "samples_per_measurement": 204800,
                "sampling_frequency_hz": 128000,
                "duration_s": 1.6,
                "stored_unit": "V",
                "conversion": "10 g/V",
            }
            for channel in CHANNELS
        },
        "csv_files": {
            f"{experiment}_meanTemperatures.csv": list(temp.columns),
            f"{experiment}_operatingConditions.csv": list(cond.columns),
        },
        "plots": [
            f"{experiment}_raw_sensor_readings_examples.png",
            f"{experiment}_sensor_readings_over_measurements.png",
        ],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(out_dir), "measurement_count": len(mat_files), "summary_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="B01")
    args = parser.parse_args()
    main(args.experiment)

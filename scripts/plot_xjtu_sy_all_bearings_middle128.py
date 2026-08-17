#!/usr/bin/env python3
"""Plot the middle N raw samples from every CSV for every XJTU-SY bearing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SENSORS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]
SENSOR_LABELS = {
    "Horizontal_vibration_signals": "Horizontal vibration",
    "Vertical_vibration_signals": "Vertical vibration",
}
COLORS = ["#2f5d8c", "#b45f06"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--center-points", type=int, default=128)
    parser.add_argument("--reference-minutes", type=int, default=20)
    return parser.parse_args()


def numeric_csvs(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("*.csv"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def compute_lhi_from_rms(
    rms_values: np.ndarray,
    reference_count: int,
    epsilon: float = 1e-6,
    range_epsilon: float = 1e-12,
) -> dict[str, np.ndarray | float | list[int]]:
    """Compute LHI from one RMS value per sensor and measurement."""
    if len(rms_values) <= reference_count:
        raise ValueError("At least one measurement after the health reference is required.")
    reference = rms_values[:reference_count]
    reference_mean = reference.mean(axis=0)
    reference_range = reference.max(axis=0) - reference.min(axis=0)
    usable = np.isfinite(reference_range) & (reference_range > range_epsilon)
    if not np.any(usable):
        raise ValueError("No usable sensor range in the first 20-minute reference.")
    normalized = np.abs(
        (rms_values[:, usable] - reference_mean[usable]) / reference_range[usable]
    )
    d_mae = normalized.mean(axis=1)
    d_rmse = np.sqrt(np.mean(normalized ** 2, axis=1))
    reference_normalized = normalized[:reference_count]
    baseline_mae = float(reference_normalized.mean())
    baseline_rmse = float(np.sqrt(np.mean(reference_normalized ** 2)))
    return {
        "d_mae": d_mae,
        "d_rmse": d_rmse,
        "lhi_mae": np.log((d_mae + epsilon) / (baseline_mae + epsilon)),
        "lhi_rmse": np.log((d_rmse + epsilon) / (baseline_rmse + epsilon)),
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "usable_sensor_indices": np.flatnonzero(usable).tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.center_points <= 0:
        raise ValueError("center-points must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    bearing_dirs = sorted(
        (path for condition in args.input_root.iterdir() if condition.is_dir()
         for path in condition.iterdir() if path.is_dir()),
        key=lambda path: (path.parent.name, path.name),
    )
    if not bearing_dirs:
        raise FileNotFoundError(f"No bearing directories found under {args.input_root}")

    summary_rows: list[dict[str, int | str]] = []
    for bearing_dir in bearing_dirs:
        condition = bearing_dir.parent.name
        bearing = bearing_dir.name
        name = f"{condition}_{bearing}"
        output_dir = args.output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)

        files = numeric_csvs(bearing_dir)
        if not files:
            continue
        selected_arrays: list[np.ndarray] = []
        full_length: int | None = None
        summary: list[dict[str, int | float | str]] = []

        for measurement, path in enumerate(files, start=1):
            frame = pd.read_csv(path)
            missing = [sensor for sensor in SENSORS if sensor not in frame.columns]
            if missing:
                raise ValueError(f"{path}: missing columns {missing}")
            values = frame[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite sensor values found")
            if full_length is None:
                full_length = len(values)
            if len(values) != full_length:
                raise ValueError(f"{path}: length {len(values)} differs from {full_length}")
            if args.center_points > len(values):
                raise ValueError(f"{path}: center-points exceeds file length")

            start = (len(values) - args.center_points) // 2
            stop = start + args.center_points
            selected = values[start:stop]
            selected_arrays.append(selected)
            summary.append({
                "measurement": measurement,
                "file": path.name,
                "full_samples": len(values),
                "selected_start_index": start,
                "selected_stop_index_exclusive": stop,
                "horizontal_full_rms": float(np.sqrt(np.mean(values[:, 0] ** 2))),
                "vertical_full_rms": float(np.sqrt(np.mean(values[:, 1] ** 2))),
                "combined_full_rms": float(np.sqrt(np.mean(values ** 2))),
                "horizontal_mean": float(selected[:, 0].mean()),
                "horizontal_std": float(selected[:, 0].std()),
                "horizontal_rms": float(np.sqrt(np.mean(selected[:, 0] ** 2))),
                "vertical_mean": float(selected[:, 1].mean()),
                "vertical_std": float(selected[:, 1].std()),
                "vertical_rms": float(np.sqrt(np.mean(selected[:, 1] ** 2))),
                "combined_rms": float(np.sqrt(np.mean(selected ** 2))),
            })

        selected_matrix = np.stack(selected_arrays, axis=0)
        summary_frame = pd.DataFrame(summary)
        prefix = f"{name}_middle{args.center_points}_all{len(files)}"
        summary_frame.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
        np.savez_compressed(
            output_dir / f"{prefix}_raw_arrays.npz",
            measurements=np.arange(1, len(files) + 1),
            files=np.asarray([path.name for path in files]),
            selected_arrays=selected_matrix,
        )

        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, dpi=180)
        x_local = np.linspace(0, 1, args.center_points, endpoint=False)
        for sensor_index, sensor in enumerate(SENSORS):
            ax = axes[sensor_index]
            for measurement, selected in enumerate(selected_arrays, start=1):
                ax.plot(
                    measurement - 1 + x_local,
                    selected[:, sensor_index],
                    color=COLORS[sensor_index],
                    linewidth=0.28,
                    alpha=0.68,
                )
            ax.set_ylabel("amplitude")
            ax.set_title(SENSOR_LABELS[sensor])
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("measurement index; adjacent CSV files are sampled 1 minute apart")
        axes[-1].set_xlim(0, len(files))
        tick_step = max(1, len(files) // 10)
        axes[-1].set_xticks(np.arange(0, len(files) + 1, tick_step))
        fig.suptitle(
            f"XJTU-SY {bearing}: middle {args.center_points} raw samples from all {len(files)} CSV files ({condition})",
            fontsize=15,
        )
        fig.text(
            0.01,
            0.01,
            f"Each CSV has {full_length:,} samples; selected indices {summary_frame['selected_start_index'].iloc[0]}:"
            f"{summary_frame['selected_stop_index_exclusive'].iloc[0]} (0-based, stop exclusive).",
            fontsize=8.5,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(output_dir / f"{prefix}_raw_waveforms.png", bbox_inches="tight")
        plt.close(fig)

        # Compare RMS computed from all 32768 samples with RMS from the middle 128 samples.
        rms_specs = [
            ("horizontal", "Horizontal RMS", "horizontal_full_rms", "horizontal_rms", "#2f5d8c"),
            ("vertical", "Vertical RMS", "vertical_full_rms", "vertical_rms", "#b45f06"),
            ("combined", "Combined RMS", "combined_full_rms", "combined_rms", "#6b4c9a"),
        ]
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, dpi=180)
        for ax, (_, title, full_column, selected_column, color) in zip(axes, rms_specs):
            ax.plot(
                summary_frame["measurement"],
                summary_frame[full_column],
                color="#111827",
                linewidth=1.35,
                label="full 32768 samples",
            )
            ax.plot(
                summary_frame["measurement"],
                summary_frame[selected_column],
                color=color,
                linewidth=1.15,
                alpha=0.9,
                label=f"middle {args.center_points} samples",
            )
            ax.set_ylabel("RMS")
            ax.set_title(title)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("measurement index; adjacent CSV files are sampled 1 minute apart")
        axes[-1].set_xlim(1, len(files))
        tick_step = max(1, len(files) // 10)
        axes[-1].set_xticks(np.arange(1, len(files) + 1, tick_step))
        fig.suptitle(
            f"XJTU-SY {bearing}: RMS before and after middle-{args.center_points} sampling ({condition})",
            fontsize=15,
        )
        fig.text(
            0.01,
            0.01,
            "Full RMS uses all samples in each CSV; sampled RMS uses only the center segment.",
            fontsize=8.5,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        rms_plot = output_dir / f"{prefix}_rms_before_after_sampling.png"
        fig.savefig(rms_plot, bbox_inches="tight")
        plt.close(fig)
        summary_frame[
            [
                "measurement", "file",
                "horizontal_full_rms", "horizontal_rms",
                "vertical_full_rms", "vertical_rms",
                "combined_full_rms", "combined_rms",
            ]
        ].to_csv(output_dir / f"{prefix}_rms_comparison.csv", index=False)

        # Compute LHI separately for full-file RMS and middle-sampled RMS.
        full_rms_values = summary_frame[["horizontal_full_rms", "vertical_full_rms"]].to_numpy(dtype=float)
        sampled_rms_values = summary_frame[["horizontal_rms", "vertical_rms"]].to_numpy(dtype=float)
        full_lhi = compute_lhi_from_rms(full_rms_values, args.reference_minutes)
        sampled_lhi = compute_lhi_from_rms(sampled_rms_values, args.reference_minutes)
        summary_frame["full_d_mae"] = full_lhi["d_mae"]
        summary_frame["full_d_rmse"] = full_lhi["d_rmse"]
        summary_frame["full_lhi_mae"] = full_lhi["lhi_mae"]
        summary_frame["full_lhi_rmse"] = full_lhi["lhi_rmse"]
        summary_frame["sampled_d_mae"] = sampled_lhi["d_mae"]
        summary_frame["sampled_d_rmse"] = sampled_lhi["d_rmse"]
        summary_frame["sampled_lhi_mae"] = sampled_lhi["lhi_mae"]
        summary_frame["sampled_lhi_rmse"] = sampled_lhi["lhi_rmse"]

        fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, dpi=180)
        for ax, metric, label in [
            (axes[0], "lhi_rmse", "LHI_RMSE"),
            (axes[1], "lhi_mae", "LHI_MAE"),
        ]:
            ax.axvspan(
                1,
                args.reference_minutes,
                color="#dbeafe",
                alpha=0.8,
                label=f"health reference: first {args.reference_minutes} minutes",
            )
            ax.plot(
                summary_frame["measurement"],
                summary_frame[f"full_{metric}"],
                color="#111827",
                linewidth=1.45,
                label="full 32768 samples",
            )
            ax.plot(
                summary_frame["measurement"],
                summary_frame[f"sampled_{metric}"],
                color="#b45f06",
                linewidth=1.25,
                label=f"middle {args.center_points} samples",
            )
            ax.axhline(0.0, color="#6b7280", linewidth=0.9)
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("measurement index; adjacent CSV files are sampled 1 minute apart")
        axes[-1].set_xlim(1, len(files))
        tick_step = max(1, len(files) // 10)
        axes[-1].set_xticks(np.arange(1, len(files) + 1, tick_step))
        fig.suptitle(
            f"XJTU-SY {bearing}: LHI before and after middle-{args.center_points} sampling ({condition})",
            fontsize=15,
        )
        fig.text(
            0.01,
            0.01,
            f"Each version uses its own first-{args.reference_minutes}-minute sensor-RMS reference; "
            "LHI is log(drift/reference drift).",
            fontsize=8.5,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        lhi_plot = output_dir / f"{prefix}_lhi_before_after_sampling.png"
        fig.savefig(lhi_plot, bbox_inches="tight")
        plt.close(fig)
        summary_frame[
            [
                "measurement", "file",
                "full_d_mae", "full_d_rmse", "full_lhi_mae", "full_lhi_rmse",
                "sampled_d_mae", "sampled_d_rmse", "sampled_lhi_mae", "sampled_lhi_rmse",
            ]
        ].to_csv(output_dir / f"{prefix}_lhi_comparison.csv", index=False)

        metadata = {
            "input_dir": str(bearing_dir),
            "condition": condition,
            "bearing": bearing,
            "file_count": len(files),
            "full_samples_per_file": int(full_length),
            "selected_samples_per_file": args.center_points,
            "selection": "center segment from every CSV",
            "selected_start_index_zero_based": int(summary_frame["selected_start_index"].iloc[0]),
            "selected_stop_index_exclusive_zero_based": int(summary_frame["selected_stop_index_exclusive"].iloc[0]),
            "sensors": SENSORS,
            "sampling_interval_between_files": "1 minute",
            "rms_comparison": "full-file RMS versus RMS from the selected center segment",
            "rms_plot": str(rms_plot),
            "lhi_reference_minutes": args.reference_minutes,
            "lhi_reference": "first 20 CSV measurements, separately for full and middle-sampled RMS",
            "lhi_formula": "log((D + epsilon) / (B + epsilon))",
            "lhi_normalization": "per-sensor min-max range within the first-20-minute RMS reference",
            "full_lhi_baseline_mae": full_lhi["baseline_mae"],
            "full_lhi_baseline_rmse": full_lhi["baseline_rmse"],
            "sampled_lhi_baseline_mae": sampled_lhi["baseline_mae"],
            "sampled_lhi_baseline_rmse": sampled_lhi["baseline_rmse"],
            "lhi_plot": str(lhi_plot),
        }
        (output_dir / f"{prefix}_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary_rows.append({
            "condition": condition,
            "bearing": bearing,
            "file_count": len(files),
            "full_samples_per_file": int(full_length),
            "selected_samples_per_file": args.center_points,
            "selected_start_index": int(summary_frame["selected_start_index"].iloc[0]),
            "selected_stop_index_exclusive": int(summary_frame["selected_stop_index_exclusive"].iloc[0]),
            "full_lhi_rmse_last": float(summary_frame["full_lhi_rmse"].iloc[-1]),
            "sampled_lhi_rmse_last": float(summary_frame["sampled_lhi_rmse"].iloc[-1]),
            "full_lhi_rmse_mean_after_reference": float(summary_frame["full_lhi_rmse"].iloc[args.reference_minutes:].mean()),
            "sampled_lhi_rmse_mean_after_reference": float(summary_frame["sampled_lhi_rmse"].iloc[args.reference_minutes:].mean()),
            "plot": str(output_dir / f"{prefix}_raw_waveforms.png"),
            "rms_plot": str(rms_plot),
            "lhi_plot": str(lhi_plot),
            "lhi_comparison": str(output_dir / f"{prefix}_lhi_comparison.csv"),
        })
        print(f"[{name}] files={len(files)}, selected indices={summary_rows[-1]['selected_start_index']}:{summary_rows[-1]['selected_stop_index_exclusive']}", flush=True)

    pd.DataFrame(summary_rows).to_csv(args.output_root / "all_bearings_middle128_summary.csv", index=False)
    print(f"Saved summary to: {args.output_root / 'all_bearings_middle128_summary.csv'}")


if __name__ == "__main__":
    main()

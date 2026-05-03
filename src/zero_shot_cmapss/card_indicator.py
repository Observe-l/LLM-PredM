from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FD_NAMES = ("FD001", "FD002", "FD003", "FD004")
SETTING_COLUMNS = ["setting1", "setting2", "setting3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = ["unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CARD health indicators from Chronos-2 C-MAPSS forecasts.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/CMAPSSData"))
    parser.add_argument("--forecast_dir", type=Path, default=Path("outputs/stride_5_robust"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--n_clusters", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--regime_reference",
        choices=["distribution", "sequence"],
        default="distribution",
        help=(
            "How to define r_t^H. distribution uses counts of KMeans classes over the horizon "
            "and is less sparse for FD002/FD004; sequence requires the exact horizon label sequence."
        ),
    )
    parser.add_argument(
        "--min_reference",
        type=int,
        default=3,
        help="Minimum same-unit, same-regime historical windows required to compute CARD.",
    )
    parser.add_argument(
        "--reference_strategy",
        choices=["knn", "exact"],
        default="knn",
        help="Reference set construction. knn uses nearest historical horizon regime distributions; exact requires equal regime key.",
    )
    parser.add_argument(
        "--knn_reference",
        type=int,
        default=30,
        help="Maximum number of historical windows used by --reference_strategy knn.",
    )
    parser.add_argument(
        "--plot_units",
        nargs="*",
        default=None,
        help="Optional unit-level CARD plots, e.g. FD001:4 FD004:3. If omitted, plots first units per FD.",
    )
    parser.add_argument("--plot_examples", type=int, default=3)
    return parser.parse_args()


def load_cmapss_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


def train_regime_classifier(data_dir: Path, n_clusters: int) -> Pipeline:
    fd002_train = load_cmapss_file(data_dir / "FD002" / "train_FD002.txt")
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=50)),
        ]
    )
    classifier.fit(fd002_train[SETTING_COLUMNS])
    return classifier


def add_regime_labels(data_dir: Path, classifier: Pipeline) -> Dict[str, pd.DataFrame]:
    labeled: Dict[str, pd.DataFrame] = {}
    for fd_name in FD_NAMES:
        train_df = load_cmapss_file(data_dir / fd_name / f"train_{fd_name}.txt")
        train_df = train_df.copy()
        train_df["regime"] = classifier.predict(train_df[SETTING_COLUMNS]).astype(int)
        labeled[fd_name] = train_df
    return labeled


def summarize_regime_counts(labeled: Dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    rows = []
    for fd_name, frame in labeled.items():
        counts = frame["regime"].value_counts().sort_index()
        rows.append(
            {
                "fd": fd_name,
                "num_classes": int(counts.size),
                "classes": " ".join(str(int(x)) for x in counts.index),
                "counts": json.dumps({str(int(k)): int(v) for k, v in counts.items()}, sort_keys=True),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "regime_class_counts.csv", index=False)
    return summary


def build_regime_lookup(labeled: Dict[str, pd.DataFrame]) -> Dict[Tuple[str, int], Dict[int, int]]:
    lookup: Dict[Tuple[str, int], Dict[int, int]] = {}
    for fd_name, frame in labeled.items():
        for unit_id, unit_df in frame.groupby("unit_id", sort=True):
            lookup[(fd_name, int(unit_id))] = {
                int(row.cycle): int(row.regime)
                for row in unit_df[["cycle", "regime"]].itertuples(index=False)
            }
    return lookup


def parse_plot_units(items: Sequence[str] | None) -> List[Tuple[str, int]]:
    if not items:
        return []
    parsed: List[Tuple[str, int]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"--plot_units entries must look like FD004:3, got {item!r}")
        fd_name, unit_text = item.split(":", 1)
        if fd_name not in FD_NAMES:
            raise ValueError(f"Unknown FD in --plot_units: {fd_name!r}")
        parsed.append((fd_name, int(unit_text)))
    return parsed


def slope(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    x = np.arange(len(values), dtype=np.float32)
    x = x - float(x.mean())
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0
    y = values.astype(np.float32) - float(np.mean(values))
    return float(np.sum(x * y) / denom)


def format_regime_key(regime_sequence: Sequence[int], n_clusters: int, regime_reference: str) -> str:
    if regime_reference == "sequence":
        return "|".join(str(int(x)) for x in regime_sequence)
    if regime_reference == "distribution":
        counts = np.bincount(np.asarray(regime_sequence, dtype=np.int64), minlength=n_clusters)
        return "|".join(str(int(x)) for x in counts)
    raise ValueError(f"Unsupported regime_reference: {regime_reference}")


def extract_window_features(
    window_forecasts: pd.DataFrame,
    regime_lookup: Dict[Tuple[str, int], Dict[int, int]],
    gamma: float,
    n_clusters: int,
    regime_reference: str,
) -> pd.DataFrame:
    rows = []
    group_cols = ["covariate_mode", "fd", "unit_id", "cutoff_cycle", "forecast_start_cycle", "sensor"]
    for key, group in window_forecasts.groupby(group_cols, sort=True):
        covariate_mode, fd_name, unit_id, cutoff_cycle, forecast_start_cycle, sensor = key
        group = group.sort_values("horizon")
        pred = group["y_pred"].to_numpy(dtype=np.float32)
        cycle_values = group["cycle"].to_numpy(dtype=np.int64)
        unit_regimes = regime_lookup.get((str(fd_name), int(unit_id)), {})
        horizon_regime = tuple(unit_regimes.get(int(cycle), -1) for cycle in cycle_values)
        if any(x < 0 for x in horizon_regime):
            continue
        regime_counts = np.bincount(np.asarray(horizon_regime, dtype=np.int64), minlength=n_clusters).astype(np.float32)
        regime_distribution = regime_counts / max(float(np.sum(regime_counts)), 1.0)
        horizon_regime_key = format_regime_key(horizon_regime, n_clusters, regime_reference)
        row = {
            "covariate_mode": covariate_mode,
            "fd": fd_name,
            "unit_id": int(unit_id),
            "cutoff_cycle": int(cutoff_cycle),
            "forecast_start_cycle": int(forecast_start_cycle),
            "sensor": sensor,
            "horizon_regime": horizon_regime_key,
            "horizon_regime_sequence": "|".join(str(x) for x in horizon_regime),
            "z_mean": float(np.mean(pred)),
            "z_slope": float(gamma * slope(pred)),
        }
        for idx, value in enumerate(regime_distribution):
            row[f"regime_p_{idx}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def select_reference_set(
    candidates: pd.DataFrame,
    row,
    reference_strategy: str,
    knn_reference: int,
    regime_cols: Sequence[str],
) -> pd.DataFrame:
    if reference_strategy == "exact":
        return candidates[candidates["horizon_regime"] == row.horizon_regime]
    if reference_strategy == "knn":
        if candidates.empty:
            return candidates
        candidate_vectors = candidates.loc[:, regime_cols].to_numpy(dtype=np.float32)
        row_vector = np.asarray([getattr(row, col) for col in regime_cols], dtype=np.float32)
        distances = np.linalg.norm(candidate_vectors - row_vector[None, :], axis=1)
        order = np.argsort(distances)[:knn_reference]
        return candidates.iloc[order]
    raise ValueError(f"Unsupported reference_strategy: {reference_strategy}")


def compute_card_scores(
    features: pd.DataFrame,
    eps: float,
    min_reference: int,
    prediction_length: int,
    reference_strategy: str,
    knn_reference: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    card_rows = []
    detail_rows = []
    feature_cols = ["z_mean", "z_slope"]
    regime_cols = [col for col in features.columns if col.startswith("regime_p_")]
    ref_gap = int(prediction_length)

    for (mode, fd_name, unit_id), unit_df in features.groupby(["covariate_mode", "fd", "unit_id"], sort=True):
        unit_df = unit_df.sort_values(["cutoff_cycle", "sensor"]).reset_index(drop=True)
        for cutoff_cycle, window_df in unit_df.groupby("cutoff_cycle", sort=True):
            card_value = 0.0
            total_reference = 0
            usable_terms = 0
            for row in window_df.itertuples(index=False):
                candidates = unit_df[
                    (unit_df["sensor"] == row.sensor)
                    & (unit_df["cutoff_cycle"] < int(cutoff_cycle) - ref_gap)
                ]
                ref = select_reference_set(
                    candidates=candidates,
                    row=row,
                    reference_strategy=reference_strategy,
                    knn_reference=knn_reference,
                    regime_cols=regime_cols,
                )
                ref_count = int(len(ref))
                total_reference += ref_count
                sensor_card = np.nan
                if ref_count >= min_reference:
                    diffs = []
                    for feature_col in feature_cols:
                        ref_values = ref[feature_col].to_numpy(dtype=np.float32)
                        ref_median = float(np.median(ref_values))
                        ref_mad = float(np.median(np.abs(ref_values - ref_median)))
                        value = float(getattr(row, feature_col))
                        diffs.append(abs(value - ref_median) / (ref_mad + eps))
                    sensor_card = float(np.sum(diffs))
                    card_value += sensor_card
                    usable_terms += 1
                detail_rows.append(
                    {
                        "covariate_mode": mode,
                        "fd": fd_name,
                        "unit_id": int(unit_id),
                        "cutoff_cycle": int(cutoff_cycle),
                        "forecast_start_cycle": int(row.forecast_start_cycle),
                        "sensor": row.sensor,
                        "horizon_regime": row.horizon_regime,
                        "reference_count": ref_count,
                        "sensor_card": sensor_card,
                    }
                )
            card_rows.append(
                {
                    "covariate_mode": mode,
                    "fd": fd_name,
                    "unit_id": int(unit_id),
                    "cutoff_cycle": int(cutoff_cycle),
                    "forecast_start_cycle": int(window_df["forecast_start_cycle"].iloc[0]),
                    "card": float(card_value) if usable_terms > 0 else np.nan,
                    "usable_sensor_terms": int(usable_terms),
                    "total_reference_count": int(total_reference),
                    "min_reference": int(min_reference),
                }
            )

    return pd.DataFrame(card_rows), pd.DataFrame(detail_rows)


def plot_card_scores(card_scores: pd.DataFrame, output_dir: Path, plot_units: Sequence[str] | None, plot_examples: int) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    explicit_units = parse_plot_units(plot_units)
    if explicit_units:
        keys = pd.DataFrame(explicit_units, columns=["fd", "unit_id"])
    else:
        keys = (
            card_scores[["fd", "unit_id"]]
            .drop_duplicates()
            .sort_values(["fd", "unit_id"])
            .groupby("fd", as_index=False)
            .head(plot_examples)
        )

    colors = {"past_only": "#1f77b4", "known_future": "#ff7f0e", "none": "#2ca02c"}
    for _, key in keys.iterrows():
        fd_name = str(key["fd"])
        unit_id = int(key["unit_id"])
        unit_df = card_scores[(card_scores["fd"] == fd_name) & (card_scores["unit_id"] == unit_id)]
        if unit_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        for mode, mode_df in unit_df.groupby("covariate_mode", sort=True):
            mode_df = mode_df.sort_values("forecast_start_cycle")
            ax.plot(
                mode_df["forecast_start_cycle"],
                mode_df["card"],
                label=mode,
                color=colors.get(str(mode)),
                linewidth=1.8,
            )
        ax.set_title(f"CARD health indicator: {fd_name} unit {unit_id}")
        ax.set_xlabel("forecast start cycle")
        ax.set_ylabel("CARD")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{fd_name}_unit{unit_id}_card.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.forecast_dir / "card")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    run_config_path = args.forecast_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    prediction_length = int(run_config.get("prediction_length", 5))

    print("Training FD002 operating-regime KMeans classifier...", flush=True)
    classifier = train_regime_classifier(args.data_dir, args.n_clusters)
    labeled = add_regime_labels(args.data_dir, classifier)
    regime_summary = summarize_regime_counts(labeled, output_dir)
    print(regime_summary.to_string(index=False), flush=True)

    regime_lookup = build_regime_lookup(labeled)
    window_forecasts_path = args.forecast_dir / "window_forecasts.csv"
    print(f"Loading forecasts from {window_forecasts_path}...", flush=True)
    window_forecasts = pd.read_csv(window_forecasts_path)
    print(f"Extracting CARD trajectory features from {len(window_forecasts):,} rows...", flush=True)
    features = extract_window_features(
        window_forecasts=window_forecasts,
        regime_lookup=regime_lookup,
        gamma=args.gamma,
        n_clusters=args.n_clusters,
        regime_reference=args.regime_reference,
    )
    features.to_csv(output_dir / "card_features.csv", index=False)

    print("Computing condition-aware reference deviation scores...", flush=True)
    card_scores, card_sensor_details = compute_card_scores(
        features=features,
        eps=args.eps,
        min_reference=args.min_reference,
        prediction_length=prediction_length,
        reference_strategy=args.reference_strategy,
        knn_reference=args.knn_reference,
    )
    card_scores.to_csv(output_dir / "card_scores.csv", index=False)
    card_sensor_details.to_csv(output_dir / "card_sensor_details.csv", index=False)
    plot_card_scores(card_scores, output_dir, args.plot_units, args.plot_examples)
    print(f"Saved CARD outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

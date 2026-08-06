from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_ROOT = ROOT / "outputs/CMAPSS/RAG/agentic_evaluation_w10"
OUTPUT_DIR = ROOT / "outputs/paper_figure"
FDS = ("FD001", "FD002", "FD003", "FD004")
PLOT_FDS = ("FD001", "FD002", "FD003")
ROUNDS = tuple(range(1, 11))

COLORS = {
    "FD001": "#1f77b4",
    "FD002": "#ff7f0e",
    "FD003": "#2ca02c",
    "FD004": "#9467bd",
    "Early": "#e69f00",
    "Wrong Component": "#d55e00",
    "Missed Maintenance": "#0072b2",
}


def load_feedback(fd: str) -> list[dict]:
    path = EVALUATOR_ROOT / fd / "feedback_logs.json"
    rows = json.loads(path.read_text())
    by_engine = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        marker = "_Engine"
        if marker not in case_id:
            continue
        engine = int(case_id.split(marker, 1)[1].split("_", 1)[0])
        by_engine[engine] = row
    return [by_engine[index] for index in sorted(by_engine)]


def failure_class(row: dict) -> str | None:
    label = str(row.get("feedback_label", ""))
    if label in {"too_early", "over_maintenance"}:
        return "Early"
    if label.startswith("missed_"):
        if "fan" in label.lower():
            return "Wrong Component"
        return "Missed Maintenance"
    return None


def round_rows(fd: str, limit: int = 100) -> list[dict]:
    rows = load_feedback(fd)[:limit]
    output = []
    cumulative_correct = 0
    for round_id in ROUNDS:
        batch = rows[(round_id - 1) * 10 : round_id * 10]
        if len(batch) < 10:
            output.append({"round": round_id, "accuracy": np.nan, "n": len(batch)})
            continue
        correct = sum(row.get("feedback_label") == "correct_maintenance" for row in batch)
        cumulative_correct += correct
        failures = {name: 0 for name in ("Early", "Wrong Component", "Missed Maintenance")}
        for row in batch:
            category = failure_class(row)
            if category is not None:
                failures[category] += 1
        output.append(
            {
                "round": round_id,
                # Accuracy at round r is cumulative over the first 10*r engines.
                "accuracy": cumulative_correct / float(round_id * 10),
                "n": len(batch),
                **failures,
            }
        )
    return output


def style_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ROUNDS)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def plot_accuracy(rows: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for fd in FDS:
        values = [row["accuracy"] for row in rows[fd]]
        ax.plot(
            ROUNDS,
            values,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            label=fd,
            color=COLORS[fd],
        )
    style_axis(ax, "Evaluation Round", "Accuracy")
    ax.set_ylim(0.6, 1.05)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_evaluator_round.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_average_accuracy(rows: dict[str, list[dict]]) -> None:
    values = [
        float(np.nanmean([rows[fd][index]["accuracy"] for fd in PLOT_FDS]))
        for index in range(10)
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        ROUNDS,
        values,
        marker="o",
        markersize=4.8,
        linewidth=2.0,
        color="#1f4e79",
    )
    style_axis(ax, "Evaluation Round", "Average accuracy")
    ax.set_ylim(0.60, 0.90)
    ax.set_yticks(np.arange(0.60, 0.901, 0.05))
    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "average_accuracy_evaluator_round.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_failure_statistics(rows: dict[str, list[dict]]) -> None:
    categories = ("Early", "Wrong Component", "Missed Maintenance")
    totals = {
        category: [sum(rows[fd][index][category] for fd in FDS) for index in range(10)]
        for category in categories
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bottom = np.zeros(10)
    for category in categories:
        values = np.asarray(totals[category], dtype=float)
        ax.bar(
            ROUNDS,
            values,
            bottom=bottom,
            width=0.72,
            color=COLORS[category],
            label=category,
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += values
    style_axis(ax, "Evaluator round (10 engines per FD)", "Number of failure cases")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "failure_statistics_evaluator_round.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = {fd: round_rows(fd) for fd in FDS}
    plot_accuracy(rows)
    plot_average_accuracy(rows)
    plot_failure_statistics(rows)
    print("Saved evaluator figures:")
    for name in (
        "accuracy_evaluator_round.png",
        "average_accuracy_evaluator_round.png",
        "failure_statistics_evaluator_round.png",
    ):
        print(OUTPUT_DIR / name)


if __name__ == "__main__":
    main()

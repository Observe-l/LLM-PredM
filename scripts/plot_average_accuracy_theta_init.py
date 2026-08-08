from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "outputs/CMAPSS/RAG/agentic_evaluation_w10_theta_init"
OUTPUT = ROOT / "outputs/paper_figure/average_accuracy_evaluator_round.png"
THETAS = ("0p25", "0p50", "0p75", "1p0", "1p25")
LABELS = (r"Initial $\theta$=0.25", r"Initial $\theta$=0.50", r"Initial $\theta$=0.75", r"Initial $\theta$=1.0", r"Initial $\theta$=1.25")
PLOT_FDS = ("FD001", "FD002", "FD003", "FD004")
ROUNDS = tuple(range(1, 11))
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def load_feedback(theta: str, fd: str) -> list[dict]:
    path = DATA_ROOT / f"theta_{theta}" / fd / "feedback_logs.json"
    rows = json.loads(path.read_text())
    by_engine: dict[int, dict] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if "_Engine" not in case_id:
            continue
        try:
            engine = int(case_id.split("_Engine", 1)[1].split("_", 1)[0])
        except (ValueError, IndexError):
            continue
        by_engine[engine] = row
    return [by_engine[index] for index in sorted(by_engine)]


def cumulative_accuracy(theta: str, fd: str, limit: int = 100) -> list[float]:
    rows = load_feedback(theta, fd)[:limit]
    values: list[float] = []
    correct = 0
    for round_id in ROUNDS:
        batch = rows[(round_id - 1) * 10 : round_id * 10]
        if len(batch) < 10:
            values.append(float("nan"))
            continue
        correct += sum(row.get("feedback_label") == "correct_maintenance" for row in batch)
        values.append(correct / float(round_id * 10))
    return values


def main() -> None:
    series: list[list[float]] = []
    for theta in THETAS:
        fd_values = [cumulative_accuracy(theta, fd) for fd in PLOT_FDS]
        series.append([float(np.nanmean([values[i] for values in fd_values])) for i in range(10)])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, color, values in zip(LABELS, COLORS, series):
        ax.plot(ROUNDS, values, marker="o", markersize=4.5, linewidth=1.8, color=color, label=label)
    ax.set_xlabel("Evaluation Round")
    ax.set_ylabel("Maintenance decision accuracy")
    ax.set_xticks(ROUNDS)
    # The four-dataset average spans 0.55--0.93; retain headroom while
    # keeping the differences between initialization thresholds readable.
    ax.set_ylim(0.50, 0.95)
    ax.set_yticks(np.arange(0.50, 0.951, 0.05))
    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")
    for label, values in zip(LABELS, series):
        print(label, [round(v, 4) if np.isfinite(v) else None for v in values])


if __name__ == "__main__":
    main()

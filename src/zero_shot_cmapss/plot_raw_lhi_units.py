from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from .plot_lhi_units import filter_monitor_cycles, load_scores, plot_unit, validate_args
    from .plot_operating_condition_clusters import FD_NAMES
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.zero_shot_cmapss.plot_lhi_units import filter_monitor_cycles, load_scores, plot_unit, validate_args
    from src.zero_shot_cmapss.plot_operating_condition_clusters import FD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw-observed C-MAPSS LHI curves.")
    parser.add_argument("--lhi_dir", type=Path, default=Path("outputs/CMAPSS/raw_lhi"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--fd", choices=list(FD_NAMES), required=True)
    parser.add_argument("--unit_start", type=int, required=True)
    parser.add_argument("--unit_end", type=int, required=True)
    parser.add_argument("--metric", choices=["rmse", "mae"], default="rmse")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--fig_width", type=float, default=12.0)
    parser.add_argument("--fig_height", type=float, default=8.0)
    parser.add_argument("--healthy_cycles", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir or (args.lhi_dir / f"raw_{args.fd}_unit{args.unit_start}_{args.unit_end}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    scores = load_scores(
        lhi_dir=args.lhi_dir,
        fd_name=args.fd,
        unit_start=args.unit_start,
        unit_end=args.unit_end,
        modes=["raw_observed"],
    )
    scores = filter_monitor_cycles(scores, args.healthy_cycles)
    if scores.empty:
        raise ValueError(f"No raw LHI rows remain after cycle > {args.healthy_cycles}.")

    plotted = 0
    skipped = []
    for unit_id in range(args.unit_start, args.unit_end + 1):
        did_plot = plot_unit(
            scores=scores,
            fd_name=args.fd,
            unit_id=unit_id,
            metric=args.metric,
            output_dir=output_dir,
            dpi=args.dpi,
            fig_size=(args.fig_width, args.fig_height),
        )
        if did_plot:
            plotted += 1
        else:
            skipped.append(unit_id)

    print(f"Saved {plotted} raw LHI unit plots to: {output_dir}", flush=True)
    if skipped:
        print(f"Skipped units with no rows: {skipped}", flush=True)


if __name__ == "__main__":
    main()

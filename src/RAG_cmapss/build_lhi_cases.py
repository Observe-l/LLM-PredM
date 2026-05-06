from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .lhi_case_adapter import (
    build_forecast_case,
    case_peak_lhi,
    iter_lhi_windows,
    load_lhi_frames,
    load_threshold_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ForecastCase JSONL from Layer-2 C-MAPSS LHI outputs.")
    parser.add_argument("--lhi_dir", type=Path, default=Path("outputs/CMAPSS/cluster_20/lhi"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/CMAPSS/RAG/lhi_cases"))
    parser.add_argument("--fds", nargs="+", default=None)
    parser.add_argument("--score_col", default="lhi_rmse_roll_mean")
    parser.add_argument("--raw_score_col", default="d_rmse")
    parser.add_argument("--lhi_col", default="lhi_rmse_roll_mean")
    parser.add_argument("--threshold_config", type=Path, default=None)
    parser.add_argument("--lhi_trigger", type=float, default=None, help="If set, only export windows whose peak LHI exceeds this value.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_dir = args.output_dir / "forecast_windows"
    window_dir.mkdir(parents=True, exist_ok=True)
    scores, top_drift = load_lhi_frames(args.lhi_dir, load_top_drift_detail=True)
    threshold_config = load_threshold_config(args.threshold_config)

    count = 0
    cases_path = args.output_dir / "forecast_cases.jsonl"
    with cases_path.open("w") as f:
        for _key, window in iter_lhi_windows(scores, fds=args.fds):
            if args.lhi_trigger is not None:
                peak_lhi = case_peak_lhi(window, args.lhi_col)
                if not pd.notna(peak_lhi) or peak_lhi <= args.lhi_trigger:
                    continue
            case = build_forecast_case(
                window=window,
                top_drift=top_drift,
                score_col=args.score_col,
                raw_score_col=args.raw_score_col,
                lhi_col=args.lhi_col,
                threshold_config=threshold_config,
                window_detail_dir=window_dir,
            )
            f.write(json.dumps(case) + "\n")
            count += 1
            if args.limit is not None and count >= args.limit:
                break
    print(json.dumps({"forecast_cases": count, "cases_jsonl": str(cases_path), "window_dir": str(window_dir)}, indent=2))


if __name__ == "__main__":
    main()

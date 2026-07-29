from __future__ import annotations

import pandas as pd

from src.zero_shot_cmapss.plot_lhi_units import cap_at_final_cycles
from src.zero_shot_cmapss.plot_paper_lhi_comparison import align_lhi_series


def test_comparison_series_use_identical_common_cycles() -> None:
    raw, chronos, occ = align_lhi_series(
        [(51, 1.0), (52, 2.0), (53, 3.0)],
        [(52, 4.0), (53, 5.0), (54, 6.0)],
        [(50, 7.0), (52, 8.0), (53, 9.0), (55, 10.0)],
    )
    assert [cycle for cycle, _ in raw] == [52, 53]
    assert [cycle for cycle, _ in chronos] == [52, 53]
    assert [cycle for cycle, _ in occ] == [52, 53]


def test_unit_lhi_is_capped_at_dataset_final_cycle() -> None:
    scores = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2],
            "cycle": [9, 10, 11, 7, 8],
            "lhi_rmse": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    capped = cap_at_final_cycles(scores, pd.Series({1: 10, 2: 7}))
    assert list(zip(capped["unit_id"], capped["cycle"])) == [(1, 9), (1, 10), (2, 7)]

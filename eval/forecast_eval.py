"""Held-out forecast evaluation: P10-P90 coverage (target 0.80) and P50 normalised MAE, with and
without online adaptive conformal calibration.  Writes eval/results/forecast_calibration.csv"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from gridmind.config import load_portfolio  # noqa: E402
from gridmind.data import get_dataset  # noqa: E402
from gridmind.forecast import Forecaster, evaluate_coverage  # noqa: E402


def main() -> None:
    p, df = load_portfolio(), get_dataset()
    start = pd.Timestamp("2025-10-15")
    f = Forecaster.load_or_fit(p, df)
    static = evaluate_coverage(f, df, start, days=60, adaptive=False)
    f.adapt = {}
    adaptive = evaluate_coverage(f, df, start, days=60, adaptive=True)
    out = pd.DataFrame({"coverage_static": static["coverage"], "coverage_adaptive": adaptive["coverage"],
                        "p50_nmae": adaptive["nmae"]}).reset_index()
    out.to_csv(Path(__file__).resolve().parent / "results" / "forecast_calibration.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

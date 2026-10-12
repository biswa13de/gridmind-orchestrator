"""Run the Monte Carlo study and save results for the dashboard.

Usage:  python eval/montecarlo.py [--runs 100] [--hours 48] [--jobs 8]
Writes eval/results/montecarlo_runs.csv and montecarlo_summary.csv/.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridmind.montecarlo import MCConfig, run_montecarlo, summarise  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args()
    t0 = time.time()
    cfg = MCConfig(runs=a.runs, hours=a.hours, jobs=a.jobs, seed=a.seed)
    df = run_montecarlo(cfg, progress=lambda f: print(f"\r{100 * f:5.1f}%", end="", flush=True))
    print()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "montecarlo_runs.csv", index=False)
    s = summarise(df)
    s.to_csv(OUT / "montecarlo_summary.csv", index=False)
    (OUT / "montecarlo_summary.md").write_text(
        f"# Monte Carlo: {a.runs} runs x {a.hours} h x {df.strategy.nunique()} strategies "
        f"(seed {a.seed}, {time.time() - t0:.0f} s)\n\n" + s.to_markdown(index=False) + "\n")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()

"""30-day backtest on held-out data (forecaster trained on Jan-Aug 2025, never saw Oct-Nov).

Controllers: rules (greedy EMS), mpc (deterministic P50 MPC, fixed weights), gridmind (full agentic
stack), perfect (MPC with perfect knowledge of the future: the theoretical ceiling).
Regimes: calm (historical replay) and stress (the same days + seeded events and advance notices).

Usage:  python eval/backtest.py [--days 30] [--start 2025-10-15] [--seed 42] [--jobs 8]
Writes eval/results/backtest_summary.csv, backtest_summary.md, backtest_<regime>_<mode>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker process
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gridmind.llm import LLM  # noqa: E402
from gridmind.orchestrator import make_orchestrator  # noqa: E402
from gridmind.scenarios import generate_events  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
MODES = ["rules", "mpc", "gridmind", "perfect"]
LABELS = {"rules": "Rule-based EMS (baseline)", "mpc": "Deterministic MPC (ablation)",
          "gridmind": "GridMind (agentic, scenario-MPC)", "perfect": "Perfect-forecast MPC (reference)"}


def run_one(args) -> dict:
    mode, regime, start, days, seed = args
    o = make_orchestrator(start, mode=mode, seed=seed, approval_policy="approve_all", llm=LLM(mode="off"))
    docs = []
    if regime == "stress":
        events, docs = generate_events(o.p, o.twin.obs.index, o.twin.i, days, seed)
        for ev in events:
            o.twin.inject(ev)
    t0 = time.time()
    repairs = 0
    sit_counts: dict[str, int] = {}
    di = 0
    for _ in range(days * 96):
        while di < len(docs) and docs[di]["at"] <= o.twin.i:
            if mode == "gridmind":          # only the agentic controller can read documents
                o.submit_document(text=docs[di]["text"], name=docs[di]["name"])
            di += 1
        rec = o.tick()
        repairs += len(rec["repairs"])
        if rec.get("situation"):
            sit_counts[rec["situation"]] = sit_counts.get(rec["situation"], 0) + 1
    s = o.twin.summary()
    s.update({"mode": mode, "regime": regime, "validator_repairs": repairs, "runtime_s": round(time.time() - t0, 1),
              "mean_solve_ms": round(float(np.mean([r["plan"]["solve_ms"] for r in o.log if r.get("plan")] or [0])), 1),
              "situations": sit_counts, "documents_read": len(docs) if mode == "gridmind" else 0,
              "approvals": len(o.approvals.items)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"backtest_{regime}_{mode}.json").write_text(json.dumps(s, indent=2, default=float))
    pd.DataFrame(o.twin.history).to_csv(OUT / f"trace_{regime}_{mode}.csv.gz", index=False)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--start", default="2025-10-15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    jobs = [(m, r, a.start, a.days, a.seed) for r in ("calm", "stress") for m in MODES]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(run_one, jobs))
    df = pd.DataFrame(results)
    rows = []
    for regime, g in df.groupby("regime", sort=False):
        base = g.loc[g["mode"] == "rules", "total_cost_inr"].iloc[0]
        ceil = g.loc[g["mode"] == "perfect", "total_cost_inr"].iloc[0]
        for _, r in g.iterrows():
            rows.append({
                "regime": regime, "strategy": LABELS[r["mode"]],
                "total_cost_lakh_inr": round(r["total_cost_inr"] / 1e5, 1),
                "saving_vs_rules_pct": round(100 * (base - r["total_cost_inr"]) / base, 2),
                "pct_of_perfect_forecast_savings": round(100 * (base - r["total_cost_inr"]) / (base - ceil), 1) if base != ceil else None,
                "operating_cost_excl_voll_lakh": round(r["operating_cost_excl_voll_inr"] / 1e5, 1),
                "carbon_t": round(r["carbon_t"], 0),
                "curtailment_mwh": round(r["curtailment_mwh"], 1),
                "renewable_util_pct": round(r["renewable_utilisation_pct"], 2),
                "unserved_mwh": round(r["unserved_mwh"], 2),
                "violations": int(r["violations"]),
                "validator_repairs": int(r["validator_repairs"]),
                "mean_solve_ms": r["mean_solve_ms"],
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "backtest_summary.csv", index=False)
    (OUT / "backtest_summary.md").write_text(
        f"# Backtest: {a.days} held-out days from {a.start} (seed {a.seed})\n\n" + summary.to_markdown(index=False) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

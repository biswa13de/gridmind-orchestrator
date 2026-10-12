"""Monte Carlo simulator (F3): simulate the portfolio over time under many input variations and
compare operating strategies as *distributions*, not single numbers.

Each run samples (a) a start day from held-out history (real weather variability), (b) a random
event schedule (storms, outages, spikes, congestion, surges...), and (c) a forecast-error seed.
Every strategy faces the identical sampled world in a given run (common random numbers), so the
differences between strategies are attributable to the strategy alone.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")

STRATEGIES: dict[str, dict] = {
    "Cost-first":        {"mode": "policy", "profile": {"weights": dict(cost=0.70, carbon=0.05, curtailment=0.05,
                                                                       degradation=0.05, reliability=0.15), "reserve": 5}},
    "Green-first":       {"mode": "policy", "profile": {"weights": dict(cost=0.15, carbon=0.60, curtailment=0.10,
                                                                       degradation=0.05, reliability=0.10), "reserve": 10}},
    "Resilience-first":  {"mode": "policy", "profile": {"weights": dict(cost=0.20, carbon=0.10, curtailment=0.05,
                                                                       degradation=0.05, reliability=0.60), "reserve": 40}},
    "GridMind adaptive": {"mode": "gridmind"},
    "Rule-based EMS":    {"mode": "rules"},
}


@dataclass
class MCConfig:
    runs: int = 100
    hours: int = 48
    seed: int = 2025
    window_start: str = "2025-10-01"
    window_end: str = "2025-12-25"
    scenarios: int = 4
    intensity: float = 1.5            # event rate multiplier (stress-test the policies)
    jobs: int = 8


def _run(args) -> list[dict]:
    run_id, start, ev_seed, strategies, hours, K, intensity = args
    from .llm import LLM
    from .orchestrator import make_orchestrator
    from .scenarios import generate_events

    out = []
    steps = hours * 4
    for name in strategies:
        spec = STRATEGIES[name]
        o = make_orchestrator(start, mode=spec["mode"], seed=ev_seed, approval_policy="approve_all",
                              scenarios=K, llm=LLM(mode="off"), profile=spec.get("profile"))
        o.H = 64                                    # 16 h look-ahead keeps 1000s of runs tractable
        events, docs = generate_events(o.p, o.twin.obs.index, o.twin.i, int(np.ceil(hours / 24)), ev_seed, intensity)
        for ev in events:
            o.twin.inject(ev)
        di = 0
        for _ in range(steps):
            while di < len(docs) and docs[di]["at"] <= o.twin.i:
                if spec["mode"] == "gridmind":
                    o.submit_document(text=docs[di]["text"], name=docs[di]["name"])
                di += 1
            o.tick()
        s = o.twin.summary()
        out.append({"run": run_id, "strategy": name, "start": str(start), "n_events": len(events),
                    "total_cost_lakh": s["total_cost_inr"] / 1e5,
                    "operating_cost_lakh": s["operating_cost_excl_voll_inr"] / 1e5,
                    "carbon_t": s["carbon_t"], "curtailment_mwh": s["curtailment_mwh"],
                    "unserved_mwh": s["unserved_mwh"], "degradation_lakh": s["degradation_inr"] / 1e5,
                    "renewable_util_pct": s["renewable_utilisation_pct"], "violations": s["violations"]})
    return out


def run_montecarlo(cfg: MCConfig, strategies: list[str] | None = None, progress=None) -> pd.DataFrame:
    strategies = strategies or list(STRATEGIES)
    rng = np.random.default_rng(cfg.seed)
    days = pd.date_range(cfg.window_start, cfg.window_end, freq="D")
    jobs = []
    for r in range(cfg.runs):
        start = days[rng.integers(0, len(days))] + pd.Timedelta(hours=int(rng.choice([0, 6, 12, 18])))
        jobs.append((r, start, int(rng.integers(0, 1_000_000)), strategies, cfg.hours, cfg.scenarios, cfg.intensity))
    rows: list[dict] = []
    if cfg.jobs <= 1:
        for k, j in enumerate(jobs):
            rows += _run(j)
            if progress:
                progress((k + 1) / len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=cfg.jobs) as ex:
            for k, res in enumerate(ex.map(_run, jobs)):
                rows += res
                if progress:
                    progress((k + 1) / len(jobs))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy distribution summary (P5/P50/P95/mean/CVaR95 of cost) + Pareto flag."""
    rows = []
    for name, g in df.groupby("strategy", sort=False):
        c = g["total_cost_lakh"].to_numpy()
        tail = np.sort(c)[int(0.95 * len(c)):]
        rows.append({
            "strategy": name, "runs": len(g),
            "cost_mean_lakh": c.mean(), "cost_p5": np.percentile(c, 5), "cost_p50": np.percentile(c, 50),
            "cost_p95": np.percentile(c, 95), "cost_cvar95": tail.mean() if len(tail) else c.max(),
            "carbon_mean_t": g["carbon_t"].mean(), "unserved_mean_mwh": g["unserved_mwh"].mean(),
            "p_any_unserved": (g["unserved_mwh"] > 0.01).mean(), "violations": int(g["violations"].sum()),
        })
    s = pd.DataFrame(rows)
    # Pareto-efficient on (mean operating cost, mean carbon, mean unserved): no other strategy is
    # at least as good on all three and strictly better on one.
    cost = s["cost_mean_lakh"].to_numpy()
    carb = s["carbon_mean_t"].to_numpy()
    uns = s["unserved_mean_mwh"].to_numpy()
    eff = []
    for i in range(len(s)):
        dom = any((cost[j] <= cost[i] and carb[j] <= carb[i] and uns[j] <= uns[i]) and
                  (cost[j] < cost[i] or carb[j] < carb[i] or uns[j] < uns[i]) for j in range(len(s)) if j != i)
        eff.append(not dom)
    s["pareto_efficient"] = eff
    return s.round(3)

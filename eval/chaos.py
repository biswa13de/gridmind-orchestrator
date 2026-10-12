"""Chaos suite: named, repeatable fault scenarios with explicit pass criteria.

Usage:  python eval/chaos.py            -> eval/results/chaos.csv + chaos.md
Each test builds a fresh orchestrator on a held-out day, injects faults, runs the agent loop and
asserts on the decision log. Runs fully offline (deterministic agents) so results are reproducible.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gridmind.llm import LLM  # noqa: E402
from gridmind.orchestrator import make_orchestrator  # noqa: E402
from gridmind.twin import Event  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
OUT = ROOT / "eval" / "results"
DAY = "2025-10-22"


def orch(start: str, mode: str = "gridmind", policy: str = "approve_all"):
    return make_orchestrator(pd.Timestamp(start), mode=mode, approval_policy=policy, llm=LLM(mode="off"))


def _violations(o) -> int:
    return len(o.twin.violations)


# ----------------------------------------------------------------------------------- tests
def t_battery_outage_mid_discharge():
    o = orch(f"{DAY} 18:30")
    o.run(2)
    o.twin.inject(Event("battery_outage", o.twin.i, 12, ["BESS-1"]))
    recs = o.run(12)
    zeroed = all(r["setpoints"]["discharge"]["BESS-1"] == 0 and r["setpoints"]["charge"]["BESS-1"] == 0 for r in recs)
    unserved = sum(r["executed"]["unserved"] for r in recs)
    ok = zeroed and _violations(o) == 0 and unserved < 1e-6
    return ok, f"re-planned on the next tick without BESS-1; unserved={unserved:.2f} MW; violations={_violations(o)}"


def t_silent_battery_fault_recovery():
    o = orch(f"{DAY} 18:00")
    o.twin.inject(Event("battery_unresponsive", o.twin.i, 40, ["BESS-2"]))
    recs = o.run(16)
    cmd = lambda r: r["setpoints"]["charge"]["BESS-2"] + r["setpoints"]["discharge"]["BESS-2"]  # noqa: E731
    first_cmd = next((k for k, r in enumerate(recs) if cmd(r) > 0.5), None)
    detect = next((k for k, r in enumerate(recs) if r["faults_detected"]), None)
    insp = [a for a in o.approvals.items if a.action == "dispatch_inspection"]
    back = "BESS-2" not in o.known_faults
    ok = (detect is not None and first_cmd is not None and detect - first_cmd <= 1 and insp
          and insp[0].status == "approved" and back and _violations(o) == 0)
    return ok, (f"first command at tick {first_cmd}, fault detected at tick {detect}; inspection request "
                f"#{insp[0].id if insp else '-'} "
                f"({insp[0].status if insp else 'none'}); battery back in service: {back}")


def t_line_congestion():
    o = orch(f"{DAY} 11:00")
    o.twin.inject(Event("line_congestion", o.twin.i, 12, ["L-NORTH"], 35.0))
    recs = o.run(12)
    flows = [sum(v for k, v in r["setpoints"]["use"].items() if k in ("SOL-1", "SOL-2", "SOL-5", "WND-2")) for r in recs]
    ok = max(flows) <= 35.0 + 0.01 and _violations(o) == 0
    stored = sum(r["executed"]["charge"]["BESS-1"] + r["executed"]["charge"]["BESS-2"] for r in recs) * 0.25
    return ok, f"max L-NORTH flow {max(flows):.1f} MW <= 35 MW; {stored:.1f} MWh stored; violations={_violations(o)}"


def t_price_spike():
    base = orch(f"{DAY} 10:00")       # a morning spike: without it the batteries would not discharge
    rb = base.run(8)
    o = orch(f"{DAY} 10:00")
    o.twin.inject(Event("price_spike", o.twin.i, 8, [], 3.0))
    rs = o.run(8)
    d_base = sum(sum(r["executed"]["discharge"].values()) for r in rb) * 0.25
    d_spike = sum(sum(r["executed"]["discharge"].values()) for r in rs) * 0.25
    labels = {r["situation"] for r in rs}
    ok = d_spike > d_base + 5 and "price_spike" in labels and _violations(o) == 0
    return ok, f"discharged {d_spike:.1f} MWh during spike vs {d_base:.1f} MWh without; situations={sorted(labels)}"


def t_two_simultaneous_failures():
    o = orch(f"{DAY} 17:00")
    o.twin.inject(Event("battery_outage", o.twin.i, 16, ["BESS-1"]))
    o.twin.inject(Event("battery_unresponsive", o.twin.i, 16, ["BESS-2"]))
    o.twin.inject(Event("demand_surge", o.twin.i, 16, [], 20.0))
    recs = o.run(16)
    labels = [r["situation"] for r in recs]
    alerts = [a for a in o.approvals.items if a.action == "dispatch_inspection"]
    ok = _violations(o) == 0 and "asset_failure" in labels and len(alerts) > 0
    return ok, (f"no crash; violations={_violations(o)}; situation={labels[-1]}; human alert raised: {bool(alerts)}; "
                f"unserved={sum(r['executed']['unserved'] for r in recs) * 0.25:.2f} MWh")


def t_sensor_dropout():
    o = orch(f"{DAY} 09:00")
    o.twin.inject(Event("sensor_dropout", o.twin.i + 1, 4, ["price", "demand"]))
    recs = o.run(6)
    flagged = [r["sensor_fallback"] for r in recs if r["sensor_fallback"]]
    ok = len(flagged) == 4 and _violations(o) == 0
    return ok, f"{len(flagged)} ticks served from last-known-good values {flagged[0] if flagged else ''}; no crash"


def t_stale_forecast():
    o = orch(f"{DAY} 06:00")
    o.twin.inject(Event("forecast_stale", o.twin.i, 24))
    r = o.run(2)[-1]
    res = r["strategy"]["reserves"]
    ok = r["situation"] == "high_uncertainty" and min(res.values()) >= 25
    return ok, f"situation={r['situation']}; reserves raised to {res}"


def t_malformed_llm_output():
    o = orch(f"{DAY} 10:00")
    o.chaos_bad_llm = True
    r = o.tick()
    s = r["strategy"]
    ok = s["source"] == "rules (fallback)" and len(s["rejections"]) == 2 and abs(sum(s["weights"].values()) - 1) < 1e-6
    return ok, f"rejected twice ({s['rejections'][0][:70]}...), fell back to rules profile"


def t_injection_in_pdf():
    clean, dirty = orch(f"{DAY} 13:00"), orch(f"{DAY} 13:00")
    rc = clean.submit_document(pdf_path=SAMPLES / "imd_storm_bulletin.pdf")
    rd = dirty.submit_document(pdf_path=SAMPLES / "imd_storm_bulletin_injected.pdf")
    a, b = clean.run(4), dirty.run(4)
    same_strategy = all(x["strategy"]["weights"] == y["strategy"]["weights"] and
                        x["strategy"]["reserves"] == y["strategy"]["reserves"] for x, y in zip(a, b))
    same_dispatch = all(abs(x["setpoints"]["buy"] - y["setpoints"]["buy"]) < 1e-6 for x, y in zip(a, b))
    ok = bool(rd.injection_flags) and same_strategy and same_dispatch and len(rc.constraints) == len(rd.constraints)
    return ok, (f"{len(rd.injection_flags)} injection patterns flagged; constraints identical "
                f"({len(rd.constraints)}); strategy and dispatch unchanged: {same_strategy and same_dispatch}")


def t_storm_with_advance_warning():
    """The value of perception: GridMind reads the bulletin and pre-charges; plain MPC cannot."""
    start = f"{DAY} 12:00"
    res = {}
    for mode in ("gridmind", "mpc"):
        o = orch(start, mode)
        s = o.twin.i + 20                                     # storm at 17:00
        o.twin.inject(Event("storm", s, 20, ["WND-1", "WND-2", "WND-3", "SOL-2"]))
        o.twin.inject(Event("demand_surge", s, 20, [], 25.0))
        if mode == "gridmind":
            o.submit_document(pdf_path=SAMPLES / "imd_storm_bulletin.pdf")
        recs = o.run(40)
        soc_at_storm = sum(recs[19]["executed"]["soc"].values())
        res[mode] = (soc_at_storm, o.twin.summary())
    g, m = res["gridmind"], res["mpc"]
    ok = g[1]["unserved_mwh"] < m[1]["unserved_mwh"] and g[1]["total_cost_inr"] < m[1]["total_cost_inr"]
    return ok, (f"SOC at storm onset: GridMind {g[0]:.0f} MWh vs MPC {m[0]:.0f} MWh; unserved "
                f"{g[1]['unserved_mwh']:.2f} vs {m[1]['unserved_mwh']:.2f} MWh; cost "
                f"Rs {g[1]['total_cost_inr'] / 1e5:.1f} vs {m[1]['total_cost_inr'] / 1e5:.1f} lakh")


def t_demand_surge():
    o = orch(f"{DAY} 19:00")
    o.twin.inject(Event("demand_surge", o.twin.i, 8, [], 30.0))
    recs = o.run(8)
    unserved = sum(r["executed"]["unserved"] for r in recs) * 0.25
    ok = _violations(o) == 0
    return ok, f"+30 MW absorbed; unserved {unserved:.2f} MWh; violations={_violations(o)}"


def t_maintenance_planning_and_delay():
    o = orch(f"{DAY} 13:00", policy="manual")
    o.submit_document(pdf_path=SAMPLES / "maintenance_work_order.pdf")
    sched = [a for a in o.approvals.items if a.action == "schedule_maintenance"]
    if not sched:
        return False, "no maintenance proposal"
    o.approvals.decide(sched[0].id, True)
    o._apply_approvals(o.twin.i)
    m0 = dict(o.maintenance[0])
    o.submit_document(text="IMD NOWCAST WARNING. A severe thunderstorm is likely over Gujarat for the next 30 hours.")
    o.tick()
    delay = [a for a in o.approvals.items if a.action == "delay_maintenance"]
    if delay:
        o.approvals.decide(delay[0].id, True)
        o.tick()
    moved = o.maintenance[0]["start_idx"] > m0["start_idx"]
    when = o.twin.obs.index[m0["start_idx"]]
    ok = bool(delay) and moved
    return ok, (f"proposed WND-2 window {when:%d %b %H:%M} (lost value Rs {sched[0].params['lost_value_inr']:,}); "
                f"storm overlap -> delay requested and approved: {moved}")


TESTS = [t_battery_outage_mid_discharge, t_silent_battery_fault_recovery, t_line_congestion, t_price_spike,
         t_two_simultaneous_failures, t_sensor_dropout, t_stale_forecast, t_malformed_llm_output,
         t_injection_in_pdf, t_storm_with_advance_warning, t_demand_surge, t_maintenance_planning_and_delay]


def main() -> int:
    rows = []
    for t in TESTS:
        t0 = time.time()
        try:
            ok, detail = t()
        except Exception as e:  # a crash is a failure, never hidden
            ok, detail = False, f"CRASH {type(e).__name__}: {e}"
        name = t.__name__[2:]
        rows.append({"test": name, "result": "PASS" if ok else "FAIL", "detail": detail,
                     "seconds": round(time.time() - t0, 1)})
        print(f"{'PASS' if ok else 'FAIL'}  {name:34s} {detail}")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "chaos.csv", index=False)
    (OUT / "chaos.md").write_text(f"# Chaos suite: {sum(df.result == 'PASS')}/{len(df)} passed\n\n"
                                  + df.to_markdown(index=False) + "\n")
    print(f"\n{sum(df.result == 'PASS')}/{len(df)} passed")
    return 0 if all(df.result == "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())

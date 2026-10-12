"""The demo storyline, run headlessly and reproducibly (also produces logs/sample_demo_day.jsonl).

22 Oct 2025, 10:00 IST -> 23:00 IST:
  10:00  normal operation
  11:00  cloud bank over three solar parks
  13:00  IMD storm bulletin (PDF) for Gujarat 17:00-22:00 arrives -> perception -> storm strategy
         demand response request -> operator approves
  14:00  O&M work order for WND-2 -> agent proposes the cheapest maintenance window -> approved
  16:45  BESS-2 silently stops following setpoints
  17:00  the storm hits (wind trips, solar collapses, prices +60%, demand +20 MW); the fault shows
         up as a setpoint deviation -> BESS-2 isolated, crew requested -> approved -> repaired ~2 h later
  19:30  price spike x2

Usage:  python scripts/demo_story.py [--llm off|auto|replay]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gridmind.llm import LLM  # noqa: E402
from gridmind.orchestrator import make_orchestrator  # noqa: E402
from gridmind.twin import Event  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="off")
    a = ap.parse_args()
    log_path = ROOT / "logs" / "sample_demo_day.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    log_path.unlink(missing_ok=True)
    o = make_orchestrator("2025-10-22 10:00", mode="gridmind", approval_policy="manual", llm=LLM(mode=a.llm),
                          log_path=log_path)
    s = ROOT / "data" / "samples"

    def show(r):
        st = r["strategy"]
        w = st["weights"]
        print(f"{r['t'][11:16]}  {r['situation']:<17} rel={w['reliability']:.2f} cost={w['cost']:.2f} "
              f"res={max(st['reserves'].values()):>4.0f}%  | " + " · ".join(r["actions"][:4]))

    def run(n):
        for _ in range(n):
            show(o.tick())

    def approve_all():
        for r in o.approvals.pending():
            print(f"        OPERATOR APPROVES #{r.id}: {r.detail}")
            o.approvals.decide(r.id, True)
        o._apply_approvals(o.twin.i)

    run(4)                                                                   # 10:00-11:00
    print("        EVENT: cloud bank over SOL-1, SOL-2, SOL-3 (25% output) for 2 h")
    o.twin.inject(Event("cloud_bank", o.twin.i, 8, ["SOL-1", "SOL-2", "SOL-3"], 0.25))
    run(8)                                                                   # 11:00-13:00
    res = o.submit_document(pdf_path=s / "imd_storm_bulletin.pdf")
    print(f"        INBOX: IMD storm bulletin -> {[(c['kind'], c['assets'], c['start'][11:16], c['end'][11:16]) for c in res.constraints]}")
    run(1)
    approve_all()
    run(3)                                                                   # -> 14:00
    res = o.submit_document(pdf_path=s / "maintenance_work_order.pdf")
    print(f"        INBOX: work order -> {[(c['kind'], c['assets'], c['value']) for c in res.constraints]}")
    approve_all()
    run(11)                                                                  # -> 16:45
    print("        EVENT (hidden from agent): BESS-2 stops following setpoints")
    o.twin.inject(Event("battery_unresponsive", o.twin.i, 40, ["BESS-2"]))
    run(1)                                                                   # -> 17:00
    print("        EVENT: storm hits Gujarat + Tamil Nadu wind (17:00-22:00), demand +20 MW")
    o.twin.inject(Event("storm", o.twin.i, 20, ["WND-1", "WND-2", "SOL-2"]))
    o.twin.inject(Event("demand_surge", o.twin.i, 20, [], 20.0))
    run(3)
    approve_all()                                                            # crew dispatch
    run(6)                                                                   # -> 19:30
    print("        EVENT: price spike x2 for 1 h")
    o.twin.inject(Event("price_spike", o.twin.i, 4, [], 2.0))
    run(14)                                                                  # -> 23:00
    sm = o.twin.summary()
    print(f"\nDay summary: cost Rs {sm['total_cost_inr'] / 1e5:.1f} lakh · carbon {sm['carbon_t']:.0f} t · "
          f"unserved {sm['unserved_mwh']:.2f} MWh · violations {sm['violations']} · log -> {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

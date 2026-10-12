"""Safety layer: deterministic validator, human-in-the-lead approval gate, and safe fallback.

Nothing reaches the twin without passing `validate`. The validator *repairs* rather than
rejects (clip to physical limits, net overlaps, re-close the energy balance) and records every
repair so the decision log shows exactly what changed and why.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .config import Portfolio


def _finite(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def validate(p: Portfolio, sp: dict, tel: dict, dr_approved_mw: float = 0.0) -> tuple[dict, list[str]]:
    """Return (safe_setpoints, repairs). `tel` is the operator telemetry (nowcast)."""
    dt, repairs = p.dt, []
    sp = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (sp or {}).items()}
    for key in ("use", "charge", "discharge", "curtail"):
        sp.setdefault(key, {})
    for key in ("buy", "sell", "dr", "shift_down", "shift_up"):
        v = _finite(sp.get(key, 0.0))
        if v != sp.get(key, 0.0):
            repairs.append(f"{key}: non-numeric value replaced with 0")
        sp[key] = max(0.0, v)

    # Renewables: 0 <= use <= available nowcast
    avail = {}
    for g in p.generators:
        a = _finite(tel.get(f"{g.kind}_{g.id}", 0.0)) * tel.get("gen_avail", {}).get(g.id, 1.0)
        avail[g.id] = a
        u = _finite(sp["use"].get(g.id, a))
        if u > a + 1e-6 or u < 0:
            repairs.append(f"{g.id}: use {u:.1f} clipped to [0, {a:.1f}]")
        sp["use"][g.id] = min(max(u, 0.0), a)
    for line, cap in tel.get("line_caps", p.lines).items():
        on = [g.id for g in p.generators if g.line == line]
        flow = sum(sp["use"][a] for a in on)
        if flow > cap + 1e-6:
            k = cap / flow
            for a in on:
                sp["use"][a] *= k
            repairs.append(f"{line}: flow {flow:.1f} MW scaled to capacity {cap:.1f} MW")
    for g in p.generators:
        sp["curtail"][g.id] = max(0.0, avail[g.id] - sp["use"][g.id])

    # Batteries: power rating, availability, no simultaneous charge/discharge, SOC headroom
    for b in p.batteries:
        c, d = max(0.0, _finite(sp["charge"].get(b.id, 0))), max(0.0, _finite(sp["discharge"].get(b.id, 0)))
        if not tel.get("bess_avail", {}).get(b.id, True):
            if c or d:
                repairs.append(f"{b.id}: offline, setpoint zeroed")
            c = d = 0.0
        if c > 0 and d > 0:
            net = c - d
            c, d = max(net, 0.0), max(-net, 0.0)
            repairs.append(f"{b.id}: simultaneous charge/discharge netted")
        soc = tel["soc"][b.id]
        c_max = min(b.power_mw, max(0.0, (b.soc_max - soc) / (b.eta_charge * dt)))
        d_max = min(b.power_mw, max(0.0, (soc - b.soc_min) * b.eta_discharge / dt))
        if c > c_max + 1e-6:
            repairs.append(f"{b.id}: charge {c:.1f} clipped to {c_max:.1f} MW (SOC/rating)")
            c = c_max
        if d > d_max + 1e-6:
            repairs.append(f"{b.id}: discharge {d:.1f} clipped to {d_max:.1f} MW (SOC/rating)")
            d = d_max
        sp["charge"][b.id], sp["discharge"][b.id] = c, d

    # Demand-side actions only within approved / contracted limits
    if sp["dr"] > dr_approved_mw + 1e-6:
        repairs.append(f"demand response {sp['dr']:.1f} MW exceeds approved {dr_approved_mw:.1f} MW")
        sp["dr"] = min(sp["dr"], dr_approved_mw)
    for k in ("shift_down", "shift_up"):
        if sp[k] > p.shiftable_mw + 1e-6:
            repairs.append(f"{k} clipped to shiftable {p.shiftable_mw} MW")
            sp[k] = p.shiftable_mw

    # Re-close the energy balance on the nowcast; grid buys/sells within limits
    demand = _finite(tel.get("demand", 0.0))
    supply = sum(sp["use"].values()) + sum(sp["discharge"].values()) - sum(sp["charge"].values())
    residual = demand - sp["dr"] - sp["shift_down"] + sp["shift_up"] - supply
    g = p.grid
    want_buy, want_sell = max(residual, 0.0), max(-residual, 0.0)
    if abs((sp["buy"] - sp["sell"]) - residual) > 0.5:
        repairs.append(f"grid exchange rebalanced to close energy balance ({residual:+.1f} MW)")
    sp["buy"], sp["sell"] = min(want_buy, g["import_limit_mw"]), min(want_sell, g["export_limit_mw"])
    if want_sell > g["export_limit_mw"]:
        # cannot export the surplus: curtail more (proportionally) instead
        excess = want_sell - g["export_limit_mw"]
        tot = sum(sp["use"].values()) or 1.0
        for a in sp["use"]:
            cut = excess * sp["use"][a] / tot
            sp["use"][a] -= cut
            sp["curtail"][a] += cut
        repairs.append(f"export limit binding: extra curtailment {excess:.1f} MW")
    sp["expected_unserved"] = max(0.0, want_buy - g["import_limit_mw"])
    return sp, repairs


def rule_based_setpoints(p: Portfolio, tel: dict, reserve_pct: float = 10.0) -> dict:
    """Greedy rule-based EMS (baseline AND safe fallback): store surplus, discharge on deficit,
    export what cannot be stored, import the rest. No price awareness."""
    dt = p.dt
    ren = {g.id: _finite(tel.get(f"{g.kind}_{g.id}", 0.0)) * tel.get("gen_avail", {}).get(g.id, 1.0)
           for g in p.generators}
    net = sum(ren.values()) - _finite(tel.get("demand", 0.0))
    ch = {b.id: 0.0 for b in p.batteries}
    dis = {b.id: 0.0 for b in p.batteries}
    for b in p.batteries:
        if not tel.get("bess_avail", {}).get(b.id, True):
            continue
        soc = tel["soc"][b.id]
        floor = b.soc_min + b.energy_mwh * reserve_pct / 100
        if net > 0:
            c = min(net, b.power_mw, max(0.0, (b.soc_max - soc) / (b.eta_charge * dt)))
            ch[b.id], net = c, net - c
        elif net < 0:
            d = min(-net, b.power_mw, max(0.0, (soc - floor) * b.eta_discharge / dt))
            dis[b.id], net = d, net + d
    return {"use": ren, "curtail": {k: 0.0 for k in ren}, "charge": ch, "discharge": dis,
            "buy": max(-net, 0.0), "sell": max(net, 0.0), "dr": 0.0, "shift_down": 0.0, "shift_up": 0.0}


# ---------------------------------------------------------------------- human in the lead
HIGH_IMPACT = {"trigger_demand_response", "schedule_maintenance", "delay_maintenance", "dispatch_inspection"}
_ids = itertools.count(1)


@dataclass
class ApprovalRequest:
    action: str
    detail: str
    params: dict
    created: str
    rationale: str = ""
    id: int = field(default_factory=lambda: next(_ids))
    status: str = "pending"          # pending | approved | rejected | expired
    decided_by: str = ""


class ApprovalQueue:
    """High-impact actions wait here for an operator. Policies let backtests run unattended:
    'manual' (UI), 'approve_all', 'reject_all'."""

    def __init__(self, policy: str = "manual"):
        self.policy = policy
        self.items: list[ApprovalRequest] = []

    def request(self, action: str, detail: str, params: dict, created: str, rationale: str = "") -> ApprovalRequest:
        for it in self.items:  # never re-ask an identical question, whatever the answer was
            same = {k: v for k, v in it.params.items() if not k.startswith("_")} == params
            if it.action == action and same:
                return it
        req = ApprovalRequest(action=action, detail=detail, params=params, created=created, rationale=rationale)
        if self.policy == "approve_all":
            req.status, req.decided_by = "approved", "policy:auto"
        elif self.policy == "reject_all":
            req.status, req.decided_by = "rejected", "policy:auto"
        self.items.append(req)
        return req

    def decide(self, req_id: int, approve: bool, who: str = "operator") -> None:
        for it in self.items:
            if it.id == req_id and it.status == "pending":
                it.status, it.decided_by = ("approved" if approve else "rejected"), who

    def pending(self) -> list[ApprovalRequest]:
        return [i for i in self.items if i.status == "pending"]

    def approved(self, action: str) -> list[ApprovalRequest]:
        return [i for i in self.items if i.status == "approved" and i.action == action]

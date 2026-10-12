"""The agent loop. One call to `tick()` = one 15-minute control cycle:

  perceive -> forecast (+online recalibration) -> assess situation -> strategist (criteria)
  -> human gate -> scenario-MPC (dispatch) -> validator -> execute on twin -> detect faults -> log

Controller modes (same loop, different brains — used by the evaluation harness):
  gridmind   strategist + scenario-MPC (K scenarios) + perception + HITL     <- the product
  mpc        deterministic MPC on P50 forecasts with fixed weights (ablation)
  rules      greedy rule-based EMS (baseline)
  perfect    MPC with perfect knowledge of the actual future (theoretical ceiling)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Portfolio
from .forecast import Forecaster
from .llm import LLM
from .optimizer import PlanInputs, solve
from .perception import Perceived, Perception
from .safety import ApprovalQueue, rule_based_setpoints, validate
from .strategist import KEYS, PROFILES, Strategist, assess
from .twin import DigitalTwin

FIXED_WEIGHTS = dict(zip(KEYS, PROFILES["normal"]["w"]))


@dataclass
class OrchestratorConfig:
    mode: str = "gridmind"
    horizon: int | None = None
    scenarios: int | None = None
    seed: int = 0
    approval_policy: str = "manual"
    log_path: Path | None = None
    profile: dict | None = None      # mode "policy": fixed {"weights": {...}, "reserve": pct} with scenario-MPC


class Orchestrator:
    def __init__(self, p: Portfolio, twin: DigitalTwin, forecaster: Forecaster | None,
                 cfg: OrchestratorConfig | None = None, llm: LLM | None = None):
        self.p, self.twin, self.f = p, twin, forecaster
        self.cfg = cfg or OrchestratorConfig()
        self.H = self.cfg.horizon or p.horizon_steps
        self.K = self.cfg.scenarios or p.scenarios
        self.rng = np.random.default_rng(self.cfg.seed)
        self.llm = llm or LLM()
        self.strategist = Strategist(p, self.llm)
        self.perception = Perception(p, self.llm)
        self.approvals = ApprovalQueue(self.cfg.approval_policy)
        self.constraints: list[dict] = []
        self.maintenance: list[dict] = []          # approved maintenance windows
        self.known_faults: set[str] = set()
        self.last_good: dict[str, float] = {}
        self.prev_fc: tuple[int, dict] | None = None
        self.log: list[dict] = []
        self.last_plan = None
        self.last_fc = None
        self.pending_docs: list[dict] = []
        self.chaos_bad_llm = False                 # chaos hook: next strategist call gets malformed output

    # ================================================================== perception
    def submit_document(self, text: str | None = None, pdf_path: str | Path | None = None, name: str = "") -> Perceived:
        now = self.twin.now
        res = self.perception.read_pdf(pdf_path, now) if pdf_path else self.perception.read_text(text or "", now)
        idx = self.twin.obs.index
        for c in res.constraints:
            s, e = pd.Timestamp(c["start"]).tz_localize(None), pd.Timestamp(c["end"]).tz_localize(None)
            c["start_idx"] = int(np.searchsorted(idx, s))
            c["end_idx"] = int(np.searchsorted(idx, e))
            c["document"] = name or (Path(pdf_path).name if pdf_path else "text notice")
            self.constraints.append(c)
            if c["kind"] == "maintenance_request":
                self._plan_maintenance(c)
        self.pending_docs.append({"document": name or (str(pdf_path) if pdf_path else "text notice"),
                                  "source": res.source, "summary": res.summary,
                                  "constraints": [{k: v for k, v in c.items() if k not in ("start_idx", "end_idx")}
                                                  for c in res.constraints],
                                  "rejected": res.rejected, "injection_flags": res.injection_flags})
        return res

    # ================================================================== main loop
    def tick(self) -> dict:
        p, twin, i = self.p, self.twin, self.twin.i
        mode = self.cfg.mode
        tel = twin.telemetry()
        sensor_fallback = self._clean_telemetry(tel)
        self._apply_maintenance_status(i)
        tel["gen_avail"] = dict(twin.gen_avail)
        for b in self.known_faults:                 # operator now knows these batteries are faulty
            tel["bess_avail"][b] = False
        tel["faulted"] = sorted(self.known_faults)

        rec: dict = {"t": str(twin.now), "i": i, "mode": mode, "sensor_fallback": sensor_fallback,
                     "documents": self.pending_docs, "approvals_created": []}
        self.pending_docs = []
        decision, meta, sit = None, {}, None

        if mode == "rules":
            sp = rule_based_setpoints(p, tel)
            plan = None
        else:
            fc = self._forecast(i, tel)
            if mode == "gridmind":
                sit = assess(p, tel, fc, self.constraints, i)
                decision, meta = self.strategist.decide(sit, force_bad_output=self.chaos_bad_llm)
                self.chaos_bad_llm = False
                self._human_gate(decision, sit, i, rec)
                weights = decision.weights.model_dump()
                reserves = {r.battery_id: r.reserve_pct for r in decision.battery_reserves}
                banned = tuple(decision.banned_actions)
            elif mode == "policy":
                weights = self.cfg.profile["weights"]
                reserves = {b.id: float(self.cfg.profile.get("reserve", 10)) for b in p.batteries}
                banned = ()
            else:
                weights, reserves, banned = FIXED_WEIGHTS, {b.id: 10.0 for b in p.batteries}, ()
            x = self._plan_inputs(i, tel, fc, weights, reserves, banned)
            plan = solve(p, x)
            self.last_plan, self.last_fc = plan, fc
            if plan.status != "optimal":
                sp = rule_based_setpoints(p, tel)
                rec["fallback"] = f"optimizer {plan.status}; rule-based safe action used"
            else:
                sp = plan.step0

        dr_ok = self._approved_dr_mw(i)
        sp, repairs = validate(p, sp, tel, dr_approved_mw=dr_ok)
        res = twin.step(sp)

        new_faults = [b for b in res.faults if b not in self.known_faults]
        for b in new_faults:
            self.known_faults.add(b)
        if new_faults:   # recovery path: isolate (above) and ask a human to send a crew
            detail = f"Dispatch inspection team to {', '.join(sorted(self.known_faults))} (not following setpoints)"
            req = self.approvals.request("dispatch_inspection", detail, {"assets": sorted(self.known_faults)},
                                         str(res.t), "Executor detected a setpoint deviation; asset isolated.")
            rec["approvals_created"].append({"id": req.id, "action": req.action, "detail": detail, "status": req.status})
            self._apply_approvals(i)
        rec.update({
            "situation": sit.label if sit else None,
            "situation_reasons": sit.reasons if sit else [],
            "features": sit.features if sit else {},
            "strategy": None if decision is None else {
                "source": meta.get("source"), "reused": meta.get("reused"), "rejections": meta.get("rejections"),
                "weights": decision.weights.model_dump(),
                "reserves": {r.battery_id: r.reserve_pct for r in decision.battery_reserves},
                "banned": decision.banned_actions, "requests": decision.request_human_approval,
                "rationale": decision.rationale, "confidence": decision.confidence},
            "plan": None if plan is None else {"status": plan.status, "solve_ms": round(plan.solve_ms, 1),
                                                "n_vars": plan.n_vars, "breakdown": plan.breakdown},
            "setpoints": _round(sp), "repairs": repairs, "executed": _round(res.dispatch),
            "metrics": _round(res.metrics), "violations": res.violations,
            "faults_detected": new_faults,
            "recovery": [f"{b} did not follow setpoint -> marked unavailable, re-planning without it"
                         for b in new_faults],
            "actions": self._describe(sp, plan, decision, tel),
            "forecast_calibration": dict(self.f.adapt) if self.f else {},
        })
        self.log.append(rec)
        if self.cfg.log_path:
            with open(self.cfg.log_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def run(self, steps: int) -> list[dict]:
        return [self.tick() for _ in range(steps)]

    # ================================================================== helpers
    def _clean_telemetry(self, tel: dict) -> list[str]:
        """Sensor dropout: replace NaN with last-known-good values and flag them."""
        flagged = []
        for k, v in list(tel.items()):
            if isinstance(v, float):
                if math.isnan(v):
                    tel[k] = self.last_good.get(k, 0.0)
                    flagged.append(k)
                else:
                    self.last_good[k] = v
        return flagged

    def _forecast(self, i: int, tel: dict) -> dict[str, np.ndarray]:
        obs = self.twin.obs
        H1 = self.H - 1
        if self.cfg.mode == "perfect":
            return {s.column: np.repeat(obs[s.column].to_numpy()[i + 1: i + 1 + H1][None, :], 3, axis=0)
                    for s in self.f.signals}
        if self.prev_fc is not None:   # online recalibration with what has been realised since
            pi, pfc = self.prev_fc
            self.f.observe(pfc, {s.column: obs[s.column].to_numpy()[pi + 1: i + 1] for s in self.f.signals})
        widen = 2.0 if tel.get("forecast_stale") else 1.0
        fc = self.f.predict(obs, i, H1, widen=widen)
        self.prev_fc = (i, fc)
        return fc

    def _plan_inputs(self, i, tel, fc, weights, reserves, banned) -> PlanInputs:
        p, H = self.p, self.H
        if self.cfg.mode in ("gridmind", "policy"):
            K = self.K
            sc = self.f.sample_scenarios(fc, K, self.rng)
            if K > 1:
                for col in sc:   # keep the P50 path as one scenario for stability
                    sc[col][0] = fc[col][1]
        else:
            K = 1
            sc = {col: fc[col][1][None, :] for col in fc}
        # Known future constraints from perception / approved maintenance reshape the scenarios.
        gen = {}
        for g in p.generators:
            col = f"{g.kind}_{g.id}"
            arr = np.hstack([np.full((K, 1), tel[col]), sc[col]])
            gen[g.id] = arr * self._gen_factor(g.id, i, H)[None, :]
        dem = np.hstack([np.full((K, 1), tel["demand"]), sc["demand"]]) + self._demand_delta(i, H)[None, :]
        price = np.hstack([np.full((K, 1), tel["price"]), sc["price"]]) * self._price_mult(i, H)[None, :]
        obs_carbon = self.twin.obs["carbon"].to_numpy()
        carbon_profile = obs_carbon[max(0, i - H + 1): i + 1]
        carbon_profile = np.concatenate([[tel.get("carbon", carbon_profile[-1])], carbon_profile[1:]])[:H]
        carbon = np.tile(np.resize(carbon_profile, H)[None, :], (K, 1))
        if self.cfg.mode == "perfect":
            carbon = obs_carbon[i: i + H][None, :]
        line_caps = {l: self._line_cap(l, tel["line_caps"][l], i, H) for l in p.lines}
        freq_res = p.grid["frequency_reserve_mw"] * (2 if tel.get("frequency", 50) < 49.9 else 1)
        return PlanInputs(gen=gen, demand=dem, price=price, carbon=carbon, soc=dict(tel["soc"]), weights=weights,
                          reserve_pct=reserves, banned=banned, bess_avail=dict(tel["bess_avail"]),
                          line_caps=line_caps, dr_allowed_mw=self._approved_dr_mw(i), freq_reserve_mw=freq_res,
                          shift_balance_mwh=self.twin.shift_balance)

    def _window(self, c, i, H) -> np.ndarray:
        t = np.arange(i, i + H)
        return (t >= c["start_idx"]) & (t < c["end_idx"])

    def _gen_factor(self, asset, i, H) -> np.ndarray:
        f = np.ones(H)
        for c in self.constraints:
            if asset not in c["assets"]:
                continue
            w = self._window(c, i, H)
            if c["kind"] == "storm_alert":
                f[w] *= 0.0 if asset.startswith("WND") else 0.3
            elif c["kind"] == "generation_derate":
                f[w] *= c["value"]
            elif c["kind"] == "asset_outage":
                f[w] = 0.0
        for m in self.maintenance:
            if m["asset"] == asset:
                t = np.arange(i, i + H)
                f[(t >= m["start_idx"]) & (t < m["end_idx"])] = 0.0
        return f

    def _demand_delta(self, i, H) -> np.ndarray:
        d = np.zeros(H)
        for c in self.constraints:
            if c["kind"] == "demand_change":
                d[self._window(c, i, H)] += c["value"]
        return d

    def _price_mult(self, i, H) -> np.ndarray:
        m = np.ones(H)
        for c in self.constraints:
            if c["kind"] == "price_alert":
                m[self._window(c, i, H)] *= c["value"]
            elif c["kind"] == "storm_alert":
                m[self._window(c, i, H)] *= 1.3
        return m

    def _line_cap(self, line, cap_now, i, H) -> np.ndarray:
        caps = np.full(H, float(self.p.lines[line]))
        caps[0] = cap_now
        for c in self.constraints:
            if c["kind"] == "line_limit" and line in c["assets"]:
                w = self._window(c, i, H)
                caps[w] = np.minimum(caps[w], c["value"])
        return caps

    # ---------------------------------------------------------------- human in the lead
    def _approved_dr_mw(self, i) -> float:
        mw = 0.0
        for r in self.approvals.approved("trigger_demand_response"):
            if r.params["start_idx"] <= i < r.params["end_idx"]:
                mw = max(mw, r.params["mw"])
        return mw

    def _human_gate(self, decision, sit, i, rec) -> None:
        now = str(self.twin.now)
        for action in decision.request_human_approval:
            if action == "trigger_demand_response":
                storm = [c for c in self.constraints if c["kind"] == "storm_alert" and c["end_idx"] > i]
                s, e = (storm[0]["start_idx"], storm[0]["end_idx"]) if storm else (i, i + 16)
                params = {"mw": float(self.p.dr_max_mw), "start_idx": int(s), "end_idx": int(e)}
                detail = (f"Enable up to {self.p.dr_max_mw:.0f} MW industrial demand response "
                          f"{self.twin.obs.index[min(s, len(self.twin.obs) - 1)]:%d %b %H:%M}-"
                          f"{self.twin.obs.index[min(e, len(self.twin.obs) - 1)]:%H:%M} "
                          f"(incentive Rs {self.p.market['dr_incentive_inr_per_mwh']}/MWh)")
            elif action == "delay_maintenance":
                storm = [c for c in self.constraints if c["kind"] == "storm_alert" and c["end_idx"] > i]
                clash = [m for m in self.maintenance if storm and m["start_idx"] < storm[0]["end_idx"]
                         and m["end_idx"] > storm[0]["start_idx"]]
                if not clash:
                    continue
                m = clash[0]
                shift = storm[0]["end_idx"] - m["start_idx"] + 8
                params = {"asset": m["asset"], "shift_steps": int(shift)}
                detail = f"Delay {m['asset']} maintenance by {shift / 4:.1f} h to clear the storm window"
            elif action == "dispatch_inspection":
                if not self.known_faults:
                    continue
                params = {"assets": sorted(self.known_faults)}
                detail = f"Dispatch inspection team to {', '.join(sorted(self.known_faults))} (not following setpoints)"
            else:
                continue
            req = self.approvals.request(action, detail, params, now, decision.rationale)
            if req.id not in [x["id"] for x in rec["approvals_created"]]:
                rec["approvals_created"].append({"id": req.id, "action": action, "detail": detail, "status": req.status})
        self._apply_approvals(i)

    def _apply_approvals(self, i) -> None:
        for r in self.approvals.items:
            if r.status != "approved" or r.params.get("_applied"):
                continue
            if r.action == "delay_maintenance":
                for m in self.maintenance:
                    if m["asset"] == r.params["asset"]:
                        m["start_idx"] += r.params["shift_steps"]
                        m["end_idx"] += r.params["shift_steps"]
            elif r.action == "schedule_maintenance":
                self.maintenance.append({"asset": r.params["asset"], "start_idx": r.params["start_idx"],
                                         "end_idx": r.params["end_idx"]})
            elif r.action == "dispatch_inspection":
                # Crew arrives in 2 h and resets the battery controller.
                for ev in self.twin.events:
                    if ev.type == "battery_unresponsive" and set(ev.assets) & set(r.params["assets"]):
                        ev.duration = max(0, min(ev.duration, i + 8 - ev.start))
                r.params["_repair_at"] = i + 8
            r.params["_applied"] = True
        for r in self.approvals.approved("dispatch_inspection"):
            if r.params.get("_repair_at") is not None and i >= r.params["_repair_at"]:
                for b in r.params["assets"]:
                    self.known_faults.discard(b)

    def _plan_maintenance(self, c: dict) -> None:
        """Pick the maintenance window that sacrifices the least expected energy value."""
        i, asset = self.twin.i, c["assets"][0]
        g = next((g for g in self.p.generators if g.id == asset), None)
        dur = int(round(c["value"] * 4))
        if g is None or self.f is None:
            return
        fc = self.f.predict(self.twin.obs, i, self.H - 1)
        gen = np.concatenate([[0.0], fc[f"{g.kind}_{g.id}"][1]])
        price = np.concatenate([[self.twin.obs["price"].iloc[i]], fc["price"][1]])
        lo, hi = max(c["start_idx"], i + 1) - i, min(c["end_idx"], i + self.H) - i - dur
        storm = [(s["start_idx"] - i, s["end_idx"] - i) for s in self.constraints if s["kind"] == "storm_alert"]
        best = None
        for s in range(lo, max(lo, hi) + 1):
            if any(s < e and s + dur > b for b, e in storm):
                continue
            v = float((gen[s:s + dur] * price[s:s + dur]).sum() * self.p.dt)
            if best is None or v < best[0]:
                best = (v, s)
        if best is None:
            return
        v, s = best
        idx = self.twin.obs.index
        detail = (f"Schedule {asset} maintenance ({c['value']:.0f} h) at {idx[i + s]:%d %b %H:%M}: "
                  f"lowest expected lost value (Rs {v:,.0f})")
        self.approvals.request("schedule_maintenance", detail,
                               {"asset": asset, "start_idx": int(i + s), "end_idx": int(i + s + dur),
                                "lost_value_inr": round(v)}, str(self.twin.now),
                               "Window chosen by scanning the P50 generation x price forecast.")
        self._apply_approvals(i)

    def _apply_maintenance_status(self, i: int) -> None:
        for g in self.p.generators:
            down = any(m["asset"] == g.id and m["start_idx"] <= i < m["end_idx"] for m in self.maintenance)
            self.twin.set_generator_availability(g.id, 0.0 if down else 1.0)

    # ---------------------------------------------------------------- explanation
    def _describe(self, sp, plan, decision, tel) -> list[str]:
        """Human-readable action cluster for the operator console."""
        out = []
        for b in self.p.batteries:
            c, d = sp["charge"].get(b.id, 0), sp["discharge"].get(b.id, 0)
            if not tel["bess_avail"].get(b.id, True):
                out.append(f"{b.id}: offline / isolated")
            elif c > 0.1:
                out.append(f"Charge {b.id} at {c:.1f} MW")
            elif d > 0.1:
                out.append(f"Discharge {b.id} at {d:.1f} MW")
            else:
                out.append(f"Hold {b.id} (SOC {100 * tel['soc'][b.id] / b.energy_mwh:.0f}%)")
        if sp["buy"] > 0.1:
            out.append(f"Buy {sp['buy']:.1f} MW from the market")
        if sp["sell"] > 0.1:
            out.append(f"Sell {sp['sell']:.1f} MW to the market")
        curt = {k: v for k, v in sp["curtail"].items() if v > 0.1}
        if curt:
            out.append("Curtail " + ", ".join(f"{k} by {v:.1f} MW" for k, v in curt.items()))
        if sp.get("dr", 0) > 0.1:
            out.append(f"Demand response: shed {sp['dr']:.1f} MW (operator-approved)")
        if sp.get("shift_down", 0) > 0.1:
            out.append(f"Shift {sp['shift_down']:.1f} MW of industrial load to later")
        if sp.get("shift_up", 0) > 0.1:
            out.append(f"Pull forward {sp['shift_up']:.1f} MW of shiftable industrial load")
        if plan is not None and plan.expected:
            soc_total = sum(tel["soc"].values())
            price_path = plan.expected["price"]
            disch_now = sum(sp["discharge"].values())
            peak = int(np.argmax(price_path[:48]))     # only the next 12 h
            if soc_total > 20 and disch_now < 0.1 and peak > 0 and price_path[peak] > 1.2 * price_path[0]:
                t_peak = self.twin.obs.index[min(self.twin.i - 1 + peak, len(self.twin.obs) - 1)]
                out.append(f"Delay selling stored energy until ~{t_peak:%H:%M} "
                           f"(expected Rs {price_path[peak]:,.0f}/MWh vs Rs {price_path[0]:,.0f} now)")
        if decision is not None and decision.banned_actions:
            out.append("Blocked by strategy: " + ", ".join(decision.banned_actions))
        return out


def _round(d):
    if isinstance(d, dict):
        return {k: _round(v) for k, v in d.items()}
    if isinstance(d, float):
        return round(d, 3)
    return d


def make_orchestrator(start: str | pd.Timestamp, mode: str = "gridmind", seed: int = 0, approval_policy: str = "manual",
                      scenarios: int | None = None, log_path: Path | None = None, llm: LLM | None = None,
                      profile: dict | None = None):
    """Convenience constructor used by the UI, the evaluation harness and the tests."""
    from .config import load_portfolio
    from .data import get_dataset
    p = load_portfolio()
    df = get_dataset()
    f = Forecaster.load_or_fit(p, df)
    f.adapt = {}
    twin = DigitalTwin(p, df, pd.Timestamp(start))
    cfg = OrchestratorConfig(mode=mode, seed=seed, approval_policy=approval_policy, scenarios=scenarios,
                             log_path=log_path, profile=profile)
    return Orchestrator(p, twin, f, cfg, llm=llm)

"""Digital twin of the portfolio + event injector.

The twin owns the *actual* world: it replays real weather-driven generation and synthetic
demand/price, applies injected events (clouds, outages, spikes...), executes setpoints with
physical limits, settles residual imbalance with the grid, and keeps the scorecard.

The controller only sees what a real operator would: telemetry for the current interval
(possibly with sensor dropouts) and asset status flags — never the future, and never a silent
battery fault until it shows up as a setpoint deviation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Portfolio

EVENT_TYPES = {
    "cloud_bank":          "Clouds reduce solar output (factor) on selected parks",
    "wind_surge":          "Wind suddenly increases (factor) on selected farms",
    "price_spike":         "Electricity prices spike (multiplier)",
    "battery_outage":      "A battery is taken offline (reported by SCADA)",
    "battery_unresponsive": "A battery silently stops following setpoints (fault)",
    "line_congestion":     "A transmission corridor is limited to capacity_mw",
    "demand_surge":        "Industrial demand increases unexpectedly (+mw)",
    "storm":               "Storm: wind farms trip on high wind, solar collapses, prices rise",
    "forecast_stale":      "Weather feed stalls; forecasts are hours old",
    "sensor_dropout":      "Telemetry returns NaN for some signals",
}


@dataclass
class Event:
    type: str
    start: int                       # absolute row index in the dataset
    duration: int                    # steps
    assets: list[str] = field(default_factory=list)
    value: float = 1.0               # factor / multiplier / MW, depending on type
    note: str = ""

    @property
    def end(self) -> int:
        return self.start + self.duration

    def to_dict(self) -> dict:
        return {"type": self.type, "start": self.start, "duration": self.duration,
                "assets": self.assets, "value": self.value, "note": self.note}


@dataclass
class StepResult:
    t: pd.Timestamp
    dispatch: dict
    metrics: dict
    faults: list[str]
    violations: list[str]


class DigitalTwin:
    def __init__(self, p: Portfolio, truth: pd.DataFrame, start: pd.Timestamp | int, deviation_penalty: float = 0.2):
        self.p = p
        self.base = truth
        self.obs = truth.copy()                      # actual world incl. injected events
        self.i = start if isinstance(start, (int, np.integer)) else int(np.searchsorted(truth.index, start))
        self.i0 = self.i
        self.soc = {b.id: b.energy_mwh * b.soc_init_pct / 100 for b in p.batteries}
        self.bess_avail = {b.id: True for b in p.batteries}       # visible to the operator
        self.bess_silent_fault: set[str] = set()                  # hidden from the operator
        self.gen_avail = {g.id: 1.0 for g in p.generators}        # planned outages / maintenance
        self.line_caps = dict(p.lines)
        self.events: list[Event] = []
        self.dropout: dict[str, tuple[int, int]] = {}             # column -> (start, end) index
        self.forecast_stale_until = -1
        self.deviation_penalty = deviation_penalty
        self.shift_balance = 0.0                                  # MWh of industrial load deferred (+)
        self.totals = {k: 0.0 for k in ("cost_inr", "energy_cost_inr", "carbon_t", "carbon_cost_inr",
                                        "curtailment_mwh", "renewable_avail_mwh", "renewable_used_mwh",
                                        "degradation_inr", "unserved_mwh", "dr_mwh", "dr_cost_inr",
                                        "deviation_mwh", "import_mwh", "export_mwh", "demand_mwh")}
        self.violations: list[str] = []
        self.history: list[dict] = []

    # ------------------------------------------------------------------ state access
    @property
    def now(self) -> pd.Timestamp:
        return self.obs.index[self.i]

    def snapshot(self) -> "DigitalTwin":
        return copy.deepcopy(self)

    def telemetry(self) -> dict:
        """Observed values for the current interval (what SCADA reports)."""
        self._apply_status_events()
        row = self.obs.iloc[self.i]
        tel = {c: float(row[c]) for c in self.obs.columns if not c.startswith(("ghi_", "ws_", "load_"))}
        for col, (start, end) in self.dropout.items():
            if start <= self.i < end and col in tel:
                tel[col] = float("nan")
        tel["soc"] = dict(self.soc)
        tel["bess_avail"] = dict(self.bess_avail)
        tel["gen_avail"] = dict(self.gen_avail)
        tel["line_caps"] = dict(self.line_caps)
        tel["forecast_stale"] = self.i < self.forecast_stale_until
        return tel

    # ------------------------------------------------------------------ events
    def inject(self, ev: Event) -> None:
        """Apply an event to the actual world from ev.start for ev.duration steps."""
        self.events.append(ev)
        sl = slice(ev.start, ev.end)
        cols = self.obs.columns
        idx = self.obs.index[sl]
        n = len(idx)
        ramp = np.minimum(1.0, np.arange(1, n + 1) / 2)  # 30-min ramp in
        if ev.type == "cloud_bank":
            targets = ev.assets or [g.id for g in self.p.solar]
            for a in targets:
                c = f"solar_{a}"
                self.obs.loc[idx, c] = self.obs.loc[idx, c] * (1 - (1 - ev.value) * ramp)
        elif ev.type == "wind_surge":
            for a in ev.assets or [g.id for g in self.p.wind]:
                cap = next(g.capacity_mw for g in self.p.wind if g.id == a)
                c = f"wind_{a}"
                self.obs.loc[idx, c] = np.minimum(cap, np.maximum(self.obs.loc[idx, c] * ev.value, 0.6 * cap * ramp))
        elif ev.type == "price_spike":
            self.obs.loc[idx, "price"] = np.minimum(10_000, self.obs.loc[idx, "price"] * (1 + (ev.value - 1) * ramp))
        elif ev.type == "demand_surge":
            share = ev.value / len(self.p.loads)
            for ld in self.p.loads:
                self.obs.loc[idx, f"load_{ld.id}"] += share * ramp
            self.obs["demand"] = self.obs[[f"load_{l.id}" for l in self.p.loads]].sum(axis=1)
        elif ev.type == "storm":
            for g in self.p.wind:
                if not ev.assets or g.id in ev.assets:
                    self.obs.loc[idx, f"wind_{g.id}"] = 0.0          # high-wind cut-out
            for g in self.p.solar:
                if not ev.assets or g.id in ev.assets:
                    self.obs.loc[idx, f"solar_{g.id}"] *= 0.25
            self.obs.loc[idx, "price"] = np.minimum(10_000, self.obs.loc[idx, "price"] * 1.6)
        elif ev.type == "sensor_dropout":
            for c in ev.assets or ["price"]:
                if c in cols:
                    self.dropout[c] = (ev.start, ev.end)
        elif ev.type == "forecast_stale":
            self.forecast_stale_until = ev.end
        # battery_outage / battery_unresponsive / line_congestion are status changes, applied per step

    def _apply_status_events(self) -> None:
        self.line_caps = dict(self.p.lines)
        for b in self.p.batteries:
            self.bess_avail[b.id] = True
        self.bess_silent_fault = set()
        for ev in self.events:
            if not (ev.start <= self.i < ev.end):
                continue
            if ev.type == "battery_outage":
                for a in ev.assets:
                    self.bess_avail[a] = False
            elif ev.type == "battery_unresponsive":
                self.bess_silent_fault.update(ev.assets)
            elif ev.type == "line_congestion":
                for a in ev.assets:
                    self.line_caps[a] = min(self.line_caps[a], ev.value)

    def set_generator_availability(self, asset: str, factor: float) -> None:
        self.gen_avail[asset] = factor

    # ------------------------------------------------------------------ execution
    def step(self, sp: dict) -> StepResult:
        """Execute setpoints for the current interval against the actual world, then advance."""
        self._apply_status_events()
        p, dt = self.p, self.p.dt
        row = self.obs.iloc[self.i]
        t = self.now
        violations = self._check_setpoints(sp, row)

        # Renewables: setpoint 'use' acts as a curtailment cap when the plan curtails.
        avail = {g.id: float(row[f"{g.kind}_{g.id}"]) * self.gen_avail[g.id] for g in p.generators}
        use = {}
        for g in p.generators:
            cap = sp["use"].get(g.id, avail[g.id]) if sp.get("curtail", {}).get(g.id, 0) > 0.05 else np.inf
            use[g.id] = min(avail[g.id], cap)
        # Automatic line runback (protection) if actual injection exceeds corridor capacity.
        runback = 0.0
        for line, cap in self.line_caps.items():
            on = [g.id for g in p.generators if g.line == line]
            flow = sum(use[a] for a in on)
            if flow > cap + 1e-6:
                k = cap / flow
                for a in on:
                    runback += use[a] * (1 - k)
                    use[a] *= k

        # Batteries: follow setpoints unless offline / silently faulted; respect SOC physics.
        ch, dis, faults = {}, {}, []
        for b in p.batteries:
            c, d = float(sp["charge"].get(b.id, 0.0)), float(sp["discharge"].get(b.id, 0.0))
            if not self.bess_avail[b.id] or b.id in self.bess_silent_fault:
                if (c > 0.5 or d > 0.5) and b.id in self.bess_silent_fault:
                    faults.append(b.id)
                c = d = 0.0
            c = min(c, b.power_mw, max(0.0, (b.soc_max - self.soc[b.id]) / (b.eta_charge * dt)))
            d = min(d, b.power_mw, max(0.0, (self.soc[b.id] - b.soc_min) * b.eta_discharge / dt))
            ch[b.id], dis[b.id] = c, d
            self.soc[b.id] += b.eta_charge * c * dt - d * dt / b.eta_discharge

        demand = float(row["demand"])
        dr = float(sp.get("dr", 0.0))
        shift = float(sp.get("shift_down", 0.0)) - float(sp.get("shift_up", 0.0))
        self.shift_balance += shift * dt
        net_load = demand - dr - shift
        supply = sum(use.values()) + sum(dis.values()) - sum(ch.values())
        residual = net_load - supply

        g = p.grid
        buy = sell = unserved = extra_curt = 0.0
        if residual >= 0:
            buy = min(residual, g["import_limit_mw"])
            unserved = residual - buy
        else:
            sell = min(-residual, g["export_limit_mw"])
            extra_curt = -residual - sell

        price, carbon = float(row["price"]), float(row["carbon"])
        sched_net = float(sp.get("buy", 0.0)) - float(sp.get("sell", 0.0))
        deviation = abs((buy - sell) - sched_net)
        mk = p.market
        energy_cost = (price * buy - g["sell_discount"] * price * sell) * dt
        dev_cost = self.deviation_penalty * price * deviation * dt
        carbon_t = carbon * buy * dt
        carbon_cost = carbon_t * mk["carbon_price_inr_per_t"]
        deg = sum((ch[b.id] + dis[b.id]) * b.degradation_inr_per_mwh for b in p.batteries) * dt
        dr_cost = dr * mk["dr_incentive_inr_per_mwh"] * dt
        shift_cost = (float(sp.get("shift_down", 0.0)) + float(sp.get("shift_up", 0.0))) * mk["load_shift_cost_inr_per_mwh"] * dt
        ren_avail = sum(avail.values())
        ren_used = sum(use.values())
        curt = ren_avail - ren_used + extra_curt
        cost = energy_cost + dev_cost + carbon_cost + deg + dr_cost + shift_cost + unserved * dt * mk["voll_inr_per_mwh"]

        m = {
            "cost_inr": cost, "energy_cost_inr": energy_cost + dev_cost, "carbon_t": carbon_t,
            "carbon_cost_inr": carbon_cost, "curtailment_mwh": curt * dt, "renewable_avail_mwh": ren_avail * dt,
            "renewable_used_mwh": (ren_used - extra_curt) * dt, "degradation_inr": deg, "unserved_mwh": unserved * dt,
            "dr_mwh": dr * dt, "dr_cost_inr": dr_cost + shift_cost, "deviation_mwh": deviation * dt,
            "import_mwh": buy * dt, "export_mwh": sell * dt, "demand_mwh": demand * dt,
        }
        for k, v in m.items():
            self.totals[k] += v
        self.violations += [f"{t}: {v}" for v in violations]

        dispatch = {"t": str(t), "use": use, "charge": ch, "discharge": dis, "soc": dict(self.soc), "buy": buy,
                    "sell": sell, "dr": dr, "shift": shift, "unserved": unserved, "curtail": curt, "runback": runback,
                    "demand": demand, "price": price, "carbon": carbon, "renewable_avail": ren_avail,
                    "frequency": float(row["frequency"])}
        self.history.append({**{k: v for k, v in dispatch.items() if not isinstance(v, dict)},
                             **{f"soc_{k}": v for k, v in self.soc.items()},
                             "charge": sum(ch.values()), "discharge": sum(dis.values()),
                             "renewable_used": ren_used, "cost": cost})
        self.i += 1
        return StepResult(t=t, dispatch=dispatch, metrics=m, faults=faults, violations=violations)

    def _check_setpoints(self, sp: dict, row) -> list[str]:
        """Independent physics audit of the setpoints the controller sent (should always be empty)."""
        p, dt, out, tol = self.p, self.p.dt, [], 1e-3
        for b in p.batteries:
            c, d = sp["charge"].get(b.id, 0.0), sp["discharge"].get(b.id, 0.0)
            if c > b.power_mw + tol or d > b.power_mw + tol:
                out.append(f"{b.id} power rating exceeded")
            if c > tol and d > tol:
                out.append(f"{b.id} simultaneous charge/discharge")
            if not self.bess_avail[b.id] and (c > tol or d > tol):
                out.append(f"{b.id} dispatched while offline")
            nxt = self.soc[b.id] + b.eta_charge * c * dt - d * dt / b.eta_discharge
            if nxt < b.soc_min - 0.01 or nxt > b.soc_max + 0.01:
                out.append(f"{b.id} SOC limit violated ({nxt:.2f} MWh)")
        if sp.get("buy", 0) > p.grid["import_limit_mw"] + tol:
            out.append("import limit exceeded")
        if sp.get("sell", 0) > p.grid["export_limit_mw"] + tol:
            out.append("export limit exceeded")
        for line, cap in self.line_caps.items():
            flow = sum(sp["use"].get(g.id, 0.0) for g in p.generators if g.line == line)
            if flow > cap + 0.01:
                out.append(f"{line} capacity exceeded ({flow:.1f}>{cap:.1f} MW)")
        if sp.get("dr", 0) > p.dr_max_mw + tol:
            out.append("demand response above contracted maximum")
        return out

    def summary(self) -> dict:
        """Run scorecard. Load still deferred/pulled-forward at the end is settled at the run's
        mean price so controllers cannot 'win' by leaving energy owed."""
        t = self.totals
        prices = [h["price"] for h in self.history] or [0.0]
        settle = abs(self.shift_balance) * float(np.mean(prices))
        return {
            "total_cost_inr": t["cost_inr"] + settle,
            "operating_cost_excl_voll_inr": t["cost_inr"] + settle - t["unserved_mwh"] * self.p.market["voll_inr_per_mwh"],
            "energy_cost_inr": t["energy_cost_inr"],
            "carbon_t": t["carbon_t"],
            "curtailment_mwh": t["curtailment_mwh"],
            "renewable_utilisation_pct": 100 * t["renewable_used_mwh"] / max(t["renewable_avail_mwh"], 1e-9),
            "degradation_inr": t["degradation_inr"],
            "unserved_mwh": t["unserved_mwh"],
            "dr_mwh": t["dr_mwh"],
            "deviation_mwh": t["deviation_mwh"],
            "violations": len(self.violations),
            "shift_balance_mwh": self.shift_balance,
            "steps": self.i - self.i0,
        }

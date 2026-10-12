"""Seeded stress-event schedules (for the backtest and Monte Carlo) and the external documents
that announce some of them (storm bulletins, grid notices) — what a real operator would receive.

Every controller sees the same physical events. Only controllers with a Perception agent can
read the advance warnings; that difference is exactly what the ablation measures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Portfolio
from .twin import Event

DAY = 96


def storm_bulletin(region: str, start: pd.Timestamp, end: pd.Timestamp, severity: str = "severe") -> str:
    return (f"IMD NOWCAST WARNING. A {severity} thunderstorm with squalls is likely over {region} from "
            f"{start:%H:%M} to {end:%H:%M}. Gusts of 70-90 km/h expected. "
            f"Wind turbines may trip on high-wind cut-out.")


def line_notice(line: str, mw: float, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return (f"GRID-INDIA OPERATIONAL NOTICE. Due to a planned shutdown of a parallel circuit, transmission "
            f"corridor {line} is limited to {mw:.0f} MW from {start:%H:%M} to {end:%H:%M}.")


def demand_notice(mw: float, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return (f"Customer notice: LD-STEEL is starting an additional furnace. Expect industrial demand to increase "
            f"by {mw:.0f} MW from {start:%H:%M} to {end:%H:%M}.")


def generate_events(p: Portfolio, index: pd.DatetimeIndex, start_i: int, days: int, seed: int,
                    intensity: float = 1.0) -> tuple[list[Event], list[dict]]:
    """Return (events, documents). documents = [{'at': idx, 'text': str, 'name': str}] delivered to the
    controller at index 'at' (before the event starts)."""
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    docs: list[dict] = []
    wind_ids = [g.id for g in p.wind]
    solar_ids = [g.id for g in p.solar]
    regions = {"Gujarat": ["WND-2", "SOL-2"], "Tamil Nadu": ["WND-1"], "Maharashtra": ["WND-3"],
               "Rajasthan": ["SOL-1"], "Karnataka": ["SOL-3"]}
    for d in range(days):
        base = start_i + d * DAY
        n_events = rng.poisson(1.3 * intensity)
        for _ in range(n_events):
            kind = rng.choice(["storm", "battery_outage", "battery_unresponsive", "price_spike", "line_congestion",
                               "demand_surge", "cloud_bank", "wind_surge", "sensor_dropout", "forecast_stale"],
                              p=[0.16, 0.10, 0.08, 0.12, 0.10, 0.12, 0.12, 0.08, 0.06, 0.06])
            s = base + int(rng.integers(0, DAY))
            if kind == "storm":
                dur = int(rng.integers(12, 25))
                s = base + int(rng.integers(60, 84))          # evening storms (15:00-21:00), the hard case
                region = rng.choice(list(regions))
                assets = regions[region] + list(rng.choice(wind_ids, 1))
                events.append(Event("storm", s, dur, sorted(set(assets)), note=f"storm over {region}"))
                events.append(Event("demand_surge", s, dur, [], float(rng.uniform(10, 25)), note="storm-driven load"))
                lead = int(rng.integers(12, 28))                # warning 3-7 h ahead
                docs.append({"at": max(start_i, s - lead), "name": f"IMD bulletin ({region})",
                             "text": storm_bulletin(region, index[s], index[min(s + dur, len(index) - 1)])})
            elif kind in ("battery_outage", "battery_unresponsive"):
                b = rng.choice([x.id for x in p.batteries])
                events.append(Event(kind, s, int(rng.integers(8, 32)), [b]))
            elif kind == "price_spike":
                s = base + int(rng.integers(68, 84))
                events.append(Event("price_spike", s, int(rng.integers(4, 12)), [], float(rng.uniform(1.6, 2.5))))
            elif kind == "line_congestion":
                line = rng.choice(list(p.lines))
                s = base + int(rng.integers(36, 56))            # midday, when solar is flowing
                dur = int(rng.integers(8, 20))
                mw = float(rng.uniform(0.35, 0.6) * p.lines[line])
                events.append(Event("line_congestion", s, dur, [line], mw))
                docs.append({"at": max(start_i, s - int(rng.integers(4, 12))), "name": "Grid notice",
                             "text": line_notice(line, mw, index[s], index[min(s + dur, len(index) - 1)])})
            elif kind == "demand_surge":
                dur = int(rng.integers(8, 24))
                mw = float(rng.uniform(12, 30))
                events.append(Event("demand_surge", s, dur, [], mw))
                if rng.random() < 0.5:                          # sometimes the customer warns us
                    docs.append({"at": max(start_i, s - 8), "name": "Customer notice",
                                 "text": demand_notice(mw, index[s], index[min(s + dur, len(index) - 1)])})
            elif kind == "cloud_bank":
                events.append(Event("cloud_bank", s, int(rng.integers(4, 16)),
                                    list(rng.choice(solar_ids, int(rng.integers(2, 5)), replace=False)),
                                    float(rng.uniform(0.15, 0.5))))
            elif kind == "wind_surge":
                events.append(Event("wind_surge", s, int(rng.integers(4, 16)),
                                    list(rng.choice(wind_ids, int(rng.integers(1, 3)), replace=False)),
                                    float(rng.uniform(1.5, 2.5))))
            elif kind == "sensor_dropout":
                events.append(Event("sensor_dropout", s, int(rng.integers(2, 8)), ["price", "demand"]))
            elif kind == "forecast_stale":
                events.append(Event("forecast_stale", s, int(rng.integers(16, 32))))
    return events, sorted(docs, key=lambda d: d["at"])

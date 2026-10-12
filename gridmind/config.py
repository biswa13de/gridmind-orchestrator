"""Portfolio configuration: typed view over config/portfolio.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "portfolio.yaml"


@dataclass
class Generator:
    id: str
    name: str
    lat: float
    lon: float
    capacity_mw: float
    line: str
    kind: str  # "solar" | "wind"


@dataclass
class Battery:
    id: str
    energy_mwh: float
    power_mw: float
    eta_charge: float
    eta_discharge: float
    soc_min_pct: float
    soc_max_pct: float
    soc_init_pct: float
    degradation_inr_per_mwh: float

    @property
    def soc_min(self) -> float:
        return self.energy_mwh * self.soc_min_pct / 100

    @property
    def soc_max(self) -> float:
        return self.energy_mwh * self.soc_max_pct / 100


@dataclass
class Load:
    id: str
    name: str
    base_mw: float
    profile: str
    dr_max_mw: float
    shiftable_mw: float


@dataclass
class Portfolio:
    timestep_minutes: int
    horizon_steps: int
    scenarios: int
    solar: list[Generator]
    wind: list[Generator]
    wind_curve: dict
    batteries: list[Battery]
    loads: list[Load]
    lines: dict[str, float]
    grid: dict
    market: dict
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def dt(self) -> float:
        """Timestep in hours."""
        return self.timestep_minutes / 60

    @property
    def generators(self) -> list[Generator]:
        return self.solar + self.wind

    @property
    def dr_max_mw(self) -> float:
        return sum(l.dr_max_mw for l in self.loads)

    @property
    def shiftable_mw(self) -> float:
        return sum(l.shiftable_mw for l in self.loads)


def load_portfolio(path: str | Path = DEFAULT_CONFIG) -> Portfolio:
    raw = yaml.safe_load(Path(path).read_text())
    return Portfolio(
        timestep_minutes=raw["timestep_minutes"],
        horizon_steps=raw["horizon_steps"],
        scenarios=raw["scenarios"],
        solar=[Generator(kind="solar", **g) for g in raw["solar"]],
        wind=[Generator(kind="wind", **g) for g in raw["wind"]],
        wind_curve=raw["wind_curve"],
        batteries=[Battery(**b) for b in raw["batteries"]],
        loads=[Load(**l) for l in raw["loads"]],
        lines={l["id"]: l["capacity_mw"] for l in raw["lines"]},
        grid=raw["grid"],
        market=raw["market"],
        raw=raw,
    )

"""Build the 15-minute 'ground truth' dataset that drives the digital twin.

Real:       solar irradiance, wind speed, temperature per site (Open-Meteo archive, 2025).
Synthetic:  industrial demand, IEX-like day-ahead price, grid carbon intensity, grid frequency.
            Each synthetic series is generated from documented, seeded formulas (see DATA.md).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import ROOT, Portfolio, load_portfolio

WEATHER_DIR = ROOT / "data" / "weather"
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)

# IEX price band: the exchange caps bids at Rs 10/kWh (= 10,000 INR/MWh).
PRICE_CAP = 10_000.0
PRICE_FLOOR = 1_000.0


def _load_weather(asset_id: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(WEATHER_DIR / f"{asset_id}.csv", parse_dates=["time"])
    df["time"] = df["time"] + IST_OFFSET
    df = df.set_index("time").sort_index()
    idx = pd.date_range(df.index[0], df.index[-1], freq=freq)
    return df.reindex(df.index.union(idx)).interpolate("time").reindex(idx)


def solar_power(ghi: pd.Series, temp: pd.Series, capacity: float) -> pd.Series:
    """PV output from irradiance with a simple temperature derate and 0.82 performance ratio."""
    cell_temp = temp + ghi / 800 * 25
    derate = 1 - 0.004 * (cell_temp - 25)
    p = capacity * (ghi / 1000) * derate * 0.82
    return p.clip(0, capacity)


def wind_power(speed: pd.Series, capacity: float, curve: dict) -> pd.Series:
    """Cubic power curve between cut-in and rated speed; zero above cut-out."""
    ci, r, co = curve["cut_in"], curve["rated"], curve["cut_out"]
    frac = ((speed**3 - ci**3) / (r**3 - ci**3)).clip(0, 1)
    frac[speed >= co] = 0
    return capacity * frac


def _load_profile(kind: str, hour: np.ndarray) -> np.ndarray:
    if kind == "three_shift":   # near-flat with shift-change dips
        return 0.92 + 0.08 * np.cos(2 * np.pi * (hour - 14) / 8)
    if kind == "two_shift":     # 06–22 heavy, night light
        return np.where((hour >= 6) & (hour < 22), 1.0, 0.55)
    if kind == "day_shift":     # 08–20
        return np.where((hour >= 8) & (hour < 20), 1.0, 0.35)
    return np.ones_like(hour, dtype=float)  # flat


def build_dataset(portfolio: Portfolio | None = None, seed: int = 7) -> pd.DataFrame:
    """Return the full-year 15-min dataset (IST index). Deterministic for a given seed."""
    p = portfolio or load_portfolio()
    freq = f"{p.timestep_minutes}min"
    rng = np.random.default_rng(seed)
    cols: dict[str, pd.Series] = {}

    for g in p.solar:
        w = _load_weather(g.id, freq)
        cols[f"solar_{g.id}"] = solar_power(w["shortwave_radiation"], w["temperature_2m"], g.capacity_mw)
        cols[f"ghi_{g.id}"] = w["shortwave_radiation"]
    for g in p.wind:
        w = _load_weather(g.id, freq)
        cols[f"wind_{g.id}"] = wind_power(w["wind_speed_100m"], g.capacity_mw, p.wind_curve)
        cols[f"ws_{g.id}"] = w["wind_speed_100m"]

    df = pd.DataFrame(cols).dropna()
    idx = df.index
    hour = (idx.hour + idx.minute / 60).to_numpy()
    doy = idx.dayofyear.to_numpy()
    weekday = idx.dayofweek.to_numpy()
    n = len(idx)

    # --- Synthetic industrial demand: shift profile x weekly x seasonal x AR(1) noise -----------
    for ld in p.loads:
        shape = _load_profile(ld.profile, hour)
        weekly = np.where(weekday == 6, 0.85, 1.0)
        seasonal = 1 + 0.05 * np.sin(2 * np.pi * (doy - 120) / 365)
        noise = _ar1(n, rng, phi=0.97, sigma=0.012)
        df[f"load_{ld.id}"] = ld.base_mw * shape * weekly * seasonal * (1 + noise)
    df["demand"] = df[[f"load_{l.id}" for l in p.loads]].sum(axis=1)

    # --- Synthetic IEX-like price (INR/MWh) ---------------------------------------------------
    # Shape: overnight ~4.5k, solar-hour trough ~2.8k, evening peak 8-10k; summer premium;
    # inverse correlation with all-portfolio solar (proxy for national solar). AR(1) noise and
    # rare scarcity spikes to the exchange cap.
    solar_frac = df[[f"solar_{g.id}" for g in p.solar]].sum(axis=1).to_numpy() / sum(
        g.capacity_mw for g in p.solar
    )
    base = (
        4500
        - 1900 * solar_frac
        + 3800 * np.exp(-0.5 * ((hour - 20.0) / 1.8) ** 2)
        + 900 * np.exp(-0.5 * ((hour - 7.5) / 1.2) ** 2)
    )
    seasonal = 1 + 0.18 * np.exp(-0.5 * ((doy - 140) / 35) ** 2) + 0.08 * np.exp(-0.5 * ((doy - 290) / 20) ** 2)
    price = base * seasonal * (1 + _ar1(n, rng, phi=0.95, sigma=0.04))
    spikes = rng.random(n) < 0.002
    price[spikes] = PRICE_CAP
    df["price"] = np.clip(price, PRICE_FLOOR, PRICE_CAP)

    # --- Grid carbon intensity (tCO2/MWh): CEA all-India average ~0.71; lower when solar is high
    df["carbon"] = 0.78 - 0.16 * solar_frac + _ar1(n, rng, phi=0.98, sigma=0.004)

    # --- Grid frequency (Hz): nominal 50 with small AR noise; dips when price (scarcity) is high
    df["frequency"] = 50.0 - 0.04 * (df["price"].to_numpy() / PRICE_CAP) + _ar1(n, rng, phi=0.9, sigma=0.01)
    return df


def _ar1(n: int, rng: np.random.Generator, phi: float, sigma: float) -> np.ndarray:
    eps = rng.normal(0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


@lru_cache(maxsize=2)
def get_dataset(seed: int = 7) -> pd.DataFrame:
    return build_dataset(seed=seed)


def column_groups(p: Portfolio) -> dict[str, list[str]]:
    return {
        "solar": [f"solar_{g.id}" for g in p.solar],
        "wind": [f"wind_{g.id}" for g in p.wind],
        "load": [f"load_{l.id}" for l in p.loads],
    }

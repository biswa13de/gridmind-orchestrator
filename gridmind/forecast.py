"""Forecast agent: probabilistic (P10/P50/P90) forecasts + scenario sampling.

One gradient-boosted quantile model per signal family (solar, wind, demand, price) x quantile,
pooled across assets by normalising each series to its capacity. Features only use information
available at issue time (no leakage): current value, recent mean, same-time-yesterday,
same-time-two-days-ago, a 7-day clear-sky proxy, and calendar terms for the target time.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import ROOT, Portfolio

QUANTILES = (0.1, 0.5, 0.9)
TRAIN_LEADS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96])
DAY = 96  # steps per day at 15-min resolution
MODEL_PATH = ROOT / "models" / "forecaster.joblib"
_MODEL_CACHE: dict[str, tuple] = {}
TRAIN_END = pd.Timestamp("2025-09-01")  # Jan-Aug train, Sep conformal calibration
CALIB_END = pd.Timestamp("2025-10-01")  # Oct-Dec held out for evaluation


@dataclass
class Signal:
    column: str
    family: str
    scale: float


def signals_for(p: Portfolio) -> list[Signal]:
    sig = [Signal(f"solar_{g.id}", "solar", g.capacity_mw) for g in p.solar]
    sig += [Signal(f"wind_{g.id}", "wind", g.capacity_mw) for g in p.wind]
    sig += [Signal("demand", "demand", 200.0), Signal("price", "price", 10_000.0)]
    return sig


def _features(y: np.ndarray, hod: np.ndarray, doy: np.ndarray, i: int, leads: np.ndarray) -> np.ndarray:
    """Feature matrix for issue index i and the given lead times. y is normalised."""
    j = i + leads
    now = y[i]
    recent = y[max(0, i - 3): i + 1].mean()
    yday = y[j - DAY]
    yday2 = y[j - 2 * DAY]
    cs = np.max(np.stack([y[j - d * DAY] for d in range(1, 8)]), axis=0)
    cs_now = np.max([y[i - d * DAY] for d in range(1, 8)])
    clear_ratio = now / cs_now if cs_now > 0.02 else 1.0
    th = hod[i] + leads * 0.25  # target hour (may exceed 24; trig handles wrap)
    return np.column_stack([
        leads,
        np.sin(2 * np.pi * th / 24), np.cos(2 * np.pi * th / 24),
        np.full(len(leads), np.sin(2 * np.pi * doy[i] / 365)),
        np.full(len(leads), np.cos(2 * np.pi * doy[i] / 365)),
        np.full(len(leads), now), np.full(len(leads), recent),
        yday, yday2, cs, np.full(len(leads), clear_ratio),
    ])


class Forecaster:
    """Quantile forecaster. `fit` once (cached to models/), then `predict` per tick."""

    def __init__(self, portfolio: Portfolio):
        self.p = portfolio
        self.signals = signals_for(portfolio)
        self.models: dict[tuple[str, float], HistGradientBoostingRegressor] = {}
        self.calib: dict[str, float] = {}  # per-family band scale from split-conformal calibration
        self.adapt: dict[str, float] = {}  # online adaptive-conformal multiplier (updated in operation)

    # ---------------------------------------------------------------- training
    def fit(self, df: pd.DataFrame, train_end: pd.Timestamp = TRAIN_END, stride: int = 6) -> "Forecaster":
        hod, doy = _calendar(df.index)
        n_train = int(np.searchsorted(df.index, train_end))
        X_by_fam: dict[str, list] = {}
        y_by_fam: dict[str, list] = {}
        for s in self.signals:
            y = df[s.column].to_numpy() / s.scale
            for i in range(7 * DAY, n_train - TRAIN_LEADS[-1], stride):
                X_by_fam.setdefault(s.family, []).append(_features(y, hod, doy, i, TRAIN_LEADS))
                y_by_fam.setdefault(s.family, []).append(y[i + TRAIN_LEADS])
        for fam in X_by_fam:
            X = np.vstack(X_by_fam[fam])
            Y = np.concatenate(y_by_fam[fam])
            for q in QUANTILES:
                m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=150,
                                                  learning_rate=0.08, max_leaf_nodes=31, random_state=0)
                self.models[(fam, q)] = m.fit(X, Y)
        return self

    def calibrate(self, df: pd.DataFrame, start: pd.Timestamp = TRAIN_END, end: pd.Timestamp = CALIB_END,
                  target: float = 0.8, stride: int = 8) -> "Forecaster":
        """Split-conformal style: per family, find the band scale whose P10-P90 coverage on the
        calibration window hits `target`."""
        self.calib = {}
        i0, i1 = np.searchsorted(df.index, [start, end])
        recs: dict[str, list] = {}
        for i in range(int(i0), int(i1) - DAY, stride):
            fc = self.predict(df, i, DAY)
            for s in self.signals:
                actual = df[s.column].to_numpy()[i + 1: i + 1 + DAY]
                p10, p50, p90 = fc[s.column]
                m = actual > 0.01 * s.scale if s.family == "solar" else np.ones_like(actual, bool)
                recs.setdefault(s.family, []).append((actual[m], p10[m], p50[m], p90[m]))
        for fam, items in recs.items():
            a, lo, mid, hi = (np.concatenate(x) for x in zip(*items))
            best = 1.0
            for w in np.arange(0.5, 3.01, 0.05):
                cov = np.mean((a >= mid - (mid - lo) * w) & (a <= mid + (hi - mid) * w))
                best = w
                if cov >= target:
                    break
            self.calib[fam] = round(float(best), 2)
        return self

    def save(self, path=MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "calib": self.calib}, path, compress=3)

    @classmethod
    def load_or_fit(cls, portfolio: Portfolio, df: pd.DataFrame, path=MODEL_PATH) -> "Forecaster":
        f = cls(portfolio)
        if str(path) in _MODEL_CACHE:          # models are read-only: share them within a process
            f.models, f.calib = _MODEL_CACHE[str(path)]
            f.calib = dict(f.calib)
            return f
        if path.exists():
            blob = joblib.load(path)
            f.models, f.calib = blob["models"], blob["calib"]
        else:
            f.fit(df).calibrate(df).save(path)
        _MODEL_CACHE[str(path)] = (f.models, dict(f.calib))
        return f

    # ---------------------------------------------------------------- inference
    def predict(self, obs: pd.DataFrame, i: int, horizon: int, widen: float = 1.0) -> dict[str, np.ndarray]:
        """Return {column: array(3, horizon)} of P10/P50/P90 for steps i+1..i+horizon.

        `obs` is the observed history (may include injected events); only rows <= i are used.
        Step 0 of the plan (interval i) is the current nowcast and is not forecast here.
        """
        hod, doy = _calendar(obs.index)
        leads = np.arange(1, horizon + 1)
        out: dict[str, np.ndarray] = {}
        by_fam: dict[str, list[tuple[Signal, np.ndarray]]] = {}
        for s in self.signals:
            y = obs[s.column].to_numpy()[: i + 1] / s.scale
            y = np.concatenate([y, np.zeros(horizon)])  # placeholders; features never read them
            by_fam.setdefault(s.family, []).append((s, _features(y, hod, doy, i, leads)))
        for fam, items in by_fam.items():
            X = np.vstack([x for _, x in items])
            qs = np.stack([self.models[(fam, q)].predict(X) for q in QUANTILES])  # (3, n*H)
            qs = np.sort(qs, axis=0)  # no quantile crossing
            for k, (s, _) in enumerate(items):
                block = qs[:, k * horizon:(k + 1) * horizon]
                p50 = block[1]
                w = widen * self.calib.get(fam, 1.0) * self.adapt.get(fam, 1.0)
                lo = p50 - (p50 - block[0]) * w
                hi = p50 + (block[2] - p50) * w
                arr = np.stack([lo, p50, hi]) * s.scale
                lower = 0.0
                upper = s.scale if fam in ("solar", "wind") else np.inf
                out[s.column] = np.clip(arr, lower, upper)
        return out

    def observe(self, fc: dict[str, np.ndarray], actual: dict[str, np.ndarray], gamma: float = 0.15,
                target: float = 0.8) -> dict[str, float]:
        """Adaptive conformal inference (Gibbs & Candes, 2021): after outcomes for an earlier
        forecast are realised, widen bands for families that missed too often, tighten otherwise.
        `actual[col]` holds the realised values for the first n leads of `fc[col]`."""
        misses: dict[str, list] = {}
        for s in self.signals:
            a = actual.get(s.column)
            if a is None or len(a) == 0:
                continue
            p10, _, p90 = fc[s.column][:, : len(a)]
            m = a > 0.01 * s.scale if s.family == "solar" else np.ones_like(a, bool)
            if m.any():
                misses.setdefault(s.family, []).extend(((a[m] < p10[m]) | (a[m] > p90[m])).tolist())
        for fam, mm in misses.items():
            rate = float(np.mean(mm))
            cur = self.adapt.get(fam, 1.0)
            self.adapt[fam] = float(np.clip(cur * np.exp(gamma * (rate - (1 - target)) * 4), 0.5, 3.0))
        return dict(self.adapt)

    def sample_scenarios(self, fc: dict[str, np.ndarray], k: int, rng: np.random.Generator,
                         phi: float = 0.92, rho: float = 0.7) -> dict[str, np.ndarray]:
        """Draw k correlated trajectories per signal from the quantile bands.

        Errors are AR(1) across lead time and share a common factor within each family
        (cloud fronts hit several solar parks together). Returns {column: array(k, H)}.
        """
        H = next(iter(fc.values())).shape[1]
        fam_common = {f: _ar1_paths(k, H, rng, phi) for f in ("solar", "wind", "demand", "price")}
        out = {}
        for s in self.signals:
            z = rho * fam_common[s.family] + np.sqrt(1 - rho**2) * _ar1_paths(k, H, rng, phi)
            p10, p50, p90 = fc[s.column]
            up = (p90 - p50) / 1.2816
            dn = (p50 - p10) / 1.2816
            val = np.where(z >= 0, p50 + z * up, p50 + z * dn)
            upper = s.scale if s.family in ("solar", "wind") else np.inf
            out[s.column] = np.clip(val, 0, upper)
        return out


def _ar1_paths(k: int, H: int, rng: np.random.Generator, phi: float) -> np.ndarray:
    eps = rng.normal(size=(k, H))
    z = np.empty((k, H))
    z[:, 0] = eps[:, 0]
    s = np.sqrt(1 - phi**2)
    for t in range(1, H):
        z[:, t] = phi * z[:, t - 1] + s * eps[:, t]
    return z


def _calendar(idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    return (idx.hour + idx.minute / 60).to_numpy(), idx.dayofyear.to_numpy()


def evaluate_coverage(f: Forecaster, df: pd.DataFrame, start: pd.Timestamp, days: int = 30,
                      horizon: int = 96, stride: int = 16, adaptive: bool = True) -> pd.DataFrame:
    """Held-out P10-P90 coverage and P50 normalised MAE per family (optionally with online ACI)."""
    i0 = int(np.searchsorted(df.index, start))
    rows = []
    prev = None
    for i in range(i0, i0 + days * DAY, stride):
        if adaptive and prev is not None:
            pi, pfc = prev
            f.observe(pfc, {s.column: df[s.column].to_numpy()[pi + 1: i + 1] for s in f.signals})
        fc = f.predict(df, i, horizon)
        prev = (i, fc)
        for s in f.signals:
            actual = df[s.column].to_numpy()[i + 1: i + 1 + horizon]
            p10, p50, p90 = fc[s.column]
            mask = actual > 0.01 * s.scale if s.family == "solar" else np.ones_like(actual, bool)
            if not mask.any():
                continue
            rows.append({
                "family": s.family,
                "coverage": np.mean((actual[mask] >= p10[mask]) & (actual[mask] <= p90[mask])),
                "nmae": np.mean(np.abs(actual[mask] - p50[mask])) / s.scale,
            })
    return pd.DataFrame(rows).groupby("family").mean().round(3)

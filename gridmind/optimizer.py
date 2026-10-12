"""Optimizer tool: rolling-horizon, scenario-based MPC formulated as a linear program (HiGHS).

The LLM never outputs megawatts. The Strategist sets *what to optimise for* (weights, reserves,
banned actions); this module computes *how* — the dispatch for every asset — and only the first
15-minute step is executed (non-anticipativity: step 0 is shared by all scenarios).

Design note: we solve an LP rather than a MILP. Simultaneous charge/discharge and simultaneous
buy/sell are never optimal here because round-trip efficiency < 1, degradation cost > 0 and the
export price is discounted; the deterministic validator nets any residual overlap anyway.
That keeps a 96-step x K-scenario solve well under a second.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .config import Portfolio

OBJECTIVES = ("cost", "carbon", "curtailment", "degradation", "reliability")
BANNABLE = ("sell_to_grid", "buy_from_grid", "demand_response", "load_shift", "battery_discharge",
            "battery_charge")


@dataclass
class PlanInputs:
    """Everything the optimizer needs for one solve. Arrays are (K, H); step 0 is the nowcast."""
    gen: dict[str, np.ndarray]            # asset id -> available renewable MW
    demand: np.ndarray
    price: np.ndarray
    carbon: np.ndarray
    soc: dict[str, float]                 # battery id -> current MWh
    weights: dict[str, float]
    reserve_pct: dict[str, float] = field(default_factory=dict)  # battery id -> % of capacity held
    banned: tuple[str, ...] = ()
    gen_avail: dict[str, float] = field(default_factory=dict)    # asset -> 0..1 availability factor
    bess_avail: dict[str, bool] = field(default_factory=dict)
    line_caps: dict[str, float | np.ndarray] | None = None  # MW, scalar or per-step
    dr_allowed_mw: float = 0.0            # DR approved by the human operator
    freq_reserve_mw: float | None = None
    shift_balance_mwh: float = 0.0        # load already deferred (+) / pulled forward (-), must be settled


@dataclass
class Plan:
    status: str
    solve_ms: float
    objective: float
    step0: dict                           # first-step setpoints (the action cluster)
    expected: dict[str, np.ndarray]       # scenario-mean trajectories for display
    breakdown: dict[str, float]           # expected objective terms (INR)
    n_vars: int = 0


class _LP:
    """Tiny sparse LP builder: vectorised variable blocks and constraint rows."""

    def __init__(self):
        self.lb: list[np.ndarray] = []
        self.ub: list[np.ndarray] = []
        self.c: list[np.ndarray] = []
        self.n = 0
        self.eq = ([], [], [], [])   # rows, cols, vals, rhs
        self.ub_rows = ([], [], [], [])
        self.n_eq = 0
        self.n_ub = 0

    def var(self, shape, lb=0.0, ub=np.inf, cost=0.0) -> np.ndarray:
        size = int(np.prod(shape))
        idx = np.arange(self.n, self.n + size).reshape(shape)
        self.n += size
        self.lb.append(np.broadcast_to(np.asarray(lb, float), shape).ravel().copy())
        self.ub.append(np.broadcast_to(np.asarray(ub, float), shape).ravel().copy())
        self.c.append(np.broadcast_to(np.asarray(cost, float), shape).ravel().copy())
        return idx

    def _add(self, store, start, terms, rhs):
        rhs = np.atleast_1d(np.asarray(rhs, float))
        R = None
        for idx, coef in terms:
            idx = np.atleast_1d(idx).ravel()
            R = len(idx) if R is None else R
        rhs = np.broadcast_to(rhs, (R,)).copy()
        rows = np.arange(start, start + R)
        for idx, coef in terms:
            idx = np.atleast_1d(idx).ravel()
            store[0].append(rows)
            store[1].append(idx)
            store[2].append(np.broadcast_to(np.asarray(coef, float), (R,)).ravel())
        store[3].append(rhs)
        return R

    def add_eq(self, terms, rhs):
        self.n_eq += self._add(self.eq, self.n_eq, terms, rhs)

    def add_le(self, terms, rhs):
        self.n_ub += self._add(self.ub_rows, self.n_ub, terms, rhs)

    def solve(self):
        c = np.concatenate(self.c)
        bounds = np.column_stack([np.concatenate(self.lb), np.concatenate(self.ub)])

        def mat(store, nrows):
            if nrows == 0:
                return None, None
            m = coo_matrix((np.concatenate(store[2]), (np.concatenate(store[0]), np.concatenate(store[1]))),
                           shape=(nrows, self.n)).tocsr()
            return m, np.concatenate(store[3])

        A_eq, b_eq = mat(self.eq, self.n_eq)
        A_ub, b_ub = mat(self.ub_rows, self.n_ub)
        return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")


def normalise_weights(w: dict[str, float]) -> dict[str, float]:
    tot = sum(max(0.0, w.get(k, 0.0)) for k in OBJECTIVES) or 1.0
    return {k: max(0.0, w.get(k, 0.0)) / tot for k in OBJECTIVES}


def solve(p: Portfolio, x: PlanInputs) -> Plan:
    t0 = time.perf_counter()
    K, H = x.demand.shape
    dt = p.dt
    prob = 1.0 / K
    w = normalise_weights(x.weights)
    # Weight multipliers: equal weights (0.2 each) -> multiplier 1.0 on every term.
    m = {k: w[k] * len(OBJECTIVES) for k in OBJECTIVES}
    mk = p.market
    banned = set(x.banned)
    line_caps = dict(p.lines) if x.line_caps is None else x.line_caps

    lp = _LP()

    # A variable that is shared across scenarios at t=0 and per-scenario for t>=1.
    def staged(lb, ub, cost):
        """Returns index array (K, H) where column 0 is the same variable for every scenario.
        lb/ub/cost are (K, H) arrays; cost already includes scenario probability."""
        lb = np.broadcast_to(lb, (K, H))
        ub = np.broadcast_to(ub, (K, H))
        cost = np.broadcast_to(cost, (K, H))
        first = lp.var((1,), lb[0, 0], ub[0, 0], cost[:, 0].sum())
        rest = lp.var((K, H - 1), lb[:, 1:], ub[:, 1:], cost[:, 1:])
        return np.hstack([np.repeat(first[None, :], K, axis=0), rest])

    gens = p.generators
    use, gen_avail_arr = {}, {}
    curt_cost = m["curtailment"] * mk["curtailment_penalty_inr_per_mwh"] * dt * prob
    for g in gens:
        avail = x.gen[g.id] * x.gen_avail.get(g.id, 1.0)
        gen_avail_arr[g.id] = avail
        # curtailment = avail - use, so penalising curtailment == rewarding use (constant dropped)
        use[g.id] = staged(0.0, avail, -curt_cost)

    ch, dis, soc, short = {}, {}, {}, {}
    for b in p.batteries:
        up = x.bess_avail.get(b.id, True)
        pmax = b.power_mw if up else 0.0
        deg = m["degradation"] * b.degradation_inr_per_mwh * dt * prob
        ch[b.id] = staged(0.0, 0.0 if "battery_charge" in banned else pmax, deg)
        dis[b.id] = staged(0.0, 0.0 if "battery_discharge" in banned else pmax, deg)
        soc0 = float(np.clip(x.soc[b.id], b.soc_min, b.soc_max))
        soc[b.id] = staged(b.soc_min, b.soc_max, 0.0)
        # reserve floor is soft: shortfall penalised at VOLL-scale so it binds unless impossible
        floor = b.soc_min + b.energy_mwh * x.reserve_pct.get(b.id, 0.0) / 100
        short_cost = m["reliability"] * mk["voll_inr_per_mwh"] * 0.05 * dt * prob
        short[b.id] = staged(0.0, max(0.0, floor - b.soc_min), short_cost)
        # dynamics: soc_t - soc_{t-1} - eta_c*dt*ch_t + dt/eta_d*dis_t = 0
        lp.add_eq([(soc[b.id][:, 0], 1.0), (ch[b.id][:, 0], -b.eta_charge * dt),
                   (dis[b.id][:, 0], dt / b.eta_discharge)], soc0)
        lp.add_eq([(soc[b.id][:, 1:], 1.0), (soc[b.id][:, :-1], -1.0), (ch[b.id][:, 1:], -b.eta_charge * dt),
                   (dis[b.id][:, 1:], dt / b.eta_discharge)], 0.0)
        lp.add_le([(soc[b.id], -1.0), (short[b.id], -1.0)], -floor)  # soc + short >= floor
        # Terminal value: energy left in the battery is worth the expected late-horizon price.
        tv = float(np.median(x.price[:, H // 2:])) * b.eta_discharge * m["cost"] * prob
        _add_cost(lp, soc[b.id][:, -1], -tv)

    g = p.grid
    sell_price = x.price * g["sell_discount"]
    buy_cost = (m["cost"] * x.price + m["carbon"] * mk["carbon_price_inr_per_t"] * x.carbon) * dt * prob
    sell_rev = -(m["cost"] * sell_price) * dt * prob
    imp_lim = 0.0 if "buy_from_grid" in banned else g["import_limit_mw"]
    exp_lim = 0.0 if "sell_to_grid" in banned else g["export_limit_mw"]
    buy = staged(0.0, imp_lim, buy_cost)
    sell = staged(0.0, exp_lim, sell_rev)

    dr_cap = 0.0 if "demand_response" in banned else min(x.dr_allowed_mw, p.dr_max_mw)
    dr = staged(0.0, dr_cap, mk["dr_incentive_inr_per_mwh"] * dt * prob)
    sh_cap = 0.0 if "load_shift" in banned else p.shiftable_mw
    shift_cost = mk["load_shift_cost_inr_per_mwh"] * dt * prob
    sh_dn = staged(0.0, sh_cap, shift_cost)
    sh_up = staged(0.0, sh_cap, shift_cost)
    unserved = staged(0.0, np.inf, m["reliability"] * mk["voll_inr_per_mwh"] * dt * prob + 1.0)

    # Energy balance per (k, t)
    terms = [(use[gg.id], 1.0) for gg in gens]
    terms += [(dis[b.id], 1.0) for b in p.batteries] + [(ch[b.id], -1.0) for b in p.batteries]
    terms += [(buy, 1.0), (sell, -1.0), (dr, 1.0), (sh_dn, 1.0), (sh_up, -1.0), (unserved, 1.0)]
    _add_rows_kh(lp, terms, x.demand, eq=True)

    # Load shifting is energy-neutral: by the horizon end every deferred MWh has been served and
    # every pulled-forward MWh has been taken back (including the balance carried in from the past).
    # The running balance is bounded (at most 2 h of shiftable load outstanding at any time).
    bal_max = 2.0 * p.shiftable_mw
    bal = staged(-bal_max, bal_max, 0.0)
    lp.add_eq([(bal[:, 0], 1.0), (sh_dn[:, 0], -dt), (sh_up[:, 0], dt)], x.shift_balance_mwh)
    lp.add_eq([(bal[:, 1:], 1.0), (bal[:, :-1], -1.0), (sh_dn[:, 1:], -dt), (sh_up[:, 1:], dt)], 0.0)
    for k in range(K):
        _add_single_row(lp, [(bal[k, -1], 1.0)], 0.0)

    # Transmission corridors: renewable injections on each line <= capacity
    for line_id, cap in line_caps.items():
        on_line = [gg for gg in gens if gg.line == line_id]
        if on_line:
            rhs = np.broadcast_to(np.asarray(cap, float), (K, H)).copy()  # scalar or per-step (H,)
            _add_rows_kh(lp, [(use[gg.id], 1.0) for gg in on_line], rhs, eq=False)

    # Grid stability: keep upward battery headroom for primary frequency response
    fres = g["frequency_reserve_mw"] if x.freq_reserve_mw is None else x.freq_reserve_mw
    pmax_up = sum(b.power_mw for b in p.batteries if x.bess_avail.get(b.id, True))
    if pmax_up > 0 and fres > 0:
        _add_rows_kh(lp, [(dis[b.id], 1.0) for b in p.batteries],
                     np.full((K, H), max(0.0, pmax_up - fres)), eq=False)

    res = lp.solve()
    solve_ms = (time.perf_counter() - t0) * 1000
    if res.status != 0:
        return Plan(status=f"failed: {res.message}", solve_ms=solve_ms, objective=float("nan"),
                    step0={}, expected={}, breakdown={}, n_vars=lp.n)

    v = res.x

    def val(idx):
        return v[idx]

    step0 = {
        "use": {gg.id: float(val(use[gg.id])[0, 0]) for gg in gens},
        "curtail": {gg.id: float(gen_avail_arr[gg.id][0, 0] - val(use[gg.id])[0, 0]) for gg in gens},
        "charge": {b.id: float(val(ch[b.id])[0, 0]) for b in p.batteries},
        "discharge": {b.id: float(val(dis[b.id])[0, 0]) for b in p.batteries},
        "soc_next": {b.id: float(val(soc[b.id])[0, 0]) for b in p.batteries},
        "buy": float(val(buy)[0, 0]),
        "sell": float(val(sell)[0, 0]),
        "dr": float(val(dr)[0, 0]),
        "shift_down": float(val(sh_dn)[0, 0]),
        "shift_up": float(val(sh_up)[0, 0]),
        "unserved": float(val(unserved)[0, 0]),
    }
    mean = lambda idx: val(idx).mean(axis=0)  # noqa: E731
    expected = {
        "renewable_used": sum(mean(use[gg.id]) for gg in gens),
        "renewable_avail": sum(gen_avail_arr[gg.id].mean(axis=0) for gg in gens),
        "charge": sum(mean(ch[b.id]) for b in p.batteries),
        "discharge": sum(mean(dis[b.id]) for b in p.batteries),
        "buy": mean(buy), "sell": mean(sell), "dr": mean(dr), "unserved": mean(unserved),
        "demand": x.demand.mean(axis=0), "price": x.price.mean(axis=0),
        **{f"soc_{b.id}": mean(soc[b.id]) for b in p.batteries},
    }
    curt = sum((gen_avail_arr[gg.id] - val(use[gg.id])).sum() for gg in gens) * dt * prob
    breakdown = {
        "energy_cost_inr": float(((x.price * val(buy) - sell_price * val(sell)) * dt * prob).sum()),
        "carbon_t": float((x.carbon * val(buy) * dt * prob).sum()),
        "curtailment_mwh": float(curt),
        "degradation_inr": float(sum(((val(ch[b.id]) + val(dis[b.id])) * b.degradation_inr_per_mwh * dt * prob).sum()
                                     for b in p.batteries)),
        "unserved_mwh": float((val(unserved) * dt * prob).sum()),
        "dr_mwh": float((val(dr) * dt * prob).sum()),
    }
    return Plan(status="optimal", solve_ms=solve_ms, objective=float(res.fun), step0=step0,
                expected=expected, breakdown=breakdown, n_vars=lp.n)


def _add_cost(lp: _LP, idx: np.ndarray, coef: float) -> None:
    """Add `coef` to the objective coefficient of each variable in idx (handles shared t=0 vars)."""
    flat = np.concatenate(lp.c)
    np.add.at(flat, np.atleast_1d(idx).ravel(), coef)
    # rebuild blocks with the same sizes
    sizes = [len(c) for c in lp.c]
    lp.c = np.split(flat, np.cumsum(sizes)[:-1])


def _add_rows_kh(lp: _LP, terms, rhs: np.ndarray, eq: bool) -> None:
    """One row per (k, t). The shared t=0 column yields K identical rows; keep only k=0 for t=0."""
    K, H = rhs.shape
    sel_rows = np.ones((K, H), bool)
    sel_rows[1:, 0] = False
    flat_terms = [(np.asarray(idx)[sel_rows], np.broadcast_to(np.asarray(c, float), (K, H))[sel_rows] if np.ndim(c) else c)
                  for idx, c in terms]
    (lp.add_eq if eq else lp.add_le)(flat_terms, rhs[sel_rows])


def _add_single_row(lp: _LP, terms, rhs: float) -> None:
    """A single equality row summing over all given indices."""
    rows, cols, vals = [], [], []
    for idx, c in terms:
        idx = np.atleast_1d(idx).ravel()
        rows.append(np.full(len(idx), lp.n_eq))
        cols.append(idx)
        vals.append(np.broadcast_to(np.asarray(c, float), idx.shape).ravel())
    lp.eq[0].extend(rows)
    lp.eq[1].extend(cols)
    lp.eq[2].extend(vals)
    lp.eq[3].append(np.array([rhs]))
    lp.n_eq += 1

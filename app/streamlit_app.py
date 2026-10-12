"""GridMind operator console (Streamlit).

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from gridmind.llm import LLM  # noqa: E402
from gridmind.orchestrator import make_orchestrator  # noqa: E402
from gridmind.twin import EVENT_TYPES, Event  # noqa: E402

RESULTS = ROOT / "eval" / "results"
SAMPLES = ROOT / "data" / "samples"
# Validated categorical palette (fixed order; see dataviz reference palette)
C = {"renewable": "#1baf7a", "battery": "#2a78d6", "grid": "#eb6834", "dr": "#6250d6"}
INK, MUTED, GRID = "#52514e", "#8a8984", "rgba(128,128,128,0.18)"
SIT_STYLE = {"normal": ("🟢", "Normal"), "renewable_surplus": ("🟢", "Renewable surplus"),
             "price_spike": ("🟠", "Price spike"), "congestion": ("🟠", "Congestion"),
             "high_uncertainty": ("🟠", "High uncertainty"), "grid_stress": ("🔴", "Grid stress"),
             "asset_failure": ("🔴", "Asset failure"), "storm": ("🔴", "Storm")}

st.set_page_config(page_title="GridMind Orchestrator", page_icon="⚡", layout="wide")


# ----------------------------------------------------------------------------- state
def new_session(start: str, llm_mode: str):
    o = make_orchestrator(pd.Timestamp(start), mode="gridmind", approval_policy="manual", llm=LLM(mode=llm_mode),
                          log_path=None)
    st.session_state.o = o
    st.session_state.start = start
    st.session_state.playing = False


def load_preset(name: str, llm_mode: str):
    """Warm-started demo states (the clock is advanced through real control cycles, nothing is faked)."""
    if name == "storm":            # 22 Oct 2025: run 10:00 -> 13:00, a cloud bank passes at 11:00
        new_session("2025-10-22 10:00", llm_mode)
        o_ = st.session_state.o
        o_.run(4)
        o_.twin.inject(Event("cloud_bank", o_.twin.i, 8, ["SOL-1", "SOL-2", "SOL-3"], 0.25, note="preset"))
        o_.run(8)
    elif name == "storm_live":     # same, plus the IMD bulletin read and the storm under way
        load_preset("storm", llm_mode)
        o_ = st.session_state.o
        o_.submit_document(pdf_path=SAMPLES / "imd_storm_bulletin.pdf", name="imd_storm_bulletin.pdf")
        o_.run(1)
        for r in o_.approvals.pending():
            o_.approvals.decide(r.id, True)
        o_._apply_approvals(o_.twin.i)
        o_.run(15)
        o_.twin.inject(Event("battery_unresponsive", o_.twin.i, 40, ["BESS-2"], note="preset"))
        o_.twin.inject(Event("storm", o_.twin.i, 20, ["WND-1", "WND-2", "SOL-2"], note="preset"))
        o_.twin.inject(Event("demand_surge", o_.twin.i, 20, [], 20.0, note="preset"))
        o_.run(8)
    st.session_state.start = str(st.session_state.o.twin.now)


def orch():
    if "o" not in st.session_state:
        preset = st.query_params.get("preset")
        if preset:
            load_preset(preset, os.environ.get("GRIDMIND_LLM", "auto"))
        else:
            new_session("2025-10-22 12:00", os.environ.get("GRIDMIND_LLM", "auto"))
    return st.session_state.o


def fig_base(height=300, title=None):
    f = go.Figure()
    f.update_layout(height=height, margin=dict(l=10, r=10, t=60 if title else 30, b=10),
                    title=dict(text=title, y=0.98, yanchor="top") if title else None,
                    hovermode="x unified", legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12))
    f.update_xaxes(showgrid=False, linecolor=GRID)
    f.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=GRID)
    return f


# ----------------------------------------------------------------------------- sidebar
o = orch()
with st.sidebar:
    st.markdown("## ⚡ GridMind")
    st.caption("Agentic renewable-energy orchestrator · **Claim: F3 / D2**")
    llm_label = {"auto": "Claude (live)", "on": "Claude (live)", "replay": "Claude (cached replay)",
                 "off": "Offline · deterministic agents"}[o.llm.mode]
    st.markdown(f"**Agents:** {llm_label}")
    with st.expander("Scenario", expanded=False):
        start = st.text_input("Start (IST)", st.session_state.get("start", "2025-10-22 12:00"))
        mode = st.selectbox("LLM mode", ["auto", "off", "replay"], index=["auto", "off", "replay"].index(
            o.llm.mode if o.llm.mode in ("auto", "off", "replay") else "off"))
        if st.button("Reset simulation", width="stretch"):
            new_session(start, mode)
            st.rerun()
    if st.button("🎬 Load demo: storm day (13:00)", width="stretch"):
        with st.spinner("Running the morning's control cycles…"):
            load_preset("storm", o.llm.mode if o.llm.mode != "on" else "auto")
        st.rerun()
    c1, c2 = st.columns(2)
    step1 = c1.button("▶ 1 tick", width="stretch")
    step4 = c2.button("⏩ 1 hour", width="stretch")
    step24 = c1.button("⏭ 6 hours", width="stretch")
    play = c2.toggle("Auto-play", value=st.session_state.get("playing", False))
    st.session_state.playing = play
    speed = st.slider("Auto-play speed (s / tick)", 0.2, 2.0, 0.6, 0.1)

    st.markdown("### Inject event")
    etype = st.selectbox("Event", list(EVENT_TYPES), format_func=lambda k: k.replace("_", " "))
    st.caption(EVENT_TYPES[etype])
    p = o.p
    if etype in ("cloud_bank",):
        assets = st.multiselect("Solar parks", [g.id for g in p.solar], default=[g.id for g in p.solar][:3])
        value = st.slider("Remaining output", 0.0, 1.0, 0.25)
    elif etype == "wind_surge":
        assets = st.multiselect("Wind farms", [g.id for g in p.wind], default=[g.id for g in p.wind][:2])
        value = st.slider("Multiplier", 1.0, 3.0, 2.0)
    elif etype == "storm":
        assets = st.multiselect("Affected assets", [g.id for g in p.generators], default=["WND-2", "SOL-2", "WND-1"])
        value = 0.0
    elif etype in ("battery_outage", "battery_unresponsive"):
        assets = [st.selectbox("Battery", [b.id for b in p.batteries])]
        value = 0.0
    elif etype == "line_congestion":
        assets = [st.selectbox("Line", list(p.lines))]
        value = st.slider("Capacity (MW)", 0.0, float(p.lines[assets[0]]), 40.0)
    elif etype == "price_spike":
        assets, value = [], st.slider("Price multiplier", 1.2, 5.0, 2.5)
    elif etype == "demand_surge":
        assets, value = [], st.slider("Extra demand (MW)", 5.0, 50.0, 25.0)
    elif etype == "sensor_dropout":
        assets, value = st.multiselect("Signals", ["price", "demand", "frequency"], default=["price"]), 0.0
    else:
        assets, value = [], 0.0
    dur_h = st.slider("Duration (h)", 0.25, 12.0, 3.0, 0.25)
    if st.button("💥 Inject now", width="stretch", type="primary"):
        o.twin.inject(Event(etype, o.twin.i, int(dur_h * 4), assets, value, note="UI"))
        st.toast(f"Injected {etype} for {dur_h} h")

# ----------------------------------------------------------------------------- advance
n = 1 if step1 else 4 if step4 else 24 if step24 else 0
if n:
    with st.spinner(f"Running {n} control cycle(s)…"):
        o.run(n)
if play:
    o.tick()

log = o.log
last = log[-1] if log else None

# ----------------------------------------------------------------------------- header KPIs
summ = o.twin.summary()
st.markdown(f"### {o.twin.now:%a %d %b %Y · %H:%M} IST")
k = st.columns(7)
sit = last["situation"] if last else "normal"
icon, label = SIT_STYLE.get(sit or "normal", ("⚪", sit))
k[0].metric("Situation", f"{icon} {label}")
k[1].metric("Cost so far", f"₹{summ['total_cost_inr'] / 1e5:,.1f} L")
k[2].metric("Carbon", f"{summ['carbon_t']:,.0f} t")
k[3].metric("Renewable use", f"{summ['renewable_utilisation_pct']:.1f}%")
k[4].metric("Unserved", f"{summ['unserved_mwh']:.2f} MWh")
k[5].metric("Violations", summ["violations"])
k[6].metric("Pending approvals", len(o.approvals.pending()))

tabs = st.tabs(["🎛 Control room", "📨 Inbox & perception", "✅ Approvals", "🧾 Decision log",
                "🎲 Simulation (F3)", "📊 Evaluation (D2)", "🏗 Architecture"])

# ============================================================================= control room
with tabs[0]:
    if not last:
        st.info("Press **▶ 1 tick** (or Auto-play) to start the control loop. Then inject events from the sidebar "
                "or drop a storm bulletin in the Inbox tab.")
    else:
        left, right = st.columns([1, 1.6])
        with left:
            st.markdown("#### Action cluster (this 15-min interval)")
            for a in last["actions"]:
                st.markdown(f"- {a}")
            if last.get("recovery"):
                for r in last["recovery"]:
                    st.error(f"Recovery: {r}")
            if last.get("sensor_fallback"):
                st.warning(f"Sensor dropout on {', '.join(last['sensor_fallback'])}: using last-known-good values")
            if last.get("fallback"):
                st.error(last["fallback"])
            if last["repairs"]:
                with st.expander(f"Validator repairs ({len(last['repairs'])})"):
                    for r in last["repairs"]:
                        st.markdown(f"- {r}")
            s = last["strategy"]
            st.markdown("#### Strategist: optimality criteria")
            src = s["source"] + (" · reused (situation unchanged)" if s.get("reused") else "")
            st.caption(f"Source: **{src}** · confidence {s['confidence']:.2f}")
            st.info(s["rationale"])
            wf = fig_base(170)
            w = s["weights"]
            wf.add_bar(x=list(w.values()), y=[k.capitalize() for k in w], orientation="h", marker_color=C["battery"],
                       text=[f"{v:.0%}" for v in w.values()], textposition="outside", hovertemplate="%{y}: %{x:.0%}<extra></extra>")
            wf.update_xaxes(range=[0, 1.05], tickformat=".0%", showgrid=True, gridcolor=GRID)
            wf.update_layout(hovermode="closest", bargap=0.35)
            st.plotly_chart(wf, width="stretch", config={"displayModeBar": False})
            st.caption("Reserve floors: " + ", ".join(f"{b} {v:.0f}%" for b, v in s["reserves"].items())
                       + (" · Blocked: " + ", ".join(s["banned"]) if s["banned"] else ""))
            if s.get("rejections"):
                st.warning("Guardrails rejected LLM output: " + " | ".join(s["rejections"]))
            if last["plan"]:
                st.caption(f"Optimizer: {last['plan']['status']} · {last['plan']['n_vars']:,} variables · "
                           f"{last['plan']['solve_ms']:.0f} ms · {o.K} scenarios × {o.H} steps")
        with right:
            hist = pd.DataFrame(o.twin.history[-96:])
            hist["t"] = pd.to_datetime(hist["t"])
            plan, now = o.last_plan, o.twin.now
            f = fig_base(330, "Dispatch: last 24 h (actual) and next 24 h (planned, scenario mean)")
            f.add_scatter(x=hist.t, y=hist.renewable_used, name="Renewables", stackgroup="a",
                          line=dict(width=0.5, color=C["renewable"]), hovertemplate="%{y:.1f} MW")
            f.add_scatter(x=hist.t, y=hist.discharge, name="Battery", stackgroup="a",
                          line=dict(width=0.5, color=C["battery"]), hovertemplate="%{y:.1f} MW")
            f.add_scatter(x=hist.t, y=hist.buy, name="Grid import", stackgroup="a",
                          line=dict(width=0.5, color=C["grid"]), hovertemplate="%{y:.1f} MW")
            f.add_scatter(x=hist.t, y=-hist.charge, name="Battery charging", stackgroup="c",
                          line=dict(width=0.5, color=C["battery"]), fillcolor="rgba(42,120,214,0.35)",
                          hovertemplate="%{y:.1f} MW")
            f.add_scatter(x=hist.t, y=hist.demand, name="Demand", line=dict(color=INK, width=2),
                          hovertemplate="%{y:.1f} MW")
            if plan is not None and plan.expected:
                ts = pd.date_range(now - pd.Timedelta(minutes=15), periods=len(plan.expected["demand"]), freq="15min")
                e = plan.expected
                f.add_scatter(x=ts, y=e["renewable_used"], showlegend=False, stackgroup="b", opacity=0.35,
                              line=dict(width=0.5, color=C["renewable"], dash="dot"), hovertemplate="plan %{y:.1f} MW")
                f.add_scatter(x=ts, y=e["discharge"], showlegend=False, stackgroup="b",
                              line=dict(width=0.5, color=C["battery"], dash="dot"), hovertemplate="plan %{y:.1f} MW")
                f.add_scatter(x=ts, y=e["buy"], showlegend=False, stackgroup="b",
                              line=dict(width=0.5, color=C["grid"], dash="dot"), hovertemplate="plan %{y:.1f} MW")
                f.add_scatter(x=ts, y=-e["charge"], showlegend=False, stackgroup="d", fillcolor="rgba(42,120,214,0.18)",
                              line=dict(width=0.5, color=C["battery"], dash="dot"), hovertemplate="plan %{y:.1f} MW")
                f.add_scatter(x=ts, y=e["demand"], showlegend=False, line=dict(color=INK, width=1.5, dash="dot"),
                              hovertemplate="plan %{y:.1f} MW")
                f.add_vline(x=now, line=dict(color=MUTED, width=1, dash="dash"))
            f.update_yaxes(title="MW (charging below zero)")
            st.plotly_chart(f, width="stretch", config={"displayModeBar": False})

            c1, c2 = st.columns(2)
            sf = fig_base(220, "Battery state of charge")
            for b, col in zip(o.p.batteries, [C["battery"], C["dr"]]):
                y = 100 * hist[f"soc_{b.id}"] / b.energy_mwh
                sf.add_scatter(x=hist.t, y=y, name=b.id, line=dict(color=col, width=2), hovertemplate="%{y:.0f}%")
                if plan is not None and plan.expected:
                    sf.add_scatter(x=ts, y=100 * plan.expected[f"soc_{b.id}"] / b.energy_mwh, showlegend=False,
                                   line=dict(color=col, width=1.5, dash="dot"), hovertemplate="plan %{y:.0f}%")
            res = max(last["strategy"]["reserves"].values())
            sf.add_hline(y=10 + res, line=dict(color=MUTED, dash="dash", width=1),
                         annotation_text=f"reserve floor {10 + res:.0f}%", annotation_position="bottom left")
            sf.update_yaxes(range=[0, 100], ticksuffix="%")
            c1.plotly_chart(sf, width="stretch", config={"displayModeBar": False})
            pf = fig_base(220, "Market price (₹/MWh)")
            pf.add_scatter(x=hist.t, y=hist.price, name="Actual", line=dict(color=C["grid"], width=2),
                           hovertemplate="₹%{y:,.0f}")
            if o.last_fc is not None:
                p10, p50, p90 = o.last_fc["price"]
                tsf = pd.date_range(now, periods=len(p50), freq="15min")
                pf.add_scatter(x=tsf, y=p90, line=dict(width=0), showlegend=False, hoverinfo="skip")
                pf.add_scatter(x=tsf, y=p10, fill="tonexty", fillcolor="rgba(235,104,52,0.15)", line=dict(width=0),
                               name="P10–P90", hoverinfo="skip")
                pf.add_scatter(x=tsf, y=p50, name="P50 forecast", line=dict(color=C["grid"], width=1.5, dash="dot"),
                               hovertemplate="P50 ₹%{y:,.0f}")
            c2.plotly_chart(pf, width="stretch", config={"displayModeBar": False})

        st.markdown("#### Asset status")
        ex = last["executed"]
        rows = []
        tel = o.twin.telemetry()
        for g in o.p.generators:
            rows.append({"Asset": g.id, "Site": g.name, "Type": g.kind, "Output MW": round(ex["use"][g.id], 1),
                         "Capacity MW": g.capacity_mw, "Status": "maintenance" if o.twin.gen_avail[g.id] == 0 else "online",
                         "Line": g.line})
        for b in o.p.batteries:
            status = "FAULT" if b.id in o.known_faults else ("offline" if not tel["bess_avail"][b.id] else "online")
            pw = ex["discharge"][b.id] - ex["charge"][b.id]
            rows.append({"Asset": b.id, "Site": "Load centre", "Type": "battery", "Output MW": round(pw, 1),
                         "Capacity MW": b.power_mw, "Status": f"{status} · SOC {100 * ex['soc'][b.id] / b.energy_mwh:.0f}%",
                         "Line": "-"})
        for l, cap in tel["line_caps"].items():
            rows.append({"Asset": l, "Site": "Transmission", "Type": "line", "Output MW": None, "Capacity MW": cap,
                         "Status": "LIMITED" if cap < o.p.lines[l] else "normal", "Line": l})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ============================================================================= inbox
with tabs[1]:
    st.markdown("#### External documents → typed constraints")
    st.caption("Storm bulletins, grid notices and work orders are read by the Perception agent. Content is treated "
               "as **data, never instructions**; every constraint is validated before it can influence planning.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sample documents**")
        for f in sorted(SAMPLES.iterdir()):
            if st.button(f"📄 {f.name}", key=f"s_{f.name}", width="stretch"):
                if f.suffix == ".pdf":
                    r = o.submit_document(pdf_path=f, name=f.name)
                else:
                    r = o.submit_document(text=f.read_text(), name=f.name)
                st.session_state.last_perc = (f.name, r)
        up = st.file_uploader("Or upload a PDF / text notice", type=["pdf", "txt"])
        if up is not None and st.button("Read uploaded document"):
            tmp = ROOT / "cache" / up.name
            tmp.parent.mkdir(exist_ok=True)
            tmp.write_bytes(up.getvalue())
            r = o.submit_document(pdf_path=tmp, name=up.name) if up.name.endswith(".pdf") else \
                o.submit_document(text=up.getvalue().decode(errors="ignore"), name=up.name)
            st.session_state.last_perc = (up.name, r)
        txt = st.text_area("Or paste a free-text notice", placeholder="e.g. Cyclone warning for Tamil Nadu 18:00 to 23:00…")
        if st.button("Read notice") and txt.strip():
            st.session_state.last_perc = ("pasted notice", o.submit_document(text=txt, name="pasted notice"))
    with c2:
        if "last_perc" in st.session_state:
            name, r = st.session_state.last_perc
            st.markdown(f"**{name}** · parsed by **{r.source}**")
            if r.injection_flags:
                st.error(f"⚠️ Prompt-injection content detected ({len(r.injection_flags)} pattern(s)). It was treated "
                         "as data and ignored; no constraint or instruction was taken from it.")
            if r.constraints:
                st.success(f"{len(r.constraints)} constraint(s) accepted")
                st.dataframe(pd.DataFrame([{k: c[k] for k in ("kind", "assets", "start", "end", "value", "severity")}
                                           for c in r.constraints]), hide_index=True, width="stretch")
                for c in r.constraints:
                    st.caption(f"Evidence: “{c['evidence']}”")
            if r.rejected:
                st.warning("Rejected by validation:")
                st.dataframe(pd.DataFrame(r.rejected)[["kind", "assets", "error"]], hide_index=True)
        st.markdown("**Active constraints**")
        act = [c for c in o.constraints if c["end_idx"] > o.twin.i]
        if act:
            st.dataframe(pd.DataFrame([{"kind": c["kind"], "assets": ", ".join(c["assets"]), "from": c["start"][5:16],
                                        "to": c["end"][5:16], "value": c["value"], "source": c.get("document")}
                                       for c in act]), hide_index=True, width="stretch")
        else:
            st.caption("None")

# ============================================================================= approvals
with tabs[2]:
    st.markdown("#### Human-in-the-lead approval queue")
    st.caption("High-impact actions (demand response, maintenance scheduling or delay, crew dispatch) never execute "
               "without an operator decision.")
    pend = o.approvals.pending()
    if not pend:
        st.success("Nothing waiting for approval.")
    for r in pend:
        with st.container(border=True):
            st.markdown(f"**#{r.id} · {r.action.replace('_', ' ').title()}**: {r.detail}")
            st.caption(f"Requested {r.created} · Agent rationale: {r.rationale}")
            a, b, _ = st.columns([1, 1, 4])
            if a.button("Approve", key=f"ap{r.id}", type="primary"):
                o.approvals.decide(r.id, True)
                o._apply_approvals(o.twin.i)
                st.rerun()
            if b.button("Reject", key=f"rj{r.id}"):
                o.approvals.decide(r.id, False)
                st.rerun()
    done = [r for r in o.approvals.items if r.status != "pending"]
    if done:
        st.markdown("**History**")
        st.dataframe(pd.DataFrame([{"id": r.id, "action": r.action, "detail": r.detail, "status": r.status,
                                    "by": r.decided_by, "requested": r.created} for r in done]),
                     hide_index=True, width="stretch")

# ============================================================================= decision log
with tabs[3]:
    st.markdown("#### Decision log (one auditable record per control cycle)")
    if log:
        df = pd.DataFrame([{"time": r["t"], "situation": r["situation"], "strategy source": r["strategy"]["source"],
                            "cost w": r["strategy"]["weights"]["cost"], "reliability w": r["strategy"]["weights"]["reliability"],
                            "solve ms": r["plan"]["solve_ms"] if r["plan"] else None, "repairs": len(r["repairs"]),
                            "violations": len(r["violations"]), "actions": " · ".join(r["actions"][:4])}
                           for r in log[::-1]])
        st.dataframe(df, hide_index=True, width="stretch", height=320)
        pick = st.selectbox("Inspect record", [r["t"] for r in log[::-1]][:200])
        st.json(next(r for r in log if r["t"] == pick), expanded=False)
        st.download_button("Download log (JSONL)", "\n".join(json.dumps(r, default=str) for r in log),
                           file_name="gridmind_decision_log.jsonl")

# ============================================================================= simulation
with tabs[4]:
    st.markdown("#### Monte Carlo: strategies compared as distributions (F3)")
    runs_p, summ_p = RESULTS / "montecarlo_runs.csv", RESULTS / "montecarlo_summary.csv"
    if runs_p.exists():
        mc, ms = pd.read_csv(runs_p), pd.read_csv(summ_p)
        st.caption(f"{mc.run.nunique()} sampled worlds × {mc.strategy.nunique()} strategies. Each world = a random "
                   "held-out start day (real weather) + a random event schedule + forecast-error seed; every strategy "
                   "faces the identical world (common random numbers).")
        order = list(ms.strategy)
        cols = dict(zip(order, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"]))
        c1, c2 = st.columns(2)
        bf = fig_base(360, "Total cost per 48 h run (₹ lakh)")
        for s_ in order:
            bf.add_box(y=mc[mc.strategy == s_].total_cost_lakh, name=s_, marker_color=cols[s_], boxpoints="outliers",
                       line=dict(width=1.5))
        bf.update_layout(showlegend=False, hovermode="closest")
        c1.plotly_chart(bf, width="stretch", config={"displayModeBar": False})
        pf = fig_base(360, "Trade-off: mean cost vs mean carbon (bars = P5–P95 cost)")
        for _, r in ms.iterrows():
            pf.add_scatter(x=[r.carbon_mean_t], y=[r.cost_mean_lakh], mode="markers+text", name=r.strategy,
                           text=[r.strategy + (" ★" if r.pareto_efficient else "")], textposition="top center",
                           marker=dict(size=13, color=cols[r.strategy], line=dict(color="white", width=2)),
                           error_y=dict(type="data", symmetric=False, array=[r.cost_p95 - r.cost_mean_lakh],
                                        arrayminus=[r.cost_mean_lakh - r.cost_p5], color=cols[r.strategy], thickness=1.5),
                           hovertemplate=f"{r.strategy}<br>cost ₹%{{y:.1f}} L<br>carbon %{{x:.0f}} t"
                                         f"<br>P(unserved)={r.p_any_unserved:.0%}<extra></extra>")
        pf.update_layout(showlegend=False, hovermode="closest")
        span = ms.carbon_mean_t.max() - ms.carbon_mean_t.min() or 1.0
        pf.update_xaxes(title="Carbon (t CO₂)", showgrid=True, gridcolor=GRID,
                        range=[ms.carbon_mean_t.min() - 0.3 * span, ms.carbon_mean_t.max() + 0.3 * span])
        pf.update_yaxes(title="Cost (₹ lakh)")
        c2.plotly_chart(pf, width="stretch", config={"displayModeBar": False})
        st.caption("★ = Pareto-efficient on (cost, carbon, unserved energy). Choosing between ★ strategies is a "
                   "policy decision; GridMind makes that choice per situation and explains it.")
        show = ms.rename(columns={"cost_mean_lakh": "mean cost (₹L)", "cost_p5": "P5", "cost_p50": "P50",
                                  "cost_p95": "P95", "cost_cvar95": "CVaR95", "carbon_mean_t": "carbon (t)",
                                  "unserved_mean_mwh": "unserved (MWh)", "p_any_unserved": "P(any unserved)"})
        st.dataframe(show, hide_index=True, width="stretch")
    else:
        st.info("Run `python eval/montecarlo.py` to generate the Monte Carlo study.")

    st.markdown("#### Quick what-if (live)")
    st.caption("Simulate the next N hours from the current state under sampled input variations (forecast-error "
               "scenarios) for the current strategy vs. cost-first, without touching the live system.")
    qa, qb = st.columns(2)
    q_runs = qa.slider("Sampled futures", 3, 20, 6)
    q_hours = qb.slider("Hours", 2, 24, 6)
    if st.button("Run what-if"):
        from gridmind.montecarlo import STRATEGIES
        res = []
        prog = st.progress(0.0)
        for r_ in range(q_runs):
            for name in ("GridMind adaptive", "Cost-first"):
                sim = o.twin.snapshot()
                spec = STRATEGIES[name]
                so = make_orchestrator(sim.now, mode=spec["mode"], seed=1000 + r_, approval_policy="approve_all",
                                       scenarios=4, llm=LLM(mode="off"), profile=spec.get("profile"))
                so.twin = sim
                so.constraints = list(o.constraints)
                so.H = 48
                # sample a perturbed future: scale renewables / demand / price paths
                rng = np.random.default_rng(r_)
                sl = slice(sim.i, sim.i + q_hours * 4 + 100)
                for col in sim.obs.columns:
                    if col.startswith(("solar_", "wind_")):
                        sim.obs.iloc[sl, sim.obs.columns.get_loc(col)] *= rng.uniform(0.6, 1.3)
                sim.obs.iloc[sl, sim.obs.columns.get_loc("price")] *= rng.uniform(0.8, 1.4)
                c0 = sim.totals["cost_inr"]
                so.run(q_hours * 4)
                res.append({"future": r_, "strategy": name, "cost_lakh": (sim.totals["cost_inr"] - c0) / 1e5,
                            "unserved_mwh": sim.totals["unserved_mwh"]})
            prog.progress((r_ + 1) / q_runs)
        wf = pd.DataFrame(res)
        g = fig_base(260, f"Cost over the next {q_hours} h across {q_runs} sampled futures")
        for name, col in (("GridMind adaptive", "#6250d6"), ("Cost-first", "#2a78d6")):
            g.add_box(x=wf[wf.strategy == name].cost_lakh, name=name, marker_color=col, orientation="h")
        g.update_layout(showlegend=False, hovermode="closest")
        g.update_xaxes(title="₹ lakh", showgrid=True, gridcolor=GRID)
        st.plotly_chart(g, width="stretch", config={"displayModeBar": False})
        st.dataframe(wf.groupby("strategy")[["cost_lakh", "unserved_mwh"]].describe().round(2), width="stretch")

# ============================================================================= evaluation
with tabs[5]:
    st.markdown("#### Backtest on held-out data (D2: demonstrable reliability)")
    bt = RESULTS / "backtest_summary.csv"
    if bt.exists():
        st.dataframe(pd.read_csv(bt), hide_index=True, width="stretch")
        st.caption("30 held-out days (forecaster never saw Oct–Nov). 'stress' = same days + seeded storms, outages, "
                   "silent faults, price spikes, congestion, demand surges, sensor dropouts and stale forecasts. "
                   "'% of perfect savings' = share of the ceiling's saving over the rule-based baseline captured.")
    else:
        st.info("Run `python eval/backtest.py`.")
    st.markdown("#### Chaos suite")
    ch = RESULTS / "chaos.csv"
    if ch.exists():
        cdf = pd.read_csv(ch)
        st.metric("Passed", f"{(cdf.result == 'PASS').sum()} / {len(cdf)}")
        st.dataframe(cdf, hide_index=True, width="stretch")
    st.markdown("#### Forecast calibration (P10–P90 coverage, target 80%)")
    fcal = RESULTS / "forecast_calibration.csv"
    if fcal.exists():
        st.dataframe(pd.read_csv(fcal), hide_index=True, width="stretch")
    if o.f is not None and o.f.adapt:
        st.caption("Live adaptive-conformal multipliers: " + ", ".join(f"{k} ×{v:.2f}" for k, v in o.f.adapt.items()))

# ============================================================================= architecture
with tabs[6]:
    arch = ROOT / "docs" / "architecture.svg"
    if arch.exists():
        st.image(str(arch), width="stretch")
    claim = ROOT / "ARCHITECTURE.md"
    if claim.exists():
        st.markdown(claim.read_text())

if play:
    time.sleep(speed)
    st.rerun()

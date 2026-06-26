"""
Combinatorial Carbon Auction Clearing — interactive demo.

An end-user tool for an auction operator: configure a multi-vintage carbon-allowance
auction, then clear it and compare an exact optimiser, a fast greedy heuristic, and a
probabilistic sampler. Shows the welfare each method clears, the winning allocation, cap
utilisation, and a bench of alternative near-optimal clearings.

This demo uses ONLY classical solvers (exact branch-and-bound, greedy, simulated annealing). The
probabilistic-hardware backend (Quantum Dice ORBIT) plugs into the same QUBO interface
in the full project but is not bundled here.

Run:  streamlit run app.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from carbon_auction import (generate_instance, generate_real_instance, instance_stats,
                            build_qubo, exact_mip, greedy, simulated_annealing)

st.set_page_config(page_title="Carbon Auction Clearing", page_icon="🌍", layout="wide")

PALETTE = {"mip": "#1b4965", "greedy": "#9a7197", "sa": "#e08a3c", "accent": "#386641"}

PRESETS = {
    "Moderate auction": dict(kind="synthetic", N=40, J=25, pkg_max=1, cap_ratio=0.6, seed=3),
    "Package bids": dict(kind="synthetic", N=40, J=20, pkg_max=3, cap_ratio=0.6, seed=3),
    "Greedy trap (hard)": dict(kind="synthetic", N=45, J=14, pkg_max=4, cap_ratio=0.45, seed=161),
    "Real: EU/UK ETS": dict(kind="real", participants=16, vintages=14, cap_ratio=0.7, seed=1),
    "Custom": None,
}


@st.cache_data(show_spinner=False)
def make_instance(cfg: dict):
    if cfg["kind"] == "real":
        inst = generate_real_instance(n_participants=cfg["participants"], bids_per_participant=3,
                                      n_vintages=cfg["vintages"], cap_ratio=cfg["cap_ratio"],
                                      seed=cfg["seed"])
    else:
        inst = generate_instance(N=cfg["N"], J=cfg["J"], pkg_max=cfg["pkg_max"],
                                 cap_ratio=cfg["cap_ratio"], seed=cfg["seed"])
    return inst


@st.cache_data(show_spinner=False)
def clear_auction(cfg: dict, sa_reads: int, sa_sweeps: int):
    """Run the three classical clearing methods; return results dict."""
    inst = make_instance(cfg)
    m = exact_mip(inst)
    g = greedy(inst)
    q, info = build_qubo(inst, alpha_pack=3.0, alpha_cap=8.0)
    sa = simulated_annealing(q, inst, info, num_reads=sa_reads, num_sweeps=sa_sweeps, seed=7)
    opt = m["objective"] or 1.0
    # distinct near-optimal feasible allocations from SA reads
    feas = [r for r in sa["reads"] if r["feasible"]]
    distinct = {tuple(r["selected"]) for r in feas if r["objective"] >= 0.9 * opt}
    return dict(
        stats=instance_stats(inst), meta=inst.meta,
        mip=dict(obj=m["objective"], rt=m["runtime"], sel=m["selected"], status=m["status"]),
        greedy=dict(obj=g["objective"], rt=g["runtime"], sel=g["selected"]),
        sa=dict(obj=sa["objective"], rt=sa["runtime"], sel=sa["selected"],
                feas_objs=[r["objective"] for r in feas], n_distinct=len(distinct)),
        opt=opt, n_lots=inst.J,
        credits=inst.credits.tolist(), values=inst.values.tolist(), cap=inst.cap,
        coverage=[list(c) for c in inst.coverage],
    )


def fmt(x, real):
    return f"{x/1000:,.0f}k" if real else f"{x:,.0f}"


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Auction setup")
preset_name = st.sidebar.selectbox("Scenario", list(PRESETS.keys()), index=0)
cfg = dict(PRESETS[preset_name]) if PRESETS[preset_name] else None

if cfg is None:  # Custom
    kind = st.sidebar.radio("Dataset", ["synthetic", "real"], horizontal=True)
    if kind == "real":
        cfg = dict(kind="real",
                   participants=st.sidebar.slider("Participants", 6, 30, 16),
                   vintages=st.sidebar.slider("Vintages (lots)", 6, 24, 14),
                   cap_ratio=st.sidebar.slider("Cap (share of supply)", 0.3, 1.0, 0.7, 0.05),
                   seed=st.sidebar.number_input("Seed", 0, 9999, 1))
    else:
        cfg = dict(kind="synthetic",
                   N=st.sidebar.slider("Bids", 10, 80, 40),
                   J=st.sidebar.slider("Lots", 6, 40, 25),
                   pkg_max=st.sidebar.slider("Max package size", 1, 5, 1),
                   cap_ratio=st.sidebar.slider("Cap tightness", 0.3, 1.0, 0.6, 0.05),
                   seed=st.sidebar.number_input("Seed", 0, 9999, 3))

st.sidebar.markdown("---")
st.sidebar.caption("Probabilistic sampler (SA) settings")
sa_reads = st.sidebar.slider("Reads", 10, 100, 40, 10)
sa_sweeps = st.sidebar.slider("Sweeps per read", 200, 2000, 1000, 200)
real = cfg["kind"] == "real"

# ----------------------------------------------------------------------------- header
st.title("🌍 Combinatorial Carbon Auction Clearing")
st.markdown(
    "Clear a multi-vintage carbon-allowance auction with **package bids**, and compare three "
    "clearing engines: an **exact optimiser**, a **fast greedy heuristic**, and a **probabilistic "
    "sampler**. The exact optimiser sets the benchmark; the heuristic is fast but can leave welfare "
    "on the table; the sampler returns a *spread* of near-optimal clearings."
)

inst = make_instance(cfg)
s = instance_stats(inst)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Bids", s["N"])
c2.metric("Lots / vintages", s["J"])
c3.metric("Cap (allowances)", fmt(inst.cap, real))
c4.metric("Bid competition", f"{s['conflict_density']:.2f}", help="conflict density: 0=no overlap, 1=all bids compete")
if real:
    st.caption(f"Calibrated to EU/UK ETS: ~€{inst.meta['price_eur']}/tCO₂, "
               f"{inst.meta['n_participants']} participants, {inst.meta['auction_mtco2']} Mt offered.")

run = st.button("⚖️  Clear the auction", type="primary", use_container_width=True)

if run:
    st.session_state["res"] = clear_auction(cfg, sa_reads, sa_sweeps)

res = st.session_state.get("res")
if res is None:
    st.info("Set up the auction in the sidebar, then click **Clear the auction**.")
    st.stop()

opt = res["opt"]
mip, grd, sa = res["mip"], res["greedy"], res["sa"]

# ----------------------------------------------------------------------------- comparison
st.markdown("### Clearing results")
k1, k2, k3 = st.columns(3)
k1.metric("Exact optimiser", fmt(mip["obj"], real), f"{mip['rt']*1000:.0f} ms  ·  optimal")
k2.metric("Greedy heuristic", fmt(grd["obj"], real),
          f"{(grd['obj']/opt-1)*100:+.1f}% vs optimum", delta_color="normal")
k3.metric("Probabilistic sampler", fmt(sa["obj"], real),
          f"{(sa['obj']/opt-1)*100:+.1f}% vs optimum", delta_color="normal")

gap = opt - grd["obj"]
if gap > 0.001 * opt:
    st.warning(f"The fast greedy heuristic leaves **{fmt(gap, real)}** "
               f"({(1-grd['obj']/opt)*100:.0f}% of welfare) unclaimed versus the exact optimum — "
               f"a case where careful optimisation pays off.")

fig = go.Figure()
fig.add_hline(y=1.0, line_dash="dash", line_color=PALETTE["mip"], annotation_text="optimum")
fig.add_trace(go.Bar(x=["Greedy", "Sampler", "Exact"],
                     y=[grd["obj"]/opt, sa["obj"]/opt, 1.0],
                     marker_color=[PALETTE["greedy"], PALETTE["sa"], PALETTE["mip"]],
                     text=[f"{grd['obj']/opt:.2f}", f"{sa['obj']/opt:.2f}", "1.00"],
                     textposition="outside"))
fig.update_layout(yaxis_title="fraction of optimum", yaxis_range=[0, 1.12],
                  height=320, margin=dict(t=20, b=20), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------- allocation
left, right = st.columns([3, 2])
with left:
    st.markdown("#### Winning allocation (exact optimum)")
    rows = []
    for i in mip["sel"]:
        rows.append(dict(bid=i, value=round(res["values"][i], 1),
                         allowances=res["credits"][i],
                         lots=" ".join(map(str, res["coverage"][i]))))
    alloc = pd.DataFrame(rows).sort_values("value", ascending=False)
    st.dataframe(alloc, use_container_width=True, height=300, hide_index=True)
    used = sum(res["credits"][i] for i in mip["sel"])
    st.caption(f"{len(mip['sel'])} winning bids · {fmt(used, real)} / {fmt(res['cap'], real)} "
               f"allowances used ({used/res['cap']*100:.0f}% of cap)")

with right:
    st.markdown("#### Alternative near-optimal clearings")
    fo = np.array(res["sa"]["feas_objs"]) / opt
    if len(fo):
        h = go.Figure(go.Histogram(x=fo, nbinsx=20, marker_color=PALETTE["sa"]))
        h.add_vline(x=grd["obj"]/opt, line_color=PALETTE["accent"], line_width=2,
                    annotation_text="greedy")
        h.update_layout(height=260, margin=dict(t=10, b=10),
                        xaxis_title="clearing welfare / optimum", yaxis_title="clearings")
        st.plotly_chart(h, use_container_width=True)
    st.caption(f"The sampler found **{res['sa']['n_distinct']}** distinct clearings within 10% of the "
               f"optimum — alternatives an operator can use for tie-breaking, fairness, or robustness.")

st.markdown("---")
st.caption("Demo uses classical solvers only (exact branch-and-bound · greedy · simulated annealing). "
           "In the full project the same QUBO drives the Quantum Dice ORBIT probabilistic sampler.")

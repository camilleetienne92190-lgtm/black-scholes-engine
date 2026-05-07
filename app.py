"""
Options Pricing Engine — Streamlit Web App  (Bloomberg Terminal Theme)
======================================================================
Interactive dashboard wrapping five backend modules:

    options_engine.py  — Black-Scholes pricing + Greeks
    implied_vol.py     — IV solver (Newton-Raphson + Brentq)
    heston.py          — Heston stochastic vol model
    backtest.py        — Covered-call backtester (yfinance)
    strategies.py      — Multi-leg strategy P&L analyser

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Project path ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from options_engine import black_scholes, compute_greeks, price_option_pair  # noqa: E402
from implied_vol import implied_vol as iv_solver                              # noqa: E402
from heston import heston_price, validate_heston_inputs                       # noqa: E402
from backtest import run_backtest                                              # noqa: E402
from strategies import strategy_payoff, find_breakevens, net_premium, analyse # noqa: E402

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BS ENGINE | Options Analytics",
    page_icon="▸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# BLOOMBERG TERMINAL CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Import Courier-alike from Google Fonts as fallback ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* ── Root variables ─────────────────────────────────────── */
:root {
    --bg-app:     #0a0e14;
    --bg-sidebar: #111820;
    --bg-card:    #141c26;
    --bg-input:   #0d1520;
    --border:     #1f2d3d;
    --c-green:    #00d4aa;
    --c-green2:   #00a884;
    --c-cyan:     #00b8d9;
    --c-red:      #ff4d6a;
    --c-yellow:   #ffd166;
    --c-text:     #e8edf2;
    --c-sec:      #8fa3b8;
    --c-muted:    #4a6278;
    --font-mono:  'Courier New', 'Share Tech Mono', monospace;
}

/* ── App background ────────────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-app) !important;
    font-family: var(--font-mono) !important;
}

/* ── Header bar ─────────────────────────────────────────────── */
[data-testid="stHeader"] {
    background-color: #0a0e14 !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    font-family: var(--font-mono) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
    color: var(--c-sec) !important;
    font-size: 10px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: var(--c-green) !important;
    font-size: 13px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* ── Global text ──────────────────────────────────────────────── */
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp span {
    font-family: var(--font-mono) !important;
    color: var(--c-text) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0px !important;
    padding: 0 8px !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--c-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 18px !important;
    transition: all 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--c-green) !important;
    background-color: rgba(0,212,170,0.04) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--c-green) !important;
    border-bottom: 2px solid var(--c-green) !important;
    background-color: rgba(0,212,170,0.06) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background-color: var(--bg-app) !important;
    padding-top: 20px !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--c-green2) 0%, var(--c-cyan) 100%) !important;
    color: #000000 !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 8px 20px !important;
    transition: transform 0.1s, box-shadow 0.1s !important;
    box-shadow: 0 2px 8px rgba(0,212,170,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(0,212,170,0.45) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Number inputs ────────────────────────────────────────────── */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--c-text) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: var(--c-green) !important;
    box-shadow: 0 0 0 1px rgba(0,212,170,0.3) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--c-text) !important;
    font-family: var(--font-mono) !important;
}

/* ── Sliders ─────────────────────────────────────────────────── */
.stSlider > div > div > div {
    background: #1f2d3d !important;
}
.stSlider > div > div > div > div {
    background: #1f2d3d !important;
}
.stSlider [data-testid="stThumb"] {
    background-color: #00d4aa !important;
    border: 2px solid #00d4aa !important;
    width: 16px !important;
    height: 16px !important;
}
.stSlider [data-testid="stTrackFill"] {
    background: linear-gradient(90deg, #00a884, #00d4aa) !important;
}
.stSlider * {
    color: #e8edf2 !important;
}
[data-testid="stSidebar"] .stSlider > div {
    background: transparent !important;
}
[data-testid="stSidebar"] > div {
    background-color: #111820 !important;
}

/* ── Slider min/max tick labels ───────────────────────────── */
.stSlider [data-testid="stTickBar"] {
    color: #4a6278 !important;
}
.stSlider div[data-testid="stTickBarMin"],
.stSlider div[data-testid="stTickBarMax"] {
    color: #4a6278 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 10px !important;
}
.stSlider p {
    color: #4a6278 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 10px !important;
}

/* ── Progress bar ────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00a884, #00d4aa) !important;
}
.stProgress > div > div > div {
    background-color: #1f2d3d !important;
}
[data-testid="stProgressBar"] p {
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    color: #8fa3b8 !important;
}

/* ── Metric containers ──────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--c-green) !important;
    border-radius: 3px !important;
    padding: 12px 14px !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 9px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--c-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    color: var(--c-green) !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
}

/* ── DataFrames / tables ──────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stDataFrame"] th {
    background-color: var(--bg-card) !important;
    color: var(--c-muted) !important;
    font-size: 9px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="stDataFrame"] td {
    color: var(--c-text) !important;
    font-size: 11px !important;
    font-family: var(--font-mono) !important;
}

/* ── Info / warning / success / error boxes ──────────────────── */
[data-testid="stAlert"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
}
.stSuccess { border-left: 3px solid var(--c-green) !important; }
.stWarning { border-left: 3px solid var(--c-yellow) !important; }
.stError   { border-left: 3px solid var(--c-red) !important; }
.stInfo    { border-left: 3px solid var(--c-cyan) !important; }

/* ── Spinner ─────────────────────────────────────────────────── */
.stSpinner > div {
    border-color: var(--c-green) transparent transparent transparent !important;
}

/* ── Divider ─────────────────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    margin: 12px 0 !important;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-app); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-muted); }

/* ── Caption / small text ────────────────────────────────────── */
.stCaption, small, .stApp .stMarkdown p small {
    font-family: var(--font-mono) !important;
    color: var(--c-muted) !important;
    font-size: 10px !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TOP BANNER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
    background: linear-gradient(90deg, #111820 0%, #0d1520 60%, #0a0e14 100%);
    border-bottom: 1px solid #1f2d3d;
    border-left: 4px solid #00d4aa;
    padding: 14px 24px;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <div>
        <span style="
            font-family:'Courier New',monospace;
            font-size: 18px;
            font-weight: 700;
            color: #00d4aa;
            letter-spacing: 0.20em;
        ">◈ BLACK-SCHOLES ENGINE</span>
        <span style="
            font-family:'Courier New',monospace;
            font-size: 10px;
            color: #4a6278;
            letter-spacing: 0.12em;
            margin-left: 20px;
            text-transform: uppercase;
        ">OPTIONS ANALYTICS TERMINAL v2.0</span>
    </div>
    <div style="
        font-family:'Courier New',monospace;
        font-size: 10px;
        color: #4a6278;
        letter-spacing: 0.08em;
        text-align: right;
    ">
        BS · IV · HESTON · BACKTEST · STRATEGIES
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═════════════════════════════════════════════════════════════════════════════
C_GREEN  = "#00d4aa"
C_GREEN2 = "#00a884"
C_CYAN   = "#00b8d9"
C_RED    = "#ff4d6a"
C_YELLOW = "#ffd166"
C_TEXT   = "#e8edf2"
C_SEC    = "#8fa3b8"
C_MUTED  = "#4a6278"
C_BG     = "#0a0e14"
C_CARD   = "#141c26"
C_BORDER = "#1f2d3d"

# ─────────────────────────────────────────────────────────────────────────────
# Plotly BBG axis / hoverlabel defaults
# ─────────────────────────────────────────────────────────────────────────────
_BBG_AXIS = dict(
    gridcolor=C_BORDER,
    linecolor=C_BORDER,
    zerolinecolor="#2a3f54",
    tickfont=dict(family="Courier New, monospace", color=C_SEC, size=10),
    title_font=dict(family="Courier New, monospace", color=C_SEC, size=11),
)
_BBG_HOVER = dict(
    bgcolor=C_CARD,
    bordercolor=C_GREEN,
    font=dict(family="Courier New, monospace", color=C_TEXT, size=11),
)


def _layout(**kw) -> dict:
    """Return a Bloomberg-themed Plotly layout dict."""
    base = dict(
        paper_bgcolor=C_BG,
        plot_bgcolor="#0d1117",
        font=dict(family="Courier New, monospace", color=C_SEC, size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(20,28,38,0.9)", bordercolor=C_BORDER, borderwidth=1,
            font=dict(family="Courier New, monospace", size=10, color=C_SEC),
        ),
        hoverlabel=_BBG_HOVER,
        xaxis=_BBG_AXIS,
        yaxis=_BBG_AXIS,
        margin=dict(l=52, r=20, t=40, b=44),
    )
    base.update(kw)
    return base


def _apply_bbg_axes(fig: go.Figure) -> go.Figure:
    """Apply BBG axis styling to all axes in a figure (for subplots)."""
    fig.update_xaxes(**_BBG_AXIS)
    fig.update_yaxes(**_BBG_AXIS)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI component helpers
# ─────────────────────────────────────────────────────────────────────────────
def bbg_header(title: str, subtitle: str = "") -> None:
    """Render a Bloomberg-style tab header with green left border."""
    sub_html = (f'<div style="font-family:\'Courier New\',monospace;font-size:9px;'
                f'letter-spacing:0.14em;color:{C_MUTED};margin-top:3px;'
                f'text-transform:uppercase;">{subtitle}</div>') if subtitle else ""
    st.markdown(f"""
    <div style="
        border-left: 4px solid {C_GREEN};
        padding: 6px 0 6px 14px;
        margin-bottom: 18px;
    ">
        <div style="
            font-family:'Courier New',monospace;
            font-size:16px;
            font-weight:700;
            color:{C_GREEN};
            letter-spacing:0.18em;
            text-transform:uppercase;
        ">{title}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str) -> None:
    """Thin muted uppercase section divider with ── prefix."""
    st.markdown(f"""
    <div style="
        display:flex;align-items:center;gap:8px;
        margin:20px 0 8px 0;
        border-bottom:1px solid {C_BORDER};
        padding-bottom:5px;
    ">
        <span style="
            color:{C_MUTED};
            font-family:'Courier New',monospace;
            font-size:9px;
            letter-spacing:0.18em;
            text-transform:uppercase;
        ">── {text}</span>
    </div>
    """, unsafe_allow_html=True)


def stat_card(label: str, value: str, color: str = C_GREEN,
              border_color: str = C_GREEN) -> str:
    """Return HTML for a stat card (render with st.markdown unsafe_allow_html)."""
    return f"""
    <div style="
        background:{C_CARD};
        border:1px solid {C_BORDER};
        border-left:3px solid {border_color};
        border-radius:3px;
        padding:10px 14px;
        margin:4px 0;
    ">
        <div style="
            font-family:'Courier New',monospace;font-size:9px;
            letter-spacing:0.15em;color:{C_MUTED};
            text-transform:uppercase;margin-bottom:5px;
        ">{label}</div>
        <div style="
            font-family:'Courier New',monospace;font-size:20px;
            font-weight:700;color:{color};
        ">{value}</div>
    </div>"""


def quote_box(call_p: float, put_p: float, S: float, K: float) -> str:
    """Return HTML for the sidebar live quote box."""
    mono = S / K
    if abs(mono - 1.0) < 0.02:
        mono_label, mono_color = "ATM", C_YELLOW
    elif mono > 1.0:
        mono_label, mono_color = "CALL ITM", C_GREEN
    else:
        mono_label, mono_color = "CALL OTM", C_RED
    return f"""
    <div style="
        background:{C_CARD};border:1px solid {C_BORDER};
        border-radius:3px;padding:12px;margin-top:10px;
    ">
        <div style="font-family:'Courier New',monospace;font-size:9px;
            letter-spacing:0.15em;color:{C_MUTED};margin-bottom:8px;">
            ── LIVE QUOTE
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
            <span style="font-family:'Courier New',monospace;font-size:9px;
                letter-spacing:0.1em;color:{C_MUTED};text-transform:uppercase;">CALL</span>
            <span style="font-family:'Courier New',monospace;font-size:16px;
                font-weight:700;color:{C_GREEN};">${call_p:.4f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="font-family:'Courier New',monospace;font-size:9px;
                letter-spacing:0.1em;color:{C_MUTED};text-transform:uppercase;">PUT</span>
            <span style="font-family:'Courier New',monospace;font-size:16px;
                font-weight:700;color:{C_RED};">${put_p:.4f}</span>
        </div>
        <div style="border-top:1px solid {C_BORDER};padding-top:6px;text-align:center;">
            <span style="font-family:'Courier New',monospace;font-size:10px;
                font-weight:700;color:{mono_color};letter-spacing:0.1em;">{mono_label}</span>
        </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Black-Scholes pricer parameters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ◈ BS PRICER")
    st.caption("Controls Tab 1 — BS Pricer & Greeks")
    S_s   = st.slider("Spot Price  S",         50.0, 300.0, 100.0,  1.0)
    K_s   = st.slider("Strike Price  K",        50.0, 300.0, 105.0,  1.0)
    T_s   = st.slider("Maturity T (years)", 0.01,   2.0,   1.0, 0.01)
    r_s   = st.slider("Risk-free Rate  r",       0.0,   0.1,  0.05, 0.001, format="%.3f")
    sig_s = st.slider("Volatility  σ",          0.05,   1.0,  0.20, 0.01,  format="%.2f")

    # ── Live quote box ────────────────────────────────────────────────────────
    try:
        _r = price_option_pair(S_s, K_s, T_s, r_s, sig_s)
        st.markdown(
            quote_box(_r.call.price, _r.put.price, S_s, K_s),
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    st.divider()
    st.markdown(f"""
    <div style="font-family:'Courier New',monospace;font-size:9px;
        color:{C_MUTED};letter-spacing:0.1em;text-transform:uppercase;
        text-align:center;">
        Black-Scholes Options<br>Pricing Engine v2.0
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "▸  BS PRICER & GREEKS",
    "▸  IV SOLVER",
    "▸  HESTON vs BS",
    "▸  BACKTEST",
    "▸  STRATEGIES",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BS Pricer & Greeks
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    bbg_header("BS PRICER & GREEKS",
               "Black-Scholes analytical pricing · all 5 Greeks · Vega surface")

    try:
        result = price_option_pair(S_s, K_s, T_s, r_s, sig_s)
        call_r = result.call
        put_r  = result.put

        # ── Prices ────────────────────────────────────────────────────────────
        section_label("OPTION PRICES")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Call Price",         f"${call_r.price:.4f}")
        pc2.metric("Put Price",          f"${put_r.price:.4f}")
        parity_ok = result.parity_error < 1e-8
        pc3.metric(
            "Put-Call Parity Error",
            f"{result.parity_error:.2e}",
            delta="VALID" if parity_ok else "CHECK INPUTS",
            delta_color="normal" if parity_ok else "inverse",
        )

        # ── Greeks table ──────────────────────────────────────────────────────
        section_label("GREEKS")
        cg = call_r.greeks
        pg = put_r.greeks
        greeks_df = pd.DataFrame({
            "GREEK": ["DELTA", "GAMMA", "VEGA", "THETA", "RHO"],
            "CALL":  [f"{cg.delta:.4f}", f"{cg.gamma:.4f}", f"{cg.vega:.4f}",
                      f"{cg.theta:.4f}", f"{cg.rho:.4f}"],
            "PUT":   [f"{pg.delta:.4f}", f"{pg.gamma:.4f}", f"{pg.vega:.4f}",
                      f"{pg.theta:.4f}", f"{pg.rho:.4f}"],
        })
        st.dataframe(greeks_df, use_container_width=True, hide_index=True)

        # ── Delta & Gamma vs Spot ─────────────────────────────────────────────
        section_label("DELTA & GAMMA VS SPOT")
        S_lo    = max(1.0, S_s * 0.50)
        S_hi    = S_s * 1.50
        S_range = np.linspace(S_lo, S_hi, 300)

        call_deltas, put_deltas, gammas = [], [], []
        for s in S_range:
            cg_ = compute_greeks(s, K_s, T_s, r_s, sig_s, "call")
            pg_ = compute_greeks(s, K_s, T_s, r_s, sig_s, "put")
            call_deltas.append(cg_.delta)
            put_deltas.append(pg_.delta)
            gammas.append(cg_.gamma)

        fig1 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["DELTA VS SPOT", "GAMMA VS SPOT"],
        )
        fig1.layout.annotations[0].update(
            font=dict(family="Courier New, monospace", color=C_MUTED, size=10))
        fig1.layout.annotations[1].update(
            font=dict(family="Courier New, monospace", color=C_MUTED, size=10))

        fig1.add_trace(go.Scatter(
            x=S_range, y=call_deltas, name="CALL DELTA",
            line=dict(color=C_GREEN, width=2),
        ), row=1, col=1)
        fig1.add_trace(go.Scatter(
            x=S_range, y=put_deltas, name="PUT DELTA",
            line=dict(color=C_RED, width=2),
        ), row=1, col=1)
        fig1.add_vline(x=K_s, line_dash="dash", line_color=C_YELLOW, line_width=1,
                       annotation_text=f"K={K_s:.0f}",
                       annotation_font=dict(family="Courier New", color=C_YELLOW, size=10),
                       row=1, col=1)
        fig1.add_trace(go.Scatter(
            x=S_range, y=gammas, name="GAMMA",
            line=dict(color=C_CYAN, width=2),
        ), row=1, col=2)
        fig1.add_vline(x=K_s, line_dash="dash", line_color=C_YELLOW, line_width=1,
                       row=1, col=2)
        fig1.update_xaxes(title_text="SPOT ($)")
        fig1.update_yaxes(title_text="DELTA", row=1, col=1)
        fig1.update_yaxes(title_text="GAMMA", row=1, col=2)
        fig1.update_layout(**_layout(height=380))
        _apply_bbg_axes(fig1)
        st.plotly_chart(fig1, use_container_width=True)

        # ── Vega heatmap (Spot × Volatility) ─────────────────────────────────
        section_label("VEGA HEATMAP  ·  SPOT × VOLATILITY")
        S_heat   = np.linspace(max(1.0, S_s * 0.60), S_s * 1.40, 55)
        sig_heat = np.linspace(0.05, 0.80, 45)
        vega_grid = np.zeros((len(sig_heat), len(S_heat)))
        for i, sv in enumerate(sig_heat):
            for j, sp in enumerate(S_heat):
                try:
                    vega_grid[i, j] = compute_greeks(
                        sp, K_s, T_s, r_s, sv, "call").vega
                except Exception:
                    vega_grid[i, j] = 0.0

        fig_heat = go.Figure(go.Heatmap(
            x=S_heat,
            y=(sig_heat * 100).tolist(),
            z=vega_grid,
            colorscale=[
                [0.00, "#0a0e14"],
                [0.15, "#0d2234"],
                [0.40, "#00354a"],
                [0.70, "#006b7a"],
                [0.85, "#00a884"],
                [1.00, "#00d4aa"],
            ],
            colorbar=dict(
                tickfont=dict(family="Courier New", color=C_SEC, size=9),
                title=dict(
                    text="VEGA",
                    font=dict(family="Courier New", color=C_SEC, size=10),
                    side="right",
                ),
                bgcolor=C_BG,
                bordercolor=C_BORDER,
                borderwidth=1,
                thickness=12,
            ),
            hovertemplate=(
                "S = %{x:.1f}<br>"
                "σ = %{y:.1f}%<br>"
                "Vega = %{z:.3f}<extra></extra>"
            ),
            zsmooth="best",
        ))
        fig_heat.add_vline(
            x=K_s, line_color=C_YELLOW, line_dash="dash", line_width=1.5,
            annotation_text=f"K={K_s:.0f}",
            annotation_font=dict(family="Courier New", color=C_YELLOW, size=10),
        )
        fig_heat.add_vline(
            x=S_s, line_color=C_GREEN, line_dash="dot", line_width=1.2,
            annotation_text=f"S={S_s:.0f}",
            annotation_font=dict(family="Courier New", color=C_GREEN, size=10),
        )
        fig_heat.update_layout(**_layout(
            height=360,
            xaxis_title="SPOT PRICE ($)",
            yaxis_title="VOLATILITY (%)",
        ))
        _apply_bbg_axes(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)

    except ValueError as exc:
        st.error(f"Pricing error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IV Solver
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    bbg_header("IMPLIED VOLATILITY SOLVER",
               "Newton-Raphson primary · Brentq fallback · 3D IV surface")

    left2, right2 = st.columns([1, 1])

    with left2:
        section_label("INPUTS")
        iv_price  = st.number_input("Market Option Price ($)",    0.01, 99999.0,   8.50, 0.01)
        iv_S      = st.number_input("Spot Price (S)",              1.0, 99999.0, 100.0,  1.0, key="iv_S")
        iv_K      = st.number_input("Strike Price (K)",            1.0, 99999.0, 105.0,  1.0, key="iv_K")
        iv_T      = st.number_input("Maturity T (years)",           0.01,     5.0,   1.0, 0.01, key="iv_T")
        iv_r      = st.number_input("Risk-free Rate (r)",          0.0,     1.0,  0.05, 0.001,
                                    format="%.3f", key="iv_r")
        iv_type   = st.selectbox("Option Type", ["call", "put"], key="iv_type")
        solve_btn = st.button("◈ SOLVE IV", type="primary", key="solve_btn")

    with right2:
        section_label("RESULT")
        if solve_btn:
            try:
                ivr = iv_solver(iv_price, iv_S, iv_K, iv_T, iv_r, iv_type)
                r1, r2 = st.columns(2)
                r1.metric("Implied Volatility",
                          f"{ivr.implied_vol:.4f}  ({ivr.implied_vol:.2%})")
                r2.metric("Method", ivr.method)
                r3, r4 = st.columns(2)
                r3.metric("Iterations",  str(ivr.iterations))
                r4.metric("Price Error", f"{ivr.price_error:.2e}")
                if ivr.converged:
                    st.success("Converged successfully")
                else:
                    st.warning("Did not fully converge — check inputs")
            except ValueError as exc:
                st.error(f"IV solver error: {exc}")
        else:
            st.info("Enter parameters and click ◈ SOLVE IV")

    # ── IV Smile 2D ───────────────────────────────────────────────────────────
    section_label("IV SMILE  ·  STRIKES 80–120")
    try:
        K_smile   = np.linspace(80.0, 120.0, 41)
        true_vols = 0.20 + 0.15 * ((K_smile / iv_S) - 1.0) ** 2
        smile_ivs = []
        for Ks, sv in zip(K_smile, true_vols):
            try:
                mp  = black_scholes(iv_S, Ks, iv_T, iv_r, sv, iv_type)
                res = iv_solver(mp, iv_S, Ks, iv_T, iv_r, iv_type, sigma_init=sv)
                smile_ivs.append(res.implied_vol * 100)
            except ValueError:
                smile_ivs.append(None)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=K_smile, y=true_vols * 100, name="TRUE VOL SURFACE",
            mode="lines", line=dict(color=C_GREEN, width=2.5),
        ))
        fig2.add_trace(go.Scatter(
            x=K_smile, y=smile_ivs, name="RECOVERED IV",
            mode="markers",
            marker=dict(color=C_CYAN, size=7, symbol="circle",
                        line=dict(color=C_BORDER, width=1)),
        ))
        fig2.add_hline(y=20.0, line_dash="dash", line_color=C_MUTED, line_width=1,
                       annotation_text="FLAT BS 20%",
                       annotation_font=dict(family="Courier New", color=C_MUTED, size=9))
        fig2.add_vline(x=iv_S, line_dash="dot", line_color=C_YELLOW, line_width=1,
                       annotation_text=f"S={iv_S:.0f}",
                       annotation_font=dict(family="Courier New", color=C_YELLOW, size=9))
        fig2.update_layout(**_layout(
            height=340,
            xaxis_title="STRIKE K ($)",
            yaxis_title="IMPLIED VOLATILITY (%)",
        ))
        _apply_bbg_axes(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as exc:
        st.error(f"Smile plot error: {exc}")

    # ── 3D IV Surface ─────────────────────────────────────────────────────────
    section_label("IV SURFACE (3D)  ·  STRIKE × TIME")
    try:
        K_3d = np.linspace(max(iv_S * 0.70, 1.0), iv_S * 1.30, 22)
        T_3d = np.linspace(0.08, 2.0, 18)
        iv_surface = np.zeros((len(T_3d), len(K_3d)))
        for i, t_val in enumerate(T_3d):
            for j, k_val in enumerate(K_3d):
                # Quadratic smile + term-structure tilt
                true_vol = (0.20
                            + 0.15 * ((k_val / iv_S) - 1.0) ** 2
                            + 0.03 * (1.0 / max(t_val, 0.1) - 0.5))
                true_vol = float(np.clip(true_vol, 0.05, 0.99))
                try:
                    mp  = black_scholes(iv_S, k_val, t_val, iv_r, true_vol, iv_type)
                    res = iv_solver(mp, iv_S, k_val, t_val, iv_r, iv_type,
                                   sigma_init=true_vol)
                    iv_surface[i, j] = res.implied_vol * 100
                except Exception:
                    iv_surface[i, j] = 20.0

        fig_3d = go.Figure(go.Surface(
            x=K_3d,
            y=T_3d,
            z=iv_surface,
            colorscale=[
                [0.00, "#0d2234"],
                [0.25, "#004d5e"],
                [0.55, "#00a884"],
                [0.80, "#00d4aa"],
                [1.00, "#e8edf2"],
            ],
            contours=dict(
                z=dict(show=True, usecolormap=True,
                       highlightcolor=C_YELLOW, project_z=True),
            ),
            opacity=0.92,
            hovertemplate=(
                "K = %{x:.1f}<br>"
                "T = %{y:.2f} yr<br>"
                "IV = %{z:.2f}%<extra></extra>"
            ),
            colorbar=dict(
                tickfont=dict(family="Courier New", color=C_SEC, size=9),
                title=dict(
                    text="IV %",
                    font=dict(family="Courier New", color=C_SEC, size=10),
                    side="right",
                ),
                bgcolor=C_BG,
                bordercolor=C_BORDER,
                thickness=12,
            ),
        ))
        fig_3d.update_layout(
            paper_bgcolor=C_BG,
            plot_bgcolor=C_BG,
            font=dict(family="Courier New, monospace", color=C_SEC, size=10),
            hoverlabel=_BBG_HOVER,
            scene=dict(
                bgcolor=C_BG,
                xaxis=dict(title="Strike ($)",      backgroundcolor=C_BG, gridcolor=C_BORDER),
                yaxis=dict(title="Maturity (yr)",   backgroundcolor=C_BG, gridcolor=C_BORDER),
                zaxis=dict(title="IV (%)",          backgroundcolor=C_BG, gridcolor=C_BORDER),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
            ),
            height=480,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as exc:
        st.error(f"3D surface error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Heston vs Black-Scholes
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    bbg_header("HESTON STOCHASTIC VOLATILITY",
               "Semi-analytical CF formula · Albrecher et al. 2007 · IV skew vs flat BS")

    h_left, h_right = st.columns([1, 2])

    with h_left:
        section_label("PARAMETERS")
        h_S       = st.number_input("Spot Price (S)",     1.0, 9999.0, 100.0,  1.0, key="h_S")
        h_K       = st.number_input("Strike Price (K)",   1.0, 9999.0, 100.0,  1.0, key="h_K")
        h_T       = st.number_input("Maturity T (years)",  0.01,   5.0,   1.0, 0.01, key="h_T")
        h_r       = st.number_input("Risk-free Rate (r)", 0.0,    1.0,  0.05, 0.001,
                                    format="%.3f", key="h_r")
        section_label("HESTON PARAMS")
        h_v0      = st.slider("v₀  initial variance",   0.01, 0.50, 0.04, 0.01, key="h_v0")
        h_kappa   = st.slider("κ  mean reversion",       0.10, 10.0, 2.00, 0.10, key="h_kappa")
        h_theta   = st.slider("θ  long-run variance",   0.01, 0.50, 0.04, 0.01, key="h_theta")
        h_sigma_v = st.slider("σv vol of vol",          0.05, 1.00, 0.30, 0.05, key="h_sigma_v")
        h_rho     = st.slider("ρ  spot-vol corr",      -0.99, 0.99, -0.70, 0.01, key="h_rho")

        feller    = 2 * h_kappa * h_theta
        feller_ok = feller >= h_sigma_v ** 2
        if feller_ok:
            st.success(f"FELLER OK  2κθ={feller:.3f} ≥ σv²={h_sigma_v**2:.3f}")
        else:
            st.warning(f"FELLER VIOLATED  2κθ={feller:.3f} < σv²={h_sigma_v**2:.3f}")

    with h_right:
        try:
            sigma_bs = float(np.sqrt(h_theta))

            section_label("ATM PRICING COMPARISON")
            with st.spinner("Pricing ATM options..."):
                h_call  = heston_price(h_S, h_K, h_T, h_r, h_v0, h_kappa,
                                       h_theta, h_sigma_v, h_rho, "call")
                h_put   = heston_price(h_S, h_K, h_T, h_r, h_v0, h_kappa,
                                       h_theta, h_sigma_v, h_rho, "put")
                bs_call = black_scholes(h_S, h_K, h_T, h_r, sigma_bs, "call")
                bs_put  = black_scholes(h_S, h_K, h_T, h_r, sigma_bs, "put")

            cmp_df = pd.DataFrame({
                "MODEL":      ["BLACK-SCHOLES", "HESTON", "DIFFERENCE"],
                "CALL PRICE": [f"${bs_call:.4f}", f"${h_call:.4f}",
                               f"${h_call - bs_call:+.4f}"],
                "PUT PRICE":  [f"${bs_put:.4f}",  f"${h_put:.4f}",
                               f"${h_put - bs_put:+.4f}"],
            })
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            section_label("IV SKEW  ·  HESTON VS FLAT BS  ·  STRIKES 75%–125% OF S")
            K_range = np.linspace(h_S * 0.75, h_S * 1.25, 31)
            heston_ivs: list[float | None] = []

            _progress = st.progress(0, text="⬡ Initializing Heston engine...")
            _status   = st.empty()
            _total    = len(K_range)

            for _idx, Kh in enumerate(K_range):
                _pct = int((_idx / _total) * 100)
                _progress.progress(
                    _pct,
                    text=f"⬡ Computing characteristic functions... strike {_idx + 1}/{_total}",
                )
                try:
                    hp   = heston_price(h_S, Kh, h_T, h_r, h_v0, h_kappa,
                                        h_theta, h_sigma_v, h_rho, "call")
                    ivr2 = iv_solver(hp, h_S, Kh, h_T, h_r, "call",
                                     sigma_init=sigma_bs)
                    heston_ivs.append(ivr2.implied_vol * 100)
                except Exception:
                    heston_ivs.append(None)

            _progress.progress(100, text="⬡ Done")
            _progress.empty()
            _status.empty()

            bs_flat  = [sigma_bs * 100] * len(K_range)
            valid    = [v for v in heston_ivs if v is not None]
            K_valid  = K_range[[v is not None for v in heston_ivs]]
            iv_diff  = [(h - sigma_bs * 100) if h is not None else None
                        for h in heston_ivs]
            bar_col  = [C_GREEN if (d is not None and d >= 0) else C_RED
                        for d in iv_diff]

            fig3 = make_subplots(
                rows=2, cols=1, row_heights=[0.68, 0.32],
                shared_xaxes=True, vertical_spacing=0.06,
                subplot_titles=["IMPLIED VOLATILITY SMILE",
                                "IV DIFFERENCE (HESTON − FLAT BS, PP)"],
            )
            for ann in fig3.layout.annotations:
                ann.update(font=dict(family="Courier New, monospace",
                                     color=C_MUTED, size=10))

            fig3.add_trace(go.Scatter(
                x=K_valid, y=valid, name="HESTON IV",
                mode="lines+markers",
                line=dict(color=C_GREEN, width=2),
                marker=dict(size=5, color=C_GREEN,
                            line=dict(color=C_BORDER, width=1)),
            ), row=1, col=1)
            fig3.add_trace(go.Scatter(
                x=K_range, y=bs_flat, name=f"FLAT BS ({sigma_bs:.0%})",
                mode="lines",
                line=dict(color=C_RED, width=2, dash="dash"),
            ), row=1, col=1)
            fig3.add_vline(x=h_S, line_dash="dot", line_color=C_YELLOW, line_width=1,
                           annotation_text=f"ATM {h_S:.0f}",
                           annotation_font=dict(family="Courier New",
                                                color=C_YELLOW, size=9))
            fig3.add_trace(go.Bar(
                x=K_range, y=iv_diff, name="IV DIFF (PP)",
                marker_color=bar_col, opacity=0.85,
            ), row=2, col=1)
            fig3.add_hline(y=0, line_color=C_MUTED, line_width=1, row=2, col=1)
            fig3.update_xaxes(title_text="STRIKE K ($)", row=2, col=1)
            fig3.update_yaxes(title_text="IV (%)",       row=1, col=1)
            fig3.update_yaxes(title_text="DIFF (PP)",    row=2, col=1)
            fig3.update_layout(**_layout(height=520))
            _apply_bbg_axes(fig3)
            st.plotly_chart(fig3, use_container_width=True)

        except ValueError as exc:
            st.error(f"Heston error: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Backtest
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    bbg_header("COVERED-CALL BACKTEST",
               "5% OTM call sold every 30 trading days · 30-day realised HV · r=5%")

    bt_ticker = st.text_input("Ticker", value="SPY", key="bt_ticker").upper().strip()
    run_bt    = st.button("▶ RUN BACKTEST", type="primary", key="run_bt")

    if run_bt:
        try:
            with st.spinner(f"Downloading {bt_ticker} data and running backtest..."):
                trades = run_backtest(bt_ticker)

            if not trades:
                st.warning("No valid trades found. Check ticker or try a different symbol.")
            else:
                total_pnl   = sum(t.pnl      for t in trades)
                total_prem  = sum(t.premium  for t in trades)
                n_exercised = sum(t.exercised for t in trades)
                win_rate    = sum(1 for t in trades if t.pnl >= 0) / len(trades)
                avg_vol     = float(np.mean([t.sigma for t in trades]))

                section_label("SUMMARY")
                m1, m2, m3, m4, m5 = st.columns(5)
                pnl_color = "normal" if total_pnl >= 0 else "inverse"
                m1.metric("Total P&L",       f"${total_pnl:+.2f}")
                m2.metric("Total Premium",   f"${total_prem:.2f}")
                m3.metric("Total Legs",      str(len(trades)))
                m4.metric("Exercised",       f"{n_exercised}/{len(trades)}")
                m5.metric("Win Rate",        f"{win_rate:.0%}")
                _, mv, _ = st.columns([1, 1, 3])
                mv.metric("Avg Realised Vol", f"{avg_vol:.1%}")

                # ── Cumulative P&L chart ──────────────────────────────────────
                section_label("P&L OVER TIME")
                dates_exp  = [pd.Timestamp(t.expiry_date) for t in trades]
                dates_ent  = [pd.Timestamp(t.entry_date)  for t in trades]
                pnls       = [t.pnl     for t in trades]
                prems      = [t.premium for t in trades]
                cum_pnl    = list(np.cumsum(pnls))
                cum_prem   = list(np.cumsum(prems))
                dot_colors = [C_RED    if t.exercised else C_GREEN for t in trades]
                # Per-leg bars: green if profit, red if loss
                bar_colors = [C_RED if p < 0 else C_GREEN for p in pnls]

                fig4 = make_subplots(
                    rows=2, cols=1, row_heights=[0.68, 0.32],
                    shared_xaxes=True, vertical_spacing=0.06,
                    subplot_titles=["CUMULATIVE P&L", "PER-LEG P&L"],
                )
                for ann in fig4.layout.annotations:
                    ann.update(font=dict(family="Courier New, monospace",
                                         color=C_MUTED, size=10))

                fig4.add_trace(go.Scatter(
                    x=dates_exp, y=cum_pnl, name="CUMULATIVE P&L",
                    mode="lines", line=dict(color=C_CYAN, width=2.5),
                    fill="tozeroy",
                    fillcolor=f"rgba(0,184,217,0.08)",
                ), row=1, col=1)
                fig4.add_trace(go.Scatter(
                    x=dates_exp, y=cum_prem, name="CUMULATIVE PREMIUM",
                    mode="lines",
                    line=dict(color=C_GREEN, width=1.5, dash="dash"),
                ), row=1, col=1)
                fig4.add_trace(go.Scatter(
                    x=dates_exp, y=cum_pnl, name="TRADE OUTCOMES",
                    mode="markers",
                    marker=dict(color=dot_colors, size=7, symbol="circle",
                                line=dict(color=C_BORDER, width=1)),
                ), row=1, col=1)
                fig4.add_hline(y=0, line_color=C_MUTED, line_width=1, row=1, col=1)

                # Per-leg bars: green/red by sign
                fig4.add_trace(go.Bar(
                    x=dates_ent, y=pnls, name="LEG P&L",
                    marker_color=bar_colors,
                    marker_line=dict(color=C_BORDER, width=0.5),
                    opacity=0.88,
                ), row=2, col=1)
                fig4.add_hline(y=0, line_color=C_MUTED, line_width=1, row=2, col=1)
                fig4.update_xaxes(title_text="DATE", row=2, col=1)
                fig4.update_yaxes(title_text="CUMULATIVE P&L ($)", row=1, col=1)
                fig4.update_yaxes(title_text="LEG P&L ($)",        row=2, col=1)
                fig4.update_layout(**_layout(height=540))
                _apply_bbg_axes(fig4)
                st.plotly_chart(fig4, use_container_width=True)

                # ── Trades table ──────────────────────────────────────────────
                section_label("TRADE-BY-TRADE DETAIL")
                trades_df = pd.DataFrame([{
                    "ENTRY":    str(t.entry_date),
                    "EXPIRY":   str(t.expiry_date),
                    "S ENTRY":  f"${t.S_entry:.2f}",
                    "STRIKE K": f"${t.K:.2f}",
                    "SIGMA":    f"{t.sigma:.1%}",
                    "PREMIUM":  f"${t.premium:.4f}",
                    "S EXPIRY": f"${t.S_expiry:.2f}",
                    "OUTCOME":  "EXERCISED" if t.exercised else "EXPIRED",
                    "P&L":      f"${t.pnl:+.2f}",
                } for t in trades])
                st.dataframe(trades_df, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Backtest error: {exc}")
    else:
        st.info("Enter a ticker and click ▶ RUN BACKTEST")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Options Strategies
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    bbg_header("OPTIONS STRATEGIES",
               "Multi-leg payoff at expiry · BS-priced premiums · breakeven analysis")

    st5_left, st5_right = st.columns([1, 2])

    with st5_left:
        section_label("PARAMETERS")
        strat_name = st.selectbox("Strategy", [
            "Long Straddle",
            "Long Strangle",
            "Iron Condor",
            "Bull Call Spread",
        ], key="strat_name")
        st5_S   = st.slider("Spot Price (S)",           50.0, 300.0, 100.0,  1.0, key="st5_S")
        st5_sig = st.slider("Volatility (σ)",           0.05,   1.0,  0.20, 0.01, key="st5_sig",
                            format="%.2f")
        st5_T   = st.slider("Maturity T (years)",       0.01,   2.0,  0.25, 0.01, key="st5_T")
        st5_r   = st.slider("Risk-free Rate (r)",        0.0,   0.1,  0.05, 0.001, key="st5_r",
                            format="%.3f")

    with st5_right:
        try:
            ratio = st5_S / 100.0

            def _prem(K: float, opt: str) -> float:
                return black_scholes(st5_S, K, st5_T, st5_r, st5_sig, opt)

            K_atm = round(st5_S,        2)
            K_105 = round(105 * ratio,  2)
            K_95  = round(95  * ratio,  2)
            K_110 = round(110 * ratio,  2)
            K_115 = round(115 * ratio,  2)
            K_90  = round(90  * ratio,  2)
            K_85  = round(85  * ratio,  2)

            all_strats: dict[str, list[dict]] = {
                "Long Straddle": [
                    {"type": "call", "K": K_atm, "qty": +1, "premium": _prem(K_atm, "call")},
                    {"type": "put",  "K": K_atm, "qty": +1, "premium": _prem(K_atm, "put")},
                ],
                "Long Strangle": [
                    {"type": "call", "K": K_105, "qty": +1, "premium": _prem(K_105, "call")},
                    {"type": "put",  "K": K_95,  "qty": +1, "premium": _prem(K_95,  "put")},
                ],
                "Iron Condor": [
                    {"type": "call", "K": K_110, "qty": -1, "premium": _prem(K_110, "call")},
                    {"type": "call", "K": K_115, "qty": +1, "premium": _prem(K_115, "call")},
                    {"type": "put",  "K": K_90,  "qty": -1, "premium": _prem(K_90,  "put")},
                    {"type": "put",  "K": K_85,  "qty": +1, "premium": _prem(K_85,  "put")},
                ],
                "Bull Call Spread": [
                    {"type": "call", "K": K_atm, "qty": +1, "premium": _prem(K_atm, "call")},
                    {"type": "call", "K": K_110, "qty": -1, "premium": _prem(K_110, "call")},
                ],
            }

            legs          = all_strats[strat_name]
            strategy_dict = {"name": strat_name, "legs": legs}

            S_lo_r  = max(1.0, st5_S * 0.70)
            S_hi_r  = st5_S * 1.30
            S_range = np.linspace(S_lo_r, S_hi_r, 600)

            ana      = analyse(strategy_dict, S_range)
            pnl      = ana["pnl"]
            net_prem = ana["net_prem"]
            bes      = ana["breakevens"]

            # ── Summary ───────────────────────────────────────────────────────
            section_label("STRATEGY SUMMARY")
            sm1, sm2 = st.columns(2)
            sm1.metric("Net Premium",
                       f"${net_prem:+.4f}  ({'DEBIT' if net_prem > 0 else 'CREDIT'})")
            sm2.metric("Breakeven(s)",
                       " / ".join(f"${b:.2f}" for b in bes) if bes else "NONE")
            sm3, sm4 = st.columns(2)
            sm3.metric("Max Profit", ana["max_profit_str"])
            sm4.metric("Max Loss",   ana["max_loss_str"])

            s_dn  = round(st5_S * 0.90, 2)
            s_up  = round(st5_S * 1.10, 2)
            p_dn  = float(np.interp(s_dn,  S_range, pnl))
            p_mid = float(np.interp(st5_S, S_range, pnl))
            p_up  = float(np.interp(s_up,  S_range, pnl))
            sp1, sp2, sp3 = st.columns(3)
            sp1.metric(f"P&L at S={s_dn:.0f} (−10%)",   f"${p_dn:+.2f}")
            sp2.metric(f"P&L at S={st5_S:.0f}",          f"${p_mid:+.2f}")
            sp3.metric(f"P&L at S={s_up:.0f} (+10%)",    f"${p_up:+.2f}")

            # ── P&L diagram — split profit (green) / loss (red) traces ────────
            section_label("P&L DIAGRAM")
            pnl_pos = np.where(pnl >= 0, pnl, 0.0)
            pnl_neg = np.where(pnl < 0,  pnl, 0.0)

            fig5 = go.Figure()

            # Green profit fill
            fig5.add_trace(go.Scatter(
                x=S_range, y=pnl_pos,
                name="PROFIT ZONE",
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.14)",
            ))
            # Red loss fill
            fig5.add_trace(go.Scatter(
                x=S_range, y=pnl_neg,
                name="LOSS ZONE",
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor="rgba(255,77,106,0.14)",
            ))
            # Main P&L line (cyan, on top)
            fig5.add_trace(go.Scatter(
                x=S_range, y=pnl,
                name="P&L",
                mode="lines",
                line=dict(color=C_CYAN, width=2.5),
            ))

            fig5.add_hline(y=0, line_color=C_MUTED, line_dash="dash", line_width=1)
            for be in bes:
                fig5.add_vline(
                    x=be, line_color=C_YELLOW, line_dash="dash", line_width=1.2,
                    annotation_text=f"BE ${be:.2f}",
                    annotation_font=dict(family="Courier New", color=C_YELLOW, size=10),
                )
            fig5.add_vline(
                x=st5_S, line_color=C_MUTED, line_dash="dot", line_width=1,
                annotation_text=f"S=${st5_S:.0f}",
                annotation_font=dict(family="Courier New", color=C_MUTED, size=9),
            )
            fig5.update_layout(**_layout(
                height=400,
                xaxis_title="SPOT PRICE AT EXPIRY ($)",
                yaxis_title="P&L ($)",
                title=dict(
                    text=(f"{strat_name.upper()}  ·  "
                          f"NET {'DEBIT' if net_prem > 0 else 'CREDIT'} "
                          f"${abs(net_prem):.4f}"),
                    font=dict(family="Courier New, monospace",
                              color=C_GREEN, size=12),
                ),
            ))
            _apply_bbg_axes(fig5)
            st.plotly_chart(fig5, use_container_width=True)

            # ── Leg breakdown ──────────────────────────────────────────────────
            section_label("LEG BREAKDOWN")
            legs_df = pd.DataFrame([{
                "SIDE":       "LONG"  if lg["qty"] > 0 else "SHORT",
                "TYPE":       lg["type"].upper(),
                "STRIKE K":   f"${lg['K']:.2f}",
                "QTY":        f"{int(lg['qty']):+d}",
                "BS PREMIUM": f"${lg['premium']:.4f}",
            } for lg in legs])
            st.dataframe(legs_df, use_container_width=True, hide_index=True)

        except ValueError as exc:
            st.error(f"Strategy error: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    border-top: 1px solid {C_BORDER};
    margin-top: 32px;
    padding: 12px 0;
    text-align: center;
    font-family: 'Courier New', monospace;
    font-size: 9px;
    letter-spacing: 0.14em;
    color: {C_MUTED};
    text-transform: uppercase;
">
    Built with Python &amp; Streamlit &nbsp;·&nbsp; Black-Scholes Options Pricing Engine
    &nbsp;·&nbsp; Bloomberg Terminal Theme
</div>
""", unsafe_allow_html=True)

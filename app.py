"""
Options Pricing Engine — Streamlit Web App
==========================================
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
    page_title="Options Pricing Engine",
    page_icon="📈",
    layout="wide",
)

# ── Colour palette (dark theme) ───────────────────────────────────────────────
BLUE   = "#4FC3F7"
ORANGE = "#FF8A65"
GREEN  = "#A5D6A7"
AMBER  = "#FFD54F"
GREY   = "#888888"
TPL    = "plotly_dark"


# ─────────────────────────────────────────────────────────────────────────────
# Helper — styled plotly layout defaults
# ─────────────────────────────────────────────────────────────────────────────
def _layout(**kw) -> dict:
    base = dict(
        template=TPL,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    base.update(kw)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Black-Scholes pricer parameters (Tab 1 only)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ BS Pricer")
    st.caption("Controls Tab 1 — BS Pricer & Greeks")
    S_s     = st.slider("Spot Price  S",          50.0,  300.0, 100.0,  1.0)
    K_s     = st.slider("Strike Price  K",         50.0,  300.0, 105.0,  1.0)
    T_s     = st.slider("Time to Expiry  T (yr)",  0.01,    2.0,   1.0, 0.01)
    r_s     = st.slider("Risk-free Rate  r",        0.0,    0.1,  0.05, 0.001, format="%.3f")
    sig_s   = st.slider("Volatility  σ",           0.05,    1.0,  0.20, 0.01,  format="%.2f")
    st.divider()
    st.caption("📈 Black-Scholes Options Pricing Engine")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 BS Pricer & Greeks",
    "🔍 IV Solver",
    "🌀 Heston vs BS",
    "📈 Backtest",
    "🎯 Options Strategies",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BS Pricer & Greeks
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Black-Scholes Pricer & Greeks")

    try:
        result  = price_option_pair(S_s, K_s, T_s, r_s, sig_s)
        call_r  = result.call
        put_r   = result.put

        # ── Prices ────────────────────────────────────────────────────────────
        st.subheader("Option Prices")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Call Price",           f"${call_r.price:.4f}")
        pc2.metric("Put Price",            f"${put_r.price:.4f}")
        parity_ok = result.parity_error < 1e-8
        pc3.metric(
            "Put-Call Parity Error",
            f"{result.parity_error:.2e}",
            delta="Valid" if parity_ok else "Check inputs",
            delta_color="normal" if parity_ok else "inverse",
        )

        # ── Greeks table ──────────────────────────────────────────────────────
        st.subheader("Greeks")
        cg = call_r.greeks
        pg = put_r.greeks
        greeks_df = pd.DataFrame({
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Call":  [f"{cg.delta:.4f}", f"{cg.gamma:.4f}", f"{cg.vega:.4f}",
                      f"{cg.theta:.4f}", f"{cg.rho:.4f}"],
            "Put":   [f"{pg.delta:.4f}", f"{pg.gamma:.4f}", f"{pg.vega:.4f}",
                      f"{pg.theta:.4f}", f"{pg.rho:.4f}"],
        })
        st.dataframe(greeks_df, use_container_width=True, hide_index=True)

        # ── Delta & Gamma vs Spot ─────────────────────────────────────────────
        st.subheader("Delta & Gamma vs Spot Price")
        S_lo = max(1.0, S_s * 0.50)
        S_hi = S_s * 1.50
        S_range = np.linspace(S_lo, S_hi, 300)

        call_deltas, put_deltas, gammas = [], [], []
        for s in S_range:
            cg_ = compute_greeks(s, K_s, T_s, r_s, sig_s, "call")
            pg_ = compute_greeks(s, K_s, T_s, r_s, sig_s, "put")
            call_deltas.append(cg_.delta)
            put_deltas.append(pg_.delta)
            gammas.append(cg_.gamma)

        fig1 = make_subplots(rows=1, cols=2,
                             subplot_titles=("Delta vs Spot", "Gamma vs Spot"))
        fig1.add_trace(go.Scatter(x=S_range, y=call_deltas, name="Call Delta",
                                  line=dict(color=BLUE, width=2)), row=1, col=1)
        fig1.add_trace(go.Scatter(x=S_range, y=put_deltas, name="Put Delta",
                                  line=dict(color=ORANGE, width=2)), row=1, col=1)
        fig1.add_vline(x=K_s, line_dash="dash", line_color=GREY,
                       annotation_text=f"K={K_s:.0f}", row=1, col=1)
        fig1.add_trace(go.Scatter(x=S_range, y=gammas, name="Gamma",
                                  line=dict(color=GREEN, width=2),
                                  showlegend=True), row=1, col=2)
        fig1.add_vline(x=K_s, line_dash="dash", line_color=GREY, row=1, col=2)
        fig1.update_xaxes(title_text="Spot Price ($)")
        fig1.update_yaxes(title_text="Delta", row=1, col=1)
        fig1.update_yaxes(title_text="Gamma", row=1, col=2)
        fig1.update_layout(**_layout(height=420))
        st.plotly_chart(fig1, use_container_width=True)

    except ValueError as exc:
        st.error(f"Pricing error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IV Solver
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Implied Volatility Solver")

    left2, right2 = st.columns([1, 1])

    with left2:
        st.subheader("Inputs")
        iv_price   = st.number_input("Market Option Price ($)",     0.01, 99999.0,   8.50, 0.01)
        iv_S       = st.number_input("Spot Price (S)",               1.0, 99999.0, 100.0,  1.0, key="iv_S")
        iv_K       = st.number_input("Strike Price (K)",             1.0, 99999.0, 105.0,  1.0, key="iv_K")
        iv_T       = st.number_input("Time to Expiry (T, yr)",      0.01,     5.0,   1.0, 0.01, key="iv_T")
        iv_r       = st.number_input("Risk-free Rate (r)",           0.0,     1.0,  0.05, 0.001,
                                     format="%.3f", key="iv_r")
        iv_type    = st.selectbox("Option Type", ["call", "put"], key="iv_type")
        solve_btn  = st.button("🔍 Solve IV", type="primary", key="solve_btn")

    with right2:
        st.subheader("Result")
        if solve_btn:
            try:
                ivr = iv_solver(iv_price, iv_S, iv_K, iv_T, iv_r, iv_type)
                r1, r2 = st.columns(2)
                r1.metric("Implied Volatility", f"{ivr.implied_vol:.4f}  ({ivr.implied_vol:.2%})")
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
            st.info("Enter parameters and click **Solve IV**")

    # ── IV Smile plot ─────────────────────────────────────────────────────────
    st.subheader("IV Smile — Strikes 80 to 120")
    try:
        K_smile    = np.linspace(80.0, 120.0, 41)
        true_vols  = 0.20 + 0.15 * ((K_smile / iv_S) - 1.0) ** 2
        smile_ivs  = []
        for Ks, sv in zip(K_smile, true_vols):
            try:
                mp  = black_scholes(iv_S, Ks, iv_T, iv_r, sv, iv_type)
                res = iv_solver(mp, iv_S, Ks, iv_T, iv_r, iv_type, sigma_init=sv)
                smile_ivs.append(res.implied_vol * 100)
            except ValueError:
                smile_ivs.append(None)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=K_smile, y=true_vols * 100, name="True vol surface",
                                  mode="lines", line=dict(color=GREEN, width=2.5)))
        fig2.add_trace(go.Scatter(x=K_smile, y=smile_ivs, name="Recovered IV",
                                  mode="markers", marker=dict(color=BLUE, size=8, symbol="circle")))
        fig2.add_hline(y=20.0, line_dash="dash", line_color=GREY,
                       annotation_text="Flat BS 20%", annotation_font_color=GREY)
        fig2.add_vline(x=iv_S, line_dash="dot", line_color=AMBER,
                       annotation_text=f"S={iv_S:.0f}", annotation_font_color=AMBER)
        fig2.update_layout(**_layout(
            height=380,
            xaxis_title="Strike K ($)",
            yaxis_title="Implied Volatility (%)",
        ))
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as exc:
        st.error(f"Smile plot error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Heston vs Black-Scholes
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Heston Stochastic Volatility vs Black-Scholes")

    h_left, h_right = st.columns([1, 2])

    with h_left:
        st.subheader("Parameters")
        h_S       = st.number_input("Spot Price (S)",      1.0, 9999.0, 100.0,  1.0, key="h_S")
        h_K       = st.number_input("Strike Price (K)",    1.0, 9999.0, 100.0,  1.0, key="h_K")
        h_T       = st.number_input("Time to Expiry (T)",  0.01,   5.0,   1.0, 0.01, key="h_T")
        h_r       = st.number_input("Risk-free Rate (r)",  0.0,    1.0,  0.05, 0.001,
                                    format="%.3f", key="h_r")
        st.markdown("**Heston Parameters**")
        h_v0      = st.slider("v\u2080  (initial variance)",   0.01, 0.5,  0.04, 0.01, key="h_v0")
        h_kappa   = st.slider("\u03ba  (mean reversion)",       0.1, 10.0,  2.0,  0.1, key="h_kappa")
        h_theta   = st.slider("\u03b8  (long-run variance)",   0.01,  0.5, 0.04, 0.01, key="h_theta")
        h_sigma_v = st.slider("\u03c3\u1d65 (vol of vol)",     0.05,  1.0, 0.30, 0.05, key="h_sigma_v")
        h_rho     = st.slider("\u03c1  (spot-vol corr)",      -0.99, 0.99, -0.70, 0.01, key="h_rho")

        feller    = 2 * h_kappa * h_theta
        feller_ok = feller >= h_sigma_v ** 2
        if feller_ok:
            st.success(f"Feller: 2\u03ba\u03b8 = {feller:.3f} \u2265 \u03c3\u1d65\u00b2 = {h_sigma_v**2:.3f}")
        else:
            st.warning(f"Feller violated: 2\u03ba\u03b8 = {feller:.3f} < \u03c3\u1d65\u00b2 = {h_sigma_v**2:.3f}")

    with h_right:
        try:
            sigma_bs = float(np.sqrt(h_theta))

            # ── ATM comparison ────────────────────────────────────────────────
            st.subheader("ATM Pricing Comparison")
            with st.spinner("Pricing ATM options..."):
                h_call  = heston_price(h_S, h_K, h_T, h_r, h_v0, h_kappa, h_theta, h_sigma_v, h_rho, "call")
                h_put   = heston_price(h_S, h_K, h_T, h_r, h_v0, h_kappa, h_theta, h_sigma_v, h_rho, "put")
                bs_call = black_scholes(h_S, h_K, h_T, h_r, sigma_bs, "call")
                bs_put  = black_scholes(h_S, h_K, h_T, h_r, sigma_bs, "put")

            cmp_df = pd.DataFrame({
                "Model":      ["Black-Scholes", "Heston", "Difference"],
                "Call Price": [f"${bs_call:.4f}", f"${h_call:.4f}", f"${h_call - bs_call:+.4f}"],
                "Put Price":  [f"${bs_put:.4f}",  f"${h_put:.4f}",  f"${h_put  - bs_put:+.4f}"],
            })
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            # ── IV Skew across strikes ─────────────────────────────────────────
            st.subheader("IV Skew: Heston vs Flat BS (strikes 75%–125% of S)")
            K_range = np.linspace(h_S * 0.75, h_S * 1.25, 31)

            heston_ivs: list[float | None] = []
            with st.spinner("Computing IV smile (~10 s)..."):
                for Kh in K_range:
                    try:
                        hp   = heston_price(h_S, Kh, h_T, h_r, h_v0, h_kappa, h_theta,
                                            h_sigma_v, h_rho, "call")
                        ivr2 = iv_solver(hp, h_S, Kh, h_T, h_r, "call", sigma_init=sigma_bs)
                        heston_ivs.append(ivr2.implied_vol * 100)
                    except Exception:
                        heston_ivs.append(None)

            bs_flat = [sigma_bs * 100] * len(K_range)
            valid   = [v for v in heston_ivs if v is not None]
            K_valid = K_range[[v is not None for v in heston_ivs]]
            iv_diff = [(h - sigma_bs * 100) if h is not None else None
                       for h in heston_ivs]
            bar_col = [BLUE if (d is not None and d >= 0) else ORANGE
                       for d in iv_diff]

            fig3 = make_subplots(
                rows=2, cols=1, row_heights=[0.68, 0.32],
                shared_xaxes=True, vertical_spacing=0.06,
                subplot_titles=("Implied Volatility Smile", "IV Difference (Heston − Flat BS, pp)"),
            )
            fig3.add_trace(go.Scatter(
                x=K_valid, y=valid, name="Heston IV",
                mode="lines+markers",
                line=dict(color=BLUE, width=2),
                marker=dict(size=5),
            ), row=1, col=1)
            fig3.add_trace(go.Scatter(
                x=K_range, y=bs_flat, name=f"Flat BS ({sigma_bs:.0%})",
                mode="lines", line=dict(color=ORANGE, width=2, dash="dash"),
            ), row=1, col=1)
            fig3.add_vline(x=h_S, line_dash="dot", line_color=GREY,
                           annotation_text=f"ATM {h_S:.0f}")
            fig3.add_trace(go.Bar(
                x=K_range, y=iv_diff, name="IV diff (pp)",
                marker_color=bar_col, opacity=0.85,
            ), row=2, col=1)
            fig3.add_hline(y=0, line_color=GREY, row=2, col=1)
            fig3.update_xaxes(title_text="Strike K ($)", row=2, col=1)
            fig3.update_yaxes(title_text="IV (%)",       row=1, col=1)
            fig3.update_yaxes(title_text="Diff (pp)",    row=2, col=1)
            fig3.update_layout(**_layout(height=520))
            st.plotly_chart(fig3, use_container_width=True)

        except ValueError as exc:
            st.error(f"Heston error: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Backtest
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Covered-Call Backtester")
    st.caption(
        "Sells a 5% OTM call every 30 trading days.  "
        "Volatility = 30-day realised HV from log returns.  r = 5%."
    )

    bt_ticker = st.text_input("Ticker", value="SPY", key="bt_ticker").upper().strip()
    run_bt    = st.button("▶ Run Backtest", type="primary", key="run_bt")

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

                # ── Summary metrics ───────────────────────────────────────────
                st.subheader("Summary")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total P&L",       f"${total_pnl:+.2f}")
                m2.metric("Total Premium",   f"${total_prem:.2f}")
                m3.metric("Total Legs",      str(len(trades)))
                m4.metric("Exercised",       f"{n_exercised}/{len(trades)}")
                m5.metric("Win Rate",        f"{win_rate:.0%}")
                _, mv, _ = st.columns([1, 1, 3])
                mv.metric("Avg Realised Vol", f"{avg_vol:.1%}")

                # ── Cumulative P&L chart ──────────────────────────────────────
                st.subheader("Cumulative P&L Over Time")
                dates_exp  = [pd.Timestamp(t.expiry_date) for t in trades]
                dates_ent  = [pd.Timestamp(t.entry_date)  for t in trades]
                pnls       = [t.pnl     for t in trades]
                prems      = [t.premium for t in trades]
                cum_pnl    = list(np.cumsum(pnls))
                cum_prem   = list(np.cumsum(prems))
                dot_colors = [ORANGE if t.exercised else GREEN for t in trades]
                bar_colors = [ORANGE if p < 0 else BLUE for p in pnls]

                fig4 = make_subplots(
                    rows=2, cols=1, row_heights=[0.68, 0.32],
                    shared_xaxes=True, vertical_spacing=0.06,
                    subplot_titles=("Cumulative P&L", "Per-Leg P&L"),
                )
                fig4.add_trace(go.Scatter(
                    x=dates_exp, y=cum_pnl, name="Cumulative P&L",
                    mode="lines", line=dict(color=BLUE, width=2.5),
                    fill="tozeroy", fillcolor="rgba(79,195,247,0.07)",
                ), row=1, col=1)
                fig4.add_trace(go.Scatter(
                    x=dates_exp, y=cum_prem, name="Cumulative Premium",
                    mode="lines", line=dict(color=GREEN, width=1.5, dash="dash"),
                ), row=1, col=1)
                fig4.add_trace(go.Scatter(
                    x=dates_exp, y=cum_pnl, name="Trade outcomes",
                    mode="markers",
                    marker=dict(color=dot_colors, size=7, symbol="circle"),
                    showlegend=True,
                ), row=1, col=1)
                fig4.add_hline(y=0, line_color=GREY, line_width=1, row=1, col=1)
                fig4.add_trace(go.Bar(
                    x=dates_ent, y=pnls, name="Leg P&L",
                    marker_color=bar_colors, opacity=0.85,
                ), row=2, col=1)
                fig4.add_hline(y=0, line_color=GREY, row=2, col=1)
                fig4.update_xaxes(title_text="Date", row=2, col=1)
                fig4.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
                fig4.update_yaxes(title_text="Leg P&L ($)",        row=2, col=1)
                fig4.update_layout(**_layout(height=540))
                st.plotly_chart(fig4, use_container_width=True)

                # ── Trades table ──────────────────────────────────────────────
                st.subheader("Trade-by-Trade Detail")
                trades_df = pd.DataFrame([{
                    "Entry":    str(t.entry_date),
                    "Expiry":   str(t.expiry_date),
                    "S Entry":  f"${t.S_entry:.2f}",
                    "Strike K": f"${t.K:.2f}",
                    "Sigma":    f"{t.sigma:.1%}",
                    "Premium":  f"${t.premium:.4f}",
                    "S Expiry": f"${t.S_expiry:.2f}",
                    "Outcome":  "Exercised" if t.exercised else "Expired",
                    "P&L":      f"${t.pnl:+.2f}",
                } for t in trades])
                st.dataframe(trades_df, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Backtest error: {exc}")
    else:
        st.info("Enter a ticker and click **Run Backtest**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Options Strategies
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Options Strategies — P&L at Expiry")

    st5_left, st5_right = st.columns([1, 2])

    with st5_left:
        st.subheader("Parameters")
        strat_name = st.selectbox("Strategy", [
            "Long Straddle",
            "Long Strangle",
            "Iron Condor",
            "Bull Call Spread",
        ], key="strat_name")
        st5_S   = st.slider("Spot Price (S)",            50.0, 300.0, 100.0,  1.0, key="st5_S")
        st5_sig = st.slider("Volatility (σ)",            0.05,   1.0,  0.20, 0.01, key="st5_sig", format="%.2f")
        st5_T   = st.slider("Time to Expiry (T, yr)",   0.01,   2.0,  0.25, 0.01, key="st5_T")
        st5_r   = st.slider("Risk-free Rate (r)",         0.0,   0.1,  0.05, 0.001, key="st5_r", format="%.3f")

    with st5_right:
        try:
            # ── Build legs scaled to the current spot ──────────────────────────
            ratio = st5_S / 100.0

            def _prem(K: float, opt: str) -> float:
                return black_scholes(st5_S, K, st5_T, st5_r, st5_sig, opt)

            # Strikes scaled proportionally (K=100 base → K=S)
            K_atm   = round(st5_S, 2)
            K_105   = round(105 * ratio, 2)
            K_95    = round(95  * ratio, 2)
            K_110   = round(110 * ratio, 2)
            K_115   = round(115 * ratio, 2)
            K_90    = round(90  * ratio, 2)
            K_85    = round(85  * ratio, 2)

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

            # P&L range: ±30% around spot
            S_lo_r  = max(1.0, st5_S * 0.70)
            S_hi_r  = st5_S * 1.30
            S_range = np.linspace(S_lo_r, S_hi_r, 600)

            ana      = analyse(strategy_dict, S_range)
            pnl      = ana["pnl"]
            net_prem = ana["net_prem"]
            bes      = ana["breakevens"]

            # ── Summary metrics ────────────────────────────────────────────────
            st.subheader("Strategy Summary")
            sm1, sm2 = st.columns(2)
            sm1.metric("Net Premium",
                       f"${net_prem:+.4f}  ({'debit' if net_prem > 0 else 'credit'})")
            sm2.metric("Breakeven(s)",
                       " / ".join(f"${b:.2f}" for b in bes) if bes else "None")
            sm3, sm4 = st.columns(2)
            sm3.metric("Max Profit", ana["max_profit_str"])
            sm4.metric("Max Loss",   ana["max_loss_str"])

            # P&L at S−10%, S, S+10%
            s_dn  = round(st5_S * 0.90, 2)
            s_up  = round(st5_S * 1.10, 2)
            p_dn  = float(np.interp(s_dn,  S_range, pnl))
            p_mid = float(np.interp(st5_S, S_range, pnl))
            p_up  = float(np.interp(s_up,  S_range, pnl))
            sp1, sp2, sp3 = st.columns(3)
            sp1.metric(f"P&L at S={s_dn:.0f} (−10%)",    f"${p_dn:+.2f}")
            sp2.metric(f"P&L at S={st5_S:.0f}",           f"${p_mid:+.2f}")
            sp3.metric(f"P&L at S={s_up:.0f} (+10%)",     f"${p_up:+.2f}")

            # ── P&L diagram ────────────────────────────────────────────────────
            st.subheader("P&L Diagram")
            pnl_pos = np.where(pnl >= 0, pnl, 0.0)
            pnl_neg = np.where(pnl < 0,  pnl, 0.0)

            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=S_range, y=pnl_pos,
                fill="tozeroy", fillcolor="rgba(79,195,247,0.13)",
                line=dict(width=0), name="Profit zone",
            ))
            fig5.add_trace(go.Scatter(
                x=S_range, y=pnl_neg,
                fill="tozeroy", fillcolor="rgba(255,138,101,0.13)",
                line=dict(width=0), name="Loss zone",
            ))
            fig5.add_trace(go.Scatter(
                x=S_range, y=pnl, name="P&L",
                mode="lines", line=dict(color=BLUE, width=2.5),
            ))
            fig5.add_hline(y=0, line_color=GREY, line_dash="dash", line_width=1)
            for be in bes:
                fig5.add_vline(x=be, line_color=AMBER, line_dash="dash",
                               annotation_text=f"BE ${be:.2f}",
                               annotation_font_color=AMBER, annotation_font_size=11)
            fig5.add_vline(x=st5_S, line_color=GREY, line_dash="dot",
                           annotation_text=f"S=${st5_S:.0f}",
                           annotation_font_color=GREY)
            fig5.update_layout(**_layout(
                height=420,
                xaxis_title="Spot Price at Expiry ($)",
                yaxis_title="P&L ($)",
                title=dict(
                    text=(f"{strat_name}  |  "
                          f"net {'debit' if net_prem > 0 else 'credit'} ${abs(net_prem):.4f}"),
                    font=dict(size=13),
                ),
            ))
            st.plotly_chart(fig5, use_container_width=True)

            # ── Leg breakdown ──────────────────────────────────────────────────
            st.subheader("Leg Breakdown")
            legs_df = pd.DataFrame([{
                "Side":     "Long"  if lg["qty"] > 0 else "Short",
                "Type":     lg["type"].title(),
                "Strike K": f"${lg['K']:.2f}",
                "Qty":      f"{int(lg['qty']):+d}",
                "BS Premium": f"${lg['premium']:.4f}",
            } for lg in legs])
            st.dataframe(legs_df, use_container_width=True, hide_index=True)

        except ValueError as exc:
            st.error(f"Strategy error: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85em;'>"
    "Built with Python &amp; Streamlit | Black-Scholes Options Pricing Engine"
    "</div>",
    unsafe_allow_html=True,
)

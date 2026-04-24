"""
Covered-Call Backtester
========================
Downloads historical price data via yfinance, computes 30-day rolling
historical volatility, and simulates a rolling covered-call strategy:

  * Every 30 trading days: sell a 5% OTM call priced by Black-Scholes
  * Track premium collected, expiry outcome, per-trade P&L
  * Print a formatted summary table to the terminal
  * Save Plot 6 → plots/plot6_backtest_pnl.png

Strategy mechanics (per 30-day leg)
-------------------------------------
  Entry  : own the stock at S_entry; sell call at K = S_entry * 1.05
  Premium: Black-Scholes call price  (T = 30/365, r = RISK_FREE_RATE)
  Expiry : 30 trading days later at S_expiry
    - If S_expiry > K  → exercised    : P&L = premium + K       - S_entry
    - If S_expiry ≤ K  → worthless    : P&L = premium + S_expiry - S_entry
  (= premium + min(S_expiry, K) - S_entry)

Usage:
    python backtest.py              # SPY, 2 years
    python backtest.py AAPL        # any ticker
    python backtest.py SPY --no-show  # headless (save PNG, skip display)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import date

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from options_engine import black_scholes  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ─────────────────────────────────────────────────────────────────
LOOKBACK_YEARS: int = 2
HV_WINDOW: int = 30          # trading days used to compute rolling HV
REBALANCE_EVERY: int = 30    # trading days between new covered-call legs
OTM_FACTOR: float = 1.05     # strike = spot * OTM_FACTOR  (5% OTM)
DAYS_TO_EXPIRY: float = 30 / 365  # T in years for each leg
RISK_FREE_RATE: float = 0.05
TRADING_DAYS_PER_YEAR: int = 252

PLOTS_DIR: str = os.path.join(os.path.dirname(__file__), "plots")

# ── Dark-theme palette (matches greeks_viz.py / implied_vol.py) ───────────────
PNL_COLOUR = "#4FC3F7"        # sky blue cumulative P&L line
FILL_POS_COLOUR = "#4FC3F7"
FILL_NEG_COLOUR = "#FF8A65"   # warm orange below zero
ZERO_COLOUR = "#888888"
TRADE_DOT_COLOUR = "#FFD54F"  # amber trade markers

plt.style.use("dark_background")


# ── Data container ─────────────────────────────────────────────────────────────
@dataclass
class TradeRecord:
    """One covered-call leg in the backtest.

    Attributes:
        entry_date:   Date the call was sold.
        expiry_date:  Date the call expired / was exercised.
        S_entry:      Stock price at entry.
        K:            Strike price (S_entry * OTM_FACTOR).
        sigma:        30-day realised vol used to price the call.
        premium:      Black-Scholes call price collected.
        S_expiry:     Stock price at expiry.
        exercised:    True if S_expiry > K.
        pnl:          Net P&L for this 30-day leg (premium + stock move, capped).
    """
    entry_date: date
    expiry_date: date
    S_entry: float
    K: float
    sigma: float
    premium: float
    S_expiry: float
    exercised: bool
    pnl: float


# ── Data download & vol calculation ───────────────────────────────────────────
def fetch_prices(ticker: str = "SPY") -> pd.Series:
    """Download adjusted close prices for *ticker* over the last 2 years.

    Args:
        ticker: Yahoo Finance ticker symbol (default ``'SPY'``).

    Returns:
        Pandas Series of adjusted close prices indexed by date.

    Raises:
        ValueError: If fewer than HV_WINDOW + REBALANCE_EVERY rows are returned.
    """
    print(f"Downloading {LOOKBACK_YEARS}-year price history for {ticker} ...")
    raw = yf.download(ticker, period=f"{LOOKBACK_YEARS}y",
                      auto_adjust=True, progress=False)

    # yfinance may return a MultiIndex when auto_adjust=True
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    close = close.dropna()
    close.index = pd.to_datetime(close.index).date  # plain date objects

    if len(close) < HV_WINDOW + REBALANCE_EVERY:
        raise ValueError(
            f"Only {len(close)} rows returned for {ticker}; need at least "
            f"{HV_WINDOW + REBALANCE_EVERY}."
        )

    print(f"  {len(close)} trading days  |  "
          f"{close.index[0]} -> {close.index[-1]}\n")
    return close


def rolling_hv(close: pd.Series) -> pd.Series:
    """Compute 30-day annualised historical volatility from log returns.

    Args:
        close: Series of adjusted close prices.

    Returns:
        Series of annualised HV values (same index, NaN for first HV_WINDOW rows).
    """
    log_ret = np.log(close / close.shift(1))
    hv = log_ret.rolling(HV_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return hv


# ── Strategy simulation ────────────────────────────────────────────────────────
def run_backtest(ticker: str = "SPY") -> list[TradeRecord]:
    """Simulate the rolling covered-call strategy.

    Args:
        ticker: Yahoo Finance ticker symbol.

    Returns:
        List of :class:`TradeRecord` objects, one per 30-day leg.
    """
    close = fetch_prices(ticker)
    hv = rolling_hv(close)

    prices = close.values
    hvs = hv.values
    dates = list(close.index)

    trades: list[TradeRecord] = []

    # Walk forward in REBALANCE_EVERY steps; need room for expiry window
    entry_indices = range(HV_WINDOW, len(prices) - REBALANCE_EVERY,
                          REBALANCE_EVERY)

    for i in entry_indices:
        j = i + REBALANCE_EVERY          # expiry index

        S_entry = float(prices[i])
        S_expiry = float(prices[j])
        sigma = float(hvs[i])

        if np.isnan(sigma) or sigma <= 0:
            continue                      # skip if vol not yet available

        K = round(S_entry * OTM_FACTOR, 2)

        try:
            premium = black_scholes(S_entry, K, DAYS_TO_EXPIRY,
                                    RISK_FREE_RATE, sigma, "call")
        except ValueError:
            continue

        exercised = S_expiry > K
        # Covered-call P&L: premium + capped stock appreciation
        pnl = premium + min(S_expiry, K) - S_entry

        trades.append(TradeRecord(
            entry_date=dates[i],
            expiry_date=dates[j],
            S_entry=S_entry,
            K=K,
            sigma=sigma,
            premium=premium,
            S_expiry=S_expiry,
            exercised=exercised,
            pnl=pnl,
        ))

    return trades


# ── Terminal output ────────────────────────────────────────────────────────────
def print_summary(trades: list[TradeRecord], ticker: str) -> None:
    """Print a formatted trade-by-trade summary table to stdout.

    Args:
        trades: List of completed trade legs.
        ticker: Ticker symbol (shown in header).
    """
    sep = "=" * 95
    header = (
        f"{'Date':<12} {'Expiry':<12} {'S':>8} {'K':>8} "
        f"{'Sigma':>7} {'Premium':>9} {'S expiry':>10} "
        f"{'Outcome':<18} {'P&L':>8}"
    )
    divider = "-" * 95

    print(sep)
    print(f"  Covered-Call Backtest — {ticker}  |  "
          f"{trades[0].entry_date} to {trades[-1].expiry_date}  |  "
          f"{len(trades)} legs")
    print(sep)
    print(header)
    print(divider)

    for t in trades:
        outcome = "Exercised   [cap]" if t.exercised else "Expired worthless"
        pnl_str = f"${t.pnl:+.2f}"
        print(
            f"{str(t.entry_date):<12} {str(t.expiry_date):<12} "
            f"{t.S_entry:>8.2f} {t.K:>8.2f} "
            f"{t.sigma:>7.3f} {t.premium:>9.4f} "
            f"{t.S_expiry:>10.2f} {outcome:<18} {pnl_str:>8}"
        )

    print(divider)

    total_pnl = sum(t.pnl for t in trades)
    total_premium = sum(t.premium for t in trades)
    n_exercised = sum(t.exercised for t in trades)
    win_rate = sum(1 for t in trades if t.pnl >= 0) / len(trades) * 100
    avg_sigma = np.mean([t.sigma for t in trades])

    print(f"\n  Total P&L        : ${total_pnl:+.2f}")
    print(f"  Total premium    : ${total_premium:.2f}  (gross income from calls sold)")
    print(f"  Legs exercised   : {n_exercised}/{len(trades)}")
    print(f"  Profitable legs  : {win_rate:.0f}%")
    print(f"  Avg realised vol : {avg_sigma:.1%}")
    print(sep)
    print()


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_pnl(trades: list[TradeRecord], ticker: str,
             show: bool = True) -> plt.Figure:
    """Plot cumulative covered-call P&L over time.

    Args:
        trades: Completed trade legs.
        ticker: Ticker symbol for title/annotation.
        show:   Display interactively if ``True``.

    Returns:
        Matplotlib Figure. PNG saved to plots/plot6_backtest_pnl.png.
    """
    dates_entry = [pd.Timestamp(t.entry_date) for t in trades]
    dates_expiry = [pd.Timestamp(t.expiry_date) for t in trades]
    pnls = [t.pnl for t in trades]
    cum_pnl = np.cumsum(pnls)
    premiums = [t.premium for t in trades]
    cum_premium = np.cumsum(premiums)

    total_pnl = cum_pnl[-1]
    start_s = trades[0].S_entry
    total_pct = total_pnl / start_s * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), facecolor="#0e0e0e",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    fig.suptitle(
        f"Plot 6 — Covered-Call Backtest  |  {ticker}  |  "
        f"{trades[0].entry_date} → {trades[-1].expiry_date}\n"
        f"5% OTM call sold every 30 trading days  ·  "
        f"T = 30 days  ·  r = {RISK_FREE_RATE:.0%}  ·  "
        f"σ = 30-day realised HV",
        fontsize=11, fontweight="bold", color="white", y=1.01,
    )

    # ── Top panel: cumulative P&L ──────────────────────────────────────────
    ax1.plot(dates_expiry, cum_pnl, color=PNL_COLOUR,
             linewidth=2.0, zorder=4, label="Cumulative P&L (covered call)")
    ax1.plot(dates_expiry, cum_premium, color="#A5D6A7",
             linewidth=1.2, linestyle="--", zorder=3,
             label="Cumulative premium collected")

    # Shade positive / negative separately
    zeros = np.zeros(len(cum_pnl))
    ax1.fill_between(dates_expiry, cum_pnl, zeros,
                     where=(np.array(cum_pnl) >= 0),
                     alpha=0.12, color=FILL_POS_COLOUR, zorder=2)
    ax1.fill_between(dates_expiry, cum_pnl, zeros,
                     where=(np.array(cum_pnl) < 0),
                     alpha=0.18, color=FILL_NEG_COLOUR, zorder=2)

    # Trade markers: green dot = expired worthless, red = exercised
    for t, ex_date, cpnl in zip(trades, dates_expiry, cum_pnl):
        colour = "#EF9A9A" if t.exercised else "#A5D6A7"
        ax1.scatter(ex_date, cpnl, color=colour, s=28, zorder=5)

    # Zero line
    ax1.axhline(0, color=ZERO_COLOUR, linewidth=0.9, linestyle="-", zorder=1)

    # Annotate final total
    ax1.annotate(
        f"Total: ${total_pnl:+.2f}  ({total_pct:+.1f}%)",
        xy=(dates_expiry[-1], cum_pnl[-1]),
        xytext=(-120, 18),
        textcoords="offset points",
        color="white", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8),
    )

    ax1.set_ylabel("Cumulative P&L  ($)", fontsize=10, labelpad=6)
    ax1.tick_params(colors="#cccccc", labelsize=9, labelbottom=False)
    ax1.grid(True, color="#2a2a2a", linewidth=0.6, linestyle="--")
    ax1.spines[:].set_color("#444444")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    # Legend with trade-marker key
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color=PNL_COLOUR, linewidth=2, label="Cumulative P&L"),
        Line2D([0], [0], color="#A5D6A7", linewidth=1.2,
               linestyle="--", label="Cumulative premium"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#A5D6A7",
               markersize=7, label="Expired worthless", linestyle="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#EF9A9A",
               markersize=7, label="Exercised (capped)", linestyle="None"),
    ]
    ax1.legend(handles=legend_items, fontsize=8, framealpha=0.3, loc="upper left")

    # ── Bottom panel: per-trade P&L bars ──────────────────────────────────
    bar_colours = ["#EF9A9A" if p < 0 else PNL_COLOUR for p in pnls]
    ax2.bar(dates_entry, pnls, width=18, color=bar_colours,
            alpha=0.85, zorder=3)
    ax2.axhline(0, color=ZERO_COLOUR, linewidth=0.8, linestyle="-", zorder=2)
    ax2.set_ylabel("Leg P&L  ($)", fontsize=9, labelpad=6)
    ax2.set_xlabel("Date", fontsize=10, labelpad=6)
    ax2.tick_params(colors="#cccccc", labelsize=8)
    ax2.grid(True, color="#2a2a2a", linewidth=0.5, linestyle="--", axis="y")
    ax2.spines[:].set_color("#444444")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    # Shared x-axis formatting
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30, ha="right")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "plot6_backtest_pnl.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [saved] {path}\n")

    if show:
        plt.show()
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────
def main(ticker: str = "SPY", show: bool = True) -> None:
    """Run the full covered-call backtest pipeline.

    Args:
        ticker: Yahoo Finance ticker (default ``'SPY'``).
        show:   Display the plot interactively.
    """
    print()
    print("=" * 54)
    print("  Covered-Call Backtester".center(54))
    print("=" * 54)
    print()

    trades = run_backtest(ticker)

    if not trades:
        print("No valid trades found. Check ticker or data availability.")
        return

    print_summary(trades, ticker)
    plot_pnl(trades, ticker, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Covered-call backtester using Black-Scholes pricing."
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="SPY",
        help="Yahoo Finance ticker symbol (default: SPY)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plot PNG without opening an interactive window.",
    )
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    main(ticker=args.ticker.upper(), show=not args.no_show)

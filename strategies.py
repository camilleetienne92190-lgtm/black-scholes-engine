"""
Multi-Strategy Options P&L Engine
===================================
Implements a generic leg-based payoff engine and four classic option
strategies priced with Black-Scholes via options_engine.py.

Strategies
----------
  Long Straddle    : long call K=100 + long put  K=100
  Long Strangle    : long call K=105 + long put  K=95
  Iron Condor      : short call K=110 / long call K=115
                     short put  K=90  / long put  K=85
  Bull Call Spread : long call  K=100 + short call K=110

Plot 8 saved to plots/plot8_strategies.png — 2x2 dark-theme grid showing
P&L curve, breakeven lines, and max profit / max loss annotations.

Usage:
    python strategies.py             # display + save
    python strategies.py --no-show   # save PNG only (headless / CI)
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from options_engine import black_scholes  # noqa: E402

# ── Default market parameters ─────────────────────────────────────────────────
S_DEFAULT: float = 100.0
R_DEFAULT: float = 0.05
T_DEFAULT: float = 0.25      # 3-month options
SIGMA_DEFAULT: float = 0.20

# ── Spot range for payoff diagrams ────────────────────────────────────────────
S_MIN: float = 70.0
S_MAX: float = 130.0
S_POINTS: int = 1200         # smooth curve

# ── Output ────────────────────────────────────────────────────────────────────
PLOTS_DIR: str = os.path.join(os.path.dirname(__file__), "plots")
SEPARATOR: str = "=" * 60

# ── Colour palette (dark theme, matches project) ──────────────────────────────
plt.style.use("dark_background")
CURVE_COLOUR = "#4FC3F7"       # sky blue — P&L line
FILL_POS = "#4FC3F7"           # blue fill — profit zone
FILL_NEG = "#FF8A65"           # orange fill — loss zone
ZERO_COLOUR = "#888888"        # grey — zero line
BE_COLOUR = "#FFD54F"          # amber — breakeven verticals
MAX_P_COLOUR = "#A5D6A7"       # soft green — max profit annotation
MAX_L_COLOUR = "#EF9A9A"       # light red — max loss annotation


# ─────────────────────────────────────────────────────────────────────────────
# Generic payoff engine
# ─────────────────────────────────────────────────────────────────────────────

def strategy_payoff(S_range: np.ndarray, legs: list[dict]) -> np.ndarray:
    """Compute net P&L across a spot-price range for an arbitrary option strategy.

    P&L is computed at option expiry (intrinsic value only).

    Args:
        S_range: 1-D array of spot prices at expiry.
        legs:    List of leg dicts, each with keys:

                 * ``type``    — ``'call'`` or ``'put'``
                 * ``K``       — strike price (float)
                 * ``qty``     — signed quantity: +1 = long, -1 = short
                 * ``premium`` — Black-Scholes price paid/received per unit

    Returns:
        1-D numpy array of net P&L values (same length as ``S_range``).

    Formula per leg::

        P&L_leg(S) = qty * (intrinsic(S) - premium)

    where intrinsic = max(S-K, 0) for calls, max(K-S, 0) for puts.
    Positive ``qty`` means long (premium was paid); negative means short
    (premium was received).
    """
    pnl = np.zeros(len(S_range), dtype=float)
    for leg in legs:
        K = float(leg["K"])
        qty = float(leg["qty"])
        premium = float(leg["premium"])
        if leg["type"] == "call":
            intrinsic = np.maximum(S_range - K, 0.0)
        elif leg["type"] == "put":
            intrinsic = np.maximum(K - S_range, 0.0)
        else:
            raise ValueError(f"leg type must be 'call' or 'put', got '{leg['type']}'")
        pnl += qty * (intrinsic - premium)
    return pnl


# ─────────────────────────────────────────────────────────────────────────────
# Strategy builders
# ─────────────────────────────────────────────────────────────────────────────

def _bs(S: float, K: float, T: float, r: float, sigma: float,
        option_type: str) -> float:
    return black_scholes(S, K, T, r, sigma, option_type)


def build_strategies(
    S: float = S_DEFAULT,
    r: float = R_DEFAULT,
    T: float = T_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
) -> list[dict]:
    """Build the four option strategies with BS-priced premiums.

    Args:
        S:     Spot price.
        r:     Risk-free rate.
        T:     Time to expiration in years.
        sigma: Implied volatility.

    Returns:
        List of strategy dicts, each with keys ``'name'`` and ``'legs'``.
    """
    c = lambda K: _bs(S, K, T, r, sigma, "call")   # noqa: E731
    p = lambda K: _bs(S, K, T, r, sigma, "put")    # noqa: E731

    return [
        {
            "name": "Long Straddle",
            "legs": [
                {"type": "call", "K": 100, "qty": +1, "premium": c(100)},
                {"type": "put",  "K": 100, "qty": +1, "premium": p(100)},
            ],
        },
        {
            "name": "Long Strangle",
            "legs": [
                {"type": "call", "K": 105, "qty": +1, "premium": c(105)},
                {"type": "put",  "K":  95, "qty": +1, "premium": p(95)},
            ],
        },
        {
            "name": "Iron Condor",
            "legs": [
                {"type": "call", "K": 110, "qty": -1, "premium": c(110)},
                {"type": "call", "K": 115, "qty": +1, "premium": c(115)},
                {"type": "put",  "K":  90, "qty": -1, "premium": p(90)},
                {"type": "put",  "K":  85, "qty": +1, "premium": p(85)},
            ],
        },
        {
            "name": "Bull Call Spread",
            "legs": [
                {"type": "call", "K": 100, "qty": +1, "premium": c(100)},
                {"type": "call", "K": 110, "qty": -1, "premium": c(110)},
            ],
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def net_premium(legs: list[dict]) -> float:
    """Net premium of a strategy.

    Returns:
        Positive value = net debit (you paid); negative = net credit (received).
    """
    return sum(leg["qty"] * leg["premium"] for leg in legs)


def find_breakevens(S_range: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Find breakeven spot prices by detecting sign changes in the P&L curve.

    Args:
        S_range: Spot price array.
        pnl:     Corresponding P&L array.

    Returns:
        Sorted list of breakeven prices (linearly interpolated at sign changes).
    """
    breakevens: list[float] = []
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i + 1] < 0.0:
            # Linear interpolation between the two neighbouring points
            slope = (pnl[i + 1] - pnl[i]) / (S_range[i + 1] - S_range[i])
            be = S_range[i] - pnl[i] / slope
            breakevens.append(round(float(be), 2))
    return sorted(set(breakevens))


def _pnl_at(S_range: np.ndarray, pnl: np.ndarray, s: float) -> float:
    return float(np.interp(s, S_range, pnl))


def _is_still_moving(pnl: np.ndarray, at_right: bool, tol: float = 0.01) -> bool:
    """True if the P&L is still changing (not flat) at the given boundary.

    Used to distinguish a flat plateau at a boundary from a genuinely
    unlimited profit or loss.
    """
    if at_right:
        return abs(pnl[-1] - pnl[-20]) > tol
    else:
        return abs(pnl[0] - pnl[20]) > tol


def _max_profit_str(pnl: np.ndarray, S_range: np.ndarray) -> tuple[float, str]:
    """Return (max_value, display_string) for max profit.

    Flags 'Unlimited' only if the maximum is at a boundary AND the curve
    is still rising there (not just a flat plateau touching the edge).
    """
    idx = int(np.argmax(pnl))
    val = float(pnl[idx])
    at_right = idx >= len(pnl) - 20
    at_left = idx <= 20
    if (at_right and _is_still_moving(pnl, at_right=True)) or \
       (at_left  and _is_still_moving(pnl, at_right=False)):
        return val, f"Unlimited (>${val:.2f} in range)"
    return val, f"${val:.2f}  at S=${S_range[idx]:.1f}"


def _max_loss_str(pnl: np.ndarray, S_range: np.ndarray) -> tuple[float, str]:
    """Return (min_value, display_string) for max loss.

    Flags 'Unlimited' only if the minimum is at a boundary AND the curve
    is still falling there (not a flat floor touching the edge).
    """
    idx = int(np.argmin(pnl))
    val = float(pnl[idx])
    at_right = idx >= len(pnl) - 20
    at_left = idx <= 20
    if (at_right and _is_still_moving(pnl, at_right=True)) or \
       (at_left  and _is_still_moving(pnl, at_right=False)):
        return val, f"Unlimited (<${val:.2f} in range)"
    return val, f"${val:.2f}  at S=${S_range[idx]:.1f}"


def analyse(strategy: dict, S_range: np.ndarray) -> dict:
    """Compute full analytics for one strategy.

    Args:
        strategy: Strategy dict with ``'name'`` and ``'legs'``.
        S_range:  Spot price range array.

    Returns:
        Dict with keys: ``pnl``, ``net_prem``, ``breakevens``,
        ``max_profit``, ``max_profit_str``, ``max_loss``, ``max_loss_str``,
        ``pnl_90``, ``pnl_100``, ``pnl_110``.
    """
    legs = strategy["legs"]
    pnl = strategy_payoff(S_range, legs)
    net = net_premium(legs)
    bes = find_breakevens(S_range, pnl)
    mp, mp_str = _max_profit_str(pnl, S_range)
    ml, ml_str = _max_loss_str(pnl, S_range)

    return {
        "pnl": pnl,
        "net_prem": net,
        "breakevens": bes,
        "max_profit": mp,
        "max_profit_str": mp_str,
        "max_loss": ml,
        "max_loss_str": ml_str,
        "pnl_90":  _pnl_at(S_range, pnl, 90.0),
        "pnl_100": _pnl_at(S_range, pnl, 100.0),
        "pnl_110": _pnl_at(S_range, pnl, 110.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Terminal summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(strategies: list[dict], analytics: list[dict]) -> None:
    """Print a formatted summary table for each strategy to stdout.

    Args:
        strategies: List of strategy dicts.
        analytics:  Corresponding list of analysis result dicts.
    """
    print()
    print(SEPARATOR)
    print("  Options Strategy Summary")
    print(f"  S={S_DEFAULT}, r={R_DEFAULT}, T={T_DEFAULT} yr, sigma={SIGMA_DEFAULT:.0%}")
    print(SEPARATOR)

    for strat, ana in zip(strategies, analytics):
        net = ana["net_prem"]
        net_label = f"${net:+.4f}  ({'debit' if net > 0 else 'credit'})"
        bes = ana["breakevens"]
        be_str = "  /  ".join(f"${b:.2f}" for b in bes) if bes else "none"

        print(f"\n  {'=' * 56}")
        print(f"  Strategy    : {strat['name']}")
        print(f"  {'=' * 56}")
        print(f"  Net Premium : {net_label}")
        print(f"  Breakeven(s): {be_str}")
        print(f"  Max Profit  : {ana['max_profit_str']}")
        print(f"  Max Loss    : {ana['max_loss_str']}")
        print(f"  {'-' * 40}")
        for spot, val in [(90, ana["pnl_90"]), (100, ana["pnl_100"]),
                          (110, ana["pnl_110"])]:
            print(f"  P&L at S={spot:<3}  : ${val:+.2f}")

        # Leg breakdown
        print(f"  {'-' * 40}")
        print(f"  {'Leg':<22} {'K':>6} {'Qty':>5} {'Premium':>10}")
        for leg in strat["legs"]:
            side = "Long" if leg["qty"] > 0 else "Short"
            desc = f"{side} {leg['type']}"
            print(f"  {desc:<22} {leg['K']:>6.0f} {leg['qty']:>+5.0f}   ${leg['premium']:>7.4f}")

    print(f"\n{SEPARATOR}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8 — 2x2 strategy payoff grid
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes) -> None:
    ax.tick_params(colors="#cccccc", labelsize=8)
    ax.grid(True, color="#252525", linewidth=0.6, linestyle="--")
    ax.spines[:].set_color("#444444")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))


def _plot_one(
    ax: plt.Axes,
    S_range: np.ndarray,
    pnl: np.ndarray,
    strategy: dict,
    ana: dict,
) -> None:
    """Render one strategy panel onto *ax*."""
    net = ana["net_prem"]
    label = "debit" if net > 0 else "credit"
    title = f"{strategy['name']}\n(net {label}: ${abs(net):.4f})"

    # ── P&L curve ──────────────────────────────────────────────────────────
    ax.plot(S_range, pnl, color=CURVE_COLOUR, linewidth=1.8, zorder=4)

    # Profit / loss fills
    ax.fill_between(S_range, pnl, 0, where=(pnl >= 0),
                    color=FILL_POS, alpha=0.15, zorder=2)
    ax.fill_between(S_range, pnl, 0, where=(pnl < 0),
                    color=FILL_NEG, alpha=0.15, zorder=2)

    # ── Zero line ──────────────────────────────────────────────────────────
    ax.axhline(0, color=ZERO_COLOUR, linewidth=0.9,
               linestyle="--", zorder=3)

    # ── Breakeven verticals ───────────────────────────────────────────────
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else max(pnl) * 1.1
    for be in ana["breakevens"]:
        ax.axvline(be, color=BE_COLOUR, linewidth=0.9,
                   linestyle="--", zorder=3, alpha=0.85)
        ax.text(be, ax.get_ylim()[0] * 0.6 if ax.get_ylim()[0] < 0 else 0,
                f" ${be:.0f}", color=BE_COLOUR, fontsize=6.5,
                va="bottom", ha="left", rotation=90, zorder=5)

    # ── Max profit annotation ─────────────────────────────────────────────
    mp_idx = int(np.argmax(pnl))
    mp_val = pnl[mp_idx]
    mp_unlimited = (mp_idx == 0 or mp_idx == len(pnl) - 1)
    mp_text = f"Max profit\nUnlimited" if mp_unlimited else f"Max profit\n${mp_val:.2f}"
    # Place text in upper region
    ax.text(0.97, 0.97, mp_text,
            transform=ax.transAxes, color=MAX_P_COLOUR,
            fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      edgecolor=MAX_P_COLOUR, alpha=0.85),
            zorder=6)

    # ── Max loss annotation ───────────────────────────────────────────────
    ml_val = float(np.min(pnl))
    ax.text(0.03, 0.03, f"Max loss\n${ml_val:.2f}",
            transform=ax.transAxes, color=MAX_L_COLOUR,
            fontsize=7, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      edgecolor=MAX_L_COLOUR, alpha=0.85),
            zorder=6)

    # ── Labels ────────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=9, fontweight="bold", color="white", pad=6)
    ax.set_xlabel("Spot Price at Expiry ($)", fontsize=8, labelpad=4)
    ax.set_ylabel("P&L ($)", fontsize=8, labelpad=4)
    _style_ax(ax)


def plot_strategies(
    strategies: list[dict],
    analytics: list[dict],
    S_range: np.ndarray,
    show: bool = True,
) -> plt.Figure:
    """Generate Plot 8 — 2x2 strategy payoff grid.

    Args:
        strategies: List of strategy dicts (from :func:`build_strategies`).
        analytics:  Corresponding analysis dicts (from :func:`analyse`).
        S_range:    Spot price range array.
        show:       Display figure interactively.

    Returns:
        Matplotlib Figure. PNG saved to ``plots/plot8_strategies.png``.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0e0e0e")
    fig.suptitle(
        f"Plot 8 — Options Strategy P&L at Expiry  |  "
        f"S={S_DEFAULT}, T={T_DEFAULT} yr, r={R_DEFAULT:.0%}, "
        f"sigma={SIGMA_DEFAULT:.0%}  |  Spot range ${S_MIN:.0f}–${S_MAX:.0f}",
        fontsize=11, fontweight="bold", color="white", y=1.01,
    )

    axes_flat = axes.flatten()
    for ax, strat, ana in zip(axes_flat, strategies, analytics):
        _plot_one(ax, S_range, ana["pnl"], strat, ana)
        # Re-run style after data (limits are set by now)
        _style_ax(ax)

    # Shared legend at bottom
    legend_items = [
        mpatches.Patch(color=FILL_POS, alpha=0.6, label="Profit zone"),
        mpatches.Patch(color=FILL_NEG, alpha=0.6, label="Loss zone"),
        plt.Line2D([0], [0], color=ZERO_COLOUR, linestyle="--",
                   linewidth=0.9, label="Zero line"),
        plt.Line2D([0], [0], color=BE_COLOUR, linestyle="--",
                   linewidth=0.9, label="Breakeven"),
    ]
    fig.legend(handles=legend_items, fontsize=8, framealpha=0.3,
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "plot8_strategies.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [saved] {path}")

    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(show: bool = True) -> None:
    """Run all four strategy analyses, print summary, save Plot 8.

    Args:
        show: Display the plot interactively.
    """
    S_range = np.linspace(S_MIN, S_MAX, S_POINTS)

    strategies = build_strategies(
        S=S_DEFAULT, r=R_DEFAULT, T=T_DEFAULT, sigma=SIGMA_DEFAULT
    )
    analytics = [analyse(s, S_range) for s in strategies]

    print_summary(strategies, analytics)
    plot_strategies(strategies, analytics, S_range, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Options strategy P&L engine — straddle, strangle, "
                    "iron condor, bull call spread."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plot PNG without opening an interactive window.",
    )
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    main(show=not args.no_show)

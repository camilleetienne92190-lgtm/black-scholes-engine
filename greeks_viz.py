"""
Greeks Visualization Module
============================
Generates 4 diagnostic plots for the Black-Scholes Options Pricing Engine:

  1. Delta & Gamma vs Spot Price
  2. Theta Decay vs Time to Expiry
  3. Vega vs Spot Price (multi-sigma heatmap)
  4. Option Price vs Spot (BS price + intrinsic payoff diagram)

Usage:
    python greeks_viz.py            # generates all 4 plots
    python greeks_viz.py --no-show  # save PNGs only, no interactive window

All PNGs are saved to ./plots/
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Allow import whether run directly or as a module
sys.path.insert(0, os.path.dirname(__file__))
from options_engine import compute_greeks, black_scholes  # noqa: E402

# ── Fixed parameters (unless a plot varies them) ──────────────────────────────
K: float = 100.0
T: float = 1.0
R: float = 0.05
SIGMA: float = 0.20

# ── Axis ranges ───────────────────────────────────────────────────────────────
S_RANGE: np.ndarray = np.linspace(50, 150, 300)
T_RANGE: np.ndarray = np.linspace(0.02, 2.0, 300)
SIGMA_LEVELS: list[float] = [0.10, 0.20, 0.30, 0.40]

# ── Output directory ──────────────────────────────────────────────────────────
PLOTS_DIR: str = os.path.join(os.path.dirname(__file__), "plots")

# ── Colour palette (works on dark background) ─────────────────────────────────
CALL_COLOUR = "#4FC3F7"      # sky blue
PUT_COLOUR = "#FF8A65"       # warm orange
GAMMA_COLOUR = "#A5D6A7"     # soft green
INTRINSIC_COLOUR = "#FFD54F" # amber
SIGMA_COLOURS = ["#EF9A9A", "#CE93D8", "#80DEEA", "#C5E1A5"]  # red→purple→cyan→green

LINEWIDTH = 2.0
ALPHA_FILL = 0.08

plt.style.use("dark_background")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> str:
    """Save *fig* as PNG in PLOTS_DIR and return the full path.

    Args:
        fig: Matplotlib figure to save.
        filename: Base filename (e.g. ``'delta_gamma.png'``).

    Returns:
        Absolute path to the saved file.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [saved] {path}")
    return path


def _greek_series(
    greek: str,
    option_type: str,
    s_values: np.ndarray = S_RANGE,
    t: float = T,
    r: float = R,
    sigma: float = SIGMA,
) -> np.ndarray:
    """Vectorise a single Greek over a spot-price array.

    Args:
        greek: One of ``'delta'``, ``'gamma'``, ``'vega'``, ``'theta'``, ``'rho'``.
        option_type: ``'call'`` or ``'put'``.
        s_values: Array of spot prices.
        t: Time to expiry.
        r: Risk-free rate.
        sigma: Volatility.

    Returns:
        NumPy array of Greek values (same length as *s_values*).
    """
    return np.array(
        [getattr(compute_greeks(s, K, t, r, sigma, option_type), greek)
         for s in s_values]
    )


def _price_series(
    option_type: str,
    s_values: np.ndarray = S_RANGE,
    t: float = T,
    r: float = R,
    sigma: float = SIGMA,
) -> np.ndarray:
    """Vectorise BS price over a spot-price array.

    Args:
        option_type: ``'call'`` or ``'put'``.
        s_values: Array of spot prices.
        t: Time to expiry.
        r: Risk-free rate.
        sigma: Volatility.

    Returns:
        NumPy array of BS prices.
    """
    return np.array(
        [black_scholes(s, K, t, r, sigma, option_type) for s in s_values]
    )


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent dark-theme styling to an Axes object.

    Args:
        ax: Target Axes.
        title: Subplot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.tick_params(colors="#cccccc", labelsize=9)
    ax.grid(True, color="#333333", linewidth=0.6, linestyle="--")
    ax.spines[:].set_color("#444444")
    # Vertical line at K
    ax.axvline(K, color="#888888", linewidth=0.8, linestyle=":", label=f"K = {K:.0f}")


def _add_zero_line(ax: plt.Axes) -> None:
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="-")


# ── Plot 1 — Delta & Gamma vs Spot Price ─────────────────────────────────────

def plot_delta_gamma(show: bool = True) -> plt.Figure:
    """Plot Delta (call + put) and Gamma vs spot price side by side.

    Fixed params: K=100, T=1.0, r=0.05, sigma=0.20.

    Args:
        show: If ``True``, display the figure interactively.

    Returns:
        Matplotlib Figure object.
    """
    call_delta = _greek_series("delta", "call")
    put_delta  = _greek_series("delta", "put")
    gamma      = _greek_series("gamma", "call")   # same for call/put

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0e0e0e")
    fig.suptitle(
        "Plot 1 — Delta & Gamma vs Spot Price\n"
        f"K={K}, T={T}, r={R}, σ={SIGMA}",
        fontsize=14, fontweight="bold", color="white", y=1.01,
    )

    # ── Delta subplot ──────────────────────────────────────────────────────
    _style_axes(ax1, "Delta", "Spot Price  S ($)", "Delta")
    ax1.plot(S_RANGE, call_delta, color=CALL_COLOUR, linewidth=LINEWIDTH, label="Call Δ")
    ax1.plot(S_RANGE, put_delta,  color=PUT_COLOUR,  linewidth=LINEWIDTH, label="Put Δ")
    ax1.fill_between(S_RANGE, call_delta, alpha=ALPHA_FILL, color=CALL_COLOUR)
    ax1.fill_between(S_RANGE, put_delta,  alpha=ALPHA_FILL, color=PUT_COLOUR)
    _add_zero_line(ax1)
    ax1.set_ylim(-1.05, 1.05)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax1.legend(fontsize=9, framealpha=0.3)

    # ── Gamma subplot ──────────────────────────────────────────────────────
    _style_axes(ax2, "Gamma (identical for call & put)", "Spot Price  S ($)", "Gamma")
    ax2.plot(S_RANGE, gamma, color=GAMMA_COLOUR, linewidth=LINEWIDTH, label="Γ (call = put)")
    ax2.fill_between(S_RANGE, gamma, alpha=ALPHA_FILL, color=GAMMA_COLOUR)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax2.legend(fontsize=9, framealpha=0.3)

    fig.tight_layout()
    _save(fig, "plot1_delta_gamma.png")
    if show:
        plt.show()
    return fig


# ── Plot 2 — Theta Decay vs Time to Expiry ───────────────────────────────────

def plot_theta_decay(show: bool = True) -> plt.Figure:
    """Plot Call and Put Theta as time to expiry varies from 0.02 to 2.0 years.

    Fixed params: S=K=100, r=0.05, sigma=0.20.

    Args:
        show: If ``True``, display the figure interactively.

    Returns:
        Matplotlib Figure object.
    """
    # Use ATM spot for clearest theta signal
    s_atm = K

    call_theta = np.array(
        [compute_greeks(s_atm, K, t, R, SIGMA, "call").theta for t in T_RANGE]
    )
    put_theta = np.array(
        [compute_greeks(s_atm, K, t, R, SIGMA, "put").theta for t in T_RANGE]
    )

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0e0e0e")
    fig.suptitle(
        "Plot 2 — Theta Decay vs Time to Expiry (ATM: S = K = 100)\n"
        f"r={R}, σ={SIGMA}",
        fontsize=14, fontweight="bold", color="white",
    )

    ax.plot(T_RANGE, call_theta, color=CALL_COLOUR, linewidth=LINEWIDTH, label="Call Θ")
    ax.plot(T_RANGE, put_theta,  color=PUT_COLOUR,  linewidth=LINEWIDTH, label="Put Θ")
    ax.fill_between(T_RANGE, call_theta, alpha=ALPHA_FILL, color=CALL_COLOUR)
    ax.fill_between(T_RANGE, put_theta,  alpha=ALPHA_FILL, color=PUT_COLOUR)

    # Mark T=1.0 reference
    ax.axvline(1.0, color="#888888", linewidth=0.8, linestyle=":", label="T = 1.0 yr")

    _add_zero_line(ax)
    ax.set_xlabel("Time to Expiry  T (years)", fontsize=10, labelpad=6)
    ax.set_ylabel("Theta  (annualised $)", fontsize=10, labelpad=6)
    ax.set_title("", fontsize=1)  # suptitle already set
    ax.tick_params(colors="#cccccc", labelsize=9)
    ax.grid(True, color="#333333", linewidth=0.6, linestyle="--")
    ax.spines[:].set_color("#444444")
    ax.legend(fontsize=9, framealpha=0.3)

    # Annotation: near-expiry acceleration
    ax.annotate(
        "Rapid decay\nnear expiry",
        xy=(0.10, call_theta[int(0.10 / 2.0 * len(T_RANGE))]),
        xytext=(0.35, call_theta[int(0.35 / 2.0 * len(T_RANGE))] - 3),
        color="#cccccc", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )

    fig.tight_layout()
    _save(fig, "plot2_theta_decay.png")
    if show:
        plt.show()
    return fig


# ── Plot 3 — Vega vs Spot Price (multi-sigma) ────────────────────────────────

def plot_vega_surface(show: bool = True) -> plt.Figure:
    """Plot Vega vs spot price for multiple volatility levels.

    One curve per sigma in SIGMA_LEVELS = [0.10, 0.20, 0.30, 0.40].
    Fixed params: K=100, T=1.0, r=0.05.

    Args:
        show: If ``True``, display the figure interactively.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0e0e0e")
    fig.suptitle(
        "Plot 3 — Vega vs Spot Price  (one curve per σ level)\n"
        f"K={K}, T={T}, r={R}",
        fontsize=14, fontweight="bold", color="white",
    )

    for sigma, colour in zip(SIGMA_LEVELS, SIGMA_COLOURS):
        vega = np.array(
            [compute_greeks(s, K, T, R, sigma, "call").vega for s in S_RANGE]
        )
        ax.plot(S_RANGE, vega, color=colour, linewidth=LINEWIDTH,
                label=f"σ = {sigma:.0%}")
        ax.fill_between(S_RANGE, vega, alpha=ALPHA_FILL, color=colour)

    ax.axvline(K, color="#888888", linewidth=0.8, linestyle=":", label=f"K = {K:.0f}")
    ax.set_xlabel("Spot Price  S ($)", fontsize=10, labelpad=6)
    ax.set_ylabel("Vega  ($ per unit σ)", fontsize=10, labelpad=6)
    ax.tick_params(colors="#cccccc", labelsize=9)
    ax.grid(True, color="#333333", linewidth=0.6, linestyle="--")
    ax.spines[:].set_color("#444444")
    ax.legend(fontsize=9, framealpha=0.3, title="Volatility", title_fontsize=9)

    fig.tight_layout()
    _save(fig, "plot3_vega_surface.png")
    if show:
        plt.show()
    return fig


# ── Plot 4 — Option Price vs Spot (Payoff Diagram) ───────────────────────────

def plot_payoff_diagram(show: bool = True) -> plt.Figure:
    """Plot BS call price, BS put price, and intrinsic (expiry) payoffs vs spot.

    Fixed params: K=100, T=1.0, r=0.05, sigma=0.20.

    Args:
        show: If ``True``, display the figure interactively.

    Returns:
        Matplotlib Figure object.
    """
    call_bs = _price_series("call")
    put_bs  = _price_series("put")

    # Intrinsic payoffs (at expiry)
    call_intrinsic = np.maximum(S_RANGE - K, 0)
    put_intrinsic  = np.maximum(K - S_RANGE, 0)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0e0e0e")
    fig.suptitle(
        "Plot 4 — Option Price vs Spot  (BS price + Intrinsic payoff)\n"
        f"K={K}, T={T}, r={R}, σ={SIGMA}",
        fontsize=14, fontweight="bold", color="white",
    )

    # BS prices
    ax.plot(S_RANGE, call_bs,  color=CALL_COLOUR,     linewidth=LINEWIDTH,    label="Call (BS)")
    ax.plot(S_RANGE, put_bs,   color=PUT_COLOUR,       linewidth=LINEWIDTH,    label="Put (BS)")

    # Intrinsic payoffs (dashed)
    ax.plot(S_RANGE, call_intrinsic, color=INTRINSIC_COLOUR, linewidth=1.2,
            linestyle="--", label="Call intrinsic (at expiry)")
    ax.plot(S_RANGE, put_intrinsic,  color=INTRINSIC_COLOUR, linewidth=1.2,
            linestyle=":",  label="Put intrinsic (at expiry)")

    # Time value shading: gap between BS price and intrinsic
    ax.fill_between(
        S_RANGE, call_bs, call_intrinsic,
        where=(call_bs >= call_intrinsic),
        alpha=0.12, color=CALL_COLOUR, label="Call time value",
    )
    ax.fill_between(
        S_RANGE, put_bs, put_intrinsic,
        where=(put_bs >= put_intrinsic),
        alpha=0.12, color=PUT_COLOUR, label="Put time value",
    )

    ax.axvline(K, color="#888888", linewidth=0.8, linestyle=":", label=f"ATM  K = {K:.0f}")
    _add_zero_line(ax)

    ax.set_xlabel("Spot Price  S ($)", fontsize=10, labelpad=6)
    ax.set_ylabel("Option Price  ($)", fontsize=10, labelpad=6)
    ax.set_ylim(bottom=-1)
    ax.tick_params(colors="#cccccc", labelsize=9)
    ax.grid(True, color="#333333", linewidth=0.6, linestyle="--")
    ax.spines[:].set_color("#444444")
    ax.legend(fontsize=8, framealpha=0.3, ncol=2)

    fig.tight_layout()
    _save(fig, "plot4_payoff_diagram.png")
    if show:
        plt.show()
    return fig


# ── Master runner ─────────────────────────────────────────────────────────────

def run_all(show: bool = True) -> list[plt.Figure]:
    """Generate all 4 Greek visualisation plots.

    Args:
        show: If ``True``, each plot is displayed interactively after saving.

    Returns:
        List of the four Matplotlib Figure objects [fig1, fig2, fig3, fig4].
    """
    print("\nBlack-Scholes Greeks Visualisation")
    print("=" * 38)
    print(f"Fixed params: K={K}, T={T}, r={R}, sigma={SIGMA}")
    print(f"Output dir  : {os.path.abspath(PLOTS_DIR)}\n")

    figs = []
    tasks = [
        ("Plot 1 — Delta & Gamma vs Spot Price",         plot_delta_gamma),
        ("Plot 2 — Theta Decay vs Time to Expiry",       plot_theta_decay),
        ("Plot 3 — Vega vs Spot (multi-sigma)",          plot_vega_surface),
        ("Plot 4 — Option Price vs Spot (Payoff)",       plot_payoff_diagram),
    ]

    for label, fn in tasks:
        print(f"Generating {label} ...")
        figs.append(fn(show=show))

    print("\nAll plots generated successfully.")
    return figs


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Black-Scholes Greeks visualisation plots."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save PNGs only; do not open interactive windows.",
    )
    args = parser.parse_args()

    # Use non-interactive backend when --no-show is passed (CI / headless)
    if args.no_show:
        matplotlib.use("Agg")

    run_all(show=not args.no_show)

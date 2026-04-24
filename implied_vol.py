"""
Implied Volatility Solver
==========================
Recovers the implied volatility (IV) embedded in a market option price
using two numerical methods:

  Primary  — Newton-Raphson  (fast, ~5 iterations when vega is healthy)
  Fallback — Brentq          (robust bracketing via scipy, used when
                               Newton-Raphson exceeds 100 iterations or
                               vega falls below the near-zero threshold)

Also provides:
  * plot_iv_smile()  — IV smile visualisation (saved to plots/plot5_iv_smile.png)
  * run_solver_cli() — interactive prompt for quick IV lookups

Usage:
    python implied_vol.py          # launches CLI
    python implied_vol.py --smile  # generates smile plot only (headless-safe)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
from options_engine import black_scholes, compute_greeks, validate_inputs  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
SIGMA_INIT: float = 0.20          # Newton-Raphson starting guess
SIGMA_MIN: float = 1e-6           # lower bound for Brentq bracket / NR clamp
SIGMA_MAX: float = 1.0            # upper bound — matches validate_inputs ceiling
NR_MAX_ITER: int = 100            # Newton-Raphson iteration cap
NR_TOL: float = 1e-8              # Newton-Raphson price convergence tolerance
VEGA_FLOOR: float = 1e-10         # below this → skip NR, go straight to Brentq

PLOTS_DIR: str = os.path.join(os.path.dirname(__file__), "plots")
SEPARATOR: str = "=" * 54

OptionType = Literal["call", "put"]

# ── Smile surface parameters ──────────────────────────────────────────────────
SMILE_SIGMA_BASE: float = 0.20    # flat ATM vol
SMILE_SKEW_COEFF: float = 0.15    # curvature coefficient: σ(K) = base + coeff*(K/S-1)²

# ── Dark-theme colour palette (matches greeks_viz.py) ────────────────────────
CALL_COLOUR = "#4FC3F7"
PUT_COLOUR = "#FF8A65"
FLAT_COLOUR = "#888888"
SMILE_COLOUR = "#A5D6A7"

plt.style.use("dark_background")


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class IVResult:
    """Result returned by :func:`implied_vol`.

    Attributes:
        implied_vol:  Recovered implied volatility (decimal, e.g. 0.2345).
        iterations:   Number of iterations consumed.
        method:       ``'Newton-Raphson'`` or ``'Brentq'``.
        converged:    ``True`` if the solver reached the tolerance target.
        price_error:  |BS(sigma_iv) - market_price| at the solution.
    """
    implied_vol: float
    iterations: int
    method: str
    converged: bool
    price_error: float


# ── Core solver ───────────────────────────────────────────────────────────────
def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    *,
    sigma_init: float = SIGMA_INIT,
    nr_max_iter: int = NR_MAX_ITER,
    nr_tol: float = NR_TOL,
) -> IVResult:
    """Recover implied volatility from a market option price.

    Uses Newton-Raphson as the primary method. Automatically falls back to
    Brentq (scipy) if NR fails to converge or if vega is near zero.

    Args:
        market_price: Observed market price of the option (must be > 0).
        S:            Spot price of the underlying.
        K:            Strike price.
        T:            Time to expiration in years.
        r:            Annualised risk-free rate (decimal).
        option_type:  ``'call'`` or ``'put'``.
        sigma_init:   Initial volatility guess for Newton-Raphson (default 0.20).
        nr_max_iter:  Maximum Newton-Raphson iterations before fallback.
        nr_tol:       Price convergence tolerance (default 1e-8).

    Returns:
        :class:`IVResult` with implied vol, iteration count, method used, and
        convergence metadata.

    Raises:
        ValueError: If inputs are invalid, the market price is below intrinsic
                    value, or no solution can be bracketed for Brentq.
    """
    # ── Validate market inputs ────────────────────────────────────────────────
    validate_inputs(S, K, T, r, sigma_init)
    if market_price <= 0:
        raise ValueError(f"market_price must be > 0, got {market_price}")

    # ── Intrinsic value check ─────────────────────────────────────────────────
    discount = K * np.exp(-r * T)
    if option_type == "call":
        intrinsic = max(S - discount, 0.0)
    else:
        intrinsic = max(discount - S, 0.0)

    if market_price < intrinsic - 1e-6:
        raise ValueError(
            f"market_price ${market_price:.4f} is below intrinsic value "
            f"${intrinsic:.4f}. No implied volatility exists."
        )

    # Upper bound sanity: price cannot exceed S (call) or K·e^{-rT} (put)
    upper_bound = S if option_type == "call" else discount
    if market_price >= upper_bound:
        raise ValueError(
            f"market_price ${market_price:.4f} exceeds the theoretical upper "
            f"bound ${upper_bound:.4f} for a {option_type}."
        )

    # ── Newton-Raphson ────────────────────────────────────────────────────────
    sigma = sigma_init
    method = "Newton-Raphson"
    iterations = 0
    converged = False

    for i in range(1, nr_max_iter + 1):
        iterations = i
        bs_price = black_scholes(S, K, T, r, sigma, option_type)
        price_diff = bs_price - market_price

        if abs(price_diff) < nr_tol:
            converged = True
            break

        vega = compute_greeks(S, K, T, r, sigma, option_type).vega

        if abs(vega) < VEGA_FLOOR:
            # Near-zero vega → NR step would explode; fall through to Brentq
            break

        sigma = sigma - price_diff / vega

        # Keep sigma in a sane range between steps
        sigma = max(SIGMA_MIN, min(sigma, SIGMA_MAX))

    # ── Brentq fallback ───────────────────────────────────────────────────────
    if not converged:
        method = "Brentq"

        def objective(s: float) -> float:
            return black_scholes(S, K, T, r, s, option_type) - market_price

        # Verify the root is bracketed
        f_low = objective(SIGMA_MIN)
        f_high = objective(SIGMA_MAX)
        if f_low * f_high > 0:
            raise ValueError(
                "Cannot bracket a solution for implied volatility. "
                "Check that the market price is arbitrage-free."
            )

        brentq_iters = [0]

        def counted_objective(s: float) -> float:
            brentq_iters[0] += 1
            return objective(s)

        sigma = brentq(counted_objective, SIGMA_MIN, SIGMA_MAX, xtol=1e-10, rtol=1e-10)
        iterations += brentq_iters[0]
        converged = True

    price_error = abs(black_scholes(S, K, T, r, sigma, option_type) - market_price)

    return IVResult(
        implied_vol=sigma,
        iterations=iterations,
        method=method,
        converged=converged,
        price_error=price_error,
    )


# ── Vol smile visualisation ───────────────────────────────────────────────────
def plot_iv_smile(
    S: float = 100.0,
    K_range: np.ndarray | None = None,
    T: float = 1.0,
    r: float = 0.05,
    option_type: OptionType = "call",
    show: bool = True,
) -> plt.Figure:
    """Plot implied volatility smile: recovered IV vs strike.

    Simulates a realistic vol surface using:
        σ_true(K) = σ_base + 0.15 · (K/S − 1)²

    BS prices are generated at these vols, then IV is recovered from each
    price. The flat BS assumption (σ = σ_base) is shown as a dashed
    horizontal reference line.

    Args:
        S:           Spot price (default 100).
        K_range:     Array of strike prices. Defaults to linspace(70, 130, 61).
        T:           Time to expiration in years (default 1.0).
        r:           Risk-free rate (default 0.05).
        option_type: ``'call'`` or ``'put'`` (default ``'call'``).
        show:        Display interactively if ``True``.

    Returns:
        Matplotlib Figure object. PNG saved to plots/plot5_iv_smile.png.
    """
    if K_range is None:
        K_range = np.linspace(70, 130, 61)

    # ── Build the "true" vol surface and market prices ────────────────────────
    true_vols = SMILE_SIGMA_BASE + SMILE_SKEW_COEFF * ((K_range / S) - 1.0) ** 2
    market_prices = np.array(
        [black_scholes(S, K, T, r, sv, option_type) for K, sv in zip(K_range, true_vols)]
    )

    # ── Recover IV from each market price ─────────────────────────────────────
    implied_vols: list[float] = []
    methods: list[str] = []
    skipped: list[float] = []

    for K, mp, tv in zip(K_range, market_prices, true_vols):
        try:
            result = implied_vol(mp, S, K, T, r, option_type, sigma_init=tv)
            implied_vols.append(result.implied_vol)
            methods.append(result.method)
        except ValueError:
            implied_vols.append(float("nan"))
            methods.append("failed")
            skipped.append(K)

    iv_arr = np.array(implied_vols)
    nr_mask = np.array([m == "Newton-Raphson" for m in methods])
    bq_mask = np.array([m == "Brentq" for m in methods])

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#0e0e0e")
    fig.suptitle(
        "Plot 5 — Implied Volatility Smile\n"
        f"S={S}, T={T}, r={r}, option_type='{option_type}'  |  "
        f"True surface: σ(K) = {SMILE_SIGMA_BASE} + {SMILE_SKEW_COEFF}·(K/S−1)²",
        fontsize=12, fontweight="bold", color="white", y=1.02,
    )

    # True vol surface (what we fed in)
    ax.plot(K_range, true_vols * 100, color=SMILE_COLOUR, linewidth=2.0,
            label="True vol surface  σ(K)", zorder=3)

    # Recovered IVs — NR points
    if nr_mask.any():
        ax.scatter(K_range[nr_mask], iv_arr[nr_mask] * 100,
                   color=CALL_COLOUR, s=22, zorder=5, label="Recovered IV — Newton-Raphson")

    # Recovered IVs — Brentq points
    if bq_mask.any():
        ax.scatter(K_range[bq_mask], iv_arr[bq_mask] * 100,
                   color=PUT_COLOUR, s=28, marker="D", zorder=5, label="Recovered IV — Brentq")

    # Flat BS assumption reference
    ax.axhline(SMILE_SIGMA_BASE * 100, color=FLAT_COLOUR, linewidth=1.2,
               linestyle="--", label=f"Flat BS assumption  σ = {SMILE_SIGMA_BASE:.0%}", zorder=2)

    # ATM marker
    ax.axvline(S, color="#666666", linewidth=0.8, linestyle=":", label=f"ATM  S = {S:.0f}")

    # Shaded region: smile premium over flat vol
    valid = ~np.isnan(iv_arr)
    ax.fill_between(K_range[valid], SMILE_SIGMA_BASE * 100, iv_arr[valid] * 100,
                    alpha=0.10, color=SMILE_COLOUR, label="Smile premium")

    ax.set_xlabel("Strike  K ($)", fontsize=10, labelpad=6)
    ax.set_ylabel("Implied Volatility (%)", fontsize=10, labelpad=6)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.tick_params(colors="#cccccc", labelsize=9)
    ax.grid(True, color="#333333", linewidth=0.6, linestyle="--")
    ax.spines[:].set_color("#444444")
    ax.legend(fontsize=8, framealpha=0.3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3)

    # Annotation: smile wings
    wing_idx = int(0.05 * len(K_range))
    ax.annotate(
        "Volatility smile\n(wings > ATM)",
        xy=(K_range[wing_idx], iv_arr[wing_idx] * 100),
        xytext=(K_range[wing_idx] + 4, iv_arr[wing_idx] * 100 + 1.5),
        color="#cccccc", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "plot5_iv_smile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [saved] {path}")

    if show:
        plt.show()
    return fig


# ── Interactive CLI ───────────────────────────────────────────────────────────
def _prompt_float(prompt: str) -> float:
    """Prompt until a valid float is entered."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  [!] Please enter a valid number.")


def _prompt_option_type() -> OptionType:
    """Prompt until 'call' or 'put' is entered."""
    while True:
        raw = input("Option type (call / put): ").strip().lower()
        if raw in ("call", "put"):
            return raw  # type: ignore[return-value]
        print("  [!] Enter 'call' or 'put'.")


def run_solver_cli() -> None:
    """Interactive CLI for the implied volatility solver.

    Prompts for market_price, S, K, T, r, option_type and prints a
    formatted result showing implied vol, iterations, method, and a
    convergence note.
    """
    print()
    print(SEPARATOR)
    print("    Implied Volatility Solver".center(54))
    print(SEPARATOR)
    print()

    while True:
        try:
            market_price = _prompt_float("Market Option Price ($): ")
            S = _prompt_float("Spot Price (S): $")
            K = _prompt_float("Strike Price (K): $")
            T = _prompt_float("Time to Maturity (T) in years: ")
            r = _prompt_float("Risk-free Rate (r) as decimal (e.g. 0.05): ")
            option_type = _prompt_option_type()
            break
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    print("\nSolving...")

    try:
        result = implied_vol(market_price, S, K, T, r, option_type)
    except ValueError as exc:
        print(f"\n  [ERROR] {exc}")
        return

    # ── Convergence note ──────────────────────────────────────────────────────
    if result.method == "Newton-Raphson" and result.iterations <= 10:
        convergence_note = f"Fast quadratic convergence ({result.iterations} iterations)."
    elif result.method == "Newton-Raphson":
        convergence_note = f"Slow NR convergence ({result.iterations} iterations). Consider checking inputs."
    else:
        convergence_note = (
            f"Newton-Raphson did not converge; Brentq bracketing used "
            f"({result.iterations} total iterations). Solution is reliable."
        )

    print()
    print(SEPARATOR)
    print("                    RESULT".center(54))
    print(SEPARATOR)
    print(f"Implied Volatility : {result.implied_vol:.4f}  ({result.implied_vol:.2%})")
    print(f"Method             : {result.method}")
    print(f"Iterations         : {result.iterations}")
    print(f"Price Error        : {result.price_error:.2e}")
    print(f"Converged          : {'Yes' if result.converged else 'No'}")
    print()
    print(f"Convergence note: {convergence_note}")
    print(SEPARATOR)
    print()

    # Offer to generate smile plot
    try:
        ans = input("Generate IV smile plot for this option? (y/n): ").strip().lower()
    except KeyboardInterrupt:
        return

    if ans == "y":
        K_range = np.linspace(max(S * 0.6, 1), S * 1.4, 81)
        plot_iv_smile(S=S, K_range=K_range, T=T, r=r, option_type=option_type)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implied Volatility Solver — Newton-Raphson + Brentq fallback."
    )
    parser.add_argument(
        "--smile",
        action="store_true",
        help="Generate IV smile plot only (no CLI prompt).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save PNG only; skip interactive display.",
    )
    args = parser.parse_args()

    if args.no_show or args.smile:
        matplotlib.use("Agg")

    if args.smile:
        print("Generating IV smile plot ...")
        plot_iv_smile(show=not args.no_show)
    else:
        run_solver_cli()

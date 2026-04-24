"""
Heston Stochastic Volatility Pricer
=====================================
Prices European options under the Heston (1993) model using the
semi-analytical characteristic-function formula with numerical integration
(scipy.integrate.quad).

Heston dynamics under the risk-neutral measure Q:
    dS   = r.S.dt + sqrtv.S.dW_S
    dv   = kappa(theta - v).dt + sigma_v.sqrtv.dW_v
    dW_S.dW_v = rho.dt

Parameters:
    v0      initial variance  (e.g. 0.04 → 20% initial vol)
    kappa   mean-reversion speed
    theta   long-run variance (long-run vol = sqrttheta)
    sigma_v volatility of variance (vol-of-vol)
    rho     spot–vol correlation (typically negative for equities, e.g. -0.7)

Pricing formula:
    C = S.P1 - K.e^{-rT}.P2

    P2 = ½ + (1/π) ∫₀^∞ Re[ e^{-iphi.lnK} . Phi(phi)   / (iphi)           ] dphi
    P1 = ½ + (1/π) ∫₀^∞ Re[ e^{-iphi.lnK} . Phi(phi-i) / (iphi . S.e^{rT}) ] dphi

    where Phi(u) is the risk-neutral characteristic function of ln(S_T)
    implemented using the numerically stable Albrecher et al. (2007)
    formulation to avoid branch-cut discontinuities.

Usage:
    python heston.py              # demo: price table + Plot 7
    python heston.py --no-show    # headless (save PNG, skip display)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.integrate import IntegrationWarning, quad

sys.path.insert(0, os.path.dirname(__file__))
from options_engine import black_scholes      # noqa: E402
from implied_vol import implied_vol as iv_solver  # noqa: E402

warnings.filterwarnings("ignore", category=IntegrationWarning)

# ── Default Heston parameters ─────────────────────────────────────────────────
DEFAULT_S: float = 100.0
DEFAULT_K: float = 100.0
DEFAULT_T: float = 1.0
DEFAULT_R: float = 0.05
DEFAULT_V0: float = 0.04        # initial variance  → 20% initial vol
DEFAULT_KAPPA: float = 2.0      # mean-reversion speed
DEFAULT_THETA: float = 0.04     # long-run variance → 20% long-run vol
DEFAULT_SIGMA_V: float = 0.30   # vol of vol
DEFAULT_RHO: float = -0.70      # spot–vol correlation (negative = equity skew)

# ── Integration settings ──────────────────────────────────────────────────────
INTEGRATION_UPPER: float = 500.0   # upper limit for Gil-Pelaez integration
INTEGRATION_LIMIT: int = 500       # max adaptive sub-intervals (scipy quad)
INTEGRATION_LOWER: float = 1e-6   # start just above 0 to avoid 1/(iphi) pole

# ── Output directory ──────────────────────────────────────────────────────────
PLOTS_DIR: str = os.path.join(os.path.dirname(__file__), "plots")

# ── Dark-theme palette (matches project style) ────────────────────────────────
plt.style.use("dark_background")
HESTON_COLOUR = "#4FC3F7"     # sky blue
BS_COLOUR = "#FF8A65"         # warm orange
ATM_COLOUR = "#888888"        # grey ATM marker
DIFF_COLOUR = "#A5D6A7"       # soft green (skew premium)


# ── Characteristic function (Albrecher et al. 2007 stable form) ──────────────
def _heston_cf(
    u: complex | float,
    S: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
) -> complex:
    """Risk-neutral characteristic function of ln(S_T) under the Heston model.

    Uses the numerically stable Albrecher et al. (2007) formulation:
    the exponent uses e^{-dT} (bounded) instead of e^{+dT} (can diverge),
    avoiding branch-cut discontinuities for long maturities.

    For a real argument ``u = phi`` this gives Phi(phi).
    For a shifted argument ``u = phi - i`` this gives Phi(phi - i), used for P1.

    Args:
        u:       Integration variable (real for P2, complex phi-i for P1).
        S:       Spot price.
        T:       Time to expiration (years).
        r:       Risk-free rate.
        v0:      Initial variance.
        kappa:   Mean-reversion speed.
        theta:   Long-run variance.
        sigma_v: Volatility of variance.
        rho:     Spot–vol correlation.

    Returns:
        Complex value of the characteristic function at ``u``.
    """
    xi = kappa - rho * sigma_v * 1j * u
    # d: square root of the Riccati discriminant (principal root, Re(d) >= 0)
    d = np.sqrt(xi ** 2 + sigma_v ** 2 * (u ** 2 + 1j * u))

    # Stable ratio (avoids e^{+dT} blowup)
    rr = (xi - d) / (xi + d)
    exp_neg_dT = np.exp(-d * T)

    # Integrated variance term (C) and instantaneous variance loading (D)
    C = (kappa * theta / sigma_v ** 2) * (
        (xi - d) * T - 2.0 * np.log((1.0 - rr * exp_neg_dT) / (1.0 - rr))
    )
    D = ((xi - d) / sigma_v ** 2) * (1.0 - exp_neg_dT) / (1.0 - rr * exp_neg_dT)

    # Full CF:  exp(C + D.v₀ + iu.(ln S + rT))
    return np.exp(C + D * v0 + 1j * u * (np.log(S) + r * T))


# ── Input validation ───────────────────────────────────────────────────────────
def validate_heston_inputs(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
) -> None:
    """Validate Heston model parameters and raise descriptive errors.

    Also emits a warning when the Feller condition (2kappatheta >= sigma_v^2) is violated,
    meaning the variance process can touch zero.

    Args:
        S, K, T, r: Standard option market inputs.
        v0:         Initial variance.
        kappa:      Mean-reversion speed.
        theta:      Long-run variance.
        sigma_v:    Vol of vol.
        rho:        Spot–vol correlation.

    Raises:
        ValueError: If any parameter is outside its valid domain.
    """
    if S <= 0:
        raise ValueError(f"S must be > 0, got {S}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if not (0.0 <= r <= 1.0):
        raise ValueError(f"r must be in [0, 1], got {r}")
    if v0 <= 0:
        raise ValueError(f"v0 (initial variance) must be > 0, got {v0}")
    if kappa <= 0:
        raise ValueError(f"kappa must be > 0, got {kappa}")
    if theta <= 0:
        raise ValueError(f"theta (long-run variance) must be > 0, got {theta}")
    if sigma_v <= 0:
        raise ValueError(f"sigma_v (vol of vol) must be > 0, got {sigma_v}")
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must be in (-1, 1), got {rho}")

    # Feller condition: 2kappatheta >= sigma_v^2  (variance process stays positive)
    feller = 2.0 * kappa * theta
    if feller < sigma_v ** 2:
        warnings.warn(
            f"Feller condition violated: 2kappatheta = {feller:.4f} < sigma_v^2 = {sigma_v**2:.4f}. "
            "Variance process may reach zero; prices may be less accurate.",
            UserWarning,
            stacklevel=3,
        )


# ── Core pricer ────────────────────────────────────────────────────────────────
def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    option_type: str = "call",
) -> float:
    """Price a European option under the Heston stochastic volatility model.

    Uses the Gil-Pelaez inversion of the risk-neutral characteristic function:

        C = S.P1 - K.e^{-rT}.P2

    with P1 and P2 computed by numerical integration (scipy quad).
    Puts are priced via put-call parity.

    Args:
        S:           Spot price.
        K:           Strike price.
        T:           Time to expiration (years).
        r:           Risk-free rate (decimal).
        v0:          Initial variance  (e.g. 0.04 → 20% vol).
        kappa:       Mean-reversion speed of variance.
        theta:       Long-run variance.
        sigma_v:     Volatility of variance (vol-of-vol).
        rho:         Spot–vol correlation (typically negative for equities).
        option_type: ``'call'`` or ``'put'``.

    Returns:
        Theoretical option price >= 0.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    validate_heston_inputs(S, K, T, r, v0, kappa, theta, sigma_v, rho)
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    lnK = np.log(K)
    S_fwd = S * np.exp(r * T)   # forward price = Phi(-i) [verified in tests]

    def _integrand_p2(phi: float) -> float:
        """P2 integrand: real part of  e^{-iphi.lnK}.Phi(phi) / (iphi)."""
        cf = _heston_cf(phi, S, T, r, v0, kappa, theta, sigma_v, rho)
        return (np.exp(-1j * phi * lnK) * cf / (1j * phi)).real

    def _integrand_p1(phi: float) -> float:
        """P1 integrand: real part of  e^{-iphi.lnK}.Phi(phi-i) / (iphi.S.e^{rT})."""
        cf = _heston_cf(phi - 1j, S, T, r, v0, kappa, theta, sigma_v, rho)
        return (np.exp(-1j * phi * lnK) * cf / (1j * phi * S_fwd)).real

    I2 = quad(_integrand_p2, INTEGRATION_LOWER, INTEGRATION_UPPER,
              limit=INTEGRATION_LIMIT)[0]
    I1 = quad(_integrand_p1, INTEGRATION_LOWER, INTEGRATION_UPPER,
              limit=INTEGRATION_LIMIT)[0]

    P2 = 0.5 + I2 / np.pi
    P1 = 0.5 + I1 / np.pi

    call = max(S * P1 - K * np.exp(-r * T) * P2, 0.0)

    if option_type == "call":
        return call
    else:                           # put via put-call parity
        return max(call - S + K * np.exp(-r * T), 0.0)


# ── IV smile comparison ────────────────────────────────────────────────────────
def compare_bs_heston(
    S: float = DEFAULT_S,
    K_range: np.ndarray | None = None,
    T: float = DEFAULT_T,
    r: float = DEFAULT_R,
    v0: float = DEFAULT_V0,
    kappa: float = DEFAULT_KAPPA,
    theta: float = DEFAULT_THETA,
    sigma_v: float = DEFAULT_SIGMA_V,
    rho: float = DEFAULT_RHO,
    option_type: str = "call",
    show: bool = True,
) -> dict:
    """Compute BS and Heston prices across strikes, extract implied vols, plot.

    For each strike in ``K_range``:
      * BS price is computed at sigma = sqrttheta (the Heston long-run vol).
      * Heston price is computed via :func:`heston_price`.
      * Implied vols are extracted from both prices via the IV solver.

    The resulting smile comparison is saved as Plot 7.

    Args:
        S:           Spot price.
        K_range:     Array of strikes (default: linspace(75, 125, 51)).
        T:           Time to expiration.
        r:           Risk-free rate.
        v0, kappa, theta, sigma_v, rho: Heston parameters.
        option_type: ``'call'`` or ``'put'``.
        show:        Display plot interactively.

    Returns:
        Dict with keys ``'strikes'``, ``'heston_prices'``, ``'bs_prices'``,
        ``'heston_ivs'``, ``'bs_sigma'``.
    """
    if K_range is None:
        K_range = np.linspace(75.0, 125.0, 51)

    sigma_bs = np.sqrt(theta)   # flat vol = Heston long-run vol

    print(f"\nComputing prices across {len(K_range)} strikes …")
    print(f"  BS flat vol    : {sigma_bs:.2%}")
    print(f"  Heston params  : v0={v0}, kappa={kappa}, theta={theta}, "
          f"sigma_v={sigma_v}, rho={rho}\n")

    heston_prices: list[float] = []
    bs_prices: list[float] = []
    heston_ivs: list[float | None] = []
    valid_K: list[float] = []

    header = (f"{'Strike':>8} {'BS Price':>10} {'Heston Price':>13} "
              f"{'BS IV':>8} {'Heston IV':>11} {'IV Diff':>9}")
    print(header)
    print("-" * len(header))

    for K in K_range:
        try:
            # ── BS price at flat vol ──────────────────────────────────────
            bs_p = black_scholes(S, K, T, r, sigma_bs, option_type)

            # ── Heston price ──────────────────────────────────────────────
            h_p = heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho,
                                option_type)

            # ── Extract IV from Heston price ──────────────────────────────
            iv_res = iv_solver(h_p, S, K, T, r, option_type,
                               sigma_init=sigma_bs)
            h_iv = iv_res.implied_vol

            bs_prices.append(bs_p)
            heston_prices.append(h_p)
            heston_ivs.append(h_iv)
            valid_K.append(K)

            iv_diff = (h_iv - sigma_bs) * 100
            sign = "+" if iv_diff >= 0 else ""
            print(
                f"{K:>8.1f} {bs_p:>10.4f} {h_p:>13.4f} "
                f"{sigma_bs*100:>7.2f}% {h_iv*100:>10.2f}% "
                f"{sign}{iv_diff:>8.2f}pp"
            )

        except (ValueError, RuntimeError) as exc:
            # Skip strikes where IV cannot be recovered (e.g. deep OTM)
            print(f"{K:>8.1f}  [skipped: {exc}]")

    print()

    result = {
        "strikes": np.array(valid_K),
        "heston_prices": np.array(heston_prices),
        "bs_prices": np.array(bs_prices),
        "heston_ivs": np.array(heston_ivs),
        "bs_sigma": sigma_bs,
    }

    _plot_heston_smile(result, S, T, r, v0, kappa, theta, sigma_v, rho,
                       option_type, show=show)
    return result


def _plot_heston_smile(
    data: dict,
    S: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    option_type: str,
    show: bool = True,
) -> plt.Figure:
    """Generate Plot 7: Heston IV smile vs flat Black-Scholes assumption.

    Args:
        data:        Output dict from :func:`compare_bs_heston`.
        S, T, r:     Market parameters.
        v0, kappa, theta, sigma_v, rho: Heston parameters.
        option_type: ``'call'`` or ``'put'``.
        show:        Display interactively.

    Returns:
        Matplotlib Figure. PNG saved to plots/plot7_heston_smile.png.
    """
    strikes = data["strikes"]
    heston_ivs = data["heston_ivs"] * 100      # → percentages
    bs_sigma_pct = data["bs_sigma"] * 100
    atm_iv = np.interp(S, strikes, heston_ivs)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), facecolor="#0e0e0e",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.10},
    )
    fig.suptitle(
        f"Plot 7 — Heston Model: IV Smile vs Flat Black-Scholes  |  "
        f"S={S}, T={T}, r={r}\n"
        f"kappa={kappa}, theta={theta} (long-run sigma={np.sqrt(theta):.0%}), "
        f"sigma_v={sigma_v}, rho={rho}, v₀={v0}",
        fontsize=11, fontweight="bold", color="white", y=1.02,
    )

    # ── Top panel: IV smile ────────────────────────────────────────────────
    ax1.plot(strikes, heston_ivs, color=HESTON_COLOUR, linewidth=2.0,
             zorder=4, label="Heston implied vol  sigma(K)")
    ax1.axhline(bs_sigma_pct, color=BS_COLOUR, linewidth=1.5,
                linestyle="--", zorder=3,
                label=f"Black-Scholes flat vol  sigma = {data['bs_sigma']:.0%}")
    ax1.axvline(S, color=ATM_COLOUR, linewidth=0.9, linestyle=":",
                label=f"ATM  S = {S:.0f}")

    # Shade skew premium (Heston above flat BS)
    ax1.fill_between(
        strikes, heston_ivs, bs_sigma_pct,
        where=(heston_ivs >= bs_sigma_pct),
        alpha=0.12, color=HESTON_COLOUR, label="Skew premium",
    )
    ax1.fill_between(
        strikes, heston_ivs, bs_sigma_pct,
        where=(heston_ivs < bs_sigma_pct),
        alpha=0.12, color=BS_COLOUR, label="Vol discount",
    )

    # Scatter dots on the smile
    ax1.scatter(strikes, heston_ivs, color=HESTON_COLOUR, s=18, zorder=5)

    # Annotate ATM Heston IV
    ax1.annotate(
        f"ATM Heston IV\n{atm_iv:.2f}%",
        xy=(S, atm_iv),
        xytext=(S + 4, atm_iv + 0.6),
        color="white", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8),
    )

    # OTM put wing annotation
    wing_k = strikes[int(0.05 * len(strikes))]
    wing_iv = heston_ivs[int(0.05 * len(strikes))]
    ax1.annotate(
        f"OTM put wing\n(rho = {rho:.1f} skew)",
        xy=(wing_k, wing_iv),
        xytext=(wing_k + 6, wing_iv - 0.8),
        color="#cccccc", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )

    ax1.set_ylabel("Implied Volatility  (%)", fontsize=10, labelpad=6)
    ax1.tick_params(colors="#cccccc", labelsize=9, labelbottom=False)
    ax1.grid(True, color="#2a2a2a", linewidth=0.6, linestyle="--")
    ax1.spines[:].set_color("#444444")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax1.legend(fontsize=8, framealpha=0.3, loc="upper right")

    # ── Bottom panel: IV difference (Heston - BS flat) ────────────────────
    iv_diff = heston_ivs - bs_sigma_pct
    bar_colours = [HESTON_COLOUR if d >= 0 else BS_COLOUR for d in iv_diff]
    ax2.bar(strikes, iv_diff, width=(strikes[1] - strikes[0]) * 0.8,
            color=bar_colours, alpha=0.85, zorder=3)
    ax2.axhline(0, color=ATM_COLOUR, linewidth=0.8, linestyle="-")
    ax2.set_ylabel("IV diff  (pp)", fontsize=9, labelpad=6)
    ax2.set_xlabel("Strike  K ($)", fontsize=10, labelpad=6)
    ax2.tick_params(colors="#cccccc", labelsize=8)
    ax2.grid(True, color="#2a2a2a", linewidth=0.5, linestyle="--", axis="y")
    ax2.spines[:].set_color("#444444")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "plot7_heston_smile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [saved] {path}\n")

    if show:
        plt.show()
    return fig


# ── Demo CLI ──────────────────────────────────────────────────────────────────
def main(show: bool = True) -> None:
    """Run the Heston model demo: spot-price table + IV smile plot."""
    sep = "=" * 54

    print()
    print(sep)
    print("  Heston Stochastic Volatility Pricer".center(54))
    print(sep)
    print()

    S, K, T, r = DEFAULT_S, DEFAULT_K, DEFAULT_T, DEFAULT_R
    v0, kappa, theta = DEFAULT_V0, DEFAULT_KAPPA, DEFAULT_THETA
    sigma_v, rho = DEFAULT_SIGMA_V, DEFAULT_RHO

    # ── Feller check ──────────────────────────────────────────────────────
    feller_lhs = 2 * kappa * theta
    feller_ok = feller_lhs >= sigma_v ** 2
    print(f"  Parameters:  S={S}, K={K}, T={T}, r={r}")
    print(f"               v0={v0}, kappa={kappa}, theta={theta}, "
          f"sigma_v={sigma_v}, rho={rho}")
    print(f"  Long-run vol : {np.sqrt(theta):.0%}")
    print(f"  Feller (2kappatheta={feller_lhs:.3f} >= sigma_v^2={sigma_v**2:.3f}): "
          f"{'satisfied' if feller_ok else 'VIOLATED — variance may hit 0'}")
    print()

    # ── Single ATM price comparison ───────────────────────────────────────
    print(sep)
    print("  Single ATM Pricing Comparison")
    print(sep)
    sigma_bs = np.sqrt(theta)
    h_call = heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, "call")
    h_put  = heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, "put")
    bs_call = black_scholes(S, K, T, r, sigma_bs, "call")
    bs_put  = black_scholes(S, K, T, r, sigma_bs, "put")
    parity_error = abs((h_call - h_put) - (S - K * np.exp(-r * T)))

    print(f"  {'':20s} {'Call':>10} {'Put':>10}")
    print(f"  {'Black-Scholes (sigma=20%):':20s} {bs_call:>10.4f} {bs_put:>10.4f}")
    print(f"  {'Heston:':20s} {h_call:>10.4f} {h_put:>10.4f}")
    print(f"  {'Difference:':20s} {h_call-bs_call:>+10.4f} {h_put-bs_put:>+10.4f}")
    print(f"  Put-call parity error (Heston): {parity_error:.2e}")
    print()

    # ── IV extraction at ATM ──────────────────────────────────────────────
    iv_res = iv_solver(h_call, S, K, T, r, "call", sigma_init=sigma_bs)
    print(f"  ATM Heston IV : {iv_res.implied_vol:.4f}  "
          f"({iv_res.implied_vol:.2%})  via {iv_res.method}  "
          f"({iv_res.iterations} iter)")
    print()

    # ── Full smile comparison ─────────────────────────────────────────────
    K_smile = np.linspace(75.0, 125.0, 51)
    compare_bs_heston(
        S=S, K_range=K_smile, T=T, r=r,
        v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho,
        option_type="call", show=show,
    )

    print(sep)
    print("  Done.")
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heston stochastic volatility pricer — IV smile vs BS."
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

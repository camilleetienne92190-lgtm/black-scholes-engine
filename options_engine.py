"""
Black-Scholes Options Pricing Engine
Computes European call/put prices and all 5 Greeks analytically.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal

# ── Constants ────────────────────────────────────────────────────────────────
DAYS_IN_YEAR: int = 365
VEGA_SCALE: float = 1.0          # raw vega (per unit vol); displayed as-is
RHO_SCALE: float = 1.0           # raw rho; displayed as-is

OptionType = Literal["call", "put"]


# ── Result containers ────────────────────────────────────────────────────────
@dataclass
class Greeks:
    """Analytical Black-Scholes Greeks for one option leg."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass
class OptionResult:
    """Full pricing result for one option leg."""
    option_type: OptionType
    price: float
    greeks: Greeks


@dataclass
class PricingResult:
    """Combined result for a call/put pair on the same underlier."""
    call: OptionResult
    put: OptionResult
    parity_error: float


# ── Input validation ─────────────────────────────────────────────────────────
def validate_inputs(S: float, K: float, T: float, r: float, sigma: float) -> None:
    """Validate Black-Scholes inputs and raise ValueError on bad values.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration in years.
        r: Annualised risk-free rate (decimal, e.g. 0.05).
        sigma: Annualised volatility (decimal, e.g. 0.20).

    Raises:
        ValueError: If any parameter is outside its valid domain.
    """
    if S <= 0:
        raise ValueError(f"Spot price S must be > 0, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be > 0, got {K}")
    if T <= 0:
        raise ValueError(f"Time to maturity T must be > 0, got {T}")
    if not (0.0 <= r <= 1.0):
        raise ValueError(f"Risk-free rate r must be in [0, 1], got {r}")
    if not (0.0 < sigma <= 1.0):
        raise ValueError(f"Volatility sigma must be in (0, 1], got {sigma}")


# ── Core helpers ─────────────────────────────────────────────────────────────
def _compute_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """Compute d1 and d2 from the Black-Scholes formula.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiration (years).
        r: Risk-free rate.
        sigma: Volatility.

    Returns:
        Tuple (d1, d2).
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


# ── Pricing ──────────────────────────────────────────────────────────────────
def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Price a European vanilla option using the Black-Scholes formula.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price.
        T: Time to expiration in years.
        r: Annualised risk-free rate (decimal).
        sigma: Annualised volatility (decimal).
        option_type: ``'call'`` or ``'put'``.

    Returns:
        Theoretical option price.

    Raises:
        ValueError: If inputs fail validation or option_type is unknown.
    """
    validate_inputs(S, K, T, r, sigma)
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    discount = K * np.exp(-r * T)

    if option_type == "call":
        return S * norm.cdf(d1) - discount * norm.cdf(d2)
    elif option_type == "put":
        return discount * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# ── Greeks ───────────────────────────────────────────────────────────────────
def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> Greeks:
    """Compute all five analytical Black-Scholes Greeks.

    Greeks are:
    * Delta — first derivative of price w.r.t. spot.
    * Gamma — second derivative of price w.r.t. spot (same for calls & puts).
    * Vega  — sensitivity to a 1-unit change in vol (annualised).
    * Theta — time decay per calendar day.
    * Rho   — sensitivity to a 1-unit change in r.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiration (years).
        r: Risk-free rate.
        sigma: Volatility.
        option_type: ``'call'`` or ``'put'``.

    Returns:
        Greeks dataclass with delta, gamma, vega, theta, rho.
    """
    validate_inputs(S, K, T, r, sigma)
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    discount = K * np.exp(-r * T)

    # ── Gamma & Vega (identical for call and put) ──────────────────────────
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T  # per 1-unit change in sigma

    if option_type == "call":
        delta = norm.cdf(d1)
        # Annualised theta (matches C++ reference output — not divided by 365)
        theta = (-S * pdf_d1 * sigma / (2.0 * sqrt_T)) - (r * discount * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        delta = norm.cdf(d1) - 1.0
        theta = (-S * pdf_d1 * sigma / (2.0 * sqrt_T)) + (r * discount * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ── Full pricing run ─────────────────────────────────────────────────────────
def price_option_pair(
    S: float, K: float, T: float, r: float, sigma: float
) -> PricingResult:
    """Price both a call and a put and verify put-call parity.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiration (years).
        r: Risk-free rate.
        sigma: Volatility.

    Returns:
        PricingResult with call, put, and parity_error.
    """
    validate_inputs(S, K, T, r, sigma)

    call_price = black_scholes(S, K, T, r, sigma, "call")
    put_price = black_scholes(S, K, T, r, sigma, "put")
    call_greeks = compute_greeks(S, K, T, r, sigma, "call")
    put_greeks = compute_greeks(S, K, T, r, sigma, "put")

    # C - P = S - K·e^{-rT}
    parity_error = abs((call_price - put_price) - (S - K * np.exp(-r * T)))

    return PricingResult(
        call=OptionResult("call", call_price, call_greeks),
        put=OptionResult("put", put_price, put_greeks),
        parity_error=parity_error,
    )


# ── Greek interpretation ─────────────────────────────────────────────────────
def interpret_delta(delta: float, option_type: OptionType) -> str:
    """Return a human-readable interpretation of Delta.

    Args:
        delta: Delta value.
        option_type: ``'call'`` or ``'put'``.

    Returns:
        Interpretation string.
    """
    abs_d = abs(delta)
    if option_type == "call":
        if abs_d < 0.3:
            return "Low delta. This option will move a bit with the stock. Out-of-the-money, cheap but risky."
        elif abs_d < 0.7:
            return "Good value. Reacts strongly to price changes and good for directional trades."
        else:
            return "Strong directional bet. Almost behaves like the stock itself. Deep ITM."
    else:  # put
        if abs_d < 0.3:
            return "Mildly bearish. Small downside exposure; out-of-the-money put."
        elif abs_d < 0.7:
            return "Moderately bearish. Balanced downside protection."
        else:
            return "Strong bearish bet. Deep in-the-money put; behaves like shorting the stock."


def interpret_gamma(gamma: float) -> str:
    """Return a human-readable interpretation of Gamma.

    Args:
        gamma: Gamma value.

    Returns:
        Interpretation string.
    """
    if gamma < 0.01:
        return "Low Gamma. Delta won't move much. Stable but limited convexity."
    elif gamma < 0.05:
        return "Healthy value as the Delta will move noticeably. Good for trading and Gamma scalping."
    else:
        return "High Gamma. Dangerous if hedging, but great for long options near expiry."


def interpret_theta(theta: float) -> str:
    """Return a human-readable interpretation of Theta (daily).

    Args:
        theta: Theta value (per day).

    Returns:
        Interpretation string.
    """
    if theta < -5.0:
        return "Heavy time decay: Good for shorting, risky for long-term."
    elif theta < -1.0:
        return "Moderate time decay. Should only hold if you expect a move soon."
    elif theta < 0.0:
        return "Mild time decay, good for holding long-term."
    else:
        return "Very slow decay. Cheap to hold."


def interpret_vega(vega: float) -> str:
    """Return a human-readable interpretation of Vega.

    Args:
        vega: Vega value.

    Returns:
        Interpretation string.
    """
    if vega < 20.0:
        return "Low sensitivity. Vol changes won't affect you much."
    elif vega < 60.0:
        return "Medium sensitivity. Good if you expect rising uncertainty."
    else:
        return "High sensitivity. Be careful if vol spikes; great for vol traders."


def interpret_rho(rho: float) -> str:
    """Return a human-readable interpretation of Rho.

    Args:
        rho: Rho value.

    Returns:
        Interpretation string.
    """
    abs_r = abs(rho)
    if abs_r < 30.0:
        return "Low rate sensitivity. Rates won't affect you much."
    elif abs_r < 100.0:
        return "Big rate sensitivity. Long-term options/high strike."
    else:
        return "Very high rate sensitivity. Long-dated or deep ITM options."

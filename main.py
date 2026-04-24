"""
Black-Scholes Options Pricing Engine — Interactive CLI
Run:  python main.py
"""

from __future__ import annotations

from options_engine import (
    price_option_pair,
    interpret_delta,
    interpret_gamma,
    interpret_theta,
    interpret_vega,
    interpret_rho,
    PricingResult,
    OptionResult,
)

# ── Display constants ─────────────────────────────────────────────────────────
SEPARATOR = "=" * 54
PARITY_PASS_THRESHOLD = 1e-10
ENGINE_VERSION = "1.0"


# ── Formatting helpers ────────────────────────────────────────────────────────
def _header(title: str) -> str:
    return f"{SEPARATOR}\n{title.center(54)}\n{SEPARATOR}"


def _prompt_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  [!] Please enter a valid number.\n")


def _prompt_int(prompt: str, min_val: int = 1) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value >= min_val:
                return value
            print(f"  [!] Must be at least {min_val}.\n")
        except ValueError:
            print("  [!] Please enter a whole number.\n")


# ── Input collection ──────────────────────────────────────────────────────────
def collect_inputs() -> tuple[float, float, float, float, float]:
    """Prompt user for all Black-Scholes parameters with validation.

    Returns:
        Tuple (S, K, T, r, sigma).
    """
    while True:
        try:
            S = _prompt_float("Enter Stock Price (S): $")
            K = _prompt_float("Enter Strike Price (K): $")
            T = _prompt_float("Enter Time to Maturity (T) in years: ")
            r = _prompt_float(
                "Enter Risk-free Rate (r) as decimal (e.g., 0.05 for 5%): "
            )
            sigma = _prompt_float(
                "Enter Volatility (sigma) as decimal (e.g., 0.20 for 20%): "
            )

            # Delegate full validation to the engine
            from options_engine import validate_inputs
            validate_inputs(S, K, T, r, sigma)
            return S, K, T, r, sigma

        except ValueError as exc:
            print(f"\n  [!] Invalid input: {exc}\n  Please re-enter all parameters.\n")


# ── Output rendering ──────────────────────────────────────────────────────────
def _print_leg(result: OptionResult) -> None:
    label = "CALL OPTION" if result.option_type == "call" else "PUT OPTION"
    g = result.greeks

    print(_header(label))
    print(f"Price:  ${result.price:.4f}")
    print(f"Delta:   {g.delta:.4f}")
    print(f"Gamma:   {g.gamma:.4f}")
    print(f"Theta:   {g.theta:.4f}")
    print(f"Vega:    {g.vega:.4f}")
    print(f"Rho:     {g.rho:.4f}")
    print()


def _print_validation(parity_error: float) -> None:
    print(_header("VALIDATION"))
    print(f"Put-Call Parity Error: {parity_error:.2e}")
    if parity_error < PARITY_PASS_THRESHOLD:
        print("Calculations verified!")
    else:
        print("WARNING: Parity error exceeds threshold — check inputs.")
    print()


def _print_analysis(result: OptionResult) -> None:
    label = "CALL ANALYSIS" if result.option_type == "call" else "PUT ANALYSIS"
    g = result.greeks

    print(_header(label))
    print(f"Delta: {interpret_delta(g.delta, result.option_type)}")
    print(f"Gamma: {interpret_gamma(g.gamma)}")
    print(f"Theta: {interpret_theta(g.theta)}")
    print(f"Vega: {interpret_vega(g.vega)}")
    print(f"Rho: {interpret_rho(g.rho)}")
    print()


def display_results(result: PricingResult) -> None:
    """Print full pricing output for one option pair.

    Args:
        result: PricingResult from price_option_pair.
    """
    print("\nCalculating...\n")
    _print_leg(result.call)
    _print_leg(result.put)
    _print_validation(result.parity_error)
    _print_analysis(result.call)
    _print_analysis(result.put)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    """Entry point: run the interactive Black-Scholes CLI."""
    print()
    print(_header(f"Black-Scholes Options Pricing Engine v{ENGINE_VERSION}"))
    print()

    n = _prompt_int("How many options would you like to run?\n", min_val=1)

    for i in range(1, n + 1):
        if n > 1:
            print(f"\n{'-' * 54}")
            print(f"  Option {i} of {n}")
            print(f"{'-' * 54}\n")
        else:
            print()

        S, K, T, r, sigma = collect_inputs()

        try:
            result = price_option_pair(S, K, T, r, sigma)
            display_results(result)
        except Exception as exc:
            print(f"\n  [!] Pricing error: {exc}\n")

    print(SEPARATOR)
    print("  Session complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()

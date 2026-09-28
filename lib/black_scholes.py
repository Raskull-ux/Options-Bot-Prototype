"""
Black-Scholes pricing and implied-volatility inversion.

Used ONLY because the account is on the free indicative option feed, which
gives real bid/ask quotes but no IV or greeks fields. Every value this module
produces is an APPROXIMATION derived from those quotes, not real OPRA IV.
That label must travel with every number this produces -- see config.py's
IV_IS_APPROXIMATE flag and propagate it into every record/alert.

No external dependencies (no scipy) -- normal CDF/PDF via math.erf.
"""
import math
from dataclasses import dataclass


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, q: float, is_call: bool) -> float:
    """European option price under Black-Scholes with continuous dividend yield q."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
        return intrinsic
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if is_call:
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * math.exp(-q * T) * _norm_pdf(d1) * sqrtT


@dataclass
class IVResult:
    iv: float | None
    converged: bool
    iterations: int
    reason: str  # "ok", "below_intrinsic", "no_convergence", "invalid_input"


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
    initial_guess: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> IVResult:
    """
    Invert Black-Scholes for sigma given an observed market price, via
    Newton-Raphson with a bisection fallback. Returns IVResult -- never
    silently returns a fabricated number; check .converged before using .iv.
    """
    if S <= 0 or K <= 0 or T <= 0 or market_price <= 0:
        return IVResult(None, False, 0, "invalid_input")

    intrinsic = max(0.0, S - K * math.exp(-r * T)) if is_call else max(0.0, K * math.exp(-r * T) - S)
    if market_price < intrinsic - 1e-8:
        # Market price below intrinsic value -- no valid IV exists (stale/bad
        # quote). Do not guess; report and let the caller discard the row.
        return IVResult(None, False, 0, "below_intrinsic")

    sigma = initial_guess
    for i in range(max_iter):
        price = bs_price(S, K, T, r, sigma, q, is_call)
        diff = price - market_price
        if abs(diff) < tol:
            return IVResult(max(sigma, 1e-6), True, i + 1, "ok")
        vega = bs_vega(S, K, T, r, sigma, q)
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.001
        if sigma > 5.0:  # 500% IV -- clearly diverging, stop and fall back
            break

    # Newton-Raphson failed to converge (can happen near-expiry / deep
    # ITM-OTM with tiny vega) -- bisection fallback over a wide, sane range.
    lo, hi = 1e-4, 5.0
    f_lo = bs_price(S, K, T, r, lo, q, is_call) - market_price
    f_hi = bs_price(S, K, T, r, hi, q, is_call) - market_price
    if f_lo * f_hi > 0:
        return IVResult(None, False, max_iter, "no_convergence")
    for i in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(S, K, T, r, mid, q, is_call) - market_price
        if abs(f_mid) < tol:
            return IVResult(mid, True, max_iter + i + 1, "ok")
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return IVResult(None, False, max_iter + 200, "no_convergence")

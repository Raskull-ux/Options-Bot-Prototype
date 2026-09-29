"""
Diagnostic for the signal engine's trend-candidate rate. Makes ZERO API
calls -- reads only what's already in data/, so it's safe and free to run
as often as needed.

Question being tested: is the ~50% trend-candidate rate real momentum, or is
it beta leakage from the market-only (SPY, 1-to-1 subtraction) sector
adjustment currently in use because sector_profiles.csv doesn't exist yet?

A 1-to-1 subtraction of SPY's return under-corrects a high-beta stock. If
the market moved meaningfully over the trend window, high-beta names would
show artificially inflated |T| that's actually just leveraged market
exposure, not real idiosyncratic momentum. This script checks for exactly
that signature: does beta predict candidacy, and is the bullish/bearish
split lopsided in the market's own direction.
"""
import sys

import numpy as np
import pandas as pd

import config


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series, lookback: int = 120) -> float:
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    aligned = aligned.iloc[-lookback:]
    if len(aligned) < 30:
        return np.nan
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    if var <= 0:
        return np.nan
    return cov / var


def main() -> int:
    print("Loading data (no API calls)...")
    bars = pd.read_csv(config.BARS_FILE, parse_dates=["date"])
    signals = pd.read_csv(config.DAILY_SIGNALS_FILE)

    latest_date = signals["date"].max()
    today_signals = signals[signals["date"] == latest_date].copy()
    scored = today_signals[today_signals["status"] == "scored"].copy()
    print(f"Diagnosing {latest_date}: {len(scored)} scored symbols")

    price_panel = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    returns_panel = np.log(price_panel).diff()

    if "SPY" not in returns_panel.columns:
        print("FATAL: SPY not in bars data, cannot compute beta")
        return 1
    spy_returns = returns_panel["SPY"]

    print("\nComputing real beta vs SPY for each scored symbol (120-day trailing regression)...")
    betas = {}
    for symbol in scored["symbol"]:
        if symbol not in returns_panel.columns:
            continue
        betas[symbol] = compute_beta(returns_panel[symbol], spy_returns)
    scored["beta_vs_spy"] = scored["symbol"].map(betas)
    scored = scored.dropna(subset=["beta_vs_spy"])
    print(f"  -> beta computed for {len(scored)} symbols")

    # --- Test 1: does beta predict candidacy? ---
    corr_beta_T = scored["beta_vs_spy"].corr(scored["trend_T"])
    print(f"\n=== TEST 1: correlation between beta and the signal ===")
    print(f"  corr(beta, trend_T)              = {corr_beta_T:+.3f}")
    if scored["trend_candidate"].nunique() > 1:
        corr_beta_candidate = scored["beta_vs_spy"].corr(scored["trend_candidate"].astype(float))
        print(f"  corr(beta, trend_candidate flag) = {corr_beta_candidate:+.3f}")
    else:
        print(f"  corr(beta, trend_candidate flag) = n/a (every symbol has the same flag value, no variation to correlate)")
    print("  (near 0 = beta isn't driving the signal; strongly positive = beta leakage is likely real)")

    # --- Test 2: candidate rate by beta quartile ---
    scored["beta_quartile"] = pd.qcut(scored["beta_vs_spy"], 4, labels=["Q1 (lowest beta)", "Q2", "Q3", "Q4 (highest beta)"])
    print(f"\n=== TEST 2: trend-candidate rate by beta quartile ===")
    rate_by_quartile = scored.groupby("beta_quartile", observed=True)["trend_candidate"].mean()
    for q, rate in rate_by_quartile.items():
        print(f"  {q}: {rate:.1%} flagged as candidates")
    print("  (roughly FLAT across quartiles = fine; rising sharply with beta = beta leakage)")

    # --- Test 3: bullish/bearish split among candidates, vs the market's own direction ---
    candidates = scored[scored["trend_candidate"]]
    n_bullish = (candidates["trend_direction"] == "bullish").sum()
    n_bearish = (candidates["trend_direction"] == "bearish").sum()
    spy_prices = price_panel["SPY"].dropna()
    window = config.TREND_LOOKBACK_DAYS + config.TREND_SKIP_RECENT_DAYS
    spy_window_return = spy_prices.iloc[-1] / spy_prices.iloc[-window - 1] - 1 if len(spy_prices) > window else np.nan
    print(f"\n=== TEST 3: direction split among candidates vs. the market's own move ===")
    print(f"  candidates: {n_bullish} bullish, {n_bearish} bearish ({n_bullish/(n_bullish+n_bearish):.1%} bullish)" if (n_bullish+n_bearish) else "  no candidates")
    print(f"  SPY's own return over the same {window}-day window: {spy_window_return:+.2%}")
    print("  (if candidates are overwhelmingly bullish AND SPY was up over this window -> strong sign of beta leakage)")

    # --- Test 4: raw T distribution vs the calibrated null ---
    print(f"\n=== TEST 4: how extreme is the T distribution, overall ===")
    print(f"  median |T| among scored symbols: {scored['trend_T'].abs().median():.3f}")
    print(f"  fraction with |T| >= threshold ({config.TREND_STRENGTH_MIN_ABS}): {(scored['trend_T'].abs() >= config.TREND_STRENGTH_MIN_ABS).mean():.1%}")
    print(f"  (calibrated null rate under pure noise is 42.4% -- meaningfully higher than that across a full market")
    print(f"   is not automatically wrong, since real momentum exists, but should be interpreted alongside tests 1-3 above)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

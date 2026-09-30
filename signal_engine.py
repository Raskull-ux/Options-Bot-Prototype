"""
Signal engine for the day-to-5-day options bot -- Phase 2.

Reads the real data daily_collect.py has been accumulating and computes, per
symbol, per day:
  - VR(T) at several horizons (Lo-MacKinlay variance ratio)
  - a sector-adjusted, volatility-normalized trend-strength score T, checked
    against the EMPIRICALLY CALIBRATED threshold (not a borrowed number --
    see config.TREND_STRENGTH_MIN_ABS's comment)
  - four regime gates, computed once per day from the whole working universe:
    panic-state (rule-based approximation of Daniel/Jagannathan/Kim's HMM),
    crowding/hot-streak (market-proxy approximation), aggregate illiquidity,
    and cross-sectional dispersion
  - the earnings pre-announcement (Fear & Greed) reversal check
  - an IV-rank check, honestly reporting "insufficient_history" until real
    IV data has actually accumulated long enough to mean anything

This produces a DIAGNOSTIC daily candidate list -- it does NOT pick a
contract, size a position, or touch the paper account. Every row shows its
own inputs and reasons, not just a pass/fail verdict.
"""
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from lib.signals import (
    variance_ratio,
    trend_strength,
    percentile_rank_of_latest,
    amihud_illiquidity,
    cross_sectional_dispersion,
    average_true_range,
)
from lib.symbol_filter import load_excluded_symbols


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def merge_csv(path: str, new_rows: list[dict], subset_keys: list[str]) -> int:
    new_df = pd.DataFrame(new_rows)
    if new_df.empty:
        try:
            return len(pd.read_csv(path))
        except FileNotFoundError:
            return 0
    try:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    except FileNotFoundError:
        combined = new_df
    combined = combined.drop_duplicates(subset=subset_keys, keep="last")
    combined.to_csv(path, index=False)
    return len(combined)


def load_price_panel() -> pd.DataFrame:
    """Wide DataFrame: index=date, columns=symbol, values=close price."""
    bars = pd.read_csv(config.BARS_FILE, parse_dates=["date"])
    panel = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    return panel.sort_index()


def load_volume_panel() -> pd.DataFrame:
    bars = pd.read_csv(config.BARS_FILE, parse_dates=["date"])
    panel = bars.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last")
    return panel.sort_index()


def load_high_low_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide DataFrames for high and low, same shape as the close price panel
    -- needed for ATR, which the close-only panel can't support."""
    bars = pd.read_csv(config.BARS_FILE, parse_dates=["date"])
    high = bars.pivot_table(index="date", columns="symbol", values="high", aggfunc="last").sort_index()
    low = bars.pivot_table(index="date", columns="symbol", values="low", aggfunc="last").sort_index()
    return high, low


def load_sector_map() -> dict[str, tuple[str, bool]]:
    """Returns {symbol: (sector_etf, is_real_sector_match)}. Empty dict, with
    every symbol falling back to SPY, if the sector-profile cache doesn't
    exist yet (build_sector_profiles.py hasn't been run)."""
    try:
        df = pd.read_csv(config.SECTOR_PROFILES_FILE, parse_dates=["fetched_at"])
    except FileNotFoundError:
        log("  NOTE: no sector_profiles.csv yet -- every symbol falls back to market-only (SPY) adjustment")
        return {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.SECTOR_PROFILE_MAX_AGE_DAYS)
    df = df[df["fetched_at"] >= cutoff]
    return {row["symbol"]: (row["sector_etf"], bool(row["is_real_sector_match"])) for _, row in df.iterrows()}


def compute_regime_state(price_panel: pd.DataFrame, volume_panel: pd.DataFrame, working_symbols: list[str]) -> dict:
    """
    Computes the four market-wide regime gates for the LATEST date in the
    panel. Returns a dict of flags/values, all derived from real data
    already collected -- no external index feed needed.
    """
    result = {"date": price_panel.index.max().strftime("%Y-%m-%d")}

    # --- Panic-state approximation (rule-based, NOT the full HMM) ---
    if "SPY" in price_panel.columns:
        spy = price_panel["SPY"].dropna()
        spy_returns = np.log(spy).diff().dropna()
        decline = (spy.iloc[-1] / spy.iloc[-config.PANIC_MARKET_DECLINE_LOOKBACK_DAYS - 1] - 1) if len(spy) > config.PANIC_MARKET_DECLINE_LOOKBACK_DAYS else np.nan
        vol_series = spy_returns.rolling(20, min_periods=10).std()
        vol_percentile = percentile_rank_of_latest(vol_series, config.PANIC_VOL_PERCENTILE_LOOKBACK_DAYS)
        prior_run = (
            spy.iloc[-1] / spy.iloc[-config.PANIC_PRIOR_RUN_LOOKBACK_DAYS - 1] - 1
        ) if len(spy) > config.PANIC_PRIOR_RUN_LOOKBACK_DAYS else np.nan

        result["spy_decline_20d"] = decline
        result["spy_vol_percentile"] = vol_percentile
        result["spy_prior_run_60d"] = prior_run
        result["panic_state_flag"] = bool(
            (not np.isnan(decline) and decline <= config.PANIC_MARKET_DECLINE_THRESHOLD)
            and (not np.isnan(vol_percentile) and vol_percentile >= config.PANIC_VOL_PERCENTILE_THRESHOLD)
        )
        result["crowding_flag"] = bool(
            not np.isnan(prior_run) and abs(prior_run) >= config.PANIC_PRIOR_RUN_THRESHOLD
        )
    else:
        result.update(
            spy_decline_20d=np.nan, spy_vol_percentile=np.nan, spy_prior_run_60d=np.nan,
            panic_state_flag=False, crowding_flag=False,
        )
        log("  WARNING: SPY not in price panel -- panic/crowding gates cannot be computed, defaulting to False (not fabricated as True)")

    # --- Aggregate illiquidity (Amihud), computed from the working universe's OWN data ---
    returns_panel = np.log(price_panel[working_symbols]).diff()
    dollar_vol_panel = price_panel[working_symbols] * volume_panel[working_symbols]
    amihud_per_symbol = pd.DataFrame(
        {sym: amihud_illiquidity(returns_panel[sym], dollar_vol_panel[sym]) for sym in working_symbols}
    )
    aggregate_illiquidity_series = amihud_per_symbol.mean(axis=1, skipna=True)
    illiquidity_percentile = percentile_rank_of_latest(aggregate_illiquidity_series, config.ILLIQUIDITY_PERCENTILE_LOOKBACK_DAYS)
    result["illiquidity_percentile"] = illiquidity_percentile
    result["illiquidity_flag"] = bool(not np.isnan(illiquidity_percentile) and illiquidity_percentile >= config.ILLIQUIDITY_PERCENTILE_THRESHOLD)

    # --- Cross-sectional dispersion, same working-universe data ---
    dispersion_series = cross_sectional_dispersion(returns_panel)
    dispersion_percentile = percentile_rank_of_latest(dispersion_series, config.DISPERSION_PERCENTILE_LOOKBACK_DAYS)
    result["dispersion_percentile"] = dispersion_percentile
    result["dispersion_flag"] = bool(not np.isnan(dispersion_percentile) and dispersion_percentile >= config.DISPERSION_PERCENTILE_THRESHOLD)

    return result


def compute_earnings_signal(symbol: str, earnings_df: pd.DataFrame, price_panel: pd.DataFrame, today: pd.Timestamp) -> dict:
    sym_earnings = earnings_df[earnings_df["symbol"] == symbol]
    if sym_earnings.empty:
        return {"has_upcoming_earnings": False, "earnings_date": None, "pre_move_return": np.nan, "earnings_signal": "none"}

    sym_earnings = sym_earnings.copy()
    sym_earnings["earnings_date"] = pd.to_datetime(sym_earnings["earnings_date"])
    upcoming = sym_earnings[
        (sym_earnings["earnings_date"] >= today)
        & (sym_earnings["earnings_date"] <= today + pd.Timedelta(days=config.EARNINGS_LOOKAHEAD_DAYS))
    ]
    if upcoming.empty:
        return {"has_upcoming_earnings": False, "earnings_date": None, "pre_move_return": np.nan, "earnings_signal": "none"}

    earnings_date = upcoming["earnings_date"].min()
    if symbol not in price_panel.columns:
        return {"has_upcoming_earnings": True, "earnings_date": earnings_date.strftime("%Y-%m-%d"), "pre_move_return": np.nan, "earnings_signal": "none"}

    prices = price_panel[symbol].dropna()
    if len(prices) <= config.EARNINGS_PRE_MOVE_LOOKBACK_DAYS:
        return {"has_upcoming_earnings": True, "earnings_date": earnings_date.strftime("%Y-%m-%d"), "pre_move_return": np.nan, "earnings_signal": "none"}

    pre_move = prices.iloc[-1] / prices.iloc[-config.EARNINGS_PRE_MOVE_LOOKBACK_DAYS - 1] - 1
    max_threshold_hit = max((t for t in config.EARNINGS_PRE_MOVE_THRESHOLDS if abs(pre_move) >= t), default=None)
    if max_threshold_hit is None:
        signal = "none"
    elif pre_move > 0:
        signal = "extreme_pre_move_winner_bearish_reversal"  # per Jansen & Nikiforov: extreme winners tend to reverse down
    else:
        signal = "extreme_pre_move_loser_bullish_reversal"   # extreme losers tend to reverse up

    return {
        "has_upcoming_earnings": True,
        "earnings_date": earnings_date.strftime("%Y-%m-%d"),
        "pre_move_return": pre_move,
        "earnings_signal": signal,
        "pre_move_threshold_hit": max_threshold_hit,
    }


def compute_iv_rank(symbol: str, iv_df: pd.DataFrame, today: str) -> dict:
    sym_iv = iv_df[(iv_df["underlying"] == symbol) & (iv_df["iv_converged"])]
    if sym_iv.empty:
        return {"iv_rank": np.nan, "iv_rank_status": "no_data", "iv_history_days": 0}

    daily_iv = sym_iv.groupby("snapshot_date")["approx_iv"].mean().sort_index()
    n_days = len(daily_iv)
    if n_days < config.IV_RANK_MIN_HISTORY_DAYS:
        return {"iv_rank": np.nan, "iv_rank_status": "insufficient_history", "iv_history_days": n_days}

    rank = percentile_rank_of_latest(daily_iv, len(daily_iv))
    return {"iv_rank": rank, "iv_rank_status": "ok", "iv_history_days": n_days, "iv_rank_is_approximate": True}


def main() -> int:
    os.makedirs(config.DATA_DIR, exist_ok=True)

    for required in (config.BARS_FILE, config.UNIVERSE_MEMBERSHIP_FILE):
        if not os.path.exists(required):
            log(f"FATAL: {required} does not exist -- run daily_collect.py first")
            return 1

    log("Loading price/volume panels...")
    price_panel = load_price_panel()
    volume_panel = load_volume_panel()
    high_panel, low_panel = load_high_low_panels()
    today = price_panel.index.max()
    log(f"  -> panel covers {price_panel.index.min().date()} to {today.date()}, {price_panel.shape[1]} symbols")

    membership = pd.read_csv(config.UNIVERSE_MEMBERSHIP_FILE)
    working_symbols = [s for s in sorted(membership["symbol"].unique()) if s in price_panel.columns]
    log(f"  -> {len(working_symbols)} working-universe symbols present in the price panel")

    excluded_fund_types, exclusion_data_available = load_excluded_symbols()
    if exclusion_data_available:
        before = len(working_symbols)
        working_symbols = [s for s in working_symbols if s not in excluded_fund_types]
        log(f"  -> excluded {before - len(working_symbols)} real ETP/closed-end-fund/open-end-fund symbols already in the historical ledger; {len(working_symbols)} remain for scoring")
    else:
        log("  WARNING: data/symbol_types.csv doesn't exist yet -- fund/ETP exclusion NOT applied this run.")

    sector_map = load_sector_map()
    log(f"  -> {len(sector_map)} symbols have a cached real sector match (rest fall back to SPY)")

    try:
        earnings_df = pd.read_csv(config.EARNINGS_FILE)
    except FileNotFoundError:
        earnings_df = pd.DataFrame(columns=["symbol", "earnings_date"])
        log("  NOTE: no earnings_calendar.csv yet")

    try:
        iv_df = pd.read_csv(config.IV_SNAPSHOTS_FILE)
    except FileNotFoundError:
        iv_df = pd.DataFrame(columns=["underlying", "snapshot_date", "approx_iv", "iv_converged"])
        log("  NOTE: no iv_snapshots.csv yet")

    log("Computing market-wide regime state...")
    regime = compute_regime_state(price_panel, volume_panel, working_symbols)
    log(
        f"  -> panic={regime['panic_state_flag']} crowding={regime['crowding_flag']} "
        f"illiquidity={regime['illiquidity_flag']} dispersion={regime['dispersion_flag']}"
    )
    merge_csv(config.REGIME_STATE_FILE, [regime], subset_keys=["date"])

    log(f"Scoring {len(working_symbols)} symbols...")
    rows = []
    for symbol in working_symbols:
        prices = price_panel[symbol].dropna()
        n_history = len(prices)
        base_row = {"date": today.strftime("%Y-%m-%d"), "symbol": symbol, "n_history_days": n_history}

        if n_history < config.TREND_LOOKBACK_DAYS + config.TREND_SKIP_RECENT_DAYS + 5:
            rows.append({**base_row, "status": "insufficient_history"})
            continue

        raw_returns = np.log(prices).diff().dropna()

        # ATR(14): stored as a reference for later stop-loss/target sizing
        # (the not-yet-built contract-selection/paper-trading phases) --
        # deliberately NOT used in the trend/regime gates above, which
        # already have their own validated volatility normalization.
        atr_status = "ok"
        atr_14 = np.nan
        atr_pct = np.nan
        if symbol in high_panel.columns and symbol in low_panel.columns:
            h = high_panel[symbol].reindex(prices.index)
            l = low_panel[symbol].reindex(prices.index)
            atr_series = average_true_range(h, l, prices, period=config.ATR_PERIOD)
            atr_latest = atr_series.iloc[-1]
            if not np.isnan(atr_latest):
                atr_14 = atr_latest
                atr_pct = atr_latest / prices.iloc[-1] if prices.iloc[-1] else np.nan
            else:
                atr_status = "insufficient_history"
        else:
            atr_status = "no_high_low_data"

        vr_row = {}
        for q in config.HORIZON_CANDIDATES_DAYS:
            vr_row[f"vr_q{q}"] = variance_ratio(raw_returns.to_numpy(), q)

        sector_etf, used_real_sector = sector_map.get(symbol, ("SPY", False))
        if sector_etf in price_panel.columns:
            sector_returns = np.log(price_panel[sector_etf]).diff()
            aligned = pd.concat([raw_returns, sector_returns], axis=1, join="inner").dropna()
            idio_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        else:
            idio_returns = raw_returns
            used_real_sector = False

        T, n_obs_trend = trend_strength(
            idio_returns, lookback=config.TREND_LOOKBACK_DAYS,
            skip_recent=config.TREND_SKIP_RECENT_DAYS, ewma_lambda=config.EWMA_VOL_LAMBDA,
        )
        meets_trend_threshold = bool(not np.isnan(T) and abs(T) >= config.TREND_STRENGTH_MIN_ABS)

        trend_candidate = meets_trend_threshold and not (
            regime["panic_state_flag"] or regime["crowding_flag"] or regime["illiquidity_flag"] or regime["dispersion_flag"]
        )
        trend_direction = "bullish" if (not np.isnan(T) and T > 0) else ("bearish" if not np.isnan(T) else None)

        earnings_info = compute_earnings_signal(symbol, earnings_df, price_panel, today)
        earnings_candidate = earnings_info["earnings_signal"] != "none"

        iv_info = compute_iv_rank(symbol, iv_df, today.strftime("%Y-%m-%d"))

        rows.append(
            {
                **base_row,
                "status": "scored",
                **vr_row,
                "trend_T": T,
                "trend_n_obs": n_obs_trend,
                "meets_trend_threshold": meets_trend_threshold,
                "trend_direction": trend_direction,
                "sector_etf_used": sector_etf,
                "used_real_sector_adjustment": used_real_sector,
                "trend_candidate": trend_candidate,
                "atr_14": atr_14,
                "atr_pct": atr_pct,
                "atr_status": atr_status,
                **{f"regime_{k}": v for k, v in regime.items() if k != "date"},
                **earnings_info,
                "earnings_candidate": earnings_candidate,
                **iv_info,
            }
        )

    scored = sum(1 for r in rows if r.get("status") == "scored")
    trend_candidates = sum(1 for r in rows if r.get("trend_candidate"))
    earnings_candidates = sum(1 for r in rows if r.get("earnings_candidate"))
    log(f"  -> {scored} scored, {len(rows) - scored} insufficient history")
    log(f"  -> {trend_candidates} trend-sleeve candidates, {earnings_candidates} earnings-sleeve candidates")

    total = merge_csv(config.DAILY_SIGNALS_FILE, rows, subset_keys=["date", "symbol"])
    log(f"  -> {total} total rows in {config.DAILY_SIGNALS_FILE}")

    log("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

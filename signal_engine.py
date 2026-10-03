"""
Signal engine -- revision 2026-10-03 (research basis: RESEARCH_RECORD_v2_OPUS.md).

Each run, after the collector:
  1. Per-symbol DIAGNOSTICS (not entry signals): variance ratios, trend
     score T, ATR, approximate-IV rank -> data/daily_signals.csv
  2. Market REGIME LABELS (panic as Daniel-Moskowitz define it, illiquidity,
     dispersion, VIX level/slope/median) -> data/regime_state.csv
  3. The five SLEEVES -> data/signals.csv. These rows are what alerts and the
     paper-trading engine are built from. Every row carries its sleeve's
     evidence grade, any deviation from the paper, and the regime labels.

Removed in this revision (v2 findings): the 1-5 day trend-continuation entry
sleeve, the crowding gate, and the old panic definition. The IV-rank check
is a logged diagnostic, not a gate.
"""
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from lib.signals import variance_ratio, trend_strength, percentile_rank_of_latest, average_true_range
from lib.symbol_filter import load_excluded_symbols
from lib import sleeves


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def merge_csv(path: str, new_rows: list[dict], subset_keys: list[str]) -> int:
    new_df = pd.DataFrame(new_rows)
    if new_df.empty:
        return 0
    try:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        combined = new_df
    combined = combined.drop_duplicates(subset=subset_keys, keep="last")
    combined.to_csv(path, index=False)
    return len(combined)


def replace_date_rows(path: str, date_col: str, date_value: str, new_rows: list[dict], columns: list[str] | None = None) -> int:
    """Full per-date snapshot: a rerun for the same date REPLACES that date's
    rows (fixes the 2026-09-30 stale-row contamination bug)."""
    new_df = pd.DataFrame(new_rows, columns=columns) if columns else pd.DataFrame(new_rows)
    try:
        existing = pd.read_csv(path)
        existing = existing[existing[date_col].astype(str) != date_value]
        combined = pd.concat([existing, new_df], ignore_index=True) if not new_df.empty else existing
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        combined = new_df
    combined.to_csv(path, index=False)
    return len(combined)


def load_panels() -> dict[str, pd.DataFrame]:
    bars = pd.read_csv(config.BARS_FILE, parse_dates=["date"])
    return {
        f: bars.pivot_table(index="date", columns="symbol", values=f, aggfunc="last").sort_index()
        for f in ("close", "volume", "high", "low")
    }


def load_sector_map() -> dict[str, tuple[str, bool]]:
    try:
        df = pd.read_csv(config.SECTOR_PROFILES_FILE, parse_dates=["fetched_at"])
    except FileNotFoundError:
        log("  NOTE: no sector_profiles.csv -- sector adjustment falls back to SPY")
        return {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.SECTOR_PROFILE_MAX_AGE_DAYS)
    df = df[df["fetched_at"] >= cutoff]
    return {r["symbol"]: (r["sector_etf"], bool(r["is_real_sector_match"])) for _, r in df.iterrows()}


def load_vix() -> pd.DataFrame | None:
    try:
        v = pd.read_csv(config.VIX_TERM_FILE, parse_dates=["date"]).set_index("date").sort_index()
        return v[["vix", "vix3m"]]
    except (FileNotFoundError, KeyError):
        log("  NOTE: no vix_term.csv yet -- VIX labels, the reversal VIX condition, and the volatility sleeve are inactive")
        return None


def iv_rank_diagnostic(symbol: str, iv_df: pd.DataFrame) -> dict:
    sym_iv = iv_df[(iv_df["underlying"] == symbol) & (iv_df["iv_converged"] == True)]  # noqa: E712
    if sym_iv.empty:
        return {"iv_rank": np.nan, "iv_rank_status": "no_data", "iv_history_days": 0}
    daily = sym_iv.groupby("snapshot_date")["approx_iv"].mean().sort_index()
    if len(daily) < config.IV_RANK_MIN_HISTORY_DAYS:
        return {"iv_rank": np.nan, "iv_rank_status": "insufficient_history", "iv_history_days": len(daily)}
    return {"iv_rank": percentile_rank_of_latest(daily, len(daily)), "iv_rank_status": "ok_diagnostic_only",
            "iv_history_days": len(daily)}


def diagnostics(symbol: str, p: dict, sector_map: dict, iv_df: pd.DataFrame, today: pd.Timestamp) -> dict:
    prices = p["close"][symbol].dropna()
    row = {"date": today.strftime("%Y-%m-%d"), "symbol": symbol, "n_history_days": len(prices)}
    if len(prices) < config.TREND_LOOKBACK_DAYS + config.TREND_SKIP_RECENT_DAYS + 5:
        row["status"] = "insufficient_history"
        return row
    raw = np.log(prices).diff().dropna()
    for q in config.HORIZON_CANDIDATES_DAYS:
        row[f"vr_q{q}"] = variance_ratio(raw.to_numpy(), q)
    etf, real = sector_map.get(symbol, ("SPY", False))
    if etf in p["close"].columns:
        aligned = pd.concat([raw, np.log(p["close"][etf]).diff()], axis=1, join="inner").dropna()
        idio = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    else:
        idio, real = raw, False
    T, n_obs = trend_strength(idio, config.TREND_LOOKBACK_DAYS, config.TREND_SKIP_RECENT_DAYS, config.EWMA_VOL_LAMBDA)
    atr = average_true_range(p["high"][symbol].reindex(prices.index), p["low"][symbol].reindex(prices.index),
                             prices, config.ATR_PERIOD).iloc[-1]
    row.update(status="scored", trend_T_diagnostic=T, trend_n_obs=n_obs, sector_etf_used=etf,
               used_real_sector_adjustment=real, atr_14=atr,
               atr_pct=(atr / prices.iloc[-1]) if not np.isnan(atr) else np.nan,
               **iv_rank_diagnostic(symbol, iv_df))
    return row


def main() -> int:
    for required in (config.BARS_FILE, config.UNIVERSE_MEMBERSHIP_FILE):
        if not os.path.exists(required):
            log(f"FATAL: {required} missing -- run daily_collect.py first")
            return 1

    log("Loading panels...")
    p = load_panels()
    close = p["close"]
    today = close.index.max()
    if "SPY" not in close.columns:
        log("FATAL: SPY missing from bars -- abnormal returns and regime labels need it")
        return 1
    date_str = today.strftime("%Y-%m-%d")
    log(f"  -> {close.index.min().date()} to {today.date()}, {close.shape[1]} symbols")

    membership = pd.read_csv(config.UNIVERSE_MEMBERSHIP_FILE)
    universe = [s for s in sorted(membership["symbol"].unique()) if s in close.columns]
    excluded, have_types = load_excluded_symbols()
    if have_types:
        universe = [s for s in universe if s not in excluded]
    else:
        log("  WARNING: symbol_types.csv missing -- ETF/fund exclusion NOT applied")
    log(f"  -> {len(universe)} symbols in the tradable universe")

    sector_map = load_sector_map()
    vix = load_vix()
    try:
        earnings = pd.read_csv(config.EARNINGS_FILE)
    except FileNotFoundError:
        earnings = pd.DataFrame(columns=["symbol", "earnings_date", "hour"])
    try:
        iv_df = pd.read_csv(config.IV_SNAPSHOTS_FILE)
    except FileNotFoundError:
        iv_df = pd.DataFrame(columns=["underlying", "snapshot_date", "approx_iv", "iv_converged"])

    # 1. Diagnostics
    diag_rows = [diagnostics(s, p, sector_map, iv_df, today) for s in universe]
    replace_date_rows(config.DAILY_SIGNALS_FILE, "date", date_str, diag_rows)
    log(f"  -> diagnostics written for {len(diag_rows)} symbols")

    # 2. Regime labels
    regime = sleeves.regime_labels(close, p["volume"], universe, vix, today)
    merge_csv(config.REGIME_STATE_FILE, [regime], ["date"])
    log("  -> regime: " + ", ".join(f"{k}={regime[k]}" for k in
        ("bear_24m", "panic_state", "illiquidity_high", "dispersion_high", "vix", "vix_inverted", "vix_above_median")))

    # 3. Sleeves
    rets = sleeves.simple_returns(close)
    ar = sleeves.market_adjusted(rets)
    uni = set(universe)
    results = {
        "earnings_reversal": sleeves.earnings_reversal(ar, earnings, uni, today),
        "post_earnings_drift": sleeves.post_earnings_drift(ar, earnings, uni, today),
        "short_term_reversal": sleeves.short_term_reversal(close, uni, sector_map, today, regime.get("vix_above_median"), earnings),
        "volume_reversal": sleeves.volume_reversal(close, p["volume"], ar, uni, today),
        "index_volatility": sleeves.index_volatility(vix, today),
    }
    signals = []
    regime_cols = {f"regime_{k}": v for k, v in regime.items() if k != "date"}
    for name, (rows, status) in results.items():
        log(f"  [{name}] {status}")
        signals += [{**r, **regime_cols} for r in rows]
    n_conf = sleeves.mark_conflicts(signals)
    if n_conf:
        log(f"  -> {n_conf} symbol(s) got opposite-direction signals today; flagged conflict=True, not to be traded")
    cols = sleeves.SIGNAL_COLUMNS + list(regime_cols.keys())
    replace_date_rows(config.SIGNALS_FILE, "date", date_str, signals, columns=cols)
    log(f"  -> {len(signals)} signals fired for entry on {sleeves.next_session(today).date()}")
    for s in signals:
        log(f"     {s['sleeve']:20s} {s['symbol']:6s} {s['direction']:16s} {s['structure']}"
            + ("   [CONFLICT - skip]" if s.get("conflict") else ""))
    log("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

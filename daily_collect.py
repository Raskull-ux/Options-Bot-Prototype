"""
Daily data collector for the day-to-5-day options bot -- Phase 1: data layer only.

This script does NOT screen for trade signals or trade. It does:
  1. Pull the REAL, current optionable US equity universe from Alpaca (no
     hardcoded ticker list -- see lib/alpaca_client.get_optionable_equities).
  2. Apply a liquidity floor (price + dollar volume) -- not a size/mega-cap
     filter -- to get today's "eligible" pool.
  3. Flag unusual activity (relative volume spike, or a volatility-normalized
     move) across the whole eligible pool -- this is the "catches viral
     movers" layer, computed from real price/volume data.
  4. Promote anything eligible or flagged into a permanent, append-only
     working-universe ledger, so history keeps accumulating once a name
     qualifies (this is the "consistency" layer).
  5. Backfill full history for the working universe, sample approximate IV
     for a prioritized subset, and pull the earnings calendar.

Everything is appended to CSV files under data/, which a workflow commits
back to the repo so history survives between runs.
"""
import os
import sys
import traceback
from datetime import datetime, timezone

import pandas as pd

import config
from lib.alpaca_client import AlpacaClient, AlpacaError
from lib.finnhub_client import FinnhubClient, FinnhubError
from lib.black_scholes import implied_vol


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def merge_csv(path: str, new_rows: list[dict], subset_keys: list[str]) -> int:
    """Append new_rows to the CSV at path, de-duplicating on subset_keys
    (keeping the newest). Returns the row count after merge."""
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


def bars_dict_to_long_df(bars_by_symbol: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for symbol, bars in bars_by_symbol.items():
        for b in bars:
            rows.append(
                {
                    "symbol": symbol,
                    "date": b["t"][:10],
                    "close": b["c"],
                    "volume": b["v"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "close", "volume"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"])


def compute_screen_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each symbol, compute today's (latest row's) price, return, trailing
    20-day realized daily-return volatility, and 20-day average volume /
    dollar volume. Returns one row per symbol (the latest date only).
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["dollar_volume"] = df["close"] * df["volume"]
    df["return"] = df.groupby("symbol")["close"].pct_change()

    grouped = df.groupby("symbol")
    df["avg_volume_20d"] = grouped["volume"].transform(
        lambda s: s.rolling(window=config.VOLATILITY_LOOKBACK_DAYS, min_periods=10).mean()
    )
    df["avg_dollar_volume_20d"] = grouped["dollar_volume"].transform(
        lambda s: s.rolling(window=config.VOLATILITY_LOOKBACK_DAYS, min_periods=10).mean()
    )
    df["trailing_vol_20d"] = grouped["return"].transform(
        lambda s: s.rolling(window=config.VOLATILITY_LOOKBACK_DAYS, min_periods=10).std()
    )

    latest = df.sort_values("date").groupby("symbol").tail(1).reset_index(drop=True)
    latest = latest.rename(columns={"close": "latest_price", "volume": "today_volume", "return": "today_return"})
    return latest[
        [
            "symbol",
            "date",
            "latest_price",
            "today_volume",
            "today_return",
            "avg_volume_20d",
            "avg_dollar_volume_20d",
            "trailing_vol_20d",
        ]
    ]


def apply_liquidity_and_activity_flags(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.copy()
    m["is_liquidity_eligible"] = (m["latest_price"] >= config.MIN_PRICE) & (
        m["avg_dollar_volume_20d"] >= config.MIN_AVG_DOLLAR_VOLUME
    )

    rel_volume = m["today_volume"] / m["avg_volume_20d"].replace(0, pd.NA)
    move_threshold = (m["trailing_vol_20d"] * config.MOVE_VOL_MULTIPLE).clip(lower=config.MOVE_ABS_FLOOR)
    unusual_volume = rel_volume >= config.REL_VOLUME_SPIKE_THRESHOLD
    unusual_move = m["today_return"].abs() >= move_threshold

    m["rel_volume"] = rel_volume
    m["move_threshold_used"] = move_threshold
    m["unusual_volume_flag"] = unusual_volume.fillna(False)
    m["unusual_move_flag"] = unusual_move.fillna(False)
    m["unusual_activity"] = (m["unusual_volume_flag"] | m["unusual_move_flag"]) & m["is_liquidity_eligible"]
    return m


def update_universe_membership(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Append-only membership ledger: a symbol that qualifies (liquidity OR
    unusual activity) today is added if new, with today's date and the
    reason. Existing members are never removed here -- membership is sticky
    on purpose (that's the "consistency" requirement). Returns the FULL
    current membership table (including symbols not new today) so the
    caller knows which symbols to backfill/sample options for.
    """
    today = metrics["date"].max()
    qualifies_today = metrics[metrics["is_liquidity_eligible"] | metrics["unusual_activity"]].copy()
    qualifies_today["reason"] = qualifies_today.apply(
        lambda r: "unusual_activity" if r["unusual_activity"] and not r["is_liquidity_eligible"] else "liquidity",
        axis=1,
    )

    try:
        existing = pd.read_csv(config.UNIVERSE_MEMBERSHIP_FILE, parse_dates=["first_qualified_date", "last_seen_date"])
    except FileNotFoundError:
        existing = pd.DataFrame(columns=["symbol", "first_qualified_date", "last_seen_date", "first_reason"])

    existing_symbols = set(existing["symbol"]) if not existing.empty else set()
    new_symbols = qualifies_today[~qualifies_today["symbol"].isin(existing_symbols)]

    new_rows = [
        {
            "symbol": r["symbol"],
            "first_qualified_date": today,
            "last_seen_date": today,
            "first_reason": r["reason"],
        }
        for _, r in new_symbols.iterrows()
    ]

    if not existing.empty:
        seen_today = set(qualifies_today["symbol"])
        existing.loc[existing["symbol"].isin(seen_today), "last_seen_date"] = today

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else existing
    combined = combined.drop_duplicates(subset=["symbol"], keep="last")
    combined.to_csv(config.UNIVERSE_MEMBERSHIP_FILE, index=False)
    return combined


def pick_atm_strikes(contracts: list[dict], spot: float, n_each_side: int) -> set[float]:
    strikes = sorted({float(c["strike_price"]) for c in contracts})
    if not strikes:
        return set()
    closest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
    lo = max(0, closest_idx - n_each_side)
    hi = min(len(strikes), closest_idx + n_each_side + 1)
    return set(strikes[lo:hi])


def collect_iv_snapshots(client: AlpacaClient, priority_symbols: list[str]) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    rows: list[dict] = []
    today = datetime.now(timezone.utc).date()

    for symbol in priority_symbols:
        try:
            spot = client.get_latest_trade_price(symbol, feed=config.SPOT_PRICE_FEED)
        except AlpacaError as e:
            errors.append(f"{symbol}: latest trade failed: {e}")
            continue
        try:
            contracts = client.get_option_contracts(symbol, min_dte=config.SNAPSHOT_MIN_DTE, max_dte=config.SNAPSHOT_MAX_DTE)
        except AlpacaError as e:
            errors.append(f"{symbol}: contracts fetch failed: {e}")
            continue
        if not contracts:
            errors.append(f"{symbol}: no contracts in {config.SNAPSHOT_MIN_DTE}-{config.SNAPSHOT_MAX_DTE} DTE window")
            continue

        expirations = sorted({c["expiration_date"] for c in contracts})
        nearest_exp = expirations[0]
        exp_contracts = [c for c in contracts if c["expiration_date"] == nearest_exp]
        target_strikes = pick_atm_strikes(exp_contracts, spot, config.SNAPSHOT_STRIKES_EACH_SIDE)
        if not target_strikes:
            continue
        strike_gte, strike_lte = min(target_strikes), max(target_strikes)

        try:
            snapshots = client.get_option_snapshots(
                symbol, feed=config.OPTION_QUOTE_FEED, expiration_date=nearest_exp,
                strike_gte=strike_gte, strike_lte=strike_lte,
            )
        except AlpacaError as e:
            errors.append(f"{symbol}: snapshot fetch failed: {e}")
            continue

        contract_by_sym = {c["symbol"]: c for c in exp_contracts}
        exp_date = datetime.strptime(nearest_exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        T_years = max(dte, 0) / 365.0

        for contract_symbol, snap in snapshots.items():
            c = contract_by_sym.get(contract_symbol)
            if c is None or float(c["strike_price"]) not in target_strikes:
                continue
            quote = snap.get("latestQuote") or {}
            bid, ask = quote.get("bp"), quote.get("ap")
            if not bid or not ask or bid <= 0 or ask <= 0:
                errors.append(f"{contract_symbol}: no usable bid/ask in snapshot")
                continue
            mid = (bid + ask) / 2.0
            is_call = c["type"] == "call"
            strike = float(c["strike_price"])

            iv_result = implied_vol(
                market_price=mid, S=spot, K=strike, T=T_years,
                r=config.RISK_FREE_RATE, q=config.DIVIDEND_YIELD_ASSUMPTION, is_call=is_call,
            )
            rows.append(
                {
                    "snapshot_date": today.isoformat(),
                    "underlying": symbol,
                    "contract_symbol": contract_symbol,
                    "expiration_date": nearest_exp,
                    "dte": dte,
                    "strike": strike,
                    "type": c["type"],
                    "underlying_price": spot,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread_pct_of_mid": (ask - bid) / mid if mid else None,
                    "approx_iv": iv_result.iv,
                    "iv_converged": iv_result.converged,
                    "iv_reason": iv_result.reason,
                    "iv_is_approximate": config.IV_IS_APPROXIMATE,
                    "quote_feed": config.OPTION_QUOTE_FEED,
                }
            )
    return rows, errors


def collect_earnings(finnhub: FinnhubClient, symbols: set[str]) -> tuple[list[dict], list[str]]:
    try:
        raw_rows = finnhub.get_earnings_calendar(days_ahead=config.SNAPSHOT_MAX_DTE + 5, symbols=symbols)
    except FinnhubError as e:
        return [], [f"earnings calendar fetch failed: {e}"]
    rows = [
        {
            "symbol": r.get("symbol"),
            "earnings_date": r.get("date"),
            "hour": r.get("hour"),
            "eps_estimate": r.get("epsEstimate"),
            "eps_actual": r.get("epsActual"),
            "revenue_estimate": r.get("revenueEstimate"),
            "revenue_actual": r.get("revenueActual"),
            "pulled_at": datetime.now(timezone.utc).isoformat(),
        }
        for r in raw_rows
    ]
    return rows, []


def main() -> int:
    for var in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "FINNHUB_API_KEY"):
        if not os.environ.get(var):
            log(f"FATAL: {var} not set")
            return 2

    os.makedirs(config.DATA_DIR, exist_ok=True)
    alpaca = AlpacaClient()
    finnhub = FinnhubClient()
    all_errors: list[str] = []
    run_started = datetime.now(timezone.utc)

    # ---- Step 1: real, current optionable universe from Alpaca ----
    log("Pulling live optionable US equity universe from Alpaca...")
    try:
        assets = alpaca.get_optionable_equities()
    except AlpacaError as e:
        log(f"FATAL: could not pull asset list: {e}")
        return 1
    symbols = sorted({a["symbol"] for a in assets})
    log(f"  -> {len(symbols)} tradable, optionable, non-OTC equities")

    # ---- Step 2: light bar pull across the WHOLE optionable universe, for
    #      the liquidity screen and unusual-activity detection ----
    log(f"Pulling {config.LIQUIDITY_LOOKBACK_DAYS}-day bars for liquidity/activity screen...")
    try:
        light_bars = alpaca.get_daily_bars(symbols, lookback_days=config.LIQUIDITY_LOOKBACK_DAYS, feed=config.STOCK_BARS_FEED)
    except AlpacaError as e:
        log(f"FATAL: light bar pull failed: {e}")
        return 1
    light_df = bars_dict_to_long_df(light_bars)
    metrics = compute_screen_metrics(light_df)
    metrics = apply_liquidity_and_activity_flags(metrics)
    n_eligible = int(metrics["is_liquidity_eligible"].sum())
    n_unusual = int(metrics["unusual_activity"].sum())
    log(f"  -> {n_eligible} liquidity-eligible today, {n_unusual} flagged unusual activity")

    unusual_rows = metrics[metrics["unusual_activity"]].copy()
    unusual_rows["date"] = unusual_rows["date"].dt.strftime("%Y-%m-%d")
    merge_csv(
        config.UNUSUAL_ACTIVITY_FILE,
        unusual_rows.to_dict("records"),
        subset_keys=["symbol", "date"],
    )

    # ---- Step 3: persistent, append-only working-universe membership ----
    membership = update_universe_membership(metrics)
    working_universe = sorted(set(membership["symbol"]))
    log(f"  -> working universe (ever qualified, cumulative): {len(working_universe)} symbols")

    # ---- Step 4: deep history backfill for the full working universe ----
    log(f"Backfilling {config.MIN_HISTORY_DAYS_FOR_SCREEN}-day history for working universe...")
    try:
        deep_bars = alpaca.get_daily_bars(
            working_universe + config.REFERENCE_ETFS,
            lookback_days=config.MIN_HISTORY_DAYS_FOR_SCREEN,
            feed=config.STOCK_BARS_FEED,
        )
    except AlpacaError as e:
        all_errors.append(f"deep bar backfill failed entirely: {e}")
        deep_bars = {}

    bar_rows = []
    for symbol, bars in deep_bars.items():
        for b in bars:
            bar_rows.append(
                {
                    "symbol": symbol, "date": b["t"][:10], "open": b["o"], "high": b["h"],
                    "low": b["l"], "close": b["c"], "volume": b["v"],
                    "trade_count": b.get("n"), "vwap": b.get("vw"), "feed": config.STOCK_BARS_FEED,
                }
            )
    missing = set(working_universe + config.REFERENCE_ETFS) - set(deep_bars.keys())
    all_errors += [f"{s}: no bars in deep backfill" for s in missing]
    bars_total = merge_csv(config.BARS_FILE, bar_rows, subset_keys=["symbol", "date"])
    log(f"  -> {len(bar_rows)} bar rows this run, {bars_total} total in file")

    # ---- Step 5: prioritized approximate-IV sampling ----
    priority = pd.concat(
        [
            metrics[metrics["unusual_activity"]].assign(_pri=0),
            metrics[~metrics["unusual_activity"] & metrics["is_liquidity_eligible"]]
            .sort_values(["rel_volume", "trailing_vol_20d"], ascending=False)
            .assign(_pri=1),
        ]
    ).sort_values("_pri")
    priority_symbols = priority["symbol"].head(config.IV_SNAPSHOT_DAILY_CAP).tolist()
    log(f"Sampling approximate IV for {len(priority_symbols)} priority symbols (cap={config.IV_SNAPSHOT_DAILY_CAP})...")
    iv_rows, iv_errors = collect_iv_snapshots(alpaca, priority_symbols)
    all_errors += iv_errors
    iv_total = merge_csv(config.IV_SNAPSHOTS_FILE, iv_rows, subset_keys=["snapshot_date", "contract_symbol"])
    log(f"  -> {len(iv_rows)} snapshot rows this run, {iv_total} total in file, {len(iv_errors)} errors")

    # ---- Step 6: earnings calendar for the current eligible pool ----
    eligible_symbols = set(metrics[metrics["is_liquidity_eligible"]]["symbol"])
    log(f"Collecting earnings calendar for {len(eligible_symbols)} eligible symbols...")
    earn_rows, earn_errors = collect_earnings(finnhub, eligible_symbols)
    all_errors += earn_errors
    earn_total = merge_csv(config.EARNINGS_FILE, earn_rows, subset_keys=["symbol", "earnings_date"])
    log(f"  -> {len(earn_rows)} rows this run, {earn_total} total in file")

    merge_csv(
        config.RUN_LOG_FILE,
        [
            {
                "run_started_utc": run_started.isoformat(),
                "optionable_universe_size": len(symbols),
                "liquidity_eligible_today": n_eligible,
                "unusual_activity_today": n_unusual,
                "working_universe_cumulative": len(working_universe),
                "bar_rows_this_run": len(bar_rows),
                "iv_snapshot_rows_this_run": len(iv_rows),
                "earnings_rows_this_run": len(earn_rows),
                "error_count": len(all_errors),
                "errors_sample": " | ".join(all_errors[:10]),
            }
        ],
        subset_keys=["run_started_utc"],
    )

    if all_errors:
        log(f"{len(all_errors)} error(s) this run (per-symbol failures do not stop the run):")
        for e in all_errors[:30]:
            log(f"  - {e}")

    log("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

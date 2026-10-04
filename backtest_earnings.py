"""
Backtest the two earnings sleeves on REAL option prices.

1. Signals: rebuilt from every earnings release since BT_START (SEC EDGAR
   timing preferred, Finnhub otherwise) using exactly the live sleeves' rules:
     earnings_reversal   |market-adjusted 5-day move into the report| >= 10%,
                         fade it, exit at the close of day +1
     post_earnings_drift CAR3 (days -1..+1) in the top/bottom 10% of the
                         trailing 63-day announcer pool (min 50), go with it,
                         hold 21 trading days
   (test_backtest_equivalence() checks these match lib/sleeves.py exactly.)
2. Contract [eng, disclosed]: single-leg call (bullish) or put (bearish),
   strike nearest the entry-day open, first expiration at least N calendar
   days after the planned exit (5 for reversal, 7 for drift).
3. Prices: Alpaca daily option bars. Entry = the contract's VWAP on the entry
   day, exit = VWAP on the exit day. A trade with no option trade on either
   day is EXCLUDED and counted (this skews toward liquid contracts).
4. Costs: results at 0/5/10/20% round-trip cost as a share of premium.

Disclosed limits: today's universe is used for all past dates (survivorship);
daily VWAP is not a fill you could have gotten exactly; earnings dates are
treated as known in advance (they are normally scheduled, but not always).

Usage: python backtest_earnings.py [max_signals]   (0 = all)
"""
import math
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from lib import sleeves
from lib.alpaca_client import AlpacaClient, AlpacaError, TRADING, DATA
from lib.earnings_data import load_earnings
from lib.symbol_filter import load_excluded_symbols


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Signal reconstruction (event-based, same rules as lib/sleeves.py)
# ---------------------------------------------------------------------------
def build_signals(close: pd.DataFrame, earnings: pd.DataFrame, start: str) -> list[dict]:
    ar = sleeves.market_adjusted(sleeves.simple_returns(close))
    idx = close.index
    start_ts = pd.Timestamp(start)
    e = earnings.copy()
    e["ed"] = pd.to_datetime(e["earnings_date"], errors="coerce").dt.normalize()
    e = e.dropna(subset=["ed"]).drop_duplicates(["symbol", "ed"])
    e = e[e["symbol"].isin(ar.columns)]
    out = []

    # Earnings reversal
    for r in e.itertuples():
        pos = idx.searchsorted(r.ed)
        if pos >= len(idx):
            continue
        hour = r.hour if isinstance(r.hour, str) else ""
        entry_pos = pos if hour.strip().lower() == "amc" else pos - 1
        exit_pos = pos + 1
        if entry_pos - config.EARN_REV_WINDOW_DAYS < 1 or exit_pos >= len(idx) or idx[entry_pos] < start_ts:
            continue
        w = ar[r.symbol].iloc[entry_pos - config.EARN_REV_WINDOW_DAYS: entry_pos]
        if w.isna().any():
            continue
        car = float(w.sum())
        if abs(car) < config.EARN_REV_THRESHOLD:
            continue
        out.append(dict(sleeve="earnings_reversal", symbol=r.symbol, direction="bearish" if car > 0 else "bullish",
                        signal_date=idx[entry_pos - 1], entry_date=idx[entry_pos], exit_date=idx[exit_pos],
                        metric=car, hour=hour, earnings_date=r.ed, source=r.source))

    # Post-earnings drift
    lo, hi = config.PEAD_CAR_WINDOW
    ev = []
    for r in e.itertuples():
        pos = idx.searchsorted(r.ed)
        if pos + lo < 0 or pos + hi >= len(idx):
            continue
        v = ar[r.symbol].iloc[pos + lo: pos + hi + 1]
        if v.isna().any():
            continue
        ev.append((pos + hi, float(v.sum()), r.symbol, r.ed, r.source))
    ev.sort(key=lambda x: x[0])
    comp = np.array([x[0] for x in ev])
    cars = np.array([x[1] for x in ev])
    for c, car3, sym, ed, src in ev:
        lo_i = np.searchsorted(comp, c - config.PEAD_POOL_LOOKBACK_DAYS, side="left")
        hi_i = np.searchsorted(comp, c, side="right")
        pool = cars[lo_i:hi_i]
        if len(pool) < config.PEAD_MIN_POOL or c + 1 >= len(idx) or idx[c + 1] < start_ts:
            continue
        pct = float((pool <= car3).mean())
        if config.PEAD_TAIL_PERCENTILE < pct < 1 - config.PEAD_TAIL_PERCENTILE:
            continue
        exit_pos = c + config.PEAD_HOLD_DAYS
        out.append(dict(sleeve="post_earnings_drift", symbol=sym,
                        direction="bullish" if pct >= 1 - config.PEAD_TAIL_PERCENTILE else "bearish",
                        signal_date=idx[c], entry_date=idx[c + 1],
                        exit_date=idx[exit_pos] if exit_pos < len(idx) else pd.NaT,
                        metric=car3, pool_pct=pct, earnings_date=ed, source=src))
    return out


# ---------------------------------------------------------------------------
# 2-3. Contract selection and pricing on real Alpaca option bars
# ---------------------------------------------------------------------------
class Pricer:
    def __init__(self, client: AlpacaClient):
        self.c = client
        self._last = 0.0

    def _get(self, url, params):
        wait = config.BT_REQUEST_INTERVAL - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        return self.c._get(url, params)

    def contract(self, sym: str, typ: str, spot: float, exp_min: date) -> dict | None:
        for status in ("inactive", "active"):
            body = self._get(f"{TRADING}/v2/options/contracts", {
                "underlying_symbols": sym, "status": status, "type": typ,
                "expiration_date_gte": exp_min.isoformat(),
                "expiration_date_lte": (exp_min + timedelta(days=21)).isoformat(),
                "strike_price_gte": round(spot * (1 - config.BT_STRIKE_BAND), 2),
                "strike_price_lte": round(spot * (1 + config.BT_STRIKE_BAND), 2),
                "limit": 1000,
            })
            cs = body.get("option_contracts") or []
            if cs:
                first_exp = min(x["expiration_date"] for x in cs)
                same = [x for x in cs if x["expiration_date"] == first_exp]
                return min(same, key=lambda x: abs(float(x["strike_price"]) - spot))
        return None

    def bars(self, csym: str, start: date, end: date) -> dict[str, dict]:
        body = self._get(f"{DATA}/v1beta1/options/bars", {
            "symbols": csym, "timeframe": "1Day", "start": start.isoformat(),
            "end": (end + timedelta(days=1)).isoformat(), "limit": 1000})
        return {b["t"][:10]: b for b in (body.get("bars") or {}).get(csym, [])}


def price_trade(p: Pricer, sig: dict, opn: pd.DataFrame, close: pd.DataFrame) -> dict:
    if pd.isna(sig["exit_date"]):
        return {"status": "still_open"}
    sym, entry, exit_ = sig["symbol"], sig["entry_date"], sig["exit_date"]
    spot = opn[sym].get(entry)
    if spot is None or pd.isna(spot) or spot <= 0:
        return {"status": "no_stock_price"}
    stock_ret = float(close[sym].get(exit_) / spot - 1)
    res = {"stock_return_in_direction": stock_ret if sig["direction"] == "bullish" else -stock_ret}
    typ = "call" if sig["direction"] == "bullish" else "put"
    exp_min = exit_.date() + timedelta(days=config.BT_MIN_DAYS_AFTER_EXIT[sig["sleeve"]])
    try:
        k = p.contract(sym, typ, float(spot), exp_min)
    except AlpacaError as e:
        return {**res, "status": f"contract_error: {str(e)[:80]}"}
    if k is None:
        return {**res, "status": "no_contract"}
    try:
        b = p.bars(k["symbol"], entry.date(), exit_.date())
    except AlpacaError as e:
        return {**res, "status": f"bars_error: {str(e)[:80]}", "contract": k["symbol"]}
    eb, xb = b.get(entry.strftime("%Y-%m-%d")), b.get(exit_.strftime("%Y-%m-%d"))
    if not eb or not xb:
        return {**res, "status": "no_option_trade_on_entry_or_exit", "contract": k["symbol"]}
    ein, xout = float(eb.get("vw") or eb["c"]), float(xb.get("vw") or xb["c"])
    if ein <= 0:
        return {**res, "status": "bad_entry_price", "contract": k["symbol"]}
    out = {**res, "status": "priced", "contract": k["symbol"], "strike": float(k["strike_price"]),
           "expiration": k["expiration_date"], "entry_premium": ein, "exit_premium": xout,
           "entry_volume": eb.get("v"), "exit_volume": xb.get("v")}
    for c in config.BT_COST_LEVELS:
        out[f"ret_cost_{int(c * 100)}"] = (xout * (1 - c / 2)) / (ein * (1 + c / 2)) - 1
    return out


# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
def describe(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) < 2:
        return f"n={len(x)}"
    lr = np.log1p(x.clip(lower=-0.9999))
    t = lr.mean() / lr.std(ddof=1) * math.sqrt(len(lr))
    return (f"n={len(x):4d}  win={(x > 0).mean():5.1%}  mean={x.mean():+7.1%}  median={x.median():+7.1%}  "
            f"mean log={lr.mean():+.3f}  t(log)={t:+.2f}")


def summarize(df: pd.DataFrame) -> None:
    print("\n===================== BACKTEST SUMMARY =====================")
    for sleeve, g in df.groupby("sleeve"):
        priced = g[g["status"] == "priced"]
        print(f"\n### {sleeve}: {len(g)} signals, {len(priced)} priced ({len(priced) / max(len(g), 1):.0%} coverage)")
        print("   excluded: " + ", ".join(f"{k}={v}" for k, v in g[g["status"] != "priced"]["status"].str.split(":").str[0].value_counts().items()))
        print(f"   stock move in signal direction (all priced): {describe(priced['stock_return_in_direction'])}")
        for c in config.BT_COST_LEVELS:
            print(f"   OPTION return, {int(c * 100):2d}% round-trip cost:      {describe(priced[f'ret_cost_{int(c * 100)}'])}")
        for y, gy in priced.groupby(priced["entry_date"].dt.year):
            print(f"     {y} @10% cost: {describe(gy['ret_cost_10'])}")
        for d, gd in priced.groupby("direction"):
            print(f"     {d:8s} @10% cost: {describe(gd['ret_cost_10'])}")
        if sleeve == "earnings_reversal":
            for h, gh in priced.groupby(priced["hour"].replace("", "unknown")):
                print(f"     report {h:7s} @10% cost: {describe(gh['ret_cost_10'])}")
    print("\nBar for calling anything an edge: t(log) >= 3 at a realistic cost level (Harvey-Liu-Zhu).")


def main(max_signals: int) -> int:
    membership = pd.read_csv(config.UNIVERSE_MEMBERSHIP_FILE)
    universe = sorted(membership["symbol"].unique())
    excluded, have_types = load_excluded_symbols()
    if have_types:
        universe = [s for s in universe if s not in excluded]
    earnings = load_earnings()
    earnings = earnings[earnings["symbol"].isin(universe)]
    log(f"earnings events: {len(earnings)} ({earnings['source'].value_counts().to_dict()})")
    syms = sorted(set(earnings["symbol"]))

    client = AlpacaClient()
    log(f"pulling daily stock bars for {len(syms)} symbols + SPY...")
    raw = client.get_daily_bars(syms + ["SPY"], lookback_days=config.BT_STOCK_LOOKBACK_DAYS, feed=config.STOCK_BARS_FEED)
    rows = [{"symbol": s, "date": b["t"][:10], "open": b["o"], "close": b["c"]} for s, bs in raw.items() for b in bs]
    bars = pd.DataFrame(rows)
    bars["date"] = pd.to_datetime(bars["date"])
    close = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    opn = bars.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
    log(f"  -> {close.index.min().date()} to {close.index.max().date()}, {close.shape[1]} symbols")

    sigs = build_signals(close, earnings, config.BT_START)
    log(f"signals: {pd.Series([s['sleeve'] for s in sigs]).value_counts().to_dict() if sigs else 0}")
    if max_signals:
        sigs = sigs[:max_signals]
        log(f"  capped at {max_signals} for this run")

    pricer = Pricer(client)
    results = []
    for i, s in enumerate(sigs):
        results.append({**s, **price_trade(pricer, s, opn, close)})
        if (i + 1) % 100 == 0:
            n_ok = sum(1 for r in results if r["status"] == "priced")
            log(f"  {i + 1}/{len(sigs)} priced attempts, {n_ok} priced")
    df = pd.DataFrame(results)
    df.to_csv(f"{config.DATA_DIR}/backtest_earnings_trades.csv", index=False)
    log(f"wrote {len(df)} rows to {config.DATA_DIR}/backtest_earnings_trades.csv")
    if not df.empty:
        summarize(df)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
    except Exception:
        traceback.print_exc()
        sys.exit(1)

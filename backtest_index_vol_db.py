"""
SPY straddle backtest v2 -- priced from QUOTES, not trade prints.

Fixes over v1 (which lost 491 of 648 days to legs with no recorded trade):
  1. Each leg is priced from Databento OPRA.PILLAR cbbo-1m (consolidated
     best bid/offer, 1-minute). Price = the last quote at or before 15:45 ET,
     no older than 15 minutes, with bid > 0 and ask >= bid. If the ATM strike
     has no valid quote on either day, the nearest strike (+/-1) is used.
  2. Costs are each leg's ACTUAL spread: buy both legs at the ask on entry,
     sell both at the bid on exit. A flat 1% round-trip case is kept as a
     sanity check, alongside gross (mid to mid).
  3. History starts 2023-03-28 (start of Databento OPRA coverage).

Contract rule [eng, disclosed]: standard monthly SPY expiration (3rd Friday;
moved to the prior trading day when the exchange is closed), the first one at
least 30 calendar days after entry. Strike = nearest $1 to SPY at 15:45 ET on
the entry day (Alpaca 1-minute SIP bar).

Timing: the VIX curve is labelled at day t's close; the straddle is bought at
15:45 ET on t+1 and sold at 15:45 ET on t+2 (next-day test), or held until the
curve returns to contango (episode test). This is one session later than
Johnson's close-to-close returns, because the bot can't trade the close it
labels on.

Cost safety: `estimate` mode only prices the query with Databento's free cost
endpoint. `run` mode re-estimates and aborts if the total exceeds --budget.
Downloaded quotes are cached (data/db_spy_quotes.csv.gz) so reruns are free.

Usage:  python backtest_index_vol_db.py estimate
        python backtest_index_vol_db.py run 15
"""
import math
import os
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from lib.alpaca_client import AlpacaClient, DATA
from backtest_index_vol import episodes

ET = ZoneInfo("America/New_York")
DATASET, SCHEMA = "OPRA.PILLAR", "cbbo-1m"
START = "2023-03-28"
QUOTE_TIME = (15, 45)
WINDOW_MIN = 15
CACHE = f"{config.DATA_DIR}/db_spy_quotes.csv.gz"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline)
# ---------------------------------------------------------------------------
def osi(expiry: date, cp: str, strike: float, root: str = "SPY") -> str:
    """OPRA/OSI raw symbol: root padded to 6, YYMMDD, C/P, strike x1000 in 8 digits."""
    return f"{root:<6}{expiry:%y%m%d}{cp}{int(round(strike * 1000)):08d}"


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7 + 14)


def monthly_expiry(min_date: date, trading_days: set, last_known: date) -> date:
    y, m = min_date.year, min_date.month
    while True:
        e = third_friday(y, m)
        if e <= last_known:
            while e not in trading_days:  # exchange holiday on the 3rd Friday -> prior trading day
                e -= timedelta(days=1)
        if e >= min_date:
            return e
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def quote_at(q: pd.DataFrame, day: date) -> tuple[float, float] | None:
    """q: one symbol's cbbo rows (index tz-aware UTC; bid_px_00, ask_px_00).
    Last valid quote at or before 15:45:59 ET, no older than 15 minutes."""
    tgt = datetime(day.year, day.month, day.day, *QUOTE_TIME, 59, tzinfo=ET)
    lo = tgt - timedelta(minutes=WINDOW_MIN)
    w = q[(q.index <= tgt) & (q.index >= lo)]
    w = w[(w["bid_px_00"] > 0) & (w["ask_px_00"] >= w["bid_px_00"])]
    if w.empty:
        return None
    r = w.iloc[-1]
    return float(r["bid_px_00"]), float(r["ask_px_00"])


def price_straddle(quotes: dict, legs_by_strike: list[tuple[float, str, str]], entry: date, exit_: date) -> dict:
    """legs_by_strike: [(strike, call_sym, put_sym)] nearest-first. quotes: {(sym, day): (bid, ask)}."""
    for k, c, p in legs_by_strike:
        qs = [quotes.get((c, entry)), quotes.get((p, entry)), quotes.get((c, exit_)), quotes.get((p, exit_))]
        if any(x is None for x in qs):
            continue
        (cb_e, ca_e), (pb_e, pa_e), (cb_x, ca_x), (pb_x, pa_x) = qs
        mid_e = (cb_e + ca_e) / 2 + (pb_e + pa_e) / 2
        mid_x = (cb_x + ca_x) / 2 + (pb_x + pa_x) / 2
        if mid_e <= 0:
            continue
        return {
            "status": "priced", "strike": k, "call": c, "put": p,
            "mid_entry": mid_e, "mid_exit": mid_x,
            "ask_entry": ca_e + pa_e, "bid_exit": cb_x + pb_x,
            "half_spread_pct_entry": ((ca_e - cb_e) + (pa_e - pb_e)) / 2 / mid_e,
            "half_spread_pct_exit": ((ca_x - cb_x) + (pa_x - pb_x)) / 2 / mid_x if mid_x > 0 else np.nan,
            "ret_gross": mid_x / mid_e - 1,
            "ret_real_spread": (cb_x + pb_x) / (ca_e + pa_e) - 1,
            "ret_flat_1pct": (mid_x * 0.995) / (mid_e * 1.005) - 1,
        }
    return {"status": "no_valid_quotes"}


def describe(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) < 2:
        return f"n={len(x)}"
    lr = np.log1p(x.clip(lower=-0.9999))
    sd = lr.std(ddof=1)
    t = lr.mean() / sd * math.sqrt(len(lr)) if sd > 0 else float("nan")
    return f"n={len(x):4d}  win={(x > 0).mean():5.1%}  mean={x.mean():+7.2%}  median={x.median():+7.2%}  t(log)={t:+.2f}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def spy_1545(client: AlpacaClient, start: str) -> pd.Series:
    """SPY price at 15:45 ET per day from Alpaca 1-minute SIP bars."""
    out, token = {}, None
    end = (datetime.now(timezone.utc) - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    while True:
        params = {"symbols": "SPY", "timeframe": "1Min", "start": start, "end": end,
                  "feed": config.STOCK_BARS_FEED, "limit": 10000}
        if token:
            params["page_token"] = token
        body = client._get(f"{DATA}/v2/stocks/bars", params)
        for b in (body.get("bars") or {}).get("SPY", []):
            t = pd.Timestamp(b["t"]).tz_convert(ET)
            if (t.hour, t.minute) == QUOTE_TIME:
                out[t.date()] = float(b["c"])
        token = body.get("next_page_token")
        if not token:
            break
    return pd.Series(out).sort_index()


def build_trades(vix: pd.DataFrame, spot: pd.Series) -> tuple[list[dict], pd.DatetimeIndex]:
    days = pd.DatetimeIndex(sorted(set(pd.to_datetime(spot.index)) & set(vix.index)))
    days = days[days >= pd.Timestamp(START)]
    lab = (vix.loc[days, "vix"] > vix.loc[days, "vix3m"])
    tset, last = {d.date() for d in days}, days[-1].date()
    trades = []

    def legs(entry_pos: int) -> list[tuple[float, str, str]]:
        e = days[entry_pos].date()
        exp = monthly_expiry(e + timedelta(days=30), tset, last)
        k0 = round(float(spot.loc[e]))
        return [(k, osi(exp, "C", k), osi(exp, "P", k)) for k in (k0, k0 - 1, k0 + 1)]

    for i in range(len(days) - 2):
        trades.append({"test": "next_day", "signal_date": days[i], "inverted": bool(lab.iloc[i]),
                       "vix": vix.loc[days[i], "vix"], "vix3m": vix.loc[days[i], "vix3m"],
                       "entry_date": days[i + 1], "exit_date": days[i + 2], "legs": legs(i + 1)})
    for e, x in episodes(lab, config.IDXVOL_MAX_HOLD_DAYS):
        trades.append({"test": "episode", "signal_date": days[e - 1], "inverted": True,
                       "vix": vix.loc[days[e - 1], "vix"], "vix3m": vix.loc[days[e - 1], "vix3m"],
                       "entry_date": days[e], "exit_date": days[x], "hold_days": x - e, "legs": legs(e)})
    return trades, days


def needs_by_day(trades: list[dict]) -> dict[date, set]:
    need: dict[date, set] = {}
    for t in trades:
        syms = {s for _, c, p in t["legs"] for s in (c, p)}
        for d in (t["entry_date"].date(), t["exit_date"].date()):
            need.setdefault(d, set()).update(syms)
    return need


def resolve_day(dbc, syms: set, d: date) -> tuple[set, list]:
    """Free Databento symbology check: which raw symbols exist on day d."""
    r = dbc.symbology.resolve(dataset=DATASET, symbols=sorted(syms), stype_in="raw_symbol",
                              stype_out="instrument_id", start_date=d.isoformat(),
                              end_date=(d + timedelta(days=1)).isoformat())
    nf = list(r.get("not_found", []) or [])
    return set(syms) - set(nf), nf


def diagnose(dbc, trades: list[dict], need: dict) -> None:
    """Show exactly why symbols fail to resolve, using only free endpoints
    (plus at most a few cents for one day of SPY contract definitions)."""
    days = sorted(need)
    probe = days[:2] + days[len(days) // 2: len(days) // 2 + 1] + days[-2:]
    for d in probe:
        syms = sorted(need[d])[:6]
        ok, nf = resolve_day(dbc, set(syms), d)
        log(f"[diag] {d}: {len(ok)}/{len(syms)} resolve as raw_symbol. tried: {syms[:2]} ... not_found: {nf[:3]}")
        alt = [x.replace(" ", "") for x in syms]  # unpadded variant
        ok2, _ = resolve_day(dbc, set(alt), d)
        log(f"[diag] {d}: unpadded format resolves {len(ok2)}/{len(alt)} (e.g. {alt[0]})")
    d = probe[len(probe) // 2]
    s0, _ = window_utc(d)
    end = (datetime.fromisoformat(s0) + timedelta(minutes=1)).isoformat()
    try:
        c = dbc.metadata.get_cost(dataset=DATASET, symbols=["SPY.OPT"], schema="definition", stype_in="parent", start=s0, end=end)
        log(f"[diag] one minute of SPY option definitions on {d} would cost ${c:.4f}")
        if c <= 0.25:
            df = dbc.timeseries.get_range(dataset=DATASET, symbols=["SPY.OPT"], schema="definition",
                                          stype_in="parent", start=s0, end=end).to_df()
            if not df.empty:
                cols = [x for x in ("raw_symbol", "symbol", "expiration", "strike_price", "instrument_class") if x in df.columns]
                log(f"[diag] {len(df)} SPY contracts defined; real symbol samples:\n{df[cols].head(8).to_string()}")
            else:
                log("[diag] no definition records in that minute (definitions may only publish at the session start)")
    except Exception as ex:
        log(f"[diag] definition probe failed: {type(ex).__name__}: {str(ex)[:200]}")


def window_utc(d: date) -> tuple[str, str]:
    s = datetime(d.year, d.month, d.day, QUOTE_TIME[0], QUOTE_TIME[1] - WINDOW_MIN, tzinfo=ET).astimezone(timezone.utc)
    e = datetime(d.year, d.month, d.day, QUOTE_TIME[0], QUOTE_TIME[1] + 1, tzinfo=ET).astimezone(timezone.utc)
    return s.isoformat(), e.isoformat()


def main(mode: str, budget: float) -> int:
    import databento as db

    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        log("FATAL: DATABENTO_API_KEY not set")
        return 2
    dbc = db.Historical(key)
    vix = pd.read_csv(config.VIX_TERM_FILE, parse_dates=["date"]).set_index("date").sort_index()
    alp = AlpacaClient()
    log("SPY 15:45 ET prices from Alpaca 1-minute bars...")
    spot = spy_1545(alp, START)
    log(f"  -> {len(spot)} days")
    trades, days = build_trades(vix, spot)
    nd = [t for t in trades if t["test"] == "next_day"]
    log(f"{len(days)} trading days {days[0].date()} to {days[-1].date()}; "
        f"{sum(t['inverted'] for t in nd)} inverted signal days, {sum(not t['inverted'] for t in nd)} contango; "
        f"{len(trades) - len(nd)} episodes")

    need = needs_by_day(trades)
    cached = pd.DataFrame()
    if os.path.exists(CACHE):
        cached = pd.read_csv(CACHE, parse_dates=["ts"])
        have = set(zip(cached["symbol"], pd.to_datetime(cached["day"]).dt.date))
        need = {d: {s for s in ss if (s, d) not in have} for d, ss in need.items()}
        need = {d: ss for d, ss in need.items() if ss}
        log(f"cache: {len(have)} symbol-days already downloaded")
    log(f"to fetch: {sum(len(v) for v in need.values())} symbol-days across {len(need)} days")

    if mode == "diagnose":
        diagnose(dbc, trades, need)
        return 0

    total, per_year, unresolved, bad_days, valid = 0.0, {}, 0, [], {}
    for d, ss in sorted(need.items()):
        try:
            ok, nf = resolve_day(dbc, ss, d)
        except Exception as ex:
            bad_days.append(f"{d}: resolve {type(ex).__name__}: {str(ex)[:100]}")
            continue
        unresolved += len(nf)
        if not ok:
            bad_days.append(f"{d}: none of {len(ss)} symbols exist (e.g. {sorted(ss)[0]!r})")
            continue
        valid[d] = ok
        s, e = window_utc(d)
        try:
            c = dbc.metadata.get_cost(dataset=DATASET, symbols=sorted(ok), schema=SCHEMA, stype_in="raw_symbol", start=s, end=e)
        except Exception as ex:
            bad_days.append(f"{d}: cost {type(ex).__name__}: {str(ex)[:100]}")
            continue
        total += c
        per_year[d.year] = per_year.get(d.year, 0) + c
    resolved = sum(len(v) for v in valid.values())
    log(f"symbol check: {resolved} symbol-days resolve, {unresolved} do not; {len(bad_days)} days with no usable symbols")
    for x in bad_days[:8]:
        log(f"  - {x}")
    if resolved == 0:
        log("FATAL: no symbols resolved on any day -- run mode 'diagnose' to see the real symbol format")
        return 1
    need = valid
    log(f"DATABENTO COST ESTIMATE: ${total:.2f}  by year: { {y: round(v, 2) for y, v in per_year.items()} }  (budget ${budget:.2f})")
    if mode == "estimate":
        log("estimate mode: nothing downloaded, nothing spent")
        return 0
    if total > budget:
        log(f"ABORT: estimate ${total:.2f} exceeds budget ${budget:.2f}; nothing downloaded")
        return 1

    new_rows, errors = [], []
    for i, (d, ss) in enumerate(sorted(need.items())):
        s, e = window_utc(d)
        try:
            df = dbc.timeseries.get_range(dataset=DATASET, symbols=sorted(ss), schema=SCHEMA,
                                          stype_in="raw_symbol", start=s, end=e).to_df()
        except Exception as ex:
            errors.append(f"{d}: {type(ex).__name__}: {str(ex)[:120]}")
            continue
        if df.empty:
            continue
        df = df.reset_index()
        tcol = "ts_recv" if "ts_recv" in df.columns else df.columns[0]
        new_rows.append(pd.DataFrame({"day": d.isoformat(), "symbol": df["symbol"], "ts": df[tcol],
                                      "bid_px_00": df["bid_px_00"], "ask_px_00": df["ask_px_00"]}))
        if (i + 1) % 100 == 0:
            log(f"  fetched {i + 1}/{len(need)} days")
    allq = pd.concat([cached] + new_rows, ignore_index=True) if new_rows else cached
    if not allq.empty:
        allq.to_csv(CACHE, index=False, compression="gzip")
    log(f"quotes: {len(allq)} rows cached; {len(errors)} fetch errors")
    for x in errors[:10]:
        log(f"  - {x}")

    allq["ts"] = pd.to_datetime(allq["ts"], utc=True)
    allq["day"] = pd.to_datetime(allq["day"]).dt.date
    quotes = {}
    for (sym, d), g in allq.groupby(["symbol", "day"]):
        q = quote_at(g.set_index("ts").sort_index(), d)
        if q:
            quotes[(sym, d)] = q

    rows = []
    for t in trades:
        r = price_straddle(quotes, t["legs"], t["entry_date"].date(), t["exit_date"].date())
        rows.append({k: v for k, v in t.items() if k != "legs"} | r)
    out = pd.DataFrame(rows)
    out.to_csv(f"{config.DATA_DIR}/backtest_index_vol_db_trades.csv", index=False)

    print("\n=========== SPY STRADDLE BACKTEST v2 (quotes, real spreads) ===========")
    ndp = out[out["test"] == "next_day"]
    pr = ndp[ndp["status"] == "priced"]
    print(f"Next-day test: {len(ndp)} days, {len(pr)} priced ({len(pr) / max(len(ndp), 1):.0%}); "
          f"median half-spread at entry {pr['half_spread_pct_entry'].median():.2%} of straddle premium")
    for lab, g in [("INVERTED curve (signal days)", pr[pr["inverted"]]), ("contango (control)", pr[~pr["inverted"]])]:
        print(f"  {lab}:")
        print(f"     gross (mid->mid):      {describe(g['ret_gross'])}")
        print(f"     real spreads:          {describe(g['ret_real_spread'])}")
        print(f"     flat 1% (sanity):      {describe(g['ret_flat_1pct'])}")
    inv, con = pr[pr["inverted"]]["ret_gross"], pr[~pr["inverted"]]["ret_gross"]
    if len(inv) > 1 and len(con) > 1:
        diff = inv.mean() - con.mean()
        se = math.sqrt(inv.var(ddof=1) / len(inv) + con.var(ddof=1) / len(con))
        print(f"  inverted minus contango, gross mean: {diff:+.2%} per day (t = {diff / se:+.2f})")
    ep = out[(out["test"] == "episode")]
    epp = ep[ep["status"] == "priced"]
    print(f"\nEpisode test: {len(ep)} episodes, {len(epp)} priced")
    print(f"   gross:        {describe(epp['ret_gross'])}")
    print(f"   real spreads: {describe(epp['ret_real_spread'])}")
    for r in epp.itertuples():
        print(f"     {r.entry_date.date()} -> {r.exit_date.date()} ({r.hold_days}d) VIX {r.vix:.1f}/{r.vix3m:.1f}  "
              f"{r.mid_entry:.2f} -> {r.mid_exit:.2f}  gross {r.ret_gross:+.1%}, after spreads {r.ret_real_spread:+.1%}")
    return 0


if __name__ == "__main__":
    try:
        m = sys.argv[1] if len(sys.argv) > 1 else "estimate"
        b = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
        sys.exit(main(m, b))
    except Exception:
        traceback.print_exc()
        sys.exit(1)

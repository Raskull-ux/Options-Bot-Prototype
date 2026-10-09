# Cross checker v2 -- follows the Tazmanian Trader rulebook.
#
# HOURLY 20/50 SETUP (clock-aligned hourly candles 4 AM-8 PM ET, pre/after-hours included)
#   event    hourly 20 SMA crosses below the 50
#   scenario trade day's 9:30 open vs the hourly 50/200 known at that time (8:00 candle):
#            open at/above the 50 | open at the 200 | below the 200 | between 200 and 50
#   entry    first tag of the hourly 50 from underneath within 36h, on a 9:00-14:00 candle
#            (rulebook 1.4: no new entries after 3:00 pm; options trade regular hours only)
#   versions A   stop = yesterday's high; target = hourly 200 (if entry above it); else out at 36h
#            A5  same, but flat at the entry day's close if red (rulebook 1.5) -- THE HEADLINE
#            B   stop = hourly close above the 50, then the next hour trades above that candle's high
#   control  the same tag of the 50 when the 20 is ABOVE the 50 (no bear cross)
#   numbers  real move (what the put earns) + the same with SPY's move removed (stocks only;
#            QQQ/SPY stay real)
# DAILY 20/50 cross and 10 EMA/10 SMA cross (daily + hourly) vs every day / any hour.
# QQQ 10-day gauge: every 10/10 bear cross date (check: Sept 2023, Feb 2025).
# CALIBRATION: every QQQ hourly bear cross in 2026, to match against Taz's nine charts.
# Stats are clustered by date. Stock moves, not option P&L.
import math
import sys
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from lib.alpaca_client import AlpacaClient, DATA
from lib.crosses import add_mas, cross_below, ten_day_gauge

ET = ZoneInfo("America/New_York")
SYMBOLS = ["QQQ", "SPY", "NVDA", "TSLA", "AMD", "META", "NFLX", "COIN", "INTC", "AAPL", "AMZN", "MSFT", "GOOGL",
           "AVGO", "PLTR", "MU", "ORCL", "CRM", "UBER", "SHOP", "MSTR", "HOOD", "ARM", "LLY", "JPM", "XOM", "BA",
           "DIS", "WMT", "COST"]
INDEXES = {"QQQ", "SPY"}
# Hourly candles run 24 hours like Taz's Thinkorswim chart: regular feed 4 AM-8 PM ET
# + Blue Ocean overnight (feed=boats) 8 PM-4 AM ET. Overnight history starts Jan 2025.
HOURLY_FROM = "2025-01-02"
TEST_FROM = "2025-02-01"        # after ~8 days of 24h warm-up for the 200-hour SMA
DAILY_FROM = "2024-10-01"       # daily tests keep their original window (daily is the same on every platform)
SPLIT = "2026-01-01"            # period check: Feb-Dec 2025 vs 2026
WINDOW_H = 36
NEAR = 0.004
ENTRY_HOURS = range(9, 15)       # 9:00 ... 14:00 candles

PASS_RULES = """PASS RULES (fixed before any result is computed):
  HOURLY 20/50 SETUP passes only if ALL of these hold for version A5 (stop A, flat at the close if red):
    1. average real move per trade > 0
    2. beats the no-cross control (same tag of the 50, 20 above 50) with t >= 2
    3. still > 0 on average with SPY's move removed
    4. beats the control in BOTH periods: Feb-Dec 2025 AND 2026
  DAILY 20/50 (at +5 days), DAILY 10/10 (+5 days), HOURLY 10/10 (+12 hours) each pass only if:
    1. beats every-day / any-hour with t >= 2 (real move)
    2. still > 0 on average with SPY's move removed"""


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def fetch_bars(client, symbols, timeframe, start, feed=None):
    rows, token = [], None
    end = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    while True:
        p = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": start, "end": end,
             "limit": 10000, "feed": feed or config.STOCK_BARS_FEED, "adjustment": "split"}
        if token:
            p["page_token"] = token
        body = client._get(f"{DATA}/v2/stocks/bars", p)
        for s, bl in (body.get("bars") or {}).items():
            rows += [{"symbol": s, "t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]} for b in bl]
        token = body.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(ET)
    return df.sort_values(["symbol", "t"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# hourly setup
# ---------------------------------------------------------------------------
def classify_open(op, s50, s200):
    if op >= s50 * (1 - NEAR):
        return "open at/above the 50"
    if abs(op / s200 - 1) <= NEAR:
        return "open at the 200"
    if op < s200 * (1 - NEAR):
        return "below the 200"
    return "between 200 and 50"


def first_tag_from_below(g, start, end_t):
    """First 9:00-14:00 candle after `start` whose high reaches the 50 coming from underneath."""
    o, h, c, s50, t = g["o"].to_numpy(), g["h"].to_numpy(), g["c"].to_numpy(), g["sma50"].to_numpy(), g["t"]
    for k in range(start, len(g)):
        if t.iloc[k] > end_t:
            return None
        if t.iloc[k].hour not in ENTRY_HOURS:
            continue
        if h[k] >= s50[k] and (o[k] < s50[k] or (k > 0 and c[k - 1] < s50[k - 1])):
            return k
    return None


def spy_short(spy_end: pd.Series | None, t_start, t_end_bar) -> float:
    """SPY short return from the start of the entry candle to the close of the exit candle.
    spy_end: SPY closes indexed by candle END time."""
    if spy_end is None:
        return 0.0
    a, b = spy_end.asof(t_start), spy_end.asof(t_end_bar + pd.Timedelta(hours=1))
    return float(a / b - 1) if a == a and b == b and b > 0 else np.nan


def trade_from_tag(g, j, end_t, prev_high, spy_end=None, is_index=False):
    """Short at the tag of the 50 on candle j. Returns real and market-removed results."""
    o, h, l, c = g["o"].to_numpy(), g["h"].to_numpy(), g["l"].to_numpy(), g["c"].to_numpy()
    s50, s200, t = g["sma50"].to_numpy(), g["sma200"].to_numpy(), g["t"]
    entry = o[j] if o[j] >= s50[j] else s50[j]
    last = j
    while last + 1 < len(g) and t.iloc[last + 1] <= end_t:
        last += 1
    day = t.iloc[j].date()
    close_k = max(k for k in range(j, last + 1) if t.iloc[k].date() == day and t.iloc[k].hour <= 15)
    mae = (max(h[j:last + 1]) - entry) / entry
    mfe = (entry - min(l[j + 1:last + 1])) / entry if last > j else 0.0
    above200 = entry > s200[j]
    tgt_k = next((k for k in range(j + 1, last + 1) if l[k] <= s200[k]), None) if above200 else None
    stopA_k = next((k for k in range(j, last + 1) if h[k] > prev_high), None) if prev_high > entry else "skip"
    stopB_k = next((k + 1 for k in range(j, last) if c[k] > s50[k] and h[k + 1] > h[k]), None)

    def run(stop_k, stop_px, flat_red):
        """-> (pnl, outcome, exit candle index)"""
        if flat_red:
            s_ok = stop_k is not None and stop_k <= close_k
            t_ok = tgt_k is not None and tgt_k <= close_k
            if s_ok and (not t_ok or stop_k <= tgt_k):
                return (entry - stop_px) / entry, "stopped", stop_k
            if t_ok:
                return (entry - s200[tgt_k]) / entry, "hit the 200", tgt_k
            at_close = entry / c[close_k] - 1
            if at_close <= 0:
                return at_close, "flat at the close (red)", close_k
        if stop_k is not None and (tgt_k is None or stop_k <= tgt_k):
            return (entry - stop_px) / entry, "stopped", stop_k
        if tgt_k is not None:
            return (entry - s200[tgt_k]) / entry, "hit the 200", tgt_k
        return entry / c[last] - 1, "time exit (36h)", last

    out = {"entry": entry, "entry_t": t.iloc[j], "mae": mae, "mfe": mfe, "above200_at_entry": above200,
           "reached_200": tgt_k is not None, "ret_36h": entry / c[last] - 1}
    for name, sk, px, fr in (("A", stopA_k, prev_high, False), ("A5", stopA_k, prev_high, True),
                             ("B", stopB_k, h[stopB_k - 1] if isinstance(stopB_k, (int, np.integer)) else np.nan, False)):
        if sk == "skip":
            out[f"pnl_{name}"], out[f"out_{name}"], ek = np.nan, "skipped (already above yesterday's high)", None
        else:
            out[f"pnl_{name}"], out[f"out_{name}"], ek = run(sk, px, fr)
        mkt = 0.0 if is_index or ek is None else spy_short(spy_end, t.iloc[j], t.iloc[ek])
        out[f"adj_{name}"] = out[f"pnl_{name}"] - mkt if out[f"pnl_{name}"] == out[f"pnl_{name}"] else np.nan
    out["adj_36h"] = out["ret_36h"] - (0.0 if is_index else spy_short(spy_end, t.iloc[j], t.iloc[last]))
    return out


def hourly_events(sym, g, daily, spy_end):
    g = g.reset_index(drop=True)
    is_index = sym in INDEXES
    dv = daily.set_index("date")
    ddates = dv.index
    dbear = dv["sma20"] < dv["sma50"]

    def day_info(after_t):
        dpos = ddates.searchsorted(pd.Timestamp(after_t.date()) + (pd.Timedelta(days=1) if after_t.hour >= 16 else pd.Timedelta(0)))
        if dpos >= len(ddates) or dpos == 0:
            return None
        return ddates[dpos], float(dv["o"].iloc[dpos]), float(dv["h"].iloc[dpos - 1]), bool(dbear.iloc[dpos - 1])

    events, controls = [], []
    xs = np.flatnonzero(cross_below(g["sma20"], g["sma50"]).to_numpy() & g["sma200"].notna().to_numpy())
    for i in xs:
        ti = g["t"].iloc[i]
        if ti < pd.Timestamp(TEST_FROM, tz=ET):
            continue
        info = day_info(ti)
        if info is None:
            continue
        day, op, prev_high, d_bear = info
        known = g[g["t"] <= pd.Timestamp(day, tz=ET) + pd.Timedelta(hours=8, minutes=30)]
        if known.empty:
            continue
        k0 = known.iloc[-1]
        end_t = ti + pd.Timedelta(hours=WINDOW_H)
        ev = {"symbol": sym, "cross_t": ti, "trade_day": day.date(),
              "cross_when": ("overnight" if ti.hour >= 20 or ti.hour < 4 else "pre-market" if ti.hour < 9
                             else "after hours" if ti.hour >= 16 else "regular hours"),
              "scenario": classify_open(op, k0["sma50"], k0["sma200"]),
              "macd_bear": bool(g["macd"].iloc[i] < g["macd_sig"].iloc[i]),
              "s50_below_200": bool(g["sma50"].iloc[i] < g["sma200"].iloc[i]), "daily_20_below_50": d_bear}
        k = first_tag_from_below(g, i + 1, end_t)
        if k is None:
            last = g.index[g["t"] <= end_t][-1]
            ev.update(tagged=False, flush_ret_36h=g["c"].iloc[i] / g["c"].iloc[last] - 1)
        else:
            ev.update(tagged=True, hours_to_tag=(g["t"].iloc[k] - ti).total_seconds() / 3600,
                      hourly_1010_bear_at_tag=bool(g["ema10"].iloc[k] < g["sma10"].iloc[k]),
                      **trade_from_tag(g, k, g["t"].iloc[k] + pd.Timedelta(hours=WINDOW_H), prev_high, spy_end, is_index))
        events.append(ev)

    o, h, c = g["o"].to_numpy(), g["h"].to_numpy(), g["c"].to_numpy()
    s20, s50, s200 = g["sma20"].to_numpy(), g["sma50"].to_numpy(), g["sma200"].to_numpy()
    seen = set()
    for k in range(1, len(g)):
        tk = g["t"].iloc[k]
        if tk < pd.Timestamp(TEST_FROM, tz=ET) or tk.hour not in ENTRY_HOURS or np.isnan(s200[k]) or not s20[k] > s50[k]:
            continue
        if not (h[k] >= s50[k] and (o[k] < s50[k] or c[k - 1] < s50[k - 1])) or tk.date() in seen:
            continue
        info = day_info(tk)
        if info is None:
            continue
        seen.add(tk.date())
        controls.append({"symbol": sym, "trade_day": info[0].date(),
                         **trade_from_tag(g, k, tk + pd.Timedelta(hours=WINDOW_H), info[2], spy_end, is_index)})
    return events, controls


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def cstat(x, days):
    d = pd.DataFrame({"x": pd.Series(x).astype(float).to_numpy(), "d": pd.Series(days).to_numpy()}).dropna()
    if len(d) < 3:
        return len(d), np.nan, np.nan, np.nan, 0
    pdm = d.groupby("d")["x"].mean()
    sd = pdm.std(ddof=1)
    t = pdm.mean() / sd * math.sqrt(len(pdm)) if len(pdm) > 1 and sd > 0 else np.nan
    return len(d), pdm.mean(), (d["x"] > 0).mean(), t, len(pdm)


def fmt_stat(x, days):
    n, mu, w, t, nd = cstat(x, days)
    return f"n={n}" if mu != mu else f"n={n:4d} ({nd:3d} days)  avg={mu:+.2%}  win={w:5.1%}  t={t:+.2f}"


def cdiff(a, ad, b, bd):
    pa = pd.DataFrame({"x": pd.Series(a).astype(float).to_numpy(), "d": pd.Series(ad).to_numpy()}).dropna().groupby("d")["x"].mean()
    pb = pd.DataFrame({"x": pd.Series(b).astype(float).to_numpy(), "d": pd.Series(bd).to_numpy()}).dropna().groupby("d")["x"].mean()
    if len(pa) < 3 or len(pb) < 3:
        return np.nan, np.nan
    d = pa.mean() - pb.mean()
    se = math.sqrt(pa.var(ddof=1) / len(pa) + pb.var(ddof=1) / len(pb))
    return d, d / se if se > 0 else np.nan


def show_diff(d, t):
    return "n/a" if d != d else f"{d:+.2%} (t={t:+.2f})"


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main():
    print(PASS_RULES)
    client = AlpacaClient()
    log(f"hourly bars for {len(SYMBOLS)} symbols since {HOURLY_FROM}...")
    day = fetch_bars(client, SYMBOLS, "1Hour", HOURLY_FROM)
    day = day[(day["t"].dt.hour >= 4) & (day["t"].dt.hour < 20)]
    log(f"  -> {len(day)} bars 4 AM-8 PM; adding overnight 8 PM-4 AM (Blue Ocean)...")
    night = fetch_bars(client, SYMBOLS, "1Hour", HOURLY_FROM, feed="boats")
    night = night[(night["t"].dt.hour >= 20) | (night["t"].dt.hour < 4)]
    hb = (pd.concat([day, night]).drop_duplicates(["symbol", "t"]).sort_values(["symbol", "t"]).reset_index(drop=True))
    log(f"  -> {len(night)} overnight bars; {len(hb)} total 24-hour candles")
    log(f"  -> {len(hb)} hourly bars")
    db = fetch_bars(client, SYMBOLS, "1Day", "2022-06-01")
    db["date"] = pd.to_datetime(db["t"].dt.date)
    log(f"  -> {len(db)} daily bars")
    H = {s: add_mas(g.reset_index(drop=True)) for s, g in hb.groupby("symbol")}
    D = {s: add_mas(g.reset_index(drop=True)) for s, g in db.groupby("symbol")}
    spy_end = pd.Series(H["SPY"]["c"].to_numpy(), index=H["SPY"]["t"] + pd.Timedelta(hours=1)) if "SPY" in H else None

    # ---------------- 1. hourly setup ----------------
    ev, ctl = [], []
    for s in H:
        if s in D:
            e, c_ = hourly_events(s, H[s], D[s], spy_end)
            ev += e
            ctl += c_
    E, C = pd.DataFrame(ev), pd.DataFrame(ctl)
    E.to_csv(f"{config.DATA_DIR}/cross_check_hourly_events.csv", index=False)
    T = E[E["tagged"]].copy()
    print("\n==================== 1. HOURLY 20/50 BEAR CROSS → TAG OF THE 50 ====================")
    print(f"{len(E)} crosses since {TEST_FROM}, {E['symbol'].nunique()} symbols. {len(T)} ({len(T) / max(len(E), 1):.0%}) tagged the 50 "
          f"from underneath on a 9:00-14:00 candle within 36h (median {T['hours_to_tag'].median():.1f}h after the cross).")
    print("\nShort at the tag (real move = what the put rides on; second line = SPY's move removed):")
    for v, lab in (("A5", "A5 HEADLINE: stop yday high, flat at close if red"), ("A", "A: stop yday high, hold up to 36h"),
                   ("B", "B: close>50 then higher = out"), ("36h", "no stop, 36h")):
        col = "ret_36h" if v == "36h" else f"pnl_{v}"
        adj = "adj_36h" if v == "36h" else f"adj_{v}"
        print(f"  {lab:48s} real    {fmt_stat(T[col], T['trade_day'])}")
        print(f"  {'':48s} −market  {fmt_stat(T[adj], T['trade_day'])}")
        if v != "36h":
            print(f"  {'':48s} outcomes {T[f'out_{v}'].value_counts(normalize=True).round(2).to_dict()}")
    print(f"  best drop available within 36h: median {T['mfe'].median():.2%} · worst move against: median {T['mae'].median():.2%}, 90th pct {T['mae'].quantile(.9):.2%}")
    print(f"  entries above the 200 that reached the 200: {T.loc[T['above200_at_entry'], 'reached_200'].mean():.0%}")
    print("\nCONTROL - same tag of the 50 with NO bear cross:")
    print(f"  A5 real    {fmt_stat(C['pnl_A5'], C['trade_day'])}")
    print(f"  A5 −market {fmt_stat(C['adj_A5'], C['trade_day'])}")
    d, dt = cdiff(T["pnl_A5"], T["trade_day"], C["pnl_A5"], C["trade_day"])
    print(f"  >>> after-cross minus no-cross (A5, real): {show_diff(d, dt)}")
    print("\nBy scenario at the open (A5, real):")
    for k, g in T.groupby("scenario"):
        print(f"  {k:22s} {fmt_stat(g['pnl_A5'], g['trade_day'])}   tagged {len(g) / max((E['scenario'] == k).sum(), 1):.0%} of these crosses")
    print("By when the cross formed (A5, real):")
    for k, g in T.groupby("cross_when"):
        print(f"  {k:14s} {fmt_stat(g['pnl_A5'], g['trade_day'])}")
    print("Combinations (A5, real):")
    for col, lab in (("daily_20_below_50", "daily 20 under 50"), ("hourly_1010_bear_at_tag", "hourly 10 EMA under 10 SMA at the tag"),
                     ("macd_bear", "hourly MACD bearish at the cross"), ("s50_below_200", "hourly 50 already under the 200")):
        for val in (True, False):
            g = T[T[col] == val]
            print(f"  {lab:40s} {'YES' if val else 'no ':3s} {fmt_stat(g['pnl_A5'], g['trade_day'])}")
    NT = E[~E["tagged"]]
    print(f"\nNo tag before 3 pm (never rallied back to the 50): {len(NT)} crosses; 36h move after the cross: {fmt_stat(NT['flush_ret_36h'], NT['trade_day'])}")
    print("  ^ HINDSIGHT, NOT A TRADE: picked because price never came back up. Shows only how big the flush cases were.")

    # verdict
    n, mu, w, t, nd = cstat(T["pnl_A5"], T["trade_day"])
    _, mu_adj, _, _, _ = cstat(T["adj_A5"], T["trade_day"])
    per = []
    for lo, hi in ((TEST_FROM, SPLIT), (SPLIT, "2100-01-01")):
        ts = T[(pd.to_datetime(T["trade_day"]) >= lo) & (pd.to_datetime(T["trade_day"]) < hi)]
        cs = C[(pd.to_datetime(C["trade_day"]) >= lo) & (pd.to_datetime(C["trade_day"]) < hi)]
        pd_, _ = cdiff(ts["pnl_A5"], ts["trade_day"], cs["pnl_A5"], cs["trade_day"])
        per.append(pd_)
    checks = [mu > 0, dt == dt and dt >= 2, mu_adj > 0, all(x == x and x > 0 for x in per)]
    print(f"\n>>> HOURLY SETUP VERDICT: {verdict(all(checks))}  "
          f"[1 avg>0: {checks[0]} · 2 beats control t>=2: {checks[1]} · 3 >0 minus market: {checks[2]} · "
          f"4 both periods: {checks[3]} ({show_diff(per[0], np.nan).split(' ')[0]} / {show_diff(per[1], np.nan).split(' ')[0]})]")

    # ---------------- 2/3. daily crosses ----------------
    spyd = D["SPY"].set_index("date")["c"]
    spy_f = {k: (spyd / spyd.shift(-k) - 1) for k in (1, 3, 5, 10)}
    rows = []
    for s, g in D.items():
        x2050, x1010 = cross_below(g["sma20"], g["sma50"]), cross_below(g["ema10"], g["sma10"])
        whip = pd.concat([(g["ema10"].shift(-q) > g["sma10"].shift(-q)) for q in (1, 2, 3)], axis=1).any(axis=1)
        for i in range(len(g)):
            dte = g["date"].iloc[i]
            if dte < pd.Timestamp(DAILY_FROM):
                continue
            r = {"symbol": s, "day": dte.date(), "x2050": bool(x2050.iloc[i]), "x1010": bool(x1010.iloc[i]), "whip": bool(whip.iloc[i])}
            for k in (1, 3, 5, 10):
                raw = g["c"].iloc[i] / g["c"].iloc[i + k] - 1 if i + k < len(g) else np.nan
                r[f"r{k}"] = raw
                r[f"a{k}"] = raw if s in INDEXES else raw - spy_f[k].get(dte, np.nan)
            rows.append(r)
    R = pd.DataFrame(rows)

    def daily_block(mask_col, title, key_k):
        A = R[R[mask_col]]
        print(f"\n==================== {title} ====================")
        if mask_col == "x1010":
            print(f"  {len(A)} crosses; reversed back within 3 days (whipsaw): {A['whip'].mean():.0%}")
        for k in (1, 3, 5, 10):
            dd, tt = cdiff(A[f"r{k}"], A["day"], R[f"r{k}"], R["day"])
            print(f"  +{k:2d}d real    {fmt_stat(A[f'r{k}'], A['day'])}  | vs every day {show_diff(dd, tt)}")
            print(f"       −market  {fmt_stat(A[f'a{k}'], A['day'])}")
        dd, tt = cdiff(A[f"r{key_k}"], A["day"], R[f"r{key_k}"], R["day"])
        _, ma, _, _, _ = cstat(A[f"a{key_k}"], A["day"])
        ok = [tt == tt and tt >= 2, ma > 0]
        print(f"  >>> VERDICT (+{key_k}d): {verdict(all(ok))}  [beats every day t>=2: {ok[0]} · >0 minus market: {ok[1]}]")

    daily_block("x2050", "2. DAILY 20/50 BEAR CROSS", 5)
    daily_block("x1010", "3a. DAILY 10 EMA / 10 SMA BEAR CROSS", 5)

    # ---------------- 3b. hourly 10/10 ----------------
    hr, rnd = [], []
    for s, g in H.items():
        tt_, cc = g["t"].to_numpy(), g["c"].to_numpy()
        x = cross_below(g["ema10"], g["sma10"]).to_numpy()
        e10, s10 = g["ema10"].to_numpy(), g["sma10"].to_numpy()
        for i in range(len(g)):
            ti = g["t"].iloc[i]
            if ti < pd.Timestamp(TEST_FROM, tz=ET):
                continue
            if not x[i] and i % 7:
                continue
            row = {"day": ti.date(), "cross": bool(x[i]), "whip": bool(any(e10[i + q] > s10[i + q] for q in (1, 2, 3) if i + q < len(g)))}
            for kh in (4, 12, 36):
                j = np.searchsorted(tt_, tt_[i] + np.timedelta64(kh, "h"))
                if j >= len(cc):
                    row[f"h{kh}"] = row[f"a{kh}"] = np.nan
                    continue
                raw = cc[i] / cc[j] - 1
                row[f"h{kh}"] = raw
                # stock: close of candle i -> close of candle j; SPY over exactly the same span
                row[f"a{kh}"] = raw if s in INDEXES else raw - spy_short(spy_end, ti + pd.Timedelta(hours=1), g["t"].iloc[j])
            (hr if x[i] else rnd).append(row)
    HR, RH = pd.DataFrame(hr), pd.DataFrame(rnd)
    print("\n==================== 3b. HOURLY 10 EMA / 10 SMA BEAR CROSS ====================")
    print(f"  {len(HR)} crosses; reversed back within 3 hours (whipsaw): {HR['whip'].mean():.0%}")
    for kh in (4, 12, 36):
        dd, tt = cdiff(HR[f"h{kh}"], HR["day"], RH[f"h{kh}"], RH["day"])
        print(f"  +{kh:2d}h real    {fmt_stat(HR[f'h{kh}'], HR['day'])}  | vs any hour {show_diff(dd, tt)}")
        print(f"       −market  {fmt_stat(HR[f'a{kh}'], HR['day'])}")
    dd, tt = cdiff(HR["h12"], HR["day"], RH["h12"], RH["day"])
    _, ma, _, _, _ = cstat(HR["a12"], HR["day"])
    print(f"  >>> VERDICT (+12h): {verdict(tt == tt and tt >= 2 and ma > 0)}  [beats any hour t>=2: {tt == tt and tt >= 2} · >0 minus market: {ma > 0}]")

    # ---------------- 5. QQQ 10-day gauge ----------------
    print("\n==================== 5. QQQ 10-DAY CANDLES: 10 EMA vs 10 SMA ====================")
    gq = ten_day_gauge(D["QQQ"].set_index("date")[["o", "h", "l", "c"]], config.TEN_DAY_ANCHOR)
    tb = gq["table"]
    for st in gq["bear_crosses"]:
        pos = tb.index[tb["start"] == st][0]
        c0 = tb["c"].iloc[pos]
        fw = [(tb["c"].iloc[pos + n] / c0 - 1) if pos + n < len(tb) else np.nan for n in (1, 2, 4)]
        print(f"  {pd.Timestamp(st).date()}  close {c0:.2f} · next 1/2/4 candles {fw[0]:+.1%} / {fw[1]:+.1%} / {fw[2]:+.1%} · "
              f"lowest low within 4 candles {tb['l'].iloc[pos:pos + 5].min() / c0 - 1:+.1%}")
    print(f"  Now: 10 EMA {gq['ema10']:.2f} vs 10 SMA {gq['sma10']:.2f} -> {'BEARISH' if gq['bearish'] else 'not bearish'}; "
          f"candle {gq['candle_days_so_far']}/10 days in. Check: your chart showed Sept 14 2023 and Feb 21 2025.")

    # ---------------- calibration ----------------
    print("\n==================== CALIBRATION: EVERY QQQ HOURLY 20/50 BEAR CROSS IN 2026 ====================")
    print("  Match these against your nine QQQ charts. If yours are here, the bot reads the cross like you do.")
    Q = E[(E["symbol"] == "QQQ") & (pd.to_datetime(E["trade_day"]) >= SPLIT)].sort_values("cross_t")
    for r in Q.itertuples():
        tag = (f"tag {r.entry_t:%a %b %d %H:%M} @ {r.entry:.2f} · A5 {r.out_A5} {r.pnl_A5:+.2%}" if r.tagged else "no tag before 3 pm")
        print(f"  cross {r.cross_t:%a %b %d %H:%M} ({r.cross_when}) · {r.scenario} · {tag}")
    print(f"\nAll {len(E)} hourly events: data/cross_check_hourly_events.csv")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

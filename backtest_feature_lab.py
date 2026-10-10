# FEATURE LAB -- why does the hourly 20/50 setup work when it works, and what adds to it?
#
# Built like a fund research process:
#   * DISCOVER on TRAIN only (Feb-Dec 2025). Every feature, every filter, every rule is
#     chosen there by fixed code, never by hand.
#   * CONFIRM once on TEST (2026), which the discovery never saw.
#   * Every feature uses only information available at the moment of the decision
#     (no peeking at a candle's close before it has closed).
#
# Hourly candles: 24 hours like Thinkorswim (regular feed 4 AM-8 PM + Blue Ocean overnight).
# The setup: hourly 20 SMA crosses under the 50; price rallies back and tags the 50 from
# underneath on a 9:00-14:00 candle within 36 hours.
#
# THREE ENTRY DOORS (Taz's three ways in):
#   E0 "tag"         short at the 50 the moment it is touched (decision before the candle closes)
#   E1 "rejection"   the tag candle CLOSES back under the 50 -> short at that close
#   E2 "pop+reject"  the tag candle closes ABOVE the 50, the next hour closes back under -> short there
# Trade management (same as the checker's A5 headline): stop = yesterday's high,
# target = hourly 200 if entry is above it, flat at the close if red, else hold up to 36h.
# Control = the same doors on tags of the 50 with NO bear cross (20 above 50).
#
# FEATURES (about 60, all known at decision time):
#   hourly  10 EMA vs 10 SMA, MACD, RSI, rally from the 200, rally size in ATR, rally volume
#           ("no buyers"), room to the 200, 50 vs 200, candle patterns on the decision candle
#   3-hour  RSI 80+/90+ in the last 10 bars, MACD/RSI bearish divergence, MACD, 10/10
#           (3h candles at 1,4,7,10,13,16,19,22 ET, overnight included)
#   daily   RSI 80+ in last 10 days, exponential run (20-day +15%, 15% over the 50), RSI/MACD
#           bearish divergence, higher high on low volume, 10/10, 20/50, below 10 SMA,
#           TD sell 9, high ATR regime, candle patterns (shooting star, engulfing, evening
#           star, dark cloud, three black crows, gravestone)
#   market  SPY daily below 20 / 10-10 bearish / 20-under-50, QQQ hourly 20-under-50, VIX > 20
#   timing  entry hour, Monday/Friday, when the cross formed, open scenario
# Stock moves, not option P&L.
import math
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import backtest_crosses as bx
from lib.alpaca_client import AlpacaClient
from lib.crosses import add_mas, cross_below
from lib import technicals as ta
from lib.signals import average_true_range

ET = bx.ET
SYMBOLS = bx.SYMBOLS + ["ADBE", "NOW", "INTU", "QCOM", "TXN", "MRVL", "AMAT", "LRCX", "KLAC", "PANW",
                        "CRWD", "SNOW", "NET", "DDOG", "ABNB", "PYPL", "SBUX", "NKE", "GS", "MS",
                        "BAC", "CVX", "CAT", "GE", "UNH", "V", "HD", "PEP", "SMCI", "RBLX"]
HOURLY_FROM, TEST_FROM, SPLIT = "2025-01-02", "2025-02-01", "2026-01-01"
WINDOW_H, NEAR = 36, 0.004
ENTRY_HOURS = range(9, 15)
MIN_SIDE = 40          # min trades on each side of a feature split to report it
RULE_MIN_N = 80        # min TRAIN trades for a rule
MAX_FILTERS = 3

PASS_RULES = """PRE-REGISTERED (fixed before any result is computed)
  Discovery uses TRAIN = Feb-Dec 2025 only. TEST = 2026 is scored once, at the end.
  Feature screen: each feature compared with vs without (day-clustered t); false discoveries
    controlled with Benjamini-Hochberg across ALL feature x door tests (q < 0.10 to be eligible).
  Rule builder (automatic): start from each entry door; add up to 3 eligible features, each step
    picking the one with the highest TRAIN average that keeps n >= 80 and beats the excluded
    trades with t >= 2. Stop when nothing qualifies.
  A rule PASSES on TEST 2026 only if ALL hold:
    1. n >= 40 trades   2. average real move > 0   3. average with SPY removed > 0
    4. beats the no-cross control (same door, same non-cross filters) with t >= 2
    5. positive average on more than half of the symbols with 3+ trades
  The 10/10 lead (found on data that included 2026) is reported separately and CANNOT be
  confirmed by 2026; its real test is forward paper-trading."""


def log(m):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {m}", flush=True)


# ---------------------------------------------------------------------------
# candle patterns (array based; i = candle index)
# ---------------------------------------------------------------------------
def _rng(o, h, l, c, i):
    return h[i] - l[i]


def shooting_star(o, h, l, c, i):
    r = _rng(o, h, l, c, i)
    if r <= 0:
        return False
    body, up, lo = abs(c[i] - o[i]), h[i] - max(o[i], c[i]), min(o[i], c[i]) - l[i]
    return up >= 2 * body and up >= 0.5 * r and lo <= 0.25 * r


def gravestone(o, h, l, c, i):
    r = _rng(o, h, l, c, i)
    return r > 0 and abs(c[i] - o[i]) <= 0.1 * r and h[i] - max(o[i], c[i]) >= 0.6 * r


def bear_engulf(o, h, l, c, i):
    return i >= 1 and c[i] < o[i] and c[i - 1] > o[i - 1] and o[i] >= c[i - 1] and c[i] <= o[i - 1]


def dark_cloud(o, h, l, c, i):
    return (i >= 1 and c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] >= c[i - 1]
            and c[i] < (o[i - 1] + c[i - 1]) / 2 and c[i] > o[i - 1])


def evening_star(o, h, l, c, i):
    if i < 2:
        return False
    b0 = abs(c[i - 2] - o[i - 2])
    return (c[i - 2] > o[i - 2] and b0 >= 0.6 * max(_rng(o, h, l, c, i - 2), 1e-12)
            and abs(c[i - 1] - o[i - 1]) <= 0.3 * b0 and c[i] < o[i] and c[i] < (o[i - 2] + c[i - 2]) / 2)


def three_black_crows(o, h, l, c, i):
    if i < 2:
        return False
    return all(c[j] < o[j] for j in (i - 2, i - 1, i)) and c[i] < c[i - 1] < c[i - 2] \
        and min(o[i - 1], o[i - 2]) <= o[i - 1] and o[i - 1] <= o[i - 2] and o[i] <= o[i - 1]


def close_low_third(o, h, l, c, i):
    r = _rng(o, h, l, c, i)
    return r > 0 and (c[i] - l[i]) <= r / 3


def bear_div(hh, rsi, macd, j, recent=5, prior=30):
    """Higher high in the last `recent` bars vs the `prior` bars before; RSI / MACD lower at that high."""
    if j - recent - prior < 0:
        return False, False
    rec = slice(j - recent + 1, j + 1)
    pri = slice(j - recent - prior + 1, j - recent + 1)
    ri = rec.start + int(np.argmax(hh[rec]))
    pi = pri.start + int(np.argmax(hh[pri]))
    if not hh[ri] > hh[pi]:
        return False, False
    return bool(rsi[ri] < rsi[pi] - 1), bool(macd[ri] < macd[pi])


# ---------------------------------------------------------------------------
# per-symbol preparation
# ---------------------------------------------------------------------------
def prep_hourly(g):
    g = add_mas(g.reset_index(drop=True))
    g["rsi"] = ta.rsi(g["c"]).to_numpy()
    g["hist"] = g["macd"] - g["macd_sig"]
    g["atr"] = average_true_range(g["h"], g["l"], g["c"], 14).to_numpy()
    return g


def prep_3h(g):
    tl = g["t"].dt.tz_localize(None)
    b = (tl - pd.Timedelta(hours=1)).dt.floor("3h") + pd.Timedelta(hours=1)
    x = g.assign(b=b).groupby("b").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last")).reset_index()
    x["end"] = (x["b"] + pd.Timedelta(hours=3)).dt.tz_localize(ET, ambiguous="NaT", nonexistent="shift_forward")
    x = x.dropna(subset=["end"]).reset_index(drop=True)
    x = add_mas(x)
    x["rsi"] = ta.rsi(x["c"]).to_numpy()
    return x


def prep_daily(d):
    d = add_mas(d.reset_index(drop=True))
    d["rsi"] = ta.rsi(d["c"]).to_numpy()
    d["atr"] = average_true_range(d["h"], d["l"], d["c"], 14).to_numpy()
    d["atr_pct"] = d["atr"] / d["c"]
    d["atr_pct_med"] = d["atr_pct"].rolling(250, min_periods=60).median()
    d["vol20"] = d["v"].rolling(20).mean()
    cc = d["c"].to_numpy()
    sell = np.zeros(len(d), dtype=int)
    for i in range(4, len(d)):
        sell[i] = sell[i - 1] + 1 if cc[i] > cc[i - 4] else 0
    d["td_sell"] = sell
    return d


def daily_feats(d, p):
    o, h, l, c, v = (d[k].to_numpy() for k in ("o", "h", "l", "c", "v"))
    rsi, macd = d["rsi"].to_numpy(), d["macd"].to_numpy()
    f = {}
    f["D RSI 80+ in last 10 days"] = bool(np.nanmax(rsi[max(0, p - 9):p + 1]) >= 80)
    f["D RSI 70+ now"] = bool(rsi[p] >= 70)
    f["D exponential run (20d +15%)"] = bool(p >= 20 and c[p] / c[p - 20] - 1 > 0.15)
    f["D 15%+ above the 50 SMA"] = bool(c[p] / d["sma50"].iloc[p] - 1 > 0.15)
    rd, md = bear_div(h, rsi, macd, p)
    f["D RSI bearish divergence"], f["D MACD bearish divergence"] = rd, md
    hi_day = p - 4 + int(np.argmax(h[p - 4:p + 1])) if p >= 20 else None
    f["D higher high on low volume"] = bool(hi_day is not None and h[hi_day] >= np.max(h[p - 19:p + 1])
                                           and v[hi_day] < d["vol20"].iloc[p])
    f["D below 10 SMA"] = bool(c[p] < d["sma10"].iloc[p])
    f["D 10 EMA under 10 SMA"] = bool(d["ema10"].iloc[p] < d["sma10"].iloc[p])
    f["D 20 under 50"] = bool(d["sma20"].iloc[p] < d["sma50"].iloc[p])
    f["D TD sell 9 (last 5 days)"] = bool(np.max(d["td_sell"].to_numpy()[max(0, p - 4):p + 1]) >= 9)
    f["D high volatility (ATR% > its median)"] = bool(d["atr_pct"].iloc[p] > d["atr_pct_med"].iloc[p])
    for name, fn in (("shooting star", shooting_star), ("bearish engulfing", bear_engulf), ("evening star", evening_star),
                     ("dark cloud cover", dark_cloud), ("three black crows", three_black_crows), ("gravestone doji", gravestone)):
        f[f"D candle: {name} (yesterday)"] = bool(fn(o, h, l, c, p))
    return f


def tf3_feats(x, end_t):
    j = x["end"].searchsorted(end_t, side="right") - 1
    f = {}
    if j < 40:
        return f
    rsi, macd, hh = x["rsi"].to_numpy(), x["macd"].to_numpy(), x["h"].to_numpy()
    mx = np.nanmax(rsi[j - 9:j + 1])
    f["3h RSI 80+ (last 10 bars)"] = bool(mx >= 80)
    f["3h RSI 90+ (last 10 bars)"] = bool(mx >= 90)
    rd, md = bear_div(hh, rsi, macd, j, recent=5, prior=20)
    f["3h RSI bearish divergence"], f["3h MACD bearish divergence"] = rd, md
    f["3h MACD under signal"] = bool(x["macd"].iloc[j] < x["macd_sig"].iloc[j])
    f["3h 10 EMA under 10 SMA"] = bool(x["ema10"].iloc[j] < x["sma10"].iloc[j])
    return f


def hourly_feats(g, d, entry_px, win_start):
    o, h, l, c, v = (g[k].to_numpy() for k in ("o", "h", "l", "c", "v"))
    t = g["t"]
    f = {}
    f["H 10 EMA under 10 SMA"] = bool(g["ema10"].iloc[d] < g["sma10"].iloc[d])
    f["H MACD under signal"] = bool(g["macd"].iloc[d] < g["macd_sig"].iloc[d])
    f["H MACD histogram falling"] = bool(g["hist"].iloc[d] < g["hist"].iloc[d - 1])
    f["H RSI above 60"] = bool(g["rsi"].iloc[d] > 60)
    f["H 50 under 200"] = bool(g["sma50"].iloc[d] < g["sma200"].iloc[d])
    w = slice(win_start, d + 1)
    m = win_start + int(np.argmin(l[w]))
    f["H rally started at the 200"] = bool(np.min(l[w] - g["sma200"].to_numpy()[w] * 1.003) <= 0)
    atr = g["atr"].iloc[d]
    f["H rally 1.5+ ATR"] = bool(atr > 0 and (entry_px - l[m]) / atr >= 1.5)
    reg = lambda k: 9 <= t.iloc[k].hour <= 15
    up = [v[k] for k in range(m + 1, d + 1) if reg(k)]
    dn = [v[k] for k in range(win_start, m + 1) if reg(k)]
    f["H rally on low volume (no buyers)"] = bool(len(up) >= 2 and len(dn) >= 2 and np.mean(up) < 0.8 * np.mean(dn))
    f["H room to the 200 (0.5%+)"] = bool(entry_px / g["sma200"].iloc[d] - 1 >= 0.005)
    for name, fn in (("shooting star", shooting_star), ("bearish engulfing", bear_engulf), ("dark cloud cover", dark_cloud),
                     ("gravestone doji", gravestone), ("close in lower third", close_low_third)):
        f[f"H candle: {name}"] = bool(fn(o, h, l, c, d))
    f["H candle: red"] = bool(c[d] < o[d])
    return f


# ---------------------------------------------------------------------------
# trade simulation (A5 headline management), general entry
# ---------------------------------------------------------------------------
def simulate(g, entry, entry_bar, intrabar, prev_high, spy_end, is_index):
    """intrabar=True: entry during candle entry_bar (stops count from that candle, target from the next).
    intrabar=False: entry at entry_bar's close (everything counts from the next candle)."""
    h, l, c = g["h"].to_numpy(), g["l"].to_numpy(), g["c"].to_numpy()
    s200, t = g["sma200"].to_numpy(), g["t"]
    end_t = t.iloc[entry_bar] + pd.Timedelta(hours=WINDOW_H)
    first = entry_bar if intrabar else entry_bar + 1
    if first >= len(g):
        return None
    last = first
    while last + 1 < len(g) and t.iloc[last + 1] <= end_t:
        last += 1
    day = t.iloc[entry_bar].date()
    same = [k for k in range(first, last + 1) if t.iloc[k].date() == day and t.iloc[k].hour <= 15]
    close_k = max(same) if same else first
    above = entry > s200[entry_bar]
    tgt_k = next((k for k in range(entry_bar + 1, last + 1) if l[k] <= s200[k]), None) if above else None
    if prev_high <= entry:
        return None                       # already above yesterday's high: no defined stop (skipped, as in the checker)
    stop_k = next((k for k in range(first, last + 1) if h[k] > prev_high), None)
    s_ok = stop_k is not None and stop_k <= close_k
    t_ok = tgt_k is not None and tgt_k <= close_k
    if s_ok and (not t_ok or stop_k <= tgt_k):
        pnl, out, ek = (entry - prev_high) / entry, "stopped", stop_k
    elif t_ok:
        pnl, out, ek = (entry - s200[tgt_k]) / entry, "hit the 200", tgt_k
    elif entry / c[close_k] - 1 <= 0:
        pnl, out, ek = entry / c[close_k] - 1, "flat at close (red)", close_k
    elif stop_k is not None and (tgt_k is None or stop_k <= tgt_k):
        pnl, out, ek = (entry - prev_high) / entry, "stopped", stop_k
    elif tgt_k is not None:
        pnl, out, ek = (entry - s200[tgt_k]) / entry, "hit the 200", tgt_k
    else:
        pnl, out, ek = entry / c[last] - 1, "time exit (36h)", last
    t_start = t.iloc[entry_bar] if intrabar else t.iloc[entry_bar] + pd.Timedelta(hours=1)
    mkt = 0.0 if is_index else bx.spy_short(spy_end, t_start, t.iloc[ek])
    return {"pnl": pnl, "adj": pnl - mkt if mkt == mkt else np.nan, "outcome": out}


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def doors_for_tag(g, k):
    """Yield (door, entry_bar, entry_price, intrabar, decision_bar, decision_time)."""
    o, c, s50, t = g["o"].to_numpy(), g["c"].to_numpy(), g["sma50"].to_numpy(), g["t"]
    yield "E0 tag", k, (o[k] if o[k] >= s50[k] else s50[k]), True, k - 1, t.iloc[k]
    if c[k] < s50[k]:
        yield "E1 rejection close", k, c[k], False, k, t.iloc[k] + pd.Timedelta(hours=1)
    elif (k + 1 < len(g) and t.iloc[k + 1] - t.iloc[k] == pd.Timedelta(hours=1)
          and t.iloc[k + 1].hour in ENTRY_HOURS and c[k + 1] < s50[k + 1]):
        yield "E2 pop then reject", k + 1, c[k + 1], False, k + 1, t.iloc[k + 1] + pd.Timedelta(hours=1)


def symbol_rows(sym, g, x3, d, mkt, spy_end):
    rows = []
    is_index = sym in bx.INDEXES
    ddates = pd.DatetimeIndex(d["date"])
    s20, s50, s200 = g["sma20"].to_numpy(), g["sma50"].to_numpy(), g["sma200"].to_numpy()
    o, h, c, t = g["o"].to_numpy(), g["h"].to_numpy(), g["c"].to_numpy(), g["t"]
    test_from = pd.Timestamp(TEST_FROM, tz=ET)

    def build(kind, k, cross_i):
        day = pd.Timestamp(t.iloc[k].date())
        p = ddates.searchsorted(day) - 1          # last completed daily bar before the trade day
        if p < 60 or ddates.searchsorted(day) >= len(ddates) or ddates[p + 1] != day:
            return
        prev_high = float(d["h"].iloc[p])
        day_open = float(d["o"].iloc[p + 1])
        known = g.index[g["t"] <= pd.Timestamp(day, tz=ET) + pd.Timedelta(hours=8, minutes=30)]
        if len(known) == 0:
            return
        k0 = known[-1]
        base = {"symbol": sym, "kind": kind, "trade_day": day.date(),
                "scenario": bx.classify_open(day_open, s50[k0], s200[k0])}
        if cross_i is not None:
            ti = t.iloc[cross_i]
            base["cross_when"] = ("overnight" if ti.hour >= 20 or ti.hour < 4 else "pre-market" if ti.hour < 9
                                  else "after hours" if ti.hour >= 16 else "regular hours")
        f_daily = daily_feats(d, p)
        f_mkt = mkt.get(day.date(), {})
        for door, e, px, intrabar, dec, dec_t in doors_for_tag(g, k):
            if dec < 30:
                continue
            res = simulate(g, px, e, intrabar, prev_high, spy_end, is_index)
            if res is None:
                continue
            f = {}
            f.update(hourly_feats(g, dec, px, max(0, dec - 24)))
            f.update(tf3_feats(x3, dec_t))
            f.update(f_daily)
            f.update(f_mkt)
            qh = mkt.get(("QQQ_H", dec_t))
            if qh is not None:
                f["M QQQ hourly 20 under 50"] = qh
            for hr in ENTRY_HOURS:
                f[f"T entry hour {hr}:00"] = t.iloc[e].hour == hr
            f["T Monday"], f["T Friday"] = day.dayofweek == 0, day.dayofweek == 4
            sc = base["scenario"]
            for s_ in ("open at/above the 50", "open at the 200", "below the 200", "between 200 and 50"):
                f[f"S {s_}"] = sc == s_
            if cross_i is not None:
                for cw in ("overnight", "pre-market", "regular hours", "after hours"):
                    f[f"C cross {cw}"] = base["cross_when"] == cw
            rows.append({**base, "door": door, "entry_t": t.iloc[e], "entry": px,
                         "h1010_at_tag_close": bool(g["ema10"].iloc[k] < g["sma10"].iloc[k]), **res, **f})

    # setup events: bear cross -> first tag from underneath within 36h
    used = set()
    xs = np.flatnonzero(cross_below(g["sma20"], g["sma50"]).to_numpy() & g["sma200"].notna().to_numpy())
    for i in xs:
        if t.iloc[i] < test_from:
            continue
        k = bx.first_tag_from_below(g, i + 1, t.iloc[i] + pd.Timedelta(hours=WINDOW_H))
        if k is None or k in used:
            continue
        used.add(k)
        build("setup", k, i)
    # control: first tag from underneath per day while the 20 is ABOVE the 50
    seen = set()
    for k in range(1, len(g)):
        tk = t.iloc[k]
        if tk < test_from or tk.hour not in ENTRY_HOURS or np.isnan(s200[k]) or not s20[k] > s50[k]:
            continue
        if tk.date() in seen or not (h[k] >= s50[k] and (o[k] < s50[k] or c[k - 1] < s50[k - 1])):
            continue
        seen.add(tk.date())
        build("control", k, None)
    return rows


def market_table(D, H):
    """Per-day market flags (as of the prior close) + QQQ hourly 20<50 by decision time."""
    out = {}
    spy = D["SPY"]
    for i in range(1, len(spy)):
        p = i - 1
        out[spy["date"].iloc[i].date()] = {
            "M SPY below 20 SMA (daily)": bool(spy["c"].iloc[p] < spy["sma20"].iloc[p]),
            "M SPY 10 EMA under 10 SMA (daily)": bool(spy["ema10"].iloc[p] < spy["sma10"].iloc[p]),
            "M SPY 20 under 50 (daily)": bool(spy["sma20"].iloc[p] < spy["sma50"].iloc[p]),
        }
    try:
        vix = pd.read_csv(config.VIX_TERM_FILE, parse_dates=["date"]).set_index("date").sort_index()
        for day in list(out):
            prior = vix[vix.index < pd.Timestamp(day)]
            if len(prior):
                out[day]["M VIX above 20"] = bool(prior["vix"].iloc[-1] > 20)
    except Exception:
        pass
    q = H["QQQ"]
    for k in range(len(q)):
        out[("QQQ_H", q["t"].iloc[k] + pd.Timedelta(hours=1))] = bool(q["sma20"].iloc[k] < q["sma50"].iloc[k])
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def pval(t):
    return math.erfc(abs(t) / math.sqrt(2)) if t == t else 1.0


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        r = n - rank
        prev = min(prev, p[idx] * n / r)
        q[idx] = prev
    return q


def feature_cols(df):
    skip = {"symbol", "kind", "trade_day", "scenario", "cross_when", "door", "entry_t", "entry", "pnl", "adj",
            "outcome", "h1010_at_tag_close"}
    return [c_ for c_ in df.columns if c_ not in skip]


def screen(df, doors):
    rows = []
    for door in doors:
        x = df[df["door"] == door]
        for f in feature_cols(df):
            if f not in x or x[f].isna().all():
                continue
            m = x[f].fillna(False).astype(bool)
            if m.sum() < MIN_SIDE or (~m).sum() < MIN_SIDE:
                continue
            dd, tt = bx.cdiff(x.loc[m, "pnl"], x.loc[m, "trade_day"], x.loc[~m, "pnl"], x.loc[~m, "trade_day"])
            n, mu, w, _, _ = bx.cstat(x.loc[m, "pnl"], x.loc[m, "trade_day"])
            rows.append({"door": door, "feature": f, "n_with": int(m.sum()), "avg_with": mu, "win_with": w,
                         "diff": dd, "t": tt, "p": pval(tt)})
    s = pd.DataFrame(rows)
    if not s.empty:
        s["q"] = bh(s["p"])
    return s


def apply_rule(x, feats):
    m = pd.Series(True, index=x.index)
    for f in feats:
        m &= x[f].fillna(False).astype(bool) if f in x else False
    return x[m]


def build_rule(x, eligible):
    feats = []
    cur = x
    for _ in range(MAX_FILTERS):
        best = None
        for f in eligible:
            if f in feats or f not in cur:
                continue
            m = cur[f].fillna(False).astype(bool)
            sub, rest = cur[m], cur[~m]
            if len(sub) < RULE_MIN_N or len(rest) < 20:
                continue
            dd, tt = bx.cdiff(sub["pnl"], sub["trade_day"], rest["pnl"], rest["trade_day"])
            if not (tt == tt and tt >= 2):
                continue
            mu = bx.cstat(sub["pnl"], sub["trade_day"])[1]
            if best is None or mu > best[1]:
                best = (f, mu)
        if best is None:
            break
        feats.append(best[0])
        cur = apply_rule(x, feats)
    return feats


def score(label, s, c):
    n, mu, w, t, nd = bx.cstat(s["pnl"], s["trade_day"])
    _, mua, _, _, _ = bx.cstat(s["adj"], s["trade_day"])
    dd, tt = bx.cdiff(s["pnl"], s["trade_day"], c["pnl"], c["trade_day"]) if len(c) else (np.nan, np.nan)
    by = s.groupby("symbol")["pnl"].agg(["mean", "size"])
    by = by[by["size"] >= 3]
    breadth = (by["mean"] > 0).mean() if len(by) else np.nan
    print(f"  {label}")
    print(f"     real      {bx.fmt_stat(s['pnl'], s['trade_day'])}")
    print(f"     -market   {bx.fmt_stat(s['adj'], s['trade_day'])}")
    print(f"     control   {bx.fmt_stat(c['pnl'], c['trade_day']) if len(c) else 'n/a'}   -> setup minus control {bx.show_diff(dd, tt)}")
    print(f"     breadth   {breadth:.0%} of {len(by)} symbols (3+ trades) positive" if len(by) else "     breadth   n/a")
    if len(s):
        print(f"     outcomes  {s['outcome'].value_counts(normalize=True).round(2).to_dict()}")
    return {"n": n, "mu": mu, "mua": mua, "t_ctrl": tt, "breadth": breadth}


def main():
    print(PASS_RULES)
    client = AlpacaClient()
    log(f"hourly bars (24h) for {len(SYMBOLS)} symbols since {HOURLY_FROM}...")
    day = bx.fetch_bars(client, SYMBOLS, "1Hour", HOURLY_FROM)
    day = day[(day["t"].dt.hour >= 4) & (day["t"].dt.hour < 20)]
    night = bx.fetch_bars(client, SYMBOLS, "1Hour", HOURLY_FROM, feed="boats")
    night = night[(night["t"].dt.hour >= 20) | (night["t"].dt.hour < 4)]
    hb = pd.concat([day, night]).drop_duplicates(["symbol", "t"]).sort_values(["symbol", "t"]).reset_index(drop=True)
    db = bx.fetch_bars(client, SYMBOLS, "1Day", "2023-06-01")
    db["date"] = pd.to_datetime(db["t"].dt.date)
    log(f"  -> {len(hb)} hourly candles, {len(db)} daily bars")

    H = {s: prep_hourly(g) for s, g in hb.groupby("symbol")}
    X3 = {s: prep_3h(g) for s, g in H.items()}
    D = {s: prep_daily(g) for s, g in db.groupby("symbol")}
    mkt = market_table(D, H)
    spy_end = pd.Series(H["SPY"]["c"].to_numpy(), index=H["SPY"]["t"] + pd.Timedelta(hours=1))
    rows = []
    for s in H:
        if s in D:
            rows += symbol_rows(s, H[s], X3[s], D[s], mkt, spy_end)
        log(f"  {s}: {len(rows)} rows so far")
    R = pd.DataFrame(rows)
    R.to_csv(f"{config.DATA_DIR}/feature_lab_rows.csv.gz", index=False, compression="gzip")
    R["is_train"] = pd.to_datetime(R["trade_day"]) < pd.Timestamp(SPLIT)
    S, C = R[R["kind"] == "setup"], R[R["kind"] == "control"]
    doors = ["E0 tag", "E1 rejection close", "E2 pop then reject"]
    St, Ct, Sx, Cx = S[S["is_train"]], C[C["is_train"]], S[~S["is_train"]], C[~C["is_train"]]

    print("\n==================== A. THE 10/10 LEAD, MEASURED HONESTLY ====================")
    e0 = S[S["door"] == "E0 tag"]
    for lab, m in (("10/10 read at the tag candle's CLOSE (what the checker did - peeks)", e0["h1010_at_tag_close"]),
                   ("10/10 read BEFORE the tag candle (tradable at the tag)", e0["H 10 EMA under 10 SMA"])):
        m = m.fillna(False).astype(bool)
        print(f"  {lab}")
        print(f"     YES {bx.fmt_stat(e0.loc[m, 'pnl'], e0.loc[m, 'trade_day'])}")
        print(f"     no  {bx.fmt_stat(e0.loc[~m, 'pnl'], e0.loc[~m, 'trade_day'])}")
    e1 = S[S["door"] == "E1 rejection close"]
    m = e1["H 10 EMA under 10 SMA"].fillna(False).astype(bool)
    print("  E1: wait for the rejection candle to CLOSE under the 50, 10/10 known at that close (tradable)")
    print(f"     YES {bx.fmt_stat(e1.loc[m, 'pnl'], e1.loc[m, 'trade_day'])}")
    print(f"     no  {bx.fmt_stat(e1.loc[~m, 'pnl'], e1.loc[~m, 'trade_day'])}")

    print("\n==================== B. ENTRY DOORS (TRAIN 2025 | TEST 2026) ====================")
    for door in doors:
        for lab, s_, c_ in (("TRAIN", St, Ct), ("TEST ", Sx, Cx)):
            s_d, c_d = s_[s_["door"] == door], c_[c_["door"] == door]
            dd, tt = bx.cdiff(s_d["pnl"], s_d["trade_day"], c_d["pnl"], c_d["trade_day"])
            print(f"  {door:20s} {lab} {bx.fmt_stat(s_d['pnl'], s_d['trade_day'])}  | vs control {bx.show_diff(dd, tt)}")

    print("\n==================== C. FEATURE SCREEN (TRAIN ONLY) ====================")
    sc = screen(St, doors)
    if sc.empty:
        print("  not enough data")
        return 0
    sc.to_csv(f"{config.DATA_DIR}/feature_lab_screen.csv", index=False)
    eligible_rows = sc[(sc["q"] < 0.10) & (sc["diff"] > 0)]
    print(f"  {len(sc)} feature x door tests. Passing false-discovery control (q < 0.10) and helping: {len(eligible_rows)}")
    print("  TOP HELPERS (TRAIN):")
    for r in sc.sort_values("t", ascending=False).head(20).itertuples():
        star = " *" if r.q < 0.10 and r.diff > 0 else ""
        print(f"    [{r.door[:2]}] {r.feature:42s} with: n={r.n_with:4d} avg={r.avg_with:+.2%} win={r.win_with:5.1%} | "
              f"diff {r.diff:+.2%} t={r.t:+.2f} q={r.q:.3f}{star}")
    print("  TOP HURTERS (TRAIN) - why it fails:")
    for r in sc.sort_values("t").head(12).itertuples():
        print(f"    [{r.door[:2]}] {r.feature:42s} with: n={r.n_with:4d} avg={r.avg_with:+.2%} | diff {r.diff:+.2%} t={r.t:+.2f} q={r.q:.3f}")

    print("\n==================== D. WINNERS vs LOSERS (TRAIN, all doors) ====================")
    wl = St.copy()
    win = wl["pnl"] > 0
    diffs = []
    for f in feature_cols(wl):
        col = pd.to_numeric(wl[f].map(lambda v: float(v) if isinstance(v, (bool, np.bool_, int, float)) else np.nan), errors="coerce")
        if col.notna().sum() < 100:
            continue
        diffs.append((f, col[win].mean(), col[~win].mean()))
    diffs.sort(key=lambda x: (x[1] - x[2]), reverse=True)
    print(f"  {int(win.sum())} winners vs {int((~win).sum())} losers. Share of trades showing each feature:")
    print("  MORE COMMON IN WINNERS:")
    for f, a, b in diffs[:12]:
        print(f"    {f:44s} winners {a:5.0%} | losers {b:5.0%}")
    print("  MORE COMMON IN LOSERS:")
    for f, a, b in diffs[-12:][::-1]:
        print(f"    {f:44s} winners {a:5.0%} | losers {b:5.0%}")

    print("\n==================== E. RULES BUILT ON TRAIN, SCORED ONCE ON TEST 2026 ====================")
    for door in doors:
        el = list(eligible_rows[eligible_rows["door"] == door]["feature"])
        feats = build_rule(St[St["door"] == door], el)
        print(f"\n  [{door}] rule = setup" + "".join(f" + {f}" for f in feats) + ("" if feats else " (no filter qualified)"))
        ctrl_feats = [f for f in feats if not f.startswith("C ")]
        tr = score("TRAIN 2025", apply_rule(St[St["door"] == door], feats), apply_rule(Ct[Ct["door"] == door], ctrl_feats))
        te = score("TEST 2026 (never seen)", apply_rule(Sx[Sx["door"] == door], feats), apply_rule(Cx[Cx["door"] == door], ctrl_feats))
        checks = [te["n"] >= 40, te["mu"] > 0, te["mua"] > 0, te["t_ctrl"] == te["t_ctrl"] and te["t_ctrl"] >= 2,
                  te["breadth"] == te["breadth"] and te["breadth"] > 0.5]
        names = ["n>=40", "avg>0", ">0 minus market", "beats control t>=2", "breadth>50%"]
        print(f"  >>> {'PASS' if all(checks) else 'FAIL'}  " + " · ".join(f"{a}: {b}" for a, b in zip(names, checks)))
        q = apply_rule(Sx[(Sx["door"] == door) & (Sx["symbol"] == "QQQ")], feats).sort_values("entry_t")
        if len(q):
            print("  QQQ 2026 trades under this rule (check on your chart):")
            for r in q.itertuples():
                print(f"     {r.entry_t:%a %b %d %H:%M} @ {r.entry:.2f}  {r.outcome:22s} {r.pnl:+.2%}")
    print(f"\nAll rows: data/feature_lab_rows.csv.gz · screen: data/feature_lab_screen.csv")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

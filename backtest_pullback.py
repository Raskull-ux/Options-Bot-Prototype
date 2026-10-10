# PULLBACK STUDY -- does tagging the hourly 50 give a pullback in the first 1-3 hours?
#
# Taz's point: the setup can work for an hour or two (a pullback off the 50) even if it
# doesn't hold for 36 hours. The earlier tests held up to 36h with a stop at yesterday's
# high, so they never asked this. This study asks ONLY the short-term question:
#
#   After the entry, how far does price pull back in the next 1, 2 and 3 hours,
#   how far does it squeeze against you, and does the pullback come before the squeeze?
#
# Same events as the feature lab (24h candles like Thinkorswim, 60 symbols):
#   SETUP    hourly 20 SMA crosses under the 50 -> first tag of the 50 from underneath on a
#            9:00-14:00 candle within 36h. Doors: E0 short at the tag, E1 tag candle closes
#            back under the 50, E2 pops over one more hour then closes back under.
#   CONTROL  the same tags of the 50 when the 20 is ABOVE the 50 (no bear cross).
#   RANDOM   every 9:00-14:00 candle, short at its open (R0) or at its close (R1).
#            Any short shows SOME pullback within 2 hours -- this is the bar to beat.
#
# Windows stay inside the regular session of the entry day (options close at 4 pm).
# For entries made DURING a candle (E0, R0) the low of that candle is NOT counted
# (it may have come before the entry); only its close is. Its high IS counted against you.
# If target and stop are both inside one candle, it counts as the STOP (conservative).
# Distances are in hourly ATR (so a QQQ move and a COIN move are comparable) and in %.
# Stock moves, not option P&L.
import sys
import traceback

import numpy as np
import pandas as pd

import config
import backtest_crosses as bx
import backtest_feature_lab as fl
from lib.alpaca_client import AlpacaClient
from lib.crosses import cross_below

ET = bx.ET
SYMBOLS = fl.SYMBOLS
ENTRY_HOURS = bx.ENTRY_HOURS
WINDOW_H = bx.WINDOW_H
SPLIT = bx.SPLIT
HORIZONS = (1, 2, 3)
BRACKET = 0.5            # ATR units for target and stop
PATH_STEPS = 5

PASS_RULES = """PRE-REGISTERED (fixed before any result is computed)
  ONE primary test: door E0 (short at the tag), bracket = take profit 0.5 hourly ATR below the entry,
  stop 0.5 ATR above it, within 2 hours (else out at the 2-hour close). Result in ATR units.
  The pullback edge is REAL only if ALL hold:
    1. average result > 0
    2. beats RANDOM shorts (R0, same entry mechanics) with t >= 2 over the full sample
    3. beats random in BOTH Feb-Dec 2025 AND 2026
    4. beats the no-cross CONTROL (diff > 0)
  Everything else (other doors, 1h/3h, QQQ+SPY only) is reported as description, not as a pass/fail."""


def window(g_t, e, intrabar, n):
    """Candle indices inside the horizon: regular session (<= 15:00 candle) of the entry day.
    intrabar: entry candle e is included first; then the next n candles."""
    day = g_t[e].date()
    out = [e] if intrabar else []
    j = e + 1
    while len(out) < n + (1 if intrabar else 0) and j < len(g_t):
        tj = g_t[j]
        if tj.date() != day or tj.hour > 15:
            break
        out.append(j)
        j += 1
    return out


def arrays(g):
    return {"h": g["h"].to_numpy(), "l": g["l"].to_numpy(), "c": g["c"].to_numpy(), "t": list(g["t"])}


def path_stats(A, e, px, intrabar, atr):
    """Pullback (MFE), squeeze (MAE), bracket results for each horizon + hour-by-hour path."""
    h, l, c, tt = A["h"], A["l"], A["c"], A["t"]
    r = {}
    if not (atr > 0) or px <= 0:
        return None
    for n in HORIZONS:
        w = window(tt, e, intrabar, n)
        if not w or (intrabar and len(w) < 2) or (not intrabar and len(w) < 1):
            r[f"mfe{n}"] = r[f"mae{n}"] = r[f"brk{n}"] = np.nan
            continue
        lows = [c[k] if (intrabar and k == e) else l[k] for k in w]
        highs = [h[k] for k in w]
        r[f"mfe{n}"] = max(0.0, (px - min(lows)) / px)
        r[f"mae{n}"] = max(0.0, (max(highs) - px) / px)
        tgt, stp = px - BRACKET * atr, px + BRACKET * atr
        res = None
        for k in w:
            if h[k] >= stp:
                res = -BRACKET
                break
            if (c[k] if (intrabar and k == e) else l[k]) <= tgt:
                res = BRACKET
                break
        r[f"brk{n}"] = res if res is not None else (px - c[w[-1]]) / atr
        r[f"brkpct{n}"] = r[f"brk{n}"] * atr / px
    # hour-by-hour path: close of each following regular-session candle (same day), in ATR
    w = window(tt, e, intrabar, PATH_STEPS)
    steps = [k for k in w if not (intrabar and k == e)]
    for i in range(1, PATH_STEPS + 1):
        r[f"p{i}"] = (px - c[steps[i - 1]]) / atr if i <= len(steps) else np.nan
    if intrabar:
        r["p0"] = (px - c[e]) / atr
    # pulled back >= 0.5 ATR within 2h, then squeezed above the entry candle's high later (up to 36h)?
    w2 = window(tt, e, intrabar, 2)
    lows2 = [c[k] if (intrabar and k == e) else l[k] for k in w2]
    r["pulled"] = bool(w2) and (px - min(lows2)) / atr >= BRACKET
    end_t = tt[e] + pd.Timedelta(hours=WINDOW_H)
    r["squeezed_after"] = False
    k = (w2[-1] if w2 else e) + 1
    while k < len(tt) and tt[k] <= end_t:
        if h[k] > h[e]:
            r["squeezed_after"] = True
            break
        k += 1
    r["atr_pct"] = atr / px
    return r


def doors(g, k):
    """(door, entry candle, price, intrabar, ATR candle) -- same doors as the feature lab."""
    for door, e, px, intrabar, dec, _ in fl.doors_for_tag(g, k):
        yield door, e, px, intrabar, (k - 1 if intrabar else e)


def symbol_rows(sym, g, test_from):
    rows = []
    A = arrays(g)
    t = g["t"]
    o, h, c = g["o"].to_numpy(), g["h"].to_numpy(), g["c"].to_numpy()
    s20, s50, s200, atr = (g[x].to_numpy() for x in ("sma20", "sma50", "sma200", "atr"))

    def add(kind, door, e, px, intrabar, ai, extra=None):
        if ai < 30:
            return
        st = path_stats(A, e, px, intrabar, atr[ai])
        if st is None:
            return
        rows.append({"symbol": sym, "kind": kind, "door": door, "entry_t": A["t"][e], "trade_day": A["t"][e].date(),
                     "entry": px, **(extra or {}), **st})

    used = set()
    xs = np.flatnonzero(cross_below(g["sma20"], g["sma50"]).to_numpy() & g["sma200"].notna().to_numpy())
    for i in xs:
        if t.iloc[i] < test_from:
            continue
        k = bx.first_tag_from_below(g, i + 1, t.iloc[i] + pd.Timedelta(hours=WINDOW_H))
        if k is None or k in used:
            continue
        used.add(k)
        for door, e, px, intrabar, ai in doors(g, k):
            add("setup", door, e, px, intrabar, ai, {"cross_t": t.iloc[i]})
    seen = set()
    for k in range(1, len(g)):
        tk = t.iloc[k]
        if tk < test_from or tk.hour not in ENTRY_HOURS or np.isnan(s200[k]) or not s20[k] > s50[k]:
            continue
        if tk.date() in seen or not (h[k] >= s50[k] and (o[k] < s50[k] or c[k - 1] < s50[k - 1])):
            continue
        seen.add(tk.date())
        for door, e, px, intrabar, ai in doors(g, k):
            add("control", door, e, px, intrabar, ai)
    for k in range(1, len(g)):
        tk = A["t"][k]
        if tk < test_from or tk.hour not in ENTRY_HOURS:
            continue
        add("random", "R0 short at a candle's open", k, o[k], True, k - 1)
        add("random", "R1 short at a candle's close", k, c[k], False, k)
    return rows


def pct(x):
    return "  n/a " if x != x else f"{x:+.2%}"


def line(label, s, col):
    n, mu, w, t, nd = bx.cstat(s[col], s["trade_day"])
    if mu != mu:
        return f"  {label:34s} n={n}"
    return f"  {label:34s} n={n:5d} ({nd:3d} days)  avg={mu:+.3f}  win={w:5.1%}  t={t:+.2f}"


def describe(title, df, base_open, base_close, ctrl):
    print(f"\n==================== {title} ====================")
    print("  Best pullback (median) / worst squeeze against you (median) within N hours, in % of price:")
    print(f"  {'':34s} {'1h pull':>8s} {'1h squeeze':>11s} {'2h pull':>8s} {'2h squeeze':>11s} {'3h pull':>8s} {'3h squeeze':>11s}")
    groups = [(f"SETUP {d}", df[(df.kind == "setup") & (df.door == d)]) for d in ("E0 tag", "E1 rejection close", "E2 pop then reject")]
    groups += [(f"CONTROL {d}", df[(df.kind == "control") & (df.door == d)]) for d in ("E0 tag", "E1 rejection close")]
    groups += [("RANDOM R0 (open)", base_open), ("RANDOM R1 (close)", base_close)]
    for lab, s in groups:
        vals = " ".join(f"{pct(s[f'mfe{n}'].median()):>8s} {pct(s[f'mae{n}'].median()):>11s}" for n in HORIZONS)
        print(f"  {lab:34s} {vals}   (n={len(s)})")
    print(f"\n  Bracket {BRACKET} ATR target vs {BRACKET} ATR stop (result in ATR units; +{BRACKET} = target first):")
    for n in HORIZONS:
        print(f"  -- within {n}h --")
        for lab, s in groups:
            print(line(lab, s, f"brk{n}"))
    print("\n  Setup minus random (same entry mechanics), bracket result, clustered by day:")
    for d, base in (("E0 tag", base_open), ("E1 rejection close", base_close), ("E2 pop then reject", base_close)):
        s = df[(df.kind == "setup") & (df.door == d)]
        cs = df[(df.kind == "control") & (df.door == d)]
        for n in HORIZONS:
            dd, tt = bx.cdiff(s[f"brk{n}"], s["trade_day"], base[f"brk{n}"], base["trade_day"])
            dc, tc = bx.cdiff(s[f"brk{n}"], s["trade_day"], cs[f"brk{n}"], cs["trade_day"]) if len(cs) else (np.nan, np.nan)
            print(f"  {d:20s} {n}h   vs random {dd:+.3f} ATR (t={tt:+.2f})   vs no-cross control {dc:+.3f} ATR (t={tc:+.2f})")
    print("\n  Hour by hour after entry: average move in your favor, in ATR (close of each candle; + = lower = good):")
    hdr = "  ".join(f"+{i}h" for i in range(1, PATH_STEPS + 1))
    print(f"  {'':34s} tag-candle close  {hdr}")
    for lab, s in groups:
        p0 = f"{s['p0'].mean():+.3f}" if "p0" in s and s["p0"].notna().any() else "  -   "
        steps = "  ".join(f"{s[f'p{i}'].mean():+.3f}" for i in range(1, PATH_STEPS + 1))
        print(f"  {lab:34s} {p0:>16s}  {steps}")
    print(f"\n  'Works for an hour, then fails': pulled back >= {BRACKET} ATR within 2h, then later (within 36h)")
    print("  traded above the entry candle's high:")
    for lab, s in groups:
        if len(s):
            p = s["pulled"].mean()
            f = s.loc[s["pulled"], "squeezed_after"].mean() if s["pulled"].any() else np.nan
            print(f"  {lab:34s} pulled back {p:5.1%}   of those, squeezed above the tag high later {f:5.1%}")


def main():
    print(PASS_RULES)
    client = AlpacaClient()
    bx.log(f"hourly bars (24h) for {len(SYMBOLS)} symbols since {bx.HOURLY_FROM}...")
    day = bx.fetch_bars(client, SYMBOLS, "1Hour", bx.HOURLY_FROM)
    day = day[(day["t"].dt.hour >= 4) & (day["t"].dt.hour < 20)]
    night = bx.fetch_bars(client, SYMBOLS, "1Hour", bx.HOURLY_FROM, feed="boats")
    night = night[(night["t"].dt.hour >= 20) | (night["t"].dt.hour < 4)]
    hb = pd.concat([day, night]).drop_duplicates(["symbol", "t"]).sort_values(["symbol", "t"]).reset_index(drop=True)
    bx.log(f"  -> {len(hb)} hourly candles")
    test_from = pd.Timestamp(bx.TEST_FROM, tz=ET)
    rows = []
    for s, g in hb.groupby("symbol"):
        rows += symbol_rows(s, fl.prep_hourly(g), test_from)
        bx.log(f"  {s}: {len(rows)} rows so far")
    df = pd.DataFrame(rows)
    df.to_csv(f"{config.DATA_DIR}/pullback_rows.csv.gz", index=False)
    R0 = df[df.door.str.startswith("R0")]
    R1 = df[df.door.str.startswith("R1")]
    ctrl = df[df.kind == "control"]

    describe("A. ALL 60 SYMBOLS (Feb 2025 - now)", df, R0, R1, ctrl)
    ix = df[df.symbol.isin(bx.INDEXES)]
    describe("B. QQQ + SPY ONLY", ix, ix[ix.door.str.startswith("R0")], ix[ix.door.str.startswith("R1")], ix[ix.kind == "control"])

    print("\n==================== C. QQQ SETUP TRADES (check them on your chart) ====================")
    print("  entry · door · best pullback 1h/2h/3h · worst squeeze 2h · bracket 2h (ATR) · then squeezed above tag high?")
    q = df[(df.symbol == "QQQ") & (df.kind == "setup")].sort_values("entry_t")
    for r in q.itertuples():
        print(f"  {r.entry_t:%a %b %d %Y %H:%M} @ {r.entry:8.2f}  {r.door:20s} pull {pct(r.mfe1)} / {pct(r.mfe2)} / {pct(r.mfe3)}"
              f"  squeeze {pct(r.mae2)}  bracket {r.brk2:+.2f}  {'squeezed later' if r.squeezed_after else ''}")

    # ---------------- verdict ----------------
    s = df[(df.kind == "setup") & (df.door == "E0 tag")]
    c = df[(df.kind == "control") & (df.door == "E0 tag")]
    n, mu, w, t, nd = bx.cstat(s["brk2"], s["trade_day"])
    dd, tt = bx.cdiff(s["brk2"], s["trade_day"], R0["brk2"], R0["trade_day"])
    dc, _ = bx.cdiff(s["brk2"], s["trade_day"], c["brk2"], c["trade_day"])
    per = []
    for lo, hi in (("2000-01-01", SPLIT), (SPLIT, "2100-01-01")):
        m = lambda x: x[(pd.to_datetime(x["trade_day"]) >= lo) & (pd.to_datetime(x["trade_day"]) < hi)]
        per.append(bx.cdiff(m(s)["brk2"], m(s)["trade_day"], m(R0)["brk2"], m(R0)["trade_day"])[0])
    checks = [mu > 0, tt == tt and tt >= 2, all(x == x and x > 0 for x in per), dc == dc and dc > 0]
    print(f"\n>>> PRIMARY (E0, {BRACKET} ATR bracket, 2h): setup avg {mu:+.3f} ATR, n={n} · vs random {dd:+.3f} (t={tt:+.2f})"
          f" · 2025 {per[0]:+.3f} / 2026 {per[1]:+.3f} · vs control {dc:+.3f}")
    print(f">>> {'PASS' if all(checks) else 'FAIL'}  [1 avg>0: {checks[0]} · 2 beats random t>=2: {checks[1]} · "
          f"3 both years: {checks[2]} · 4 beats control: {checks[3]}]")
    print(f"    Typical hourly ATR here is {s['atr_pct'].median():.2%} of price, so {BRACKET} ATR is about "
          f"{BRACKET * s['atr_pct'].median():.2%} of the stock's price.")
    print(f"\nAll rows: data/pullback_rows.csv.gz")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

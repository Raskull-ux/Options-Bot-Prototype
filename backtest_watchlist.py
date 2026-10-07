# Backtest the put scanner (daily_watchlist.put_setup) on historical daily bars.
#
# Every trading day from BT_START: rebuild that day's universe (core list + top
# 100 by trailing 20-day dollar volume, price >= $20, ETFs/funds excluded),
# run the exact live put_setup() and filters (score, targets, reward/risk,
# earnings inside 14 days excluded), rank exactly like the live post, and keep
# the top WL_TOP_N as "posted" ideas.
#
# Each idea is traded at the STOCK level with Taz's plan:
#   entry  = trigger touch (or the open, if price gaps below the trigger)
#   exits  = half at T1, stop to breakeven, rest at T2;
#            stop = daily CLOSE above invalid; max 5 sessions, then exit at close
#   result = R multiple, 1R = invalid - entry
# Conservative: on a day where the close is above invalid, the stop counts
# even if T1 printed earlier that day.
#
# Baseline: for every posted idea, BL_DRAWS random names from that day's same
# universe get the IDENTICAL geometry (same % distances to T1/T2/invalid from
# their own day low). If the scanner's picks don't beat random names with the
# same trade shape, the selection rules add nothing.
#
# Limits (disclosed): today's universe membership is used for all past dates
# (survivorship); results are stock moves, not option P&L (theta/IV not
# included); fills at the trigger price are assumed.
import math
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import daily_watchlist as dw
from lib import technicals as ta
from lib.alpaca_client import AlpacaClient
from lib.earnings_data import load_earnings

BT_FROM = "2024-03-01"
LOOKBACK_BARS = 300
BL_DRAWS = 5
HIT_KEYS = [("RSI ", "RSI 80+ tier"), ("stretched", "stretched"), ("RSI bear div", "RSI div"),
            ("MACD bear div", "MACD div"), ("close < 10 SMA", "close<10 SMA"), ("lower high", "lower high"),
            ("trend breaker", "trend breaker/10<20"), ("10<20", "trend breaker/10<20"), ("rejected at", "rejection"),
            ("failed gap-up", "failed gap-up"), ("red on", "red heavy vol"), ("gap below", "gap below"),
            ("TD sell 9", "TD sell 9")]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def simulate_short(after: pd.DataFrame, trigger: float, t1: float, t2: float, invalid: float,
                   max_sessions: int = 5) -> dict:
    """after: bars strictly after the signal day (open/high/low/close)."""
    w = after.iloc[:max_sessions]
    entry = None
    half_done = False
    realized = 0.0
    risk = None
    for k, (d, b) in enumerate(w.iterrows()):
        if entry is None:
            if b["low"] > trigger:
                continue
            entry = min(b["open"], trigger)
            risk = invalid - entry
            if risk <= 0:
                return {"outcome": "gap_through_invalid", "R": np.nan, "days": k + 1}
        # 1) stop: daily close above invalid (before T1 only; after T1 the stop is breakeven)
        if not half_done and b["close"] >= invalid:
            return {"outcome": "stopped", "R": (entry - b["close"]) / risk, "days": k + 1}
        # 2) targets
        if not half_done and b["low"] <= t1:
            half_done = True
            realized += 0.5 * (entry - t1) / risk
            if b["low"] <= t2:
                realized += 0.5 * (entry - t2) / risk
                return {"outcome": "t2", "R": realized, "days": k + 1}
            continue
        if half_done:
            if b["low"] <= t2:
                return {"outcome": "t2", "R": realized + 0.5 * (entry - t2) / risk, "days": k + 1}
            if b["high"] >= entry:
                return {"outcome": "t1_then_breakeven", "R": realized, "days": k + 1}
    if entry is None:
        return {"outcome": "no_trigger", "R": np.nan, "days": len(w)}
    last = w["close"].iloc[-1]
    rest = (0.5 if half_done else 1.0) * (entry - last) / risk
    return {"outcome": "t1_then_expired" if half_done else "expired", "R": realized + rest, "days": len(w)}


def stats(r: pd.Series) -> str:
    r = r.dropna()
    if len(r) < 2:
        return f"n={len(r)}"
    t = r.mean() / r.std(ddof=1) * math.sqrt(len(r)) if r.std(ddof=1) > 0 else float("nan")
    return f"n={len(r):4d}  avg R={r.mean():+.3f}  median R={r.median():+.2f}  win={(r > 0).mean():5.1%}  t={t:+.2f}"


def outcome_mix(df: pd.DataFrame) -> str:
    trig = df[df["outcome"] != "no_trigger"]
    if trig.empty:
        return "no triggers"
    v = trig["outcome"].value_counts(normalize=True)
    return (f"triggered {len(trig) / len(df):.0%} of ideas · T2 {v.get('t2', 0):.0%} · "
            f"T1 then BE/expired {v.get('t1_then_breakeven', 0) + v.get('t1_then_expired', 0):.0%} · "
            f"stopped {v.get('stopped', 0):.0%} · expired no target {v.get('expired', 0):.0%}")


def main() -> int:
    client = AlpacaClient()
    uni = sorted(dw.universe_symbols() | set(config.CORE_WATCHLIST))
    log(f"pulling ~4 years of daily bars for {len(uni)} symbols...")
    raw = client.get_daily_bars(uni, lookback_days=1000, feed=config.STOCK_BARS_FEED)
    rows = [{"symbol": s, "date": pd.Timestamp(b["t"][:10]), "open": b["o"], "high": b["h"], "low": b["l"],
             "close": b["c"], "volume": b["v"]} for s, bl in raw.items() for b in bl]
    frames = dw.load_frames(pd.DataFrame(rows))
    spy = frames["SPY"]["close"]
    days = spy.index[spy.index >= pd.Timestamp(BT_FROM)][:-config.WL_TRACK_SESSIONS]
    log(f"  -> {len(frames)} symbols, testing {len(days)} days {days[0].date()} to {days[-1].date()}")

    e = load_earnings()
    e["d"] = pd.to_datetime(e["earnings_date"], errors="coerce")
    earn_by_sym = {s: g["d"].dropna().sort_values().to_numpy() for s, g in e.groupby("symbol")}

    dv = pd.DataFrame({s: (f["close"] * f["volume"]).rolling(20).mean() for s, f in frames.items()})
    px = pd.DataFrame({s: f["close"] for s, f in frames.items()})
    rng = np.random.default_rng(7)
    ideas, base = [], []
    t0 = time.time()
    for di, d in enumerate(days):
        elig = dv.loc[d][(px.loc[d] >= config.WL_MIN_PRICE)].dropna()
        names = list(dict.fromkeys(config.CORE_WATCHLIST + list(elig.sort_values(ascending=False).index[:config.WL_UNIVERSE_TOP])))
        pool, cands = [], []
        for s in names:
            if s in ("SPY", "QQQ") or s not in frames:
                continue
            f = frames[s]
            pos = f.index.searchsorted(d, side="right")
            if pos < 260 or f.index[pos - 1] != d:
                continue
            ed = earn_by_sym.get(s)
            if ed is not None and ((ed > np.datetime64(d)) & (ed <= np.datetime64(d + pd.Timedelta(days=config.WL_EARNINGS_WARN_DAYS)))).any():
                continue
            pool.append(s)
            m = ta.metrics(f.iloc[max(0, pos - LOOKBACK_BARS):pos], spy[:d])
            if m is None:
                continue
            st = dw.put_setup(m)
            if st is None or st["score"] < config.WL_MIN_PUT_SCORE or not st["targets"]:
                continue
            if not (st["rr"] == st["rr"] and st["rr"] >= config.WL_MIN_RR):
                continue
            cands.append((st["score"], st["stage"] == "Rolling over", st["rr"], s, m, st, pos))
        cands.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        for rank, (sc, rolling, rr, s, m, st, pos) in enumerate(cands):
            t1 = st["targets"][0][0]
            t2 = st["targets"][1][0] if len(st["targets"]) > 1 else t1
            res = simulate_short(frames[s].iloc[pos:], st["trigger"], t1, t2, st["invalid"], config.WL_TRACK_SESSIONS)
            keys = sorted({lab for h in st["hits"] for pre, lab in HIT_KEYS if h.startswith(pre)})
            ideas.append({"date": d, "symbol": s, "posted": rank < config.WL_TOP_N, "score": sc, "door": st["door"],
                          "stage": st["stage"], "rr": rr, "trigger": st["trigger"], "t1": t1, "t2": t2,
                          "invalid": st["invalid"], "hits": "|".join(keys), **res})
            if rank < config.WL_TOP_N and len(pool) > 1:
                g1, g2, gi = t1 / st["trigger"], t2 / st["trigger"], st["invalid"] / st["trigger"]
                others = [x for x in pool if x != s]
                for rs in rng.choice(others, size=min(BL_DRAWS, len(others)), replace=False):
                    rf = frames[rs]
                    rpos = rf.index.searchsorted(d, side="right")
                    trig = rf["low"].iloc[rpos - 1]
                    base.append({"date": d, "symbol": rs, **simulate_short(rf.iloc[rpos:], trig, trig * g1, trig * g2, trig * gi,
                                                                          config.WL_TRACK_SESSIONS)})
        if (di + 1) % 50 == 0:
            log(f"  {di + 1}/{len(days)} days, {sum(1 for x in ideas if x['posted'])} posted ideas so far "
                f"({time.time() - t0:.0f}s)")

    I = pd.DataFrame(ideas)
    B = pd.DataFrame(base)
    I.to_csv(f"{config.DATA_DIR}/backtest_watchlist_ideas.csv", index=False)
    P = I[I["posted"]]
    print("\n================ PUT SCANNER BACKTEST (stock level, Taz's exit plan) ================")
    print(f"Days tested: {len(days)}  ·  posted ideas: {len(P)}  ·  all qualified ideas: {len(I)}")
    print(f"\nPOSTED ideas:      {outcome_mix(P)}")
    print(f"  R per triggered trade: {stats(P['R'])}")
    print(f"RANDOM baseline:   {outcome_mix(B)}")
    print(f"  R per triggered trade: {stats(B['R'])}")
    pr, br = P["R"].dropna(), B["R"].dropna()
    if len(pr) > 1 and len(br) > 1:
        diff = pr.mean() - br.mean()
        se = math.sqrt(pr.var(ddof=1) / len(pr) + br.var(ddof=1) / len(br))
        print(f"\n>>> Scanner minus random: {diff:+.3f} R per trade (t = {diff / se:+.2f}). "
              f"Edge needs t >= 2 here, and the scanner's own avg R above zero.")
    print("\nPosted ideas by year:")
    for y, g in P.groupby(P["date"].dt.year):
        print(f"  {y}: {stats(g['R'])}")
    print("By door:")
    for k, g in P.groupby("door"):
        print(f"  {k:12s} {stats(g['R'])}")
    print("By stage:")
    for k, g in P.groupby("stage"):
        print(f"  {k:28s} {stats(g['R'])}")
    print("By score:")
    for k, g in P.groupby("score"):
        print(f"  {k:2d}/12  {stats(g['R'])}")
    print("Which confirmations matter (all qualified ideas, avg R WITH vs WITHOUT):")
    Iq = I.dropna(subset=["R"])
    for lab in dict.fromkeys(l for _, l in HIT_KEYS):
        has = Iq["hits"].str.contains(lab, regex=False)
        if has.sum() >= 10 and (~has).sum() >= 10:
            print(f"  {lab:20s} with {Iq.loc[has, 'R'].mean():+.3f} (n={has.sum()})   without {Iq.loc[~has, 'R'].mean():+.3f} (n={(~has).sum()})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

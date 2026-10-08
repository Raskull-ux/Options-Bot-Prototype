# Which put rule actually works? Decided by data, not opinion.
#
# 1. For every trading day since BT_FROM and every name in that day's universe
#    (core list + top 100 by 20-day dollar volume, price >= $20, ETFs/funds out,
#    SPY/QQQ out, earnings within 14 days out), compute every put confirmation
#    (daily_watchlist.put_features) and trade the live geometry (trigger, targets,
#    invalid; reward/risk >= 0.3) with Taz's exit plan -> R multiple.
#    The SAME code the live scanner runs.
# 2. Score every rule in daily_watchlist.PUT_RULES: top 6 per day by the live
#    ranking, and a stock can't be picked again for 5 sessions (no double counting).
# 3. Baseline = every eligible stock-day (each with its own levels).
# 4. The winner is chosen on TRAIN (2024-03..2025-12) ONLY, by the code,
#    then scored once on TEST (2026), which the choice never saw.
import math
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import daily_watchlist as dw
from backtest_watchlist import simulate_short
from lib import technicals as ta
from lib.alpaca_client import AlpacaClient
from lib.earnings_data import load_earnings

BT_FROM = "2024-03-01"
TEST_FROM = "2026-01-01"
LOOKBACK_BARS = 300
COOLDOWN = 5
MIN_TRAIN_N = 150
FEATS = ["rsi80", "stretched", "rsi_div", "macd_div", "below10", "lower_high", "trend", "rej", "rej_key",
         "failed_gap", "red_vol", "gap_below", "td9", "td_buy9", "at_top", "near_high", "score12", "core4", "door"]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def stats(r: pd.Series) -> tuple[int, float, float, float]:
    r = r.dropna()
    if len(r) < 2:
        return len(r), np.nan, np.nan, np.nan
    sd = r.std(ddof=1)
    return len(r), r.mean(), (r > 0).mean(), r.mean() / sd * math.sqrt(len(r)) if sd > 0 else np.nan


def diff_t(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d / se if se > 0 else np.nan


def pick(rows: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Live selection: rule -> rank (score, rolling, rr) -> top N per day -> 5-session cooldown."""
    fn = dw.PUT_RULES[rule]
    rows = rows.copy()
    rows["door"] = rows["door"].astype(object).where(rows["door"].notna(), None)  # NaN must never count as a door
    mask = rows.apply(lambda r: bool(fn(r)), axis=1)
    sel = rows[mask].copy()
    if sel.empty:
        return sel
    sel["score"] = sel["score12"] if rule == "v1_live" else sel["core4"]
    sel = sel.sort_values(["date", "score", "rolling", "rr"], ascending=[True, False, False, False])
    out, last_pick = [], {}
    for d, g in sel.groupby("date", sort=True):
        taken = 0
        for r in g.itertuples():
            if taken == config.WL_TOP_N:
                break
            lp = last_pick.get(r.symbol)
            if lp is not None and r.day_idx - lp < COOLDOWN:
                continue
            last_pick[r.symbol] = r.day_idx
            out.append(r.Index)
            taken += 1
    return rows.loc[out]


def main() -> int:
    client = AlpacaClient()
    uni = sorted(dw.universe_symbols() | set(config.CORE_WATCHLIST))
    log(f"pulling daily bars for {len(uni)} symbols...")
    raw = client.get_daily_bars(uni, lookback_days=1000, feed=config.STOCK_BARS_FEED)
    rows = [{"symbol": s, "date": pd.Timestamp(b["t"][:10]), "open": b["o"], "high": b["h"], "low": b["l"],
             "close": b["c"], "volume": b["v"]} for s, bl in raw.items() for b in bl]
    frames = dw.load_frames(pd.DataFrame(rows))
    spy = frames["SPY"]["close"]
    days = spy.index[spy.index >= pd.Timestamp(BT_FROM)][:-10]
    log(f"  -> {len(frames)} symbols; {len(days)} days {days[0].date()} to {days[-1].date()}")

    e = load_earnings()
    e["d"] = pd.to_datetime(e["earnings_date"], errors="coerce")
    earn = {s: g["d"].dropna().to_numpy() for s, g in e.groupby("symbol")}
    dv = pd.DataFrame({s: (f["close"] * f["volume"]).rolling(20).mean() for s, f in frames.items()})
    px = pd.DataFrame({s: f["close"] for s, f in frames.items()})

    out = []
    t0 = time.time()
    for di, d in enumerate(days):
        elig = dv.loc[d][px.loc[d] >= config.WL_MIN_PRICE].dropna()
        names = list(dict.fromkeys(config.CORE_WATCHLIST + list(elig.sort_values(ascending=False).index[:config.WL_UNIVERSE_TOP])))
        for s in names:
            if s in ("SPY", "QQQ") or s not in frames:
                continue
            f = frames[s]
            pos = f.index.searchsorted(d, side="right")
            if pos < 260 or f.index[pos - 1] != d:
                continue
            ed = earn.get(s)
            if ed is not None and ((ed > np.datetime64(d)) & (ed <= np.datetime64(d + pd.Timedelta(days=config.WL_EARNINGS_WARN_DAYS)))).any():
                continue
            m = ta.metrics(f.iloc[pos - LOOKBACK_BARS:pos], spy[:d])
            if m is None:
                continue
            feats, _, rej = dw.put_features(m)
            g = dw.put_geometry(m, rej)
            if not g["targets"] or not (g["rr"] == g["rr"] and g["rr"] >= config.WL_MIN_RR):
                continue
            t1 = g["targets"][0][0]
            t2 = g["targets"][1][0] if len(g["targets"]) > 1 else t1
            after = f.iloc[pos:]
            r5 = simulate_short(after, g["trigger"], t1, t2, g["invalid"], 5)
            r10 = simulate_short(after, g["trigger"], t1, t2, g["invalid"], 10)
            out.append({"date": d, "day_idx": di, "symbol": s, **{k: feats[k] for k in FEATS},
                        "rolling": feats["below10"] or feats["lower_high"] or feats["failed_gap"] or (feats["rej"] and m["red"]),
                        "rr": g["rr"], "R": r5["R"], "outcome": r5["outcome"], "R10": r10["R"]})
        if (di + 1) % 50 == 0:
            log(f"  {di + 1}/{len(days)} days, {len(out)} eligible stock-days ({time.time() - t0:.0f}s)")

    R = pd.DataFrame(out)
    R.to_csv(f"{config.DATA_DIR}/research_stockdays.csv.gz", index=False, compression="gzip")
    train, test = R[R["date"] < TEST_FROM], R[R["date"] >= TEST_FROM]

    print("\n==================== PUT RULE RESEARCH ====================")
    print(f"Eligible stock-days: {len(R)} (train {len(train)}, test {len(test)}). R = stock-level, Taz's exits, 5 sessions.")
    for lab, part in (("TRAIN 2024-03..2025-12", train), ("TEST 2026", test)):
        n, mu, w, t = stats(part["R"])
        print(f"Baseline (all eligible stock-days), {lab}: n={n} avg R={mu:+.3f} win={w:.1%}")

    print("\nTRAIN results (the only numbers used to choose):")
    res = {}
    for rule in dw.PUT_RULES:
        p = pick(train, rule)
        n, mu, w, t = stats(p["R"])
        d, dt = diff_t(p["R"], train["R"])
        res[rule] = (n, mu, d, dt)
        print(f"  {rule:18s} n={n:4d}  avg R={mu:+.3f}  win={w:5.1%}  vs baseline {d:+.3f} (t={dt:+.2f})")
    ok = {r: v for r, v in res.items() if v[0] >= MIN_TRAIN_N and v[2] == v[2]}
    if not ok:
        print(f"\nNo rule had {MIN_TRAIN_N}+ train trades; nothing chosen.")
        return 0
    winner = max(ok, key=lambda r: ok[r][2])
    print(f"\n>>> CHOSEN ON TRAIN ONLY: {winner}  (beat baseline by {ok[winner][2]:+.3f} R, t={ok[winner][3]:+.2f})")

    print("\nTEST 2026 — out-of-sample (chosen rule first; others shown for transparency, NOT for re-picking):")
    for rule in [winner] + [r for r in dw.PUT_RULES if r != winner]:
        p = pick(test, rule)
        n, mu, w, t = stats(p["R"])
        d, dt = diff_t(p["R"], test["R"])
        n10, mu10, _, _ = stats(p["R10"])
        tag = "  <-- CHOSEN" if rule == winner else ""
        print(f"  {rule:18s} n={n:4d}  avg R={mu:+.3f}  win={w:5.1%}  vs baseline {d:+.3f} (t={dt:+.2f})  | 10-session hold avg R={mu10:+.3f}{tag}")
    pw = pick(test, winner)
    n, mu, w, t = stats(pw["R"])
    d, dt = diff_t(pw["R"], test["R"])
    verdict = ("PASS — beats baseline out-of-sample and is positive on its own" if dt >= 2 and mu > 0 else
               "EDGE vs baseline, but still negative on its own (shorts in a bull year)" if dt >= 2 else
               "NOT CONFIRMED out-of-sample")
    print(f"\nVERDICT for {winner} on 2026: {verdict}")
    print("\nWhat the chosen rule's 2026 trades looked like:")
    print("  " + pw["outcome"].value_counts(normalize=True).round(2).to_string().replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

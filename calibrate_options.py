"""
Tazmanian Trader — Option Estimate Calibration
==============================================
The setup backtest and event study price options with Black-Scholes, using
20-day realized volatility x 1.15 as the implied-volatility stand-in. This
script checks that estimate against REAL option prices already paid for:
results/bar_cache holds hourly OPRA bars (Databento) for every contract Taz
traded. No new Databento spend.

For each real contract-hour it pulls the underlying's price for the same hour
(Alpaca, free) and compares:
  1. LEVEL   — real option price vs the Black-Scholes estimate
               (ratio > 1 means real options cost more than estimated: IV proxy too low)
  2. RETURNS — real % change over the next hour, and over the rest of the day,
               vs the estimated % change from the same stock move
               (this is what the backtests actually use)

Env: ALPACA keys.  Out: results/calibration/report.md, pairs.csv
"""
from __future__ import annotations
import os, glob, gzip, pickle, math
import numpy as np
import pandas as pd
from scipy.special import ndtr
import setup_backtest as sb

OUT = "results/calibration"
MAX_DTE, MAX_MONEY = 30, 0.10    # contracts up to 30 days out, strikes within 10% of spot
MIN_VOL = 5


def bs(S, K, T, sig, put):
    T = np.maximum(T, 1e-6); v = sig * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * v * v) / v; d2 = d1 - v
    call = S * ndtr(d1) - K * ndtr(d2)
    return np.where(put, call - S + K, call)


def load_options():
    frames = []
    for f in sorted(glob.glob("results/bar_cache/*.pkl.gz")):
        try:
            d = pickle.load(gzip.open(f))
            if isinstance(d, pd.DataFrame) and len(d):
                frames.append(d[["close", "volume", "symbol"]].reset_index())
        except Exception:
            pass
    d = pd.concat(frames, ignore_index=True)
    d = d.groupby(["symbol", "ts_event"]).agg(close=("close", "median"),
                                              volume=("volume", "sum")).reset_index()
    s = d["symbol"].astype(str)
    d["root"] = s.str[:6].str.strip()
    d["expiry"] = pd.to_datetime(s.str[6:12], format="%y%m%d", errors="coerce")
    d["put"] = s.str[12] == "P"
    d["K"] = pd.to_numeric(s.str[13:21], errors="coerce") / 1000
    d = d.dropna(subset=["expiry", "K"])
    d = d[d["volume"] >= MIN_VOL]
    d["ts"] = pd.to_datetime(d["ts_event"], utc=True)
    print(f"Option bars: {len(d):,} contract-hours, {d.symbol.nunique():,} contracts, "
          f"{d.root.nunique()} underlyings")
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    o = load_options()
    start = (o.ts.min() - pd.Timedelta(days=45)).date().isoformat()
    end = (o.ts.max() + pd.Timedelta(days=1)).date().isoformat()
    roots = sorted(o.root.unique())
    hrs, dys = [], []
    for i in range(0, len(roots), 40):
        ch = roots[i:i + 40]
        try:
            hrs.append(sb.fetch(ch, "1Hour", start, end)); dys.append(sb.fetch(ch, "1Day", start, end))
        except Exception as ex:
            print("fetch", ch[:3], ex)
        print(f"  underlyings {min(i + 40, len(roots))}/{len(roots)}")
    H = pd.concat(hrs); D = pd.concat(dys)
    H["ts"] = H["t"].dt.tz_convert("UTC")
    D = D.sort_values(["symbol", "t"])
    D["day"] = D["t"].dt.date
    D["rv"] = D.groupby("symbol")["c"].transform(
        lambda x: np.log(x).diff().rolling(20).std() * math.sqrt(252) * 1.15).clip(lower=0.15)
    D["rv"] = D.groupby("symbol")["rv"].shift()

    m = o.merge(H[["symbol", "ts", "c"]].rename(columns={"symbol": "root", "c": "S"}),
                on=["root", "ts"], how="inner")
    m["day"] = m["ts"].dt.tz_convert(sb.NY).dt.date
    m = m.merge(D[["symbol", "day", "rv"]].rename(columns={"symbol": "root"}), on=["root", "day"], how="left")
    m = m.dropna(subset=["rv", "S"])
    ny = m["ts"].dt.tz_convert(sb.NY)
    end_of_bar = ny + pd.Timedelta(hours=1)
    mins_left = (16 * 60 - (end_of_bar.dt.hour * 60 + end_of_bar.dt.minute)).clip(lower=5)
    bdays = np.busday_count(m["day"].values.astype("datetime64[D]"),
                            m["expiry"].dt.date.values.astype("datetime64[D]"))
    m["dte"] = bdays
    m = m[(m.dte >= 0) & (m.dte <= MAX_DTE) & ((m.K / m.S - 1).abs() <= MAX_MONEY)]
    m["T"] = (mins_left.loc[m.index] + 390 * m["dte"]) / (390 * 252)
    m["est"] = bs(m.S.values, m.K.values, m["T"].values, m.rv.values, m.put.values)
    m = m[m.est > 0.05]
    m["ratio"] = m["close"] / m["est"]

    # returns: next hour, and to the last bar of the same day
    m = m.sort_values(["symbol", "ts"])
    g = m.groupby(["symbol", "day"])
    rows = []
    for (sym, day), x in g:
        if len(x) < 2:
            continue
        x = x.reset_index(drop=True)
        for a, b_, kind in [(k, k + 1, "next_hour") for k in range(len(x) - 1)] + [(0, len(x) - 1, "rest_of_day")]:
            p0, p1 = x.loc[a], x.loc[b_]
            est1 = bs(np.array([p1.S]), p0.K, np.array([p1["T"]]), p0.rv, np.array([p0.put]))[0]
            rows.append(dict(symbol=sym, day=day, kind=kind, dte=p0.dte, put=p0.put,
                             real=p1.close / p0.close - 1, est=est1 / p0.est - 1,
                             stock=p1.S / p0.S - 1))
    R = pd.DataFrame(rows)
    R = R[(R.real.abs() < 5) & (R.est.abs() < 5)]
    R.to_csv(f"{OUT}/pairs.csv", index=False)

    def bucket(d):
        return pd.cut(d, [-1, 0, 1, 3, 7, 30], labels=["0DTE", "1", "2-3", "4-7", "8-30"])

    L = ["# Option Estimate Calibration",
         f"\n{len(m):,} real contract-hours from Taz's Databento cache "
         f"({m.symbol.nunique():,} contracts), strikes within {MAX_MONEY:.0%} of spot, up to {MAX_DTE} days out.\n",
         "## 1. Price level: real option price ÷ estimate\n",
         "Above 1.00 = real options cost more than the estimate (IV proxy too low).\n",
         "| Days to expiry | Puts: median ratio | Calls: median ratio | n |", "|---|---|---|---|"]
    m["b"] = bucket(m.dte)
    for lab in m.b.cat.categories:
        x = m[m.b == lab]
        L.append(f"| {lab} | {x[x.put].ratio.median():.2f} | {x[~x.put].ratio.median():.2f} | {len(x):,} |")
    L += ["\n## 2. Returns: real vs estimated % change (what the backtests use)\n",
          "| Window | Days to expiry | n | Real avg | Estimated avg | Median error (real − est) | Correlation |",
          "|---|---|---|---|---|---|---|"]
    R["b"] = bucket(R.dte)
    for kind in ("next_hour", "rest_of_day"):
        for lab in R.b.cat.categories:
            x = R[(R.kind == kind) & (R.b == lab) & R.put]
            if len(x) < 30:
                continue
            L.append(f"| {kind.replace('_', ' ')} (puts) | {lab} | {len(x):,} | {x.real.mean()*100:+.1f}% | "
                     f"{x.est.mean()*100:+.1f}% | {(x.real - x.est).median()*100:+.1f} pts | "
                     f"{x.real.corr(x.est):.2f} |")
    x = R[R.put & (R.kind == "rest_of_day")]
    if len(x):
        up = x[x.est > 0.3]
        L += ["\n## 3. When the estimate says a put gained 30%+, what did the real put do?\n",
              f"- Cases: {len(up):,}",
              f"- Real put also gained 30%+: {(up.real >= 0.3).mean():.0%}" if len(up) else "- none",
              f"- Median real gain in those cases: {up.real.median()*100:+.0f}%" if len(up) else ""]
    L.append("\nHow to read: if the median error is near 0 and correlation is high, the backtest's "
             "option numbers are trustworthy. A consistent negative error means the estimate is too "
             "optimistic and the backtests must be scaled down by about that much.")
    open(f"{OUT}/report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""
Backtest the SPY volatility sleeve (Johnson 2017) on REAL SPY option prices.

Two tests, same contract rule: an at-the-money SPY straddle (call + put, same
strike nearest the entry-day open), first expiration at least 30 calendar
days out. Prices = each leg's daily VWAP from Alpaca; both legs must trade
on both days or the trade is excluded and counted.

  A. Next-day test (Johnson's own horizon): for EVERY trading day, label the
     VIX curve at the close, buy the straddle next session, sell the session
     after. Inverted days vs contango days side by side. The contango days are
     the control: the research says they should lose (variance risk premium).
  B. Episode test (how the live sleeve trades): enter the session after the
     curve first inverts, hold until it returns to contango (exit the session
     after) or 21 trading days, one position at a time.

Expect a SMALL inverted sample in 2024-2026; the output says so plainly.
"""
import math
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from lib.alpaca_client import AlpacaClient, AlpacaError, TRADING, DATA


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


class StraddlePricer:
    def __init__(self, client: AlpacaClient):
        self.c, self._last, self.today = client, 0.0, date.today()

    def _get(self, url, params):
        wait = config.BT_REQUEST_INTERVAL - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        return self.c._get(url, params)

    def straddle(self, spot: float, exp_min: date) -> tuple[dict, dict] | None:
        statuses = ["inactive", "active"] if exp_min < self.today else ["active", "inactive"]
        for status in statuses:
            body = self._get(f"{TRADING}/v2/options/contracts", {
                "underlying_symbols": "SPY", "status": status,
                "expiration_date_gte": exp_min.isoformat(),
                "expiration_date_lte": (exp_min + timedelta(days=10)).isoformat(),
                "strike_price_gte": round(spot * 0.97, 2), "strike_price_lte": round(spot * 1.03, 2),
                "limit": 1000})
            cs = body.get("option_contracts") or []
            if not cs:
                continue
            exp = min(x["expiration_date"] for x in cs)
            calls = {float(x["strike_price"]): x for x in cs if x["expiration_date"] == exp and x["type"] == "call"}
            puts = {float(x["strike_price"]): x for x in cs if x["expiration_date"] == exp and x["type"] == "put"}
            both = sorted(set(calls) & set(puts), key=lambda k: abs(k - spot))
            if both:
                return calls[both[0]], puts[both[0]]
        return None

    def bars(self, syms: list[str], start: date, end: date) -> dict[str, dict[str, dict]]:
        body = self._get(f"{DATA}/v1beta1/options/bars", {
            "symbols": ",".join(syms), "timeframe": "1Day", "start": start.isoformat(),
            "end": (end + timedelta(days=1)).isoformat(), "limit": 1000})
        return {s: {b["t"][:10]: b for b in bl} for s, bl in (body.get("bars") or {}).items()}


def vw(bar: dict) -> float:
    return float(bar.get("vw") or bar["c"])


def straddle_return(p: StraddlePricer, spot: float, entry: pd.Timestamp, exit_: pd.Timestamp) -> dict:
    try:
        legs = p.straddle(spot, entry.date() + timedelta(days=30))
    except AlpacaError as e:
        return {"status": f"contract_error: {str(e)[:60]}"}
    if not legs:
        return {"status": "no_contract"}
    c, pt = legs
    try:
        b = p.bars([c["symbol"], pt["symbol"]], entry.date(), exit_.date())
    except AlpacaError as e:
        return {"status": f"bars_error: {str(e)[:60]}"}
    e_s, x_s = entry.strftime("%Y-%m-%d"), exit_.strftime("%Y-%m-%d")
    try:
        cin, pin = vw(b[c["symbol"]][e_s]), vw(b[pt["symbol"]][e_s])
        cout, pout = vw(b[c["symbol"]][x_s]), vw(b[pt["symbol"]][x_s])
    except KeyError:
        return {"status": "leg_missing_trade", "strike": float(c["strike_price"])}
    cost_in, val_out = cin + pin, cout + pout
    out = {"status": "priced", "strike": float(c["strike_price"]), "expiration": c["expiration_date"],
           "entry_premium": cost_in, "exit_value": val_out}
    for k in config.BT_COST_LEVELS:
        out[f"ret_cost_{int(k * 100)}"] = (val_out * (1 - k / 2)) / (cost_in * (1 + k / 2)) - 1
    return out


def episodes(labels: pd.Series, max_hold: int) -> list[tuple[int, int]]:
    """labels: bool Series (True = inverted at that close) on trading days.
    Returns (entry_pos, exit_pos): enter the session after the first inverted
    close; exit the session after the first contango close, or after max_hold
    sessions; no overlapping positions."""
    out, pos, n = [], 0, len(labels)
    v = labels.to_numpy()
    while pos < n - 1:
        if v[pos] and (pos == 0 or not v[pos - 1]):
            entry = pos + 1
            exit_ = None
            for j in range(entry, min(entry + max_hold, n)):
                if not v[j]:
                    exit_ = j + 1
                    break
            if exit_ is None:
                exit_ = entry + max_hold - 1
            exit_ = min(exit_, n - 1)
            if exit_ > entry:
                out.append((entry, exit_))
            pos = exit_
        else:
            pos += 1
    return out


def describe(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) < 2:
        return f"n={len(x)}" + (f"  return={x.iloc[0]:+.1%}" if len(x) == 1 else "")
    lr = np.log1p(x.clip(lower=-0.9999))
    t = lr.mean() / lr.std(ddof=1) * math.sqrt(len(lr)) if lr.std(ddof=1) > 0 else float("nan")
    return (f"n={len(x):4d}  win={(x > 0).mean():5.1%}  mean={x.mean():+7.2%}  median={x.median():+7.2%}  t(log)={t:+.2f}")


def main() -> int:
    vix = pd.read_csv(config.VIX_TERM_FILE, parse_dates=["date"]).set_index("date").sort_index()
    client = AlpacaClient()
    raw = client.get_daily_bars(["SPY"], lookback_days=config.BT_STOCK_LOOKBACK_DAYS, feed=config.STOCK_BARS_FEED)
    spy = pd.DataFrame(raw["SPY"])
    spy.index = pd.to_datetime(spy["t"].str[:10])
    spy = spy[spy.index >= pd.Timestamp(config.BT_START) - pd.Timedelta(days=10)]
    df = spy[["o"]].join(vix[["vix", "vix3m"]], how="inner")
    df["inverted"] = df["vix"] > df["vix3m"]
    df = df[df.index >= pd.Timestamp(config.BT_START)]
    idx = df.index
    log(f"{len(idx)} trading days {idx[0].date()} to {idx[-1].date()}; inverted closes: {int(df['inverted'].sum())}")

    p = StraddlePricer(client)

    # A. next-day test on every day
    rows = []
    for i in range(len(idx) - 2):
        entry, exit_ = idx[i + 1], idx[i + 2]
        r = straddle_return(p, float(df["o"].iloc[i + 1]), entry, exit_)
        rows.append({"test": "next_day", "signal_date": idx[i], "inverted": bool(df["inverted"].iloc[i]),
                     "vix": df["vix"].iloc[i], "vix3m": df["vix3m"].iloc[i],
                     "entry_date": entry, "exit_date": exit_, **r})
        if (i + 1) % 100 == 0:
            log(f"  next-day test: {i + 1}/{len(idx) - 2} days")

    # B. episodes
    for e, x in episodes(df["inverted"], config.IDXVOL_MAX_HOLD_DAYS):
        r = straddle_return(p, float(df["o"].iloc[e]), idx[e], idx[x])
        rows.append({"test": "episode", "signal_date": idx[e - 1], "inverted": True,
                     "vix": df["vix"].iloc[e - 1], "vix3m": df["vix3m"].iloc[e - 1],
                     "entry_date": idx[e], "exit_date": idx[x], "hold_days": x - e, **r})

    out = pd.DataFrame(rows)
    out.to_csv(f"{config.DATA_DIR}/backtest_index_vol_trades.csv", index=False)

    print("\n================ SPY STRADDLE BACKTEST ================")
    nd = out[out["test"] == "next_day"]
    print(f"Next-day test: {len(nd)} days, {int((nd['status'] == 'priced').sum())} priced "
          f"(excluded: {nd[nd['status'] != 'priced']['status'].str.split(':').str[0].value_counts().to_dict()})")
    ndp = nd[nd["status"] == "priced"]
    for lab, g in [("INVERTED curve (signal days)", ndp[ndp["inverted"]]), ("contango (control)", ndp[~ndp["inverted"]])]:
        print(f"  {lab}:")
        for k in (0, 10):
            print(f"     {k:2d}% cost: {describe(g[f'ret_cost_{k}'])}")
    ep = out[(out["test"] == "episode") & (out["status"] == "priced")]
    print(f"\nEpisode test (how the live sleeve trades): {int((out['test'] == 'episode').sum())} episodes, {len(ep)} priced")
    for k in (0, 10):
        print(f"   {k:2d}% cost: {describe(ep[f'ret_cost_{k}'])}")
    for r in ep.itertuples():
        print(f"     {r.entry_date.date()} -> {r.exit_date.date()} ({r.hold_days}d)  VIX {r.vix:.1f} vs VIX3M {r.vix3m:.1f}"
              f"  straddle {r.entry_premium:.2f} -> {r.exit_value:.2f}  = {r.ret_cost_0:+.1%} gross, {r.ret_cost_10:+.1%} at 10% cost")
    print("\nBar for an edge: t(log) >= 3. With few inverted days in 2024-2026, a pass here is unlikely on sample size alone;")
    print("what to look for is (1) inverted days beating the contango control, and (2) the episodes' direction.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

# Pre-market post (~9:15 AM ET in summer, 8:15 AM ET in winter; GitHub cron runs in UTC).
#
# For the core watchlist plus yesterday's ranked picks: the pre-market price,
# where it sits against LAST WEEK'S CLOSE (touch from above = puts, from below =
# calls), the nearest open gaps from yesterday's close, and whether yesterday's
# trigger is already through. Pre-market prices are IEX trades (thin before the
# open); a name with no trade yet today shows yesterday's close.
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from daily_readout import post
from daily_watchlist import load_frames, gap_txt, fmt
from lib import technicals as ta
from lib.alpaca_client import AlpacaClient, DATA

ET = ZoneInfo("America/New_York")


def premarket_prices(client: AlpacaClient, symbols: list[str]) -> dict[str, tuple[float, bool]]:
    """{symbol: (price, traded_today)} from Alpaca IEX snapshots."""
    body = client._get(f"{DATA}/v2/stocks/snapshots", {"symbols": ",".join(symbols), "feed": "iex"})
    snaps = body.get("snapshots", body)
    today = datetime.now(ET).date()
    out = {}
    for s, snap in (snaps or {}).items():
        t = (snap or {}).get("latestTrade") or {}
        if t.get("p"):
            ts = pd.Timestamp(t["t"]).tz_convert(ET).date()
            out[s] = (float(t["p"]), ts == today)
    return out


def line(sym: str, m: dict, px: float | None, live: bool, idea: dict | None) -> str:
    c = m["close"]
    p = px if px is not None else c
    head = f"**{sym} {fmt(p)}**" + (f" ({p / c - 1:+.1%} pre)" if live else " (no pre-market trade yet)")
    parts = [head]
    lw = m.get("lw_close", np.nan)
    if not np.isnan(lw):
        if p > lw:
            parts.append(f"{(p / lw - 1):+.1%} above last wk close {fmt(lw)} → touch = **puts**")
        else:
            parts.append(f"{(p / lw - 1):+.1%} below last wk close {fmt(lw)} → touch = **calls**")
    gap_dir = "gapping UP over yesterday's body" if p > max(m["prev_body_hi"], c) and live else (
        "gapping DOWN under yesterday's body" if p < m["prev_body_lo"] and live else "")
    if gap_dir:
        parts.append(gap_dir)
    m2 = dict(m, close=p)  # gaps relative to the pre-market price
    parts.append(gap_txt(m2, 1))
    if idea is not None:
        trig, side = float(idea["trigger"]), idea["side"]
        through = p >= trig if side == "bull" else p <= trig
        parts.append(f"yesterday: {'above' if side == 'bull' else 'below'} {fmt(trig)} → {'calls' if side == 'bull' else 'puts'}"
                     + (" — **already through pre-market**" if through and live else " — not yet"))
    return " · ".join(parts)


def build(client) -> list[str]:
    bars = pd.read_csv(config.BARS_FILE, parse_dates=["date"])
    try:
        ideas = pd.read_csv(config.WL_IDEAS_FILE)
        last = ideas[ideas["date"] == ideas["date"].max()] if not ideas.empty else ideas
    except FileNotFoundError:
        last = pd.DataFrame(columns=["symbol", "side", "trigger"])
    symbols = list(dict.fromkeys(config.CORE_WATCHLIST + list(last["symbol"])))
    missing = [s for s in symbols if s not in set(bars["symbol"])]
    if missing:
        raw = client.get_daily_bars(missing, lookback_days=300, feed=config.STOCK_BARS_FEED)
        bars = pd.concat([bars, pd.DataFrame([{"symbol": s, "date": pd.Timestamp(b["t"][:10]), "open": b["o"], "high": b["h"],
                                               "low": b["l"], "close": b["c"], "volume": b["v"]}
                                              for s, bl in raw.items() for b in bl])], ignore_index=True)
    frames = load_frames(bars)
    spy = frames["SPY"]["close"] if "SPY" in frames else None
    px = premarket_prices(client, symbols)
    if px and not any(live for _, live in px.values()):
        return []  # nothing traded today: market holiday or too early
    lines = []
    for s in symbols:
        f = frames.get(s)
        m = ta.metrics(f, spy) if f is not None else None
        if m is None:
            continue
        o, c = f["open"].iloc[-1], f["close"].iloc[-1]
        m["prev_body_hi"], m["prev_body_lo"] = max(o, c), min(o, c)
        idea = last[last["symbol"] == s].iloc[0] if s in set(last["symbol"]) else None
        p, live = px.get(s, (None, False))
        lines.append(line(s, m, p, live, idea))
    from daily_watchlist import chunk
    return chunk(f"🌅 **Pre-market** — {datetime.now(ET):%a %b %d, %I:%M %p} ET (IEX prices, thin before the open)", lines, sep="\n")


def main() -> int:
    msgs = build(AlpacaClient())
    if not msgs:
        print("No pre-market trades today (holiday or too early) - nothing posted.")
        return 0
    url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not url or "--dry-run" in sys.argv:
        print("\n\n".join(msgs))
        return 0
    for m in msgs:
        post(m, url)
        time.sleep(1.0)
    print(f"Posted {len(msgs)} pre-market messages.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

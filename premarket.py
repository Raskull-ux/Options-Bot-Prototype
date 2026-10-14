# Pre-market post, 9:15 AM ET.
#
# Timing guard: it only posts if the job starts between 8:00 and 9:35 AM ET on a
# trading day (config.PREMARKET_WINDOW). GitHub's free scheduler often starts
# jobs hours late; a late start posts nothing instead of afternoon "pre-market"
# levels. A manual run with force=true posts anyway, labeled "Levels check".
#
# For the core watchlist plus yesterday's ranked picks: the pre-market price,
# where it sits against last week's close (shown as a level), the nearest open
# gaps from yesterday's close, and whether yesterday's trigger is already
# through. Pre-market prices are IEX trades (thin before the
# open); a name with no trade yet today shows yesterday's close.
import os
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from daily_readout import post
from daily_watchlist import load_frames, gap_txt, fmt
from lib import technicals as ta
from lib.alpaca_client import AlpacaClient, DATA
from lib.market_clock import in_window, is_trading_day, now_et

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


def line(sym: str, m: dict, px: float | None, live: bool, idea: dict | None, tag: str = "pre") -> str:
    c = m["close"]
    p = px if px is not None else c
    head = f"**{sym} {fmt(p)}**" + (f" ({p / c - 1:+.1%} {tag})" if live else " (no trade yet today)")
    parts = [head]
    lw = m.get("lw_close", np.nan)
    if not np.isnan(lw):
        parts.append(f"{(p / lw - 1):+.1%} vs last wk close {fmt(lw)}")
    gap_dir = "gapping UP over yesterday's body" if p > max(m["prev_body_hi"], c) and live else (
        "gapping DOWN under yesterday's body" if p < m["prev_body_lo"] and live else "")
    if gap_dir:
        parts.append(gap_dir)
    m2 = dict(m, close=p)  # gaps relative to the pre-market price
    parts.append(gap_txt(m2, 1))
    if idea is not None:
        trig, side = float(idea["trigger"]), idea["side"]
        through = p >= trig if side == "bull" else p <= trig
        parts.append(f"last night's watch level {fmt(trig)} ({'above' if side == 'bull' else 'below'}): "
                     + (f"**already through ({tag})**" if through and live else "not yet"))
    return " · ".join(parts)


def build(client, header: str, tag: str = "pre") -> list[str]:
    bars = pd.read_csv(config.BARS_FILE, parse_dates=["date"])
    try:
        ideas = pd.read_csv(config.WL_IDEAS_FILE)
        # only last night's names; an older list is not "last night's watch level"
        last = ideas[ideas["date"] == bars["date"].max().strftime("%Y-%m-%d")]
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
        lines.append(line(s, m, p, live, idea, tag))
    from daily_watchlist import chunk
    return chunk(header, lines, sep="\n")


def main() -> int:
    force = "--force" in sys.argv or os.environ.get("FORCE_POST", "").strip().lower() == "true"
    now = now_et()
    client = AlpacaClient()
    start, end = (datetime.strptime(x, "%H:%M").time() for x in config.PREMARKET_WINDOW)
    on_time = in_window(now, start, end)
    if not force:
        if not is_trading_day(now.date(), client):
            print(f"{now:%a %b %d}: market closed today. Nothing posted.")
            return 0
        if not on_time:
            print(f"Started {now:%I:%M %p} ET, outside the {config.PREMARKET_WINDOW[0]}-{config.PREMARKET_WINDOW[1]} ET "
                  f"window (GitHub schedule delay). Nothing posted.")
            return 0
    if on_time:
        header = (f"🌅 **Pre-market** · {now:%a %b %d, %I:%M %p} ET · IEX prices (thin before the open). "
                  f"Levels to watch, not trade calls.")
        mention = config.PREMARKET_MENTION.strip()
        if mention:
            header = f"{mention}\n{header}"
    else:
        header = f"📍 **Levels check** · {now:%a %b %d, %I:%M %p} ET · latest IEX prices. Levels to watch, not trade calls."
    msgs = build(client, header, "pre" if on_time else "today")
    if not msgs:
        print("No trades yet today (holiday or too early). Nothing posted.")
        return 0
    url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not url or "--dry-run" in sys.argv:
        print("\n\n".join(msgs))
        return 0
    for m in msgs:
        post(m, url, config.PREMARKET_MENTION if on_time else "")
        time.sleep(1.0)
    print(f"Posted {len(msgs)} pre-market messages.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

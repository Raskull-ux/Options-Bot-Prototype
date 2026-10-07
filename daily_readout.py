# Daily market readout -> Discord.
#
# Reads what the daily collector and signal engine already wrote and posts one
# plain message: SPY trend, VIX term structure, the regime labels (panic state,
# liquidity, dispersion), big earnings in the next week, and any research flags.
# Facts only. It makes no trade calls: every research sleeve tested so far
# failed on real option prices, and the message says so.
#
# Needs env DISCORD_WEBHOOK_URL. Without it (or with --dry-run) the message is
# printed to the log instead of posted, and the run still succeeds.
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config
from lib.earnings_data import load_earnings

DISCORD_LIMIT = 1990


def pct(x, signed=True) -> str:
    return "n/a" if x is None or pd.isna(x) else (f"{x:+.1%}" if signed else f"{x:.0%}")


def flag(v) -> bool:
    return str(v).strip().lower() in ("true", "1")


def spy_section(bars: pd.DataFrame) -> list[str]:
    spy = bars[bars["symbol"] == "SPY"].sort_values("date").set_index("date")["close"].dropna()
    if len(spy) < 2:
        return ["**SPY:** no data"]
    last, prev = spy.iloc[-1], spy.iloc[-2]
    out = [f"**SPY** {last:.2f}  ({pct(last / prev - 1)} day"
           + (f", {pct(last / spy.iloc[-6] - 1)} week)" if len(spy) > 5 else ")")]
    trend = []
    for n in (50, 200):
        if len(spy) >= n:
            sma = spy.iloc[-n:].mean()
            trend.append(f"{'above' if last > sma else 'BELOW'} {n}-day avg ({pct(last / sma - 1)})")
    if trend:
        out.append("Trend: " + ", ".join(trend))
    return out


def vix_section(r: pd.Series) -> list[str]:
    if pd.isna(r.get("vix")):
        return ["**VIX:** no data"]
    inverted = flag(r.get("vix_inverted"))
    state = ("**INVERTED** - near-term fear above 3-month. Stress regime (about 8% of days historically)."
             if inverted else "normal (contango) - near-term fear below 3-month.")
    lvl = "above" if flag(r.get("vix_above_median")) else "below"
    return [f"**VIX** {r['vix']:.1f} / VIX3M {r['vix3m']:.1f} - curve {state}",
            f"VIX is {lvl} its 10-year median ({r['vix_10y_median']:.1f})."]


def regime_section(r: pd.Series) -> list[str]:
    out = []
    panic = flag(r.get("panic_state"))
    out.append(f"**Panic state:** {'ON' if panic else 'off'}  (24-month SPY return {pct(r.get('spy_return_24m'))}, "
               f"volatility at {pct(r.get('market_var_percentile'), False)} percentile)")
    if panic:
        out.append("Panic state = down market + high volatility. Momentum historically crashes here and rebounds are violent.")
    il, dp = flag(r.get("illiquidity_high")), flag(r.get("dispersion_high"))
    out.append(f"**Liquidity:** {'THIN' if il else 'normal'} ({pct(r.get('illiquidity_percentile'), False)} pct)   "
               f"**Dispersion:** {'HIGH' if dp else 'normal'} ({pct(r.get('dispersion_percentile'), False)} pct)")
    if dp:
        out.append("High dispersion = stocks moving on their own news more than with the market.")
    return out


def earnings_section(today: pd.Timestamp, bars: pd.DataFrame, top_n: int = 10) -> list[str]:
    e = load_earnings()
    if e.empty:
        return []
    e = e.assign(d=pd.to_datetime(e["earnings_date"], errors="coerce"))
    e = e[(e["d"] > today) & (e["d"] <= today + timedelta(days=7))].drop_duplicates(["symbol", "d"])
    if e.empty:
        return ["**Earnings next 7 days:** none in the universe"]
    recent = bars[bars["date"] > today - timedelta(days=35)]
    dollar_vol = (recent["close"] * recent["volume"]).groupby(recent["symbol"]).mean()
    e = e.assign(dv=e["symbol"].map(dollar_vol).fillna(0)).sort_values("dv", ascending=False)
    names = [f"{r.symbol} {r.d:%a}{' ' + r.hour if isinstance(r.hour, str) and r.hour else ''}" for r in e.head(top_n).itertuples()]
    return [f"**Earnings next 7 days:** {len(e)} reporting. Biggest: " + ", ".join(names)]


def flags_section(signals: pd.DataFrame, today_str: str) -> list[str]:
    if signals.empty:
        return []
    s = signals[signals["date"] == today_str]
    if s.empty:
        return []
    live = s[s["sleeve"].isin(["short_term_reversal", "index_volatility"])]
    shadow = s[s["sleeve"].isin(config.SHADOW_SLEEVES)]
    out = []
    if not live.empty:
        out.append("**Research flags (UNTESTED, not trade calls):** " + ", ".join(
            f"{r.sleeve} {r.symbol} {r.direction}" for r in live.head(8).itertuples()))
    if not shadow.empty:
        out.append(f"Shadow sleeves logged {len(shadow)} signals (failed backtests; not alerts).")
    return out


def build_message() -> str:
    reg = pd.read_csv(config.REGIME_STATE_FILE).sort_values("date")
    r = reg.iloc[-1]
    today = pd.Timestamp(r["date"])
    bars = pd.read_csv(config.BARS_FILE, parse_dates=["date"])
    try:
        signals = pd.read_csv(config.SIGNALS_FILE)
    except FileNotFoundError:
        signals = pd.DataFrame()

    lines = [f"__**Market readout - {today:%a %b %d, %Y}**__"]
    age = (pd.Timestamp(datetime.now(timezone.utc).date()) - today).days
    if age > 4:
        lines.append(f"WARNING: data is {age} days old - the daily collector may have failed.")
    lines += spy_section(bars) + [""] + vix_section(r) + [""] + regime_section(r)
    er = earnings_section(today, bars)
    fl = flags_section(signals, r["date"])
    if er:
        lines += [""] + er
    if fl:
        lines += [""] + fl
    lines += ["", "_Data readout, not trade advice. No research signal has passed a real-price backtest._"]
    stress = flag(r.get("vix_inverted")) or flag(r.get("panic_state"))
    mention = config.READOUT_MENTION.strip()
    if mention and (config.READOUT_PING_WHEN == "always" or stress):
        lines.insert(0, mention + (" - STRESS CONDITIONS" if stress else ""))
    msg = "\n".join(lines)
    return msg if len(msg) <= DISCORD_LIMIT else msg[:DISCORD_LIMIT - 3] + "..."


def post(msg: str, url: str) -> None:
    m = config.READOUT_MENTION.strip()
    allowed = {"parse": ["everyone"]} if m in ("@everyone", "@here") else (
        {"parse": [], "roles": [m[3:-1]]} if m.startswith("<@&") else {"parse": []})
    r = requests.post(url, json={"content": msg, "allowed_mentions": allowed}, timeout=20)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Discord returned {r.status_code}: {r.text[:200]}")


def all_messages() -> list[str]:
    msgs = [build_message()]
    if getattr(config, "WATCHLIST_ENABLED", False):
        import daily_watchlist
        reg = pd.read_csv(config.REGIME_STATE_FILE).sort_values("date")
        today = pd.Timestamp(reg.iloc[-1]["date"])
        try:
            from lib.alpaca_client import AlpacaClient
            client = AlpacaClient() if os.environ.get("APCA_API_KEY_ID") else None
        except Exception:
            client = None
        try:
            msgs += daily_watchlist.build_messages(client, today)
        except Exception as ex:
            traceback.print_exc()
            msgs.append(f"(watchlist failed today: {type(ex).__name__} - check the Actions log)")
    return msgs


def main() -> int:
    msgs = all_messages()
    url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if "--dry-run" in sys.argv or not url:
        print(("(no DISCORD_WEBHOOK_URL set - printing instead)\n" if not url else "(dry run)\n")
              + "\n\n---- next message ----\n\n".join(msgs))
        return 0
    import time
    for m in msgs:
        post(m, url)
        time.sleep(1.0)
    print(f"Posted {len(msgs)} messages to Discord.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

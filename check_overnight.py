# One-time check: can this Alpaca account get OVERNIGHT (8 PM - 4 AM ET) hourly bars?
# Needed so the bot's hourly candles can run 24 hours like Taz's Thinkorswim chart.
# Changes nothing in the bot. Prints PASS/FAIL lines.
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from lib.alpaca_client import AlpacaClient, AlpacaError, DATA, TRADING

ET = ZoneInfo("America/New_York")
SYMBOLS = ["QQQ", "SPY", "NVDA", "TSLA"]


def bars(client, feed, start, end, symbol="QQQ", limit=1000):
    body = client._get(f"{DATA}/v2/stocks/bars", {"symbols": symbol, "timeframe": "1Hour", "start": start,
                                                  "end": end, "limit": limit, "feed": feed})
    return [(datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET), b) for b in (body.get("bars") or {}).get(symbol, [])]


def main() -> int:
    c = AlpacaClient()
    now = datetime.now(timezone.utc)
    end = (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    week = (now - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=== 1. Which names trade overnight (Assets API) ===")
    for s in SYMBOLS:
        try:
            a = c._get(f"{TRADING}/v2/assets/{s}", {})
            print(f"  {s}: overnight_tradable = {a.get('overnight_tradable')}")
        except AlpacaError as e:
            print(f"  {s}: could not read asset ({str(e)[:120]})")

    print("\n=== 2. Last week of QQQ OVERNIGHT hourly bars (feed=boats) ===")
    ok = False
    try:
        b = bars(c, "boats", week, end)
        night = [x for x in b if x[0].hour >= 20 or x[0].hour < 4]
        print(f"  {len(b)} bars returned, {len(night)} of them between 8 PM and 4 AM ET")
        for t, x in night[:6]:
            print(f"    {t:%a %b %d %H:%M} ET  o {x['o']} h {x['h']} l {x['l']} c {x['c']} v {x['v']}")
        ok = len(night) > 0
        print(f"  >>> {'PASS: overnight bars available' if ok else 'FAIL: no overnight bars came back'}")
    except AlpacaError as e:
        print(f"  >>> FAIL: {str(e)[:300]}")

    print("\n=== 3. How far back does overnight history go? ===")
    if ok:
        for start in ("2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"):
            try:
                b = bars(c, "boats", f"{start}T00:00:00Z", f"{start[:8]}28T00:00:00Z", limit=200)
                first = b[0][0].strftime("%Y-%m-%d %H:%M") if b else "none"
                print(f"  month of {start[:7]}: {len(b)} bars (first {first})")
            except AlpacaError as e:
                print(f"  month of {start[:7]}: error {str(e)[:150]}")
    else:
        print("  skipped (no overnight access)")

    print("\n=== 4. Same week, regular feed (sip), for comparison: hours covered ===")
    try:
        b = bars(c, "sip", week, end)
        hrs = sorted({x[0].hour for x in b})
        print(f"  {len(b)} bars; hours present (ET): {hrs}")
    except AlpacaError as e:
        print(f"  sip error: {str(e)[:200]}")

    print("\nSUMMARY: " + ("overnight data AVAILABLE -> the bot can build 24-hour candles like Thinkorswim"
                         if ok else "overnight data NOT available on this plan -> next step: cost/alternatives"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

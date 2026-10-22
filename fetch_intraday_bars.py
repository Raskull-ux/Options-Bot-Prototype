# Pulls raw intraday bars (5Min, 15Min, 1Hour) for the trade-review study.
# SIP feed, extended hours included (Alpaca returns 4 AM - 8 PM ET bars by default).
# Writes data/intraday/{SYMBOL}_{TF}.csv. Changes nothing else in the bot.
import csv
import os
import sys
from datetime import datetime, timedelta, timezone

from lib.alpaca_client import AlpacaClient, DATA

SYMBOLS = os.environ.get("STUDY_SYMBOLS", "QQQ,AAPL,SMH,HOOD").split(",")
TIMEFRAMES = ["5Min", "15Min", "1Hour"]
# Long daily history for bar-pattern analogs (e.g. MRNA 2020 vs today)
DAILY_SYMBOLS = [s for s in os.environ.get("DAILY_SYMBOLS", "MRNA,QQQ,AAPL,HOOD").split(",") if s]
DAILY_START = os.environ.get("DAILY_START", "2019-01-01")
START = os.environ.get("STUDY_START", "2026-08-01")
OUT_DIR = "data/intraday"


def fetch(client, symbol, tf, start, end):
    rows, token = [], None
    while True:
        params = {"symbols": symbol, "timeframe": tf, "start": start, "end": end,
                  "feed": "sip", "limit": 10000, "adjustment": "split"}
        if token:
            params["page_token"] = token
        body = client._get(f"{DATA}/v2/stocks/bars", params)
        rows.extend((body.get("bars") or {}).get(symbol, []))
        token = body.get("next_page_token")
        if not token:
            return rows


def main() -> int:
    client = AlpacaClient()
    end = (datetime.now(timezone.utc) - timedelta(minutes=16)).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = f"{START}T00:00:00Z"
    os.makedirs(OUT_DIR, exist_ok=True)
    failed = 0
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                bars = fetch(client, sym, tf, start, end)
            except Exception as e:  # report and keep going
                print(f"{sym} {tf}: FAILED {str(e)[:200]}")
                failed += 1
                continue
            path = f"{OUT_DIR}/{sym}_{tf}.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "o", "h", "l", "c", "v", "n", "vw"])
                for b in bars:
                    w.writerow([b["t"], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("n"), b.get("vw")])
            print(f"{sym} {tf}: {len(bars)} bars -> {path}")
    for sym in DAILY_SYMBOLS:
        try:
            bars = fetch(client, sym, "1Day", f"{DAILY_START}T00:00:00Z", end)
        except Exception as e:
            print(f"{sym} 1Day: FAILED {str(e)[:200]}")
            failed += 1
            continue
        path = f"{OUT_DIR}/{sym}_1Day.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "o", "h", "l", "c", "v", "n", "vw"])
            for b in bars:
                w.writerow([b["t"], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("n"), b.get("vw")])
        print(f"{sym} 1Day: {len(bars)} bars -> {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

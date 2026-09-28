"""
Alpaca entitlements check for the day-to-5-day options bot.

Purpose: find out, from REAL responses, what this account can actually do
before any pipeline code is written. Read-only. Places no orders.
Never prints credentials.

Env vars:
  APCA_API_KEY_ID       (required)
  APCA_API_SECRET_KEY   (required)
  FINNHUB_API_KEY       (optional; tests earnings calendar if set)
"""
import os
import sys
import json
from datetime import datetime, timedelta, timezone

import requests

KEY = os.environ.get("APCA_API_KEY_ID", "").strip()
SECRET = os.environ.get("APCA_API_SECRET_KEY", "").strip()
FINNHUB = os.environ.get("FINNHUB_API_KEY", "").strip()

TRADING = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"
HEADERS = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}

results = []  # (name, status, detail)


def record(name, status, detail=""):
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))


def get(url, params=None, headers=HEADERS):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=25)
    except requests.RequestException as e:
        return None, f"network error: {type(e).__name__}"
    try:
        body = r.json()
    except ValueError:
        body = r.text[:200]
    return r.status_code, body


def short(body, n=220):
    s = body if isinstance(body, str) else json.dumps(body)
    return s[:n]


def main():
    if not KEY or not SECRET:
        print("FATAL: APCA_API_KEY_ID / APCA_API_SECRET_KEY not set in environment.")
        sys.exit(2)

    now = datetime.now(timezone.utc)
    iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
    day = lambda d: d.strftime("%Y-%m-%d")

    # ---------- 1. Trading API: account ----------
    code, acct = get(f"{TRADING}/v2/account")
    if code != 200:
        record("paper account", "FAIL", f"HTTP {code} {short(acct)}")
        print("Cannot continue without a working paper account call. Check secrets / that keys are PAPER keys.")
        sys.exit(1)
    keep = ["status", "equity", "cash", "trading_blocked", "account_blocked",
            "options_approved_level", "options_trading_level", "options_buying_power",
            "pattern_day_trader", "multiplier"]
    record("paper account", "OK", {k: acct.get(k) for k in keep if k in acct}.__repr__())

    code, cfg = get(f"{TRADING}/v2/account/configurations")
    if code == 200:
        record("account configurations", "OK", f"max_options_trading_level={cfg.get('max_options_trading_level')}")
    else:
        record("account configurations", "WARN", f"HTTP {code} {short(cfg)}")

    code, clk = get(f"{TRADING}/v2/clock")
    if code == 200:
        record("market clock", "OK", f"is_open={clk.get('is_open')} next_open={clk.get('next_open')}")

    code, asset = get(f"{TRADING}/v2/assets/AAPL")
    if code == 200:
        record("asset flags (AAPL)", "OK",
               f"tradable={asset.get('tradable')} shortable={asset.get('shortable')} attributes={asset.get('attributes')}")

    # ---------- 2. Stock bars: IEX vs SIP ----------
    start = day(now - timedelta(days=12))
    vols = {}
    for feed, end in [("iex", None), ("sip", iso(now - timedelta(minutes=30))), ("sip", None)]:
        label = f"stock bars feed={feed} " + ("(historical, end=now-30m)" if end else "(no end -> recent)")
        params = {"symbols": "SPY", "timeframe": "1Day", "start": start, "feed": feed, "limit": 10}
        if end:
            params["end"] = end
        code, body = get(f"{DATA}/v2/stocks/bars", params)
        if code == 200 and body.get("bars", {}).get("SPY"):
            last = body["bars"]["SPY"][-1]
            vols[(feed, bool(end))] = last["v"]
            record(label, "OK", f"{len(body['bars']['SPY'])} bars, last t={last['t']} volume={last['v']}")
        else:
            record(label, "FAIL", f"HTTP {code} {short(body)}")

    if ("iex", False) in vols and ("sip", True) in vols:
        ratio = vols[("iex", False)] / max(vols[("sip", True)], 1)
        record("IEX vs SIP volume (SPY, same-ish day)", "INFO",
               f"IEX volume is {ratio:.1%} of SIP volume -> volume signals need SIP if this is small")

    # ---------- 3. Sector/market ETFs bars ----------
    code, body = get(f"{DATA}/v2/stocks/bars", {
        "symbols": "SPY,XLK,XLF,XLE,VIXY", "timeframe": "1Day", "start": start,
        "feed": "iex", "limit": 5})
    if code == 200:
        record("ETF bars (SPY, sector ETFs, VIXY)", "OK", f"symbols returned: {sorted(body.get('bars', {}).keys())}")
    else:
        record("ETF bars", "FAIL", f"HTTP {code} {short(body)}")

    # ---------- 4. Option contracts (trading API) ----------
    code, body = get(f"{TRADING}/v2/options/contracts", {
        "underlying_symbols": "SPY", "status": "active", "limit": 5,
        "expiration_date_lte": day(now + timedelta(days=10))})
    sym = None
    if code == 200 and body.get("option_contracts"):
        c = body["option_contracts"][0]
        sym = c["symbol"]
        record("option contracts list", "OK", f"example={sym} exp={c.get('expiration_date')} type={c.get('type')}")
    else:
        record("option contracts list", "FAIL", f"HTTP {code} {short(body)}")

    # ---------- 5. Option snapshots: indicative vs opra ----------
    for feed in ("indicative", "opra"):
        code, body = get(f"{DATA}/v1beta1/options/snapshots/SPY", {"feed": feed, "limit": 25})
        if code == 200:
            snaps = body.get("snapshots", {})
            n = len(snaps)
            with_iv = sum(1 for s in snaps.values() if s.get("impliedVolatility") is not None)
            with_greeks = sum(1 for s in snaps.values() if s.get("greeks"))
            with_quote = sum(
                1 for s in snaps.values()
                if (s.get("latestQuote") or {}).get("ap", 0) and (s.get("latestQuote") or {}).get("bp", 0))
            keys = sorted(next(iter(snaps.values())).keys()) if snaps else []
            record(f"option snapshots feed={feed}", "OK",
                   f"n={n} with_IV={with_iv} with_greeks={with_greeks} with_bid&ask={with_quote} keys={keys}")
        else:
            record(f"option snapshots feed={feed}", "FAIL", f"HTTP {code} {short(body)}")

    # ---------- 6. Option historical bars (needed to test IV/price history) ----------
    if sym:
        code, body = get(f"{DATA}/v1beta1/options/bars", {
            "symbols": sym, "timeframe": "1Day", "start": day(now - timedelta(days=30)), "limit": 30})
        if code == 200:
            n = len(body.get("bars", {}).get(sym, []))
            record("option historical bars", "OK", f"{n} daily bars for {sym} (0 is normal for a fresh/illiquid contract)")
        else:
            record("option historical bars", "FAIL", f"HTTP {code} {short(body)}")

    # ---------- 7. Earnings calendar (Finnhub, optional) ----------
    if FINNHUB:
        code, body = get("https://finnhub.io/api/v1/calendar/earnings",
                         {"from": day(now), "to": day(now + timedelta(days=7)), "token": FINNHUB},
                         headers={})
        if code == 200 and isinstance(body, dict):
            rows = body.get("earningsCalendar", [])
            record("Finnhub earnings calendar", "OK", f"{len(rows)} rows next 7 days; sample={short(rows[:1], 200)}")
        else:
            record("Finnhub earnings calendar", "FAIL", f"HTTP {code} {short(body)}")
    else:
        record("Finnhub earnings calendar", "SKIP", "FINNHUB_API_KEY not set")

    # ---------- Summary ----------
    print("\n===== SUMMARY =====")
    for name, status, _ in results:
        print(f"{status:5} {name}")
    fails = [n for n, s, _ in results if s == "FAIL"]
    print(f"\n{len(fails)} failure(s).")


if __name__ == "__main__":
    main()

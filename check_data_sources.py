"""
Read-only data-source check. Places no orders, writes nothing to the repo.

Test 1 -- VIX term structure (needed for the index-volatility sleeve,
Johnson 2017: long vol is positive only when the VIX curve is inverted):
  - Cboe official daily CSVs for VIX and VIX3M
  - FRED backup copies (VIXCLS, VXVCLS)
  Reports: reachable?, row count, date range, latest values, and whether
  the curve is currently inverted (VIX > VIX3M).

Test 2 -- Alpaca historical bars for EXPIRED option contracts, at several
ages. Decides whether backtests can run on Alpaca or need Databento.
"""
import io
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

KEY = os.environ.get("APCA_API_KEY_ID", "").strip()
SECRET = os.environ.get("APCA_API_SECRET_KEY", "").strip()
HEADERS = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}
TRADING = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"
UA = {"User-Agent": "Mozilla/5.0 (options-bot data check)"}

summary: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    summary.append((name, status, detail))
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""), flush=True)


# ---------------------------------------------------------------------------
# Test 1: VIX term structure
# ---------------------------------------------------------------------------
def parse_cboe(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    return df.set_index("DATE")["CLOSE"].astype(float).sort_index()


def parse_fred(text: str, series_id: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    s = pd.to_numeric(df[series_id], errors="coerce")  # FRED uses "." for missing
    return pd.Series(s.values, index=df[date_col]).dropna().sort_index()


def fetch(url: str) -> tuple[int | None, str]:
    try:
        r = requests.get(url, headers=UA, timeout=30)
        return r.status_code, r.text
    except requests.RequestException as e:
        return None, f"network error: {type(e).__name__}"


def test_vix() -> None:
    sources = {
        "cboe": {
            "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
            "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
        },
        "fred": {
            "VIX": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
            "VIX3M": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VXVCLS",
        },
    }
    fred_ids = {"VIX": "VIXCLS", "VIX3M": "VXVCLS"}

    for src, urls in sources.items():
        series = {}
        for name, url in urls.items():
            code, text = fetch(url)
            if code != 200:
                record(f"{src} {name}", "FAIL", f"HTTP {code} {text[:150]}")
                continue
            try:
                s = parse_cboe(text) if src == "cboe" else parse_fred(text, fred_ids[name])
            except Exception as e:
                record(f"{src} {name}", "FAIL", f"parse error {type(e).__name__}: {e}. First 200 chars: {text[:200]!r}")
                continue
            series[name] = s
            record(
                f"{src} {name}",
                "OK",
                f"{len(s)} rows, {s.index.min().date()} to {s.index.max().date()}, latest close {s.iloc[-1]:.2f}",
            )

        if "VIX" in series and "VIX3M" in series:
            joined = pd.concat([series["VIX"], series["VIX3M"]], axis=1, join="inner").dropna()
            joined.columns = ["VIX", "VIX3M"]
            if joined.empty:
                record(f"{src} term structure", "FAIL", "no overlapping dates")
                continue
            last = joined.iloc[-1]
            slope = last["VIX3M"] - last["VIX"]
            inverted_days = int((joined["VIX"] > joined["VIX3M"]).sum())
            record(
                f"{src} term structure",
                "OK",
                f"latest {joined.index[-1].date()}: VIX {last['VIX']:.2f}, VIX3M {last['VIX3M']:.2f}, "
                f"slope {slope:+.2f} ({'INVERTED' if slope < 0 else 'normal contango'}); "
                f"inverted on {inverted_days} of {len(joined)} overlapping days ({inverted_days / len(joined):.1%})",
            )


# ---------------------------------------------------------------------------
# Test 2: Alpaca historical bars for expired option contracts
# ---------------------------------------------------------------------------
def alpaca_get(url: str, params: dict) -> tuple[int | None, object]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    except requests.RequestException as e:
        return None, f"network error: {type(e).__name__}"
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text[:200]


def spy_close_on_or_before(d: date) -> float | None:
    code, body = alpaca_get(
        f"{DATA}/v2/stocks/bars",
        {"symbols": "SPY", "timeframe": "1Day", "start": (d - timedelta(days=10)).isoformat(),
         "end": (d + timedelta(days=1)).isoformat(), "feed": "sip", "limit": 20},
    )
    if code != 200 or not isinstance(body, dict):
        return None
    bars = (body.get("bars") or {}).get("SPY") or []
    return float(bars[-1]["c"]) if bars else None


def find_expired_contract(target: date) -> tuple[dict | None, str]:
    """Find an expired SPY call near the money, expiring on or just after target."""
    for offset in range(0, 7):
        exp = target + timedelta(days=offset)
        if exp.weekday() >= 5:
            continue
        spot = spy_close_on_or_before(exp)
        if spot is None:
            continue
        code, body = alpaca_get(
            f"{TRADING}/v2/options/contracts",
            {"underlying_symbols": "SPY", "status": "inactive", "type": "call",
             "expiration_date_gte": exp.isoformat(), "expiration_date_lte": exp.isoformat(),
             "strike_price_gte": round(spot - 5), "strike_price_lte": round(spot + 5), "limit": 100},
        )
        if code != 200:
            return None, f"contracts endpoint HTTP {code}: {str(body)[:200]}"
        contracts = body.get("option_contracts") or []
        if contracts:
            best = min(contracts, key=lambda c: abs(float(c["strike_price"]) - spot))
            return best, f"spot at expiry ~{spot:.2f}"
    return None, "no inactive SPY contract found within 7 days of target date"


def test_expired_option_bars() -> None:
    if not KEY or not SECRET:
        record("alpaca expired-option test", "SKIP", "Alpaca keys not set")
        return
    today = datetime.now(timezone.utc).date()
    ages = {"~2 weeks ago": 14, "~3 months ago": 90, "~9 months ago": 270,
            "~18 months ago": 540, "~30 months ago": 900}
    for label, days_back in ages.items():
        target = today - timedelta(days=days_back)
        contract, note = find_expired_contract(target)
        if contract is None:
            record(f"expired option {label}", "FAIL", note)
            continue
        sym = contract["symbol"]
        exp = date.fromisoformat(contract["expiration_date"])
        code, body = alpaca_get(
            f"{DATA}/v1beta1/options/bars",
            {"symbols": sym, "timeframe": "1Day", "start": (exp - timedelta(days=45)).isoformat(),
             "end": (exp + timedelta(days=1)).isoformat(), "limit": 1000},
        )
        if code != 200 or not isinstance(body, dict):
            record(f"expired option {label}", "FAIL", f"{sym} bars HTTP {code}: {str(body)[:200]}")
            continue
        bars = (body.get("bars") or {}).get(sym) or []
        if not bars:
            record(f"expired option {label}", "EMPTY", f"{sym} ({note}): request OK but 0 bars returned")
            continue
        record(
            f"expired option {label}",
            "OK",
            f"{sym} ({note}): {len(bars)} daily bars, {bars[0]['t'][:10]} to {bars[-1]['t'][:10]}, "
            f"last close {bars[-1]['c']}",
        )


def main() -> int:
    print("===== TEST 1: VIX term structure =====")
    test_vix()
    print("\n===== TEST 2: Alpaca historical bars for expired options =====")
    test_expired_option_bars()
    print("\n===== SUMMARY =====")
    for name, status, _ in summary:
        print(f"{status:5} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

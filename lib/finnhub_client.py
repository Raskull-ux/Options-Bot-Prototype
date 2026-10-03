"""Finnhub earnings calendar client -- the only piece Alpaca can't supply."""
import os
from datetime import datetime, timedelta, timezone

import requests

BASE = "https://finnhub.io/api/v1"


class FinnhubError(RuntimeError):
    pass


class FinnhubClient:
    def __init__(self, api_key: str | None = None, timeout: int = 25):
        self.api_key = api_key or os.environ["FINNHUB_API_KEY"]
        self.timeout = timeout

    def get_earnings_calendar(self, days_ahead: int = 10, symbols: set[str] | None = None) -> list[dict]:
        """
        Returns raw rows from Finnhub's calendar/earnings endpoint, each with
        symbol, date, hour ("bmo"/"amc"/""), epsEstimate, epsActual, etc.
        If `symbols` is given, filters to that set (Finnhub returns the
        whole market's calendar, not just tickers you ask for).
        """
        today = datetime.now(timezone.utc).date()
        params = {
            "from": today.isoformat(),
            "to": (today + timedelta(days=days_ahead)).isoformat(),
            "token": self.api_key,
        }
        r = requests.get(f"{BASE}/calendar/earnings", params=params, timeout=self.timeout)
        if r.status_code != 200:
            raise FinnhubError(f"HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        rows = body.get("earningsCalendar", [])
        if symbols:
            rows = [row for row in rows if row.get("symbol") in symbols]
        return rows

    def get_earnings_range(self, start, end, symbols: set[str] | None = None, chunk_days: int = 30) -> list[dict]:
        """
        Earnings calendar between two dates (inclusive), requested in chunks
        so a long backfill doesn't hit a per-response row cap. Past dates are
        needed by the post-earnings-drift sleeve (it measures the reaction
        around announcements that already happened).
        """
        rows: list[dict] = []
        cur = start
        while cur <= end:
            chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
            r = requests.get(
                f"{BASE}/calendar/earnings",
                params={"from": cur.isoformat(), "to": chunk_end.isoformat(), "token": self.api_key},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                raise FinnhubError(f"HTTP {r.status_code} for {cur}..{chunk_end}: {r.text[:300]}")
            rows.extend(r.json().get("earningsCalendar", []))
            cur = chunk_end + timedelta(days=1)
        if symbols:
            rows = [row for row in rows if row.get("symbol") in symbols]
        return rows

    def get_company_profile(self, symbol: str) -> dict:
        """
        Returns Finnhub's /stock/profile2 response for a symbol, notably
        'finnhubIndustry' -- a real industry classification string, but NOT
        one with a published, guaranteed-complete enum of values. Treat the
        returned industry as raw data to be mapped (see lib/sector_map.py),
        not as a known-safe key into a hardcoded table.
        """
        r = requests.get(f"{BASE}/stock/profile2", params={"symbol": symbol, "token": self.api_key}, timeout=self.timeout)
        if r.status_code != 200:
            raise FinnhubError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def get_us_symbol_types(self) -> list[dict]:
        """
        ONE bulk call (not rate-limited per symbol) that lists every symbol
        Finnhub supports on the US exchange, each with a real security-type
        classification (OpenFIGI standard: "Common Stock", "ETP", "ADR",
        "REIT", etc.). This is the real, authoritative way to tell a common
        stock apart from an ETF/fund -- NOT something to infer indirectly
        from a blank industry field, which is also blank for other reasons
        (new IPOs Finnhub hasn't profiled yet, some ADRs, etc).
        """
        r = requests.get(f"{BASE}/stock/symbol", params={"exchange": "US", "token": self.api_key}, timeout=60)
        if r.status_code != 200:
            raise FinnhubError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

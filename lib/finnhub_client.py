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

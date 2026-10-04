"""
Historical earnings-release dates from SEC EDGAR (free, official).

A US company announcing results files an 8-K under Item 2.02 ("Results of
Operations and Financial Condition"), normally at the moment the release goes
out. The filing's acceptance timestamp tells us WHEN: before the open, during
the session, or after the close. That timing is what the earnings sleeves need,
and Finnhub's free calendar both lacks history and often leaves it blank.

Limits (disclosed):
  - Foreign issuers (ADRs like NVO, SAP, INFY) file 6-K, not 8-K -> not covered;
    Finnhub stays their source.
  - A few companies file the 8-K some minutes after the release; the timestamp
    is then a close approximation of release time.
SEC rules: identify yourself in the User-Agent; stay under 10 requests/second.
"""
import time
from datetime import datetime

import pandas as pd
import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
MIN_INTERVAL = 0.12  # ~8 requests/second, under SEC's 10/s limit


class EdgarClient:
    def __init__(self, contact: str):
        if not contact or "@" not in contact:
            raise ValueError("SEC requires a contact email in the User-Agent (set SEC_CONTACT_EMAIL)")
        self.headers = {"User-Agent": f"TazmanianTrader options-bot {contact}", "Accept-Encoding": "gzip, deflate"}
        self._last = 0.0

    def _get(self, url: str) -> dict:
        wait = MIN_INTERVAL - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            self._last = time.time()
            r = requests.get(url, headers=self.headers, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}

    def ticker_to_cik(self) -> dict[str, int]:
        data = self._get(TICKERS_URL)
        # SEC writes class shares with a dash (BRK-B); Alpaca uses a dot (BRK.B)
        return {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in data.values()}

    def earnings_filings(self, cik: str | int, since: str) -> list[dict]:
        """All 8-K / Item 2.02 filings since `since` (YYYY-MM-DD), raw timestamps."""
        sub = self._get(SUBMISSIONS_URL.format(cik=int(cik)))
        blocks = [sub.get("filings", {}).get("recent", {})]
        for f in sub.get("filings", {}).get("files", []):
            if f.get("filingTo", "9999") >= since:  # only pull older pages that reach our window
                blocks.append(self._get(SUBMISSIONS_FILE_URL.format(name=f["name"])))
        out = []
        for b in blocks:
            forms = b.get("form", [])
            for i, form in enumerate(forms):
                if form != "8-K":
                    continue
                items = (b.get("items") or [""] * len(forms))[i] or ""
                if "2.02" not in items.split(","):
                    continue
                fdate = b["filingDate"][i]
                if fdate < since:
                    continue
                out.append({"accession": b["accessionNumber"][i], "filing_date": fdate,
                            "acceptance_raw": b["acceptanceDateTime"][i]})
        return out


def wall_clock(raw: str) -> datetime:
    """'2026-07-30T16:31:07.000Z' -> naive datetime of the digits as written."""
    return datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S")


def detect_timezone(raw_values: list[str]) -> tuple[str, float, float]:
    """
    Earnings releases cluster at 06:00-09:29 and 16:00-17:59 Eastern. If the
    digits are already Eastern, most land there; if they are UTC, the same
    releases show up 4-5 hours later (10:00-13:59 and 20:00-22:59).
    Returns ("eastern" | "utc", share_eastern_pattern, share_utc_pattern).
    """
    hrs = pd.Series([wall_clock(r).hour + wall_clock(r).minute / 60 for r in raw_values])
    if hrs.empty:
        return "eastern", 0.0, 0.0
    east = (((hrs >= 6) & (hrs < 9.5)) | ((hrs >= 16) & (hrs < 18))).mean()
    utc = (((hrs >= 10) & (hrs < 14)) | ((hrs >= 20) & (hrs < 23))).mean()
    return ("utc" if utc > east else "eastern"), float(east), float(utc)


def to_eastern(raw: str, tz: str) -> pd.Timestamp:
    ts = pd.Timestamp(wall_clock(raw))
    if tz == "utc":
        return ts.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
    return ts


def classify_hour(et: pd.Timestamp) -> str:
    """bmo = before 9:30 ET, amc = 16:00 ET or later, dmh = during market hours."""
    minutes = et.hour * 60 + et.minute
    if minutes < 9 * 60 + 30:
        return "bmo"
    if minutes >= 16 * 60:
        return "amc"
    return "dmh"

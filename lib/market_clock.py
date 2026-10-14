# Market clock for the Discord posts: Eastern time, trading days, and the
# "is this post on time?" checks.
#
# Why this exists: GitHub's free scheduler starts jobs hours late (the 9:15 AM
# pre-market job has been starting between 2:47 PM and 5:42 PM ET). A late job
# must not post "pre-market" levels in the afternoon, ping everyone at night,
# or repeat yesterday's readout on a market holiday.
#
# Trading days come from Alpaca's market calendar (free, uses the same keys).
# If that call fails, a built-in NYSE holiday list is used instead.
import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# NYSE full-day closures (fallback only; Alpaca's calendar is checked first).
NYSE_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def now_et() -> datetime:
    """Current time in New York. BOT_NOW (e.g. '2026-10-14 09:10') overrides it for tests."""
    fake = os.environ.get("BOT_NOW", "").strip()
    if fake:
        return datetime.fromisoformat(fake).replace(tzinfo=ET)
    return datetime.now(ET)


def _calendar_days(client, start: date, end: date) -> set[str] | None:
    if client is None:
        return None
    try:
        from lib.alpaca_client import TRADING
        rows = client._get(f"{TRADING}/v2/calendar", {"start": start.isoformat(), "end": end.isoformat()})
        return {r["date"] for r in rows} if isinstance(rows, list) else None
    except Exception as e:  # never let the calendar check break a post
        print(f"(market calendar unavailable, using built-in holiday list: {type(e).__name__})")
        return None


def is_trading_day(d: date, client=None) -> bool:
    days = _calendar_days(client, d, d)
    if days is not None:
        return d.isoformat() in days
    return d.weekday() < 5 and d.isoformat() not in NYSE_HOLIDAYS


def latest_closed_session(now: datetime, client=None) -> date:
    """The most recent trading day whose 4:00 PM close has passed."""
    start = now.date() - timedelta(days=10)
    days = _calendar_days(client, start, now.date())
    d = now.date() if now.time() >= time(16, 0) else now.date() - timedelta(days=1)
    for _ in range(12):
        ok = (d.isoformat() in days) if days is not None else (d.weekday() < 5 and d.isoformat() not in NYSE_HOLIDAYS)
        if ok:
            return d
        d -= timedelta(days=1)
    return d


def in_window(now: datetime, start: time, end: time) -> bool:
    return start <= now.time() < end

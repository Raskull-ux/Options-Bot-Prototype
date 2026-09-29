"""
Populates data/sector_profiles.csv: symbol -> real Finnhub industry -> sector
ETF, for the trend signal's sector-adjustment step.

Deliberately SEPARATE from daily_collect.py: Finnhub's free tier is rate-
limited, and the working universe can run 1,000-2,500+ symbols, so a full
pass takes real time. Sector rarely changes, so this doesn't need to run
daily -- run it manually or on a slower schedule (weekly), and the signal
engine falls back to a market-only adjustment (clearly flagged) for any
symbol not yet cached, or cached longer ago than SECTOR_PROFILE_MAX_AGE_DAYS.
"""
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

import config
from lib.finnhub_client import FinnhubClient, FinnhubError
from lib.sector_map import map_industry_to_etf

FINNHUB_FREE_RATE_LIMIT_PER_MIN = 60
SLEEP_BETWEEN_CALLS = 60.0 / FINNHUB_FREE_RATE_LIMIT_PER_MIN * 1.15  # small safety margin


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def load_existing() -> pd.DataFrame:
    try:
        return pd.read_csv(config.SECTOR_PROFILES_FILE, parse_dates=["fetched_at"])
    except FileNotFoundError:
        return pd.DataFrame(columns=["symbol", "industry", "sector_etf", "is_real_sector_match", "fetched_at"])


def symbols_needing_refresh(all_symbols: list[str], existing: pd.DataFrame) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.SECTOR_PROFILE_MAX_AGE_DAYS)
    fresh = set()
    if not existing.empty:
        fresh_rows = existing[existing["fetched_at"] >= cutoff]
        fresh = set(fresh_rows["symbol"])
    return [s for s in all_symbols if s not in fresh]


def main(limit: int | None = None) -> int:
    if not os.environ.get("FINNHUB_API_KEY"):
        log("FATAL: FINNHUB_API_KEY not set")
        return 2

    os.makedirs(config.DATA_DIR, exist_ok=True)
    finnhub = FinnhubClient()

    try:
        membership = pd.read_csv(config.UNIVERSE_MEMBERSHIP_FILE)
    except FileNotFoundError:
        log("FATAL: no universe_membership.csv yet -- run daily_collect.py first")
        return 1
    all_symbols = sorted(membership["symbol"].unique())

    existing = load_existing()
    todo = symbols_needing_refresh(all_symbols, existing)
    if limit:
        todo = todo[:limit]
    log(f"{len(all_symbols)} symbols in working universe, {len(todo)} need a profile fetch this run")

    new_rows = []
    unmapped_rows = []
    errors = []
    for i, symbol in enumerate(todo):
        try:
            profile = finnhub.get_company_profile(symbol)
        except FinnhubError as e:
            errors.append(f"{symbol}: {e}")
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        industry = profile.get("finnhubIndustry")
        etf, is_real_match = map_industry_to_etf(industry)
        now = datetime.now(timezone.utc)
        new_rows.append(
            {"symbol": symbol, "industry": industry, "sector_etf": etf, "is_real_sector_match": is_real_match, "fetched_at": now}
        )
        if not is_real_match:
            unmapped_rows.append({"symbol": symbol, "raw_industry": industry, "seen_at": now.isoformat()})

        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(todo)} profiles fetched...")
        time.sleep(SLEEP_BETWEEN_CALLS)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined = combined.drop_duplicates(subset=["symbol"], keep="last")
        combined.to_csv(config.SECTOR_PROFILES_FILE, index=False)
        log(f"Wrote {len(combined)} total sector profiles ({len(new_rows)} new/refreshed this run)")

    if unmapped_rows:
        try:
            existing_unmapped = pd.read_csv(config.UNMAPPED_INDUSTRIES_FILE)
            combined_unmapped = pd.concat([existing_unmapped, pd.DataFrame(unmapped_rows)], ignore_index=True)
        except FileNotFoundError:
            combined_unmapped = pd.DataFrame(unmapped_rows)
        combined_unmapped = combined_unmapped.drop_duplicates(subset=["symbol"], keep="last")
        combined_unmapped.to_csv(config.UNMAPPED_INDUSTRIES_FILE, index=False)
        log(f"{len(unmapped_rows)} symbols this run had an unmapped/unrecognized industry -- logged for review")

    if errors:
        log(f"{len(errors)} fetch errors:")
        for e in errors[:20]:
            log(f"  - {e}")

    log("Done.")
    return 0


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        sys.exit(main(limit=lim))
    except Exception:
        traceback.print_exc()
        sys.exit(1)

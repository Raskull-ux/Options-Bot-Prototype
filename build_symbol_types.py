"""
Fetches Finnhub's real security-type list for the US exchange -- ONE bulk
API call, not rate-limited per symbol -- and caches it to
data/symbol_types.csv. This is the authoritative source for telling a
common stock apart from an ETF/fund/REIT/ADR, used to check (and later fix)
whether the tradable universe has been including instruments the
underlying-selection research doesn't apply to.
"""
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

import config
from lib.finnhub_client import FinnhubClient, FinnhubError


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def main() -> int:
    if not os.environ.get("FINNHUB_API_KEY"):
        log("FATAL: FINNHUB_API_KEY not set")
        return 2

    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists(config.SYMBOL_TYPES_FILE):
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            os.path.getmtime(config.SYMBOL_TYPES_FILE), tz=timezone.utc
        )
        if age < timedelta(days=config.SYMBOL_TYPE_MAX_AGE_DAYS):
            log(f"Cache is {age.days} day(s) old (< {config.SYMBOL_TYPE_MAX_AGE_DAYS}-day max) -- skipping refetch")
            return 0

    finnhub = FinnhubClient()
    log("Fetching full US exchange symbol list from Finnhub (one call)...")
    try:
        rows = finnhub.get_us_symbol_types()
    except FinnhubError as e:
        log(f"FATAL: {e}")
        return 1

    if not rows:
        log("FATAL: empty response from Finnhub")
        return 1

    sample_keys = sorted(rows[0].keys())
    log(f"Received {len(rows)} symbols. Response fields present: {sample_keys}")

    # Defensive key lookup -- Finnhub's field name for this has been "type"
    # in documented examples, but this is verified against the REAL response
    # here rather than assumed, and this log line is the tripwire if that
    # assumption is ever wrong.
    type_key = "type" if "type" in rows[0] else ("securityType" if "securityType" in rows[0] else None)
    if type_key is None:
        log(f"FATAL: could not find a security-type field in the response. Full sample record: {rows[0]}")
        return 1
    log(f"Using '{type_key}' as the security-type field")

    df = pd.DataFrame(
        [
            {
                "symbol": r.get("symbol"),
                "description": r.get("description"),
                "security_type": r.get(type_key),
                "mic": r.get("mic"),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            for r in rows
        ]
    )
    df.to_csv(config.SYMBOL_TYPES_FILE, index=False)
    log(f"Wrote {len(df)} rows to {config.SYMBOL_TYPES_FILE}")

    log("\nSecurity-type breakdown across the FULL exchange list:")
    print(df["security_type"].value_counts(dropna=False).to_string())

    log("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

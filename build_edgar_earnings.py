"""
Build data/edgar_earnings.csv from SEC EDGAR: every 8-K Item 2.02 (earnings
release) since EDGAR_SINCE for the tradable universe, with the release's
Eastern-time timestamp and before-open / during / after-close timing.

Needs env SEC_CONTACT_EMAIL (SEC requires a contact in the User-Agent).
Roughly one request per company (~2,000), about 5 minutes at SEC's rate limit.
"""
import os
import sys
import traceback
from datetime import datetime, timezone

import pandas as pd

import config
from lib.edgar import EdgarClient, detect_timezone, to_eastern, classify_hour
from lib.symbol_filter import load_excluded_symbols


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def main() -> int:
    client = EdgarClient(os.environ.get("SEC_CONTACT_EMAIL", "").strip())
    membership = pd.read_csv(config.UNIVERSE_MEMBERSHIP_FILE)
    universe = sorted(membership["symbol"].unique())
    excluded, have_types = load_excluded_symbols()
    if have_types:
        universe = [s for s in universe if s not in excluded]
    log(f"{len(universe)} symbols in the tradable universe")

    cik_map = client.ticker_to_cik()
    with_cik = [s for s in universe if s in cik_map]
    log(f"{len(with_cik)} have an SEC CIK; {len(universe) - len(with_cik)} do not (mostly foreign issuers -> Finnhub covers them)")

    raw_rows, errors, no_filings = [], [], 0
    for i, sym in enumerate(with_cik):
        try:
            filings = client.earnings_filings(cik_map[sym], config.EDGAR_SINCE)
        except Exception as e:
            errors.append(f"{sym}: {type(e).__name__}: {e}")
            continue
        if not filings:
            no_filings += 1
        for f in filings:
            raw_rows.append({"symbol": sym, "cik": cik_map[sym], **f})
        if (i + 1) % 250 == 0:
            log(f"  {i + 1}/{len(with_cik)} companies pulled, {len(raw_rows)} earnings filings so far")

    if not raw_rows:
        log("FATAL: no filings retrieved")
        for e in errors[:20]:
            log(f"  - {e}")
        return 1

    tz, east_share, utc_share = detect_timezone([r["acceptance_raw"] for r in raw_rows])
    log(f"Timestamp check: {east_share:.0%} of filings fall in Eastern release windows if read as Eastern, "
        f"{utc_share:.0%} if read as UTC -> treating timestamps as {tz.upper()}")

    df = pd.DataFrame(raw_rows)
    df["acceptance_et"] = df["acceptance_raw"].map(lambda r: to_eastern(r, tz))
    df["earnings_date"] = df["acceptance_et"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["acceptance_et"].map(classify_hour)
    df["source"] = "edgar"
    df = df.drop_duplicates(["symbol", "earnings_date"]).sort_values(["symbol", "earnings_date"])
    df[["symbol", "earnings_date", "hour", "acceptance_et", "accession", "filing_date", "cik", "source"]].to_csv(
        config.EDGAR_EARNINGS_FILE, index=False)

    log(f"Wrote {len(df)} earnings releases for {df['symbol'].nunique()} companies to {config.EDGAR_EARNINGS_FILE}")
    log(f"  timing split: {df['hour'].value_counts().to_dict()}")
    log(f"  {no_filings} companies with a CIK had no Item 2.02 filings since {config.EDGAR_SINCE}")
    for sym in ("AAPL", "MSFT", "JPM", "WMT"):
        last = df[df["symbol"] == sym].tail(2)
        for _, r in last.iterrows():
            log(f"  sanity {sym}: {r['acceptance_et']}  -> {r['hour']}")
    if errors:
        log(f"{len(errors)} fetch errors (first 20):")
        for e in errors[:20]:
            log(f"  - {e}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

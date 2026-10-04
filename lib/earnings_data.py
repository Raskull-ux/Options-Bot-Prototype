"""
One earnings table for the whole bot (live engine, replay, backtest).

  - SEC EDGAR (data/edgar_earnings.csv): actual releases with real timing.
    Preferred whenever it has the event.
  - Finnhub (data/earnings_calendar.csv): upcoming scheduled dates (EDGAR only
    knows a release after it happens) and foreign issuers EDGAR doesn't cover.
A Finnhub row is dropped if EDGAR has the same company within +/-3 days
(same event; Finnhub dates/hours are sometimes off by a day or blank).
"""
import pandas as pd

import config

COLS = ["symbol", "earnings_date", "hour", "source"]


def load_earnings() -> pd.DataFrame:
    try:
        ed = pd.read_csv(config.EDGAR_EARNINGS_FILE)[["symbol", "earnings_date", "hour"]].assign(source="edgar")
    except (FileNotFoundError, KeyError):
        ed = pd.DataFrame(columns=COLS)
    try:
        fh = pd.read_csv(config.EARNINGS_FILE)[["symbol", "earnings_date", "hour"]].assign(source="finnhub")
    except (FileNotFoundError, KeyError):
        fh = pd.DataFrame(columns=COLS)
    if ed.empty:
        return fh.reset_index(drop=True)
    if fh.empty:
        return ed.reset_index(drop=True)

    ed_dates = ed.assign(d=pd.to_datetime(ed["earnings_date"], errors="coerce"))
    fh = fh.assign(d=pd.to_datetime(fh["earnings_date"], errors="coerce")).dropna(subset=["d"])
    m = fh.reset_index().merge(ed_dates[["symbol", "d"]], on="symbol", how="left", suffixes=("", "_ed"))
    dup = m[(m["d_ed"] - m["d"]).abs() <= pd.Timedelta(days=3)]["index"].unique()
    fh = fh.drop(index=dup).drop(columns="d")
    return pd.concat([ed, fh], ignore_index=True)[COLS]

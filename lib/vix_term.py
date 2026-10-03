"""
VIX / VIX3M daily closes. Cboe's official CSVs are primary; FRED's copies
(VIXCLS, VXVCLS) are the backup. Both confirmed reachable from GitHub
Actions on 2026-10-03 (check_data_sources.py).
"""
import io

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (options-bot)"}
CBOE = {
    "vix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "vix3m": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
}
FRED = {
    "vix": ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", "VIXCLS"),
    "vix3m": ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VXVCLS", "VXVCLS"),
}


def _parse_cboe(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    return df.set_index("DATE")["CLOSE"].astype(float).sort_index()


def _parse_fred(text: str, series_id: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    d = df.columns[0]
    s = pd.to_numeric(df[series_id], errors="coerce")
    return pd.Series(s.values, index=pd.to_datetime(df[d])).dropna().sort_index()


def _get(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_vix_term() -> tuple[pd.DataFrame, str]:
    """Returns (DataFrame indexed by date with columns vix, vix3m), source name.
    Raises RuntimeError only if BOTH sources fail."""
    errors = []
    try:
        vix = _parse_cboe(_get(CBOE["vix"]))
        vix3m = _parse_cboe(_get(CBOE["vix3m"]))
        df = pd.concat([vix, vix3m], axis=1, join="inner").dropna()
        df.columns = ["vix", "vix3m"]
        if not df.empty:
            return df, "cboe"
        errors.append("cboe: no overlapping rows")
    except Exception as e:
        errors.append(f"cboe: {type(e).__name__}: {e}")
    try:
        vix = _parse_fred(_get(FRED["vix"][0]), FRED["vix"][1])
        vix3m = _parse_fred(_get(FRED["vix3m"][0]), FRED["vix3m"][1])
        df = pd.concat([vix, vix3m], axis=1, join="inner").dropna()
        df.columns = ["vix", "vix3m"]
        if not df.empty:
            return df, "fred"
        errors.append("fred: no overlapping rows")
    except Exception as e:
        errors.append(f"fred: {type(e).__name__}: {e}")
    raise RuntimeError("Both VIX sources failed: " + " | ".join(errors))

"""
Thin wrapper over the real Alpaca endpoints this bot uses, using exactly the
feeds confirmed working in the 2026-09-28 entitlements check:
  - stock bars: feed=sip  (full consolidated volume, free for daily bars)
  - option snapshots/contracts: feed=indicative (real quotes, no real IV/greeks)

Every function here either returns real data or raises -- no silent
fallback to fabricated numbers.
"""
import os
import time
from datetime import datetime, timedelta, timezone

import requests

TRADING = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"


class AlpacaError(RuntimeError):
    pass


class AlpacaClient:
    def __init__(self, key: str | None = None, secret: str | None = None, timeout: int = 25):
        self.key = key or os.environ["APCA_API_KEY_ID"]
        self.secret = secret or os.environ["APCA_API_SECRET_KEY"]
        self.headers = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}
        self.timeout = timeout

    def _get(self, url: str, params: dict | None = None, retries: int = 3) -> dict:
        last_err = None
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, headers=self.headers, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 429:  # rate limited -- back off and retry
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise AlpacaError(f"GET {url} -> HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        raise AlpacaError(f"GET {url} failed after {retries} retries: {last_err}")

    # -------------------------------------------------------------------
    # Stock bars -- always SIP, always daily, for the signal-construction layer
    # -------------------------------------------------------------------
    def get_daily_bars(self, symbols: list[str], lookback_days: int, feed: str = "sip") -> dict[str, list[dict]]:
        """
        Returns {symbol: [bar, ...]} with bar = {t, o, h, l, c, v, n, vw}.
        Alpaca allows many symbols per request; batch to stay well under
        the free tier's 200 req/min and avoid URL length limits.
        """
        out: dict[str, list[dict]] = {}
        start = (datetime.now(timezone.utc) - timedelta(days=int(lookback_days * 1.6) + 10)).strftime("%Y-%m-%d")
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            page_token = None
            while True:
                params = {
                    "symbols": ",".join(batch),
                    "timeframe": "1Day",
                    "start": start,
                    "feed": feed,
                    "limit": 10000,
                    "adjustment": "split",
                }
                if page_token:
                    params["page_token"] = page_token
                body = self._get(f"{DATA}/v2/stocks/bars", params)
                for sym, bars in (body.get("bars") or {}).items():
                    out.setdefault(sym, []).extend(bars)
                page_token = body.get("next_page_token")
                if not page_token:
                    break
        return out

    def get_latest_trade_price(self, symbol: str, feed: str = "sip") -> float:
        body = self._get(f"{DATA}/v2/stocks/{symbol}/trades/latest", {"feed": feed})
        trade = body.get("trade")
        if not trade or "p" not in trade:
            raise AlpacaError(f"No latest trade for {symbol}: {body}")
        return float(trade["p"])

    # -------------------------------------------------------------------
    # Full asset list -- source of truth for the tradable universe.
    # No hardcoded ticker list anywhere: this pulls the REAL, current set
    # of optionable US equities from Alpaca every time it's called.
    # -------------------------------------------------------------------
    def get_optionable_equities(self) -> list[dict]:
        """
        Returns active, tradable US equities that have listed options,
        straight from Alpaca's asset list. Excludes OTC (pink-sheet-style)
        exchanges as an extra safety net alongside the has_options filter.
        """
        body = self._get(
            f"{TRADING}/v2/assets",
            {"status": "active", "asset_class": "us_equity"},
        )
        # This endpoint returns a plain list, not a paginated object.
        assets = body if isinstance(body, list) else body.get("assets", [])
        out = []
        for a in assets:
            if not a.get("tradable"):
                continue
            if a.get("exchange") == "OTC":
                continue
            attrs = a.get("attributes") or []
            if "has_options" not in attrs:
                continue
            out.append(a)
        return out

    # -------------------------------------------------------------------
    # Options
    # -------------------------------------------------------------------
    def get_option_contracts(
        self, underlying: str, min_dte: int, max_dte: int, limit: int = 200
    ) -> list[dict]:
        today = datetime.now(timezone.utc).date()
        exp_gte = (today + timedelta(days=min_dte)).isoformat()
        exp_lte = (today + timedelta(days=max_dte)).isoformat()
        params = {
            "underlying_symbols": underlying,
            "status": "active",
            "expiration_date_gte": exp_gte,
            "expiration_date_lte": exp_lte,
            "limit": limit,
        }
        contracts: list[dict] = []
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            body = self._get(f"{TRADING}/v2/options/contracts", params)
            contracts.extend(body.get("option_contracts", []))
            page_token = body.get("next_page_token")
            if not page_token:
                break
        return contracts

    def get_option_snapshots(
        self,
        underlying: str,
        feed: str = "indicative",
        expiration_date: str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        limit: int = 100,
    ) -> dict[str, dict]:
        """Returns {contract_symbol: snapshot_dict}."""
        params = {"feed": feed, "limit": limit}
        if expiration_date:
            params["expiration_date"] = expiration_date
        if strike_gte is not None:
            params["strike_price_gte"] = strike_gte
        if strike_lte is not None:
            params["strike_price_lte"] = strike_lte
        snapshots: dict[str, dict] = {}
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            body = self._get(f"{DATA}/v1beta1/options/snapshots/{underlying}", params)
            snapshots.update(body.get("snapshots", {}))
            page_token = body.get("next_page_token")
            if not page_token:
                break
        return snapshots

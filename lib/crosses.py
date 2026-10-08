# Moving-average crosses and the 10-day gauge. Pure functions, no I/O.
import numpy as np
import pandas as pd


def add_mas(df: pd.DataFrame, price: str = "c") -> pd.DataFrame:
    """SMA 10/20/50/200, EMA 10, MACD(12,26,9) on one symbol's bars (in time order)."""
    df = df.copy()
    p = df[price]
    for n in (10, 20, 50, 200):
        df[f"sma{n}"] = p.rolling(n).mean()
    df["ema10"] = p.ewm(span=10, adjust=False).mean()
    line = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    df["macd"], df["macd_sig"] = line, line.ewm(span=9, adjust=False).mean()
    return df


def cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where a moves from >= b to < b."""
    return (a < b) & (a.shift(1) >= b.shift(1)) & a.notna() & b.notna() & a.shift(1).notna()


def ten_day_candles(daily: pd.DataFrame, anchor_end: str) -> pd.DataFrame:
    """Group trading days into 10-day candles so that one candle ENDS on
    anchor_end (a trading day). daily: index = dates, columns o/h/l/c.
    The newest candle may be partial (still forming), like on the chart."""
    d = daily.sort_index()
    idx = d.index
    a = pd.Timestamp(anchor_end)
    # position of the anchor; days after the last known date count forward as trading days
    if a in idx:
        apos = idx.get_loc(a)
    else:
        extra = len(pd.bdate_range(idx[-1] + pd.Timedelta(days=1), a))
        apos = len(idx) - 1 + extra
    block = (np.arange(len(idx)) - apos - 1) // 10   # block ending at apos has id -1
    g = d.assign(block=block).groupby("block")
    out = pd.DataFrame({"start": g.apply(lambda x: x.index[0]), "end": g.apply(lambda x: x.index[-1]),
                        "o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(),
                        "days": g.size()})
    return out.reset_index(drop=True)


def ten_day_gauge(daily: pd.DataFrame, anchor_end: str) -> dict:
    """10 EMA vs 10 SMA on 10-day candles: current state, when it last crossed
    below, and every past bear cross (start date of the 10-day candle)."""
    t = ten_day_candles(daily, anchor_end)
    t["ema10"] = t["c"].ewm(span=10, adjust=False).mean()
    t["sma10"] = t["c"].rolling(10).mean()
    xs = cross_below(t["ema10"], t["sma10"])
    last = t.iloc[-1]
    bear = bool(last["ema10"] < last["sma10"])
    return {"bearish": bear, "ema10": float(last["ema10"]), "sma10": float(last["sma10"]),
            "candle_start": last["start"], "candle_end_planned": pd.Timestamp(last["start"]) + pd.offsets.BDay(9),
            "candle_days_so_far": int(last["days"]),
            "bear_crosses": [r.start for r in t[xs].itertuples()], "table": t}

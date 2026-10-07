# Technical engine for the daily watchlist. Pure functions, no I/O.
#
# Rules follow Taz's own conventions: 10/20/50 moving-average structure, the
# "trend breaker" (10 under 20, 10 under 50), RSI, MACD, a red day on elevated
# volume, and RSI exhaustion tiers (80 early, 85 elevated, 90+ extreme).
import numpy as np
import pandas as pd

from lib.signals import average_true_range

SMA_PERIODS = (8, 10, 20, 50, 200)
FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)
EXHAUSTION_TIERS = ((90, "EXTREME"), (85, "ELEVATED"), (80, "EARLY"))


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (the standard charting-platform RSI)."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + up / dn)
    return out.where(dn != 0, 100.0)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, line - sig


def metrics(df: pd.DataFrame, spy_close: pd.Series | None = None) -> dict | None:
    """df: one symbol's daily bars indexed by date with high, low, close, volume.
    Returns the latest day's indicator values, or None if history is too short."""
    df = df.dropna(subset=["close"]).sort_index()
    if len(df) < 200:
        return None
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
    r = rsi(c)
    ml, ms, mh = macd(c)
    atr = average_true_range(h, lo, c, 14)
    m = {
        "date": df.index[-1], "close": float(c.iloc[-1]), "prev_close": float(c.iloc[-2]),
        "day_high": float(h.iloc[-1]), "day_low": float(lo.iloc[-1]),
        "rsi": float(r.iloc[-1]), "rsi_max_prior20": float(r.iloc[-21:-1].max()),
        "macd": float(ml.iloc[-1]), "macd_signal": float(ms.iloc[-1]),
        "macd_hist": float(mh.iloc[-1]), "macd_hist_prev": float(mh.iloc[-2]),
        "atr": float(atr.iloc[-1]), "volume": float(v.iloc[-1]), "vol20": float(v.iloc[-21:-1].mean()),
        "hi20": float(h.iloc[-20:].max()), "lo20": float(lo.iloc[-20:].min()),
        "hi60": float(h.iloc[-60:].max()), "lo60": float(lo.iloc[-60:].min()),
        "ret20": float(c.iloc[-1] / c.iloc[-21] - 1),
        "dollar_vol20": float((c.iloc[-20:] * v.iloc[-20:]).mean()),
    }
    for n in SMA_PERIODS:
        m[f"sma{n}"] = float(c.iloc[-n:].mean())
    sma20_s = c.rolling(20).mean()
    ext = (c - sma20_s) / atr
    m.update(
        prev_low=float(lo.iloc[-2]), prior_hi20=float(h.iloc[-21:-1].max()),
        rsi_max5=float(r.iloc[-5:].max()), rsi_max10=float(r.iloc[-10:].max()),
        ext_atr_max5=float(ext.iloc[-5:].max()), pct_above_sma20=float(c.iloc[-1] / sma20_s.iloc[-1] - 1),
        pct_above_sma20_max5=float((c / sma20_s - 1).iloc[-5:].max()),
        red=bool(c.iloc[-1] < c.iloc[-2]),
    )
    m.update(highs_compare(df, r, ml))
    if "open" in df.columns:
        m.update(gap_and_week_info(df))
    else:
        m.update(gaps=[], lw_close=np.nan, td=td_setup(c), failed_gap_up=False)
    if spy_close is not None and len(spy_close.dropna()) > 21:
        s = spy_close.dropna()
        m["rs20"] = m["ret20"] - float(s.iloc[-1] / s.iloc[-21] - 1)
    else:
        m["rs20"] = np.nan
    return m


def score(m: dict, side: str) -> tuple[int, list[str]]:
    """9 yes/no checks. Bullish and bearish are mirror images."""
    green = m["close"] >= m["prev_close"]
    vol_x = m["volume"] / m["vol20"] if m["vol20"] > 0 else 0
    if side == "bull":
        checks = [
            (m["close"] > m["sma10"], "above 10"),
            (m["sma10"] > m["sma20"], "10>20"),
            (m["sma20"] > m["sma50"], "20>50"),
            (m["close"] > m["sma200"], "above 200"),
            (55 <= m["rsi"] < 80, f"RSI {m['rsi']:.0f}"),
            (m["macd"] > m["macd_signal"] and m["macd_hist"] > m["macd_hist_prev"], "MACD rising"),
            (m["rs20"] > 0, f"{m['rs20']:+.1%} vs SPY"),
            (green and vol_x >= 1.2, f"green on {vol_x:.1f}x vol"),
            (m["close"] >= 0.97 * m["hi20"], "near 20d high"),
        ]
    else:
        checks = [
            (m["close"] < m["sma10"], "below 10"),
            (m["sma10"] < m["sma20"], "10<20"),
            (m["sma10"] < m["sma50"], "10<50 trend breaker"),
            (m["close"] < m["sma50"], "below 50"),
            (20 < m["rsi"] <= 45, f"RSI {m['rsi']:.0f}"),
            (m["macd"] < m["macd_signal"] and m["macd_hist"] < m["macd_hist_prev"], "MACD falling"),
            (m["rs20"] < 0, f"{m['rs20']:+.1%} vs SPY"),
            ((not green) and vol_x >= 1.5, f"red on {vol_x:.1f}x vol"),
            (m["close"] <= 1.03 * m["lo20"], "near 20d low"),
        ]
    return sum(1 for ok, _ in checks if ok), [lab for ok, lab in checks if ok]


def levels(m: dict) -> list[tuple[float, str]]:
    raw = [(m["day_high"], "day high"), (m["day_low"], "day low"),
           (m["hi20"], "20d high"), (m["lo20"], "20d low")]
    raw += [(m[f"sma{n}"], f"{n} SMA") for n in SMA_PERIODS]
    if not np.isnan(m.get("lw_close", np.nan)):
        raw.append((m["lw_close"], "last wk close"))
    c = m["close"]
    for g in m.get("gaps", []):  # the edge price reaches first is the level
        edge = g["lo"] if g["lo"] > c else g["hi"] if g["hi"] < c else None
        if edge is not None:
            raw.append((edge, f"{g['tf']} gap{' ⭐' if g['skipped'] else ''}"))
    span = m["hi60"] - m["lo60"]
    if span > 0:
        raw += [(m["hi60"] - r * span, f"fib {str(r)[1:]}") for r in FIB_RATIOS]
    raw.sort()
    merged: list[list] = []
    for p, lab in raw:  # merge levels within 0.25% of each other into one
        if merged and abs(p / merged[-1][0] - 1) < 0.0025:
            merged[-1][1] += f"/{lab}"
        else:
            merged.append([p, lab])
    return [(float(p), _clean_label(lab)) for p, lab in merged]


def _clean_label(lab: str) -> str:
    """'D gap/D gap ⭐/W gap/8 SMA' -> 'D gap ⭐/W gap/8 SMA' (one entry per kind, star kept)."""
    out: list[str] = []
    for part in lab.split("/"):
        base = part.replace(" ⭐", "")
        hit = next((i for i, x in enumerate(out) if x.replace(" ⭐", "") == base), None)
        if hit is None:
            out.append(part)
        elif "⭐" in part and "⭐" not in out[hit]:
            out[hit] = part
    return "/".join(out)


def split_levels(m: dict, lv: list[tuple[float, str]]) -> tuple[list, list]:
    res = [x for x in lv if x[0] > m["close"] * 1.001]
    sup = [x for x in lv if x[0] < m["close"] * 0.999][::-1]
    return res, sup


def idea(m: dict, side: str, res: list, sup: list) -> dict:
    """'Above X calls / below Y puts' with targets at the next levels and
    invalidation at the nearest level on the other side. If the nearest level
    is more than 1.5 ATR away, the day's high/low is the trigger instead."""
    a, c = m["atr"], m["close"]
    if side == "bull":
        trig = res[0] if res and res[0][0] - c <= 1.5 * a else (m["day_high"], "day high")
        beyond = [x for x in res if x[0] > trig[0] * 1.001]
        t1 = beyond[0] if beyond else (trig[0] + a, "+1 ATR")
        t2 = beyond[1] if len(beyond) > 1 else (t1[0] + a, "+1 ATR")
        inv = sup[0] if sup else (c - a, "-1 ATR")
        flip = sup[1] if len(sup) > 1 else (inv[0] - a, "-1 ATR")
    else:
        trig = sup[0] if sup and c - sup[0][0] <= 1.5 * a else (m["day_low"], "day low")
        beyond = [x for x in sup if x[0] < trig[0] * 0.999]
        t1 = beyond[0] if beyond else (trig[0] - a, "-1 ATR")
        t2 = beyond[1] if len(beyond) > 1 else (t1[0] - a, "-1 ATR")
        inv = res[0] if res else (c + a, "+1 ATR")
        flip = res[1] if len(res) > 1 else (inv[0] + a, "+1 ATR")
    return {"side": side, "trigger": trig, "t1": t1, "t2": t2, "invalid": inv, "flip_target": flip}


def exhaustion(m: dict) -> dict | None:
    tier = next((name for lvl, name in EXHAUSTION_TIERS if m["rsi"] >= lvl), None)
    if tier is None:
        return None
    return {
        "tier": tier,
        "extension_atr": (m["close"] - m["sma20"]) / m["atr"] if m["atr"] > 0 else np.nan,
        # price at/near its 20-day high while RSI is below its prior 20-day peak
        "divergence": m["close"] >= 0.98 * m["hi20"] and m["rsi"] < m["rsi_max_prior20"] - 3,
    }


# ---------------------------------------------------------------------------
# Gaps (Taz's rules), last week's close, TD Sequential setup
# ---------------------------------------------------------------------------
# A gap is the space between two candle BODIES (wicks ignored). It is filled
# only when a later solid body covers the whole gap; wicking into it or
# gapping across it leaves it open. A gap is "skipped" (top-priority magnet)
# when price gapped again in the same direction beyond it while it was open.
# Fills are checked on the same timeframe the gap formed on.
def body_gaps(df: pd.DataFrame, lookback: int, tf: str) -> list[dict]:
    df = df.dropna(subset=["open", "close"]).iloc[-lookback:]
    o, c, idx = df["open"].to_numpy(), df["close"].to_numpy(), df.index
    bh, bl = np.maximum(o, c), np.minimum(o, c)
    gaps = []
    for i in range(1, len(df)):
        if bl[i] > bh[i - 1]:
            gaps.append({"dir": "up", "lo": float(bh[i - 1]), "hi": float(bl[i]), "i": i})
        elif bh[i] < bl[i - 1]:
            gaps.append({"dir": "down", "lo": float(bh[i]), "hi": float(bl[i - 1]), "i": i})
    open_gaps = []
    for g in gaps:
        later = range(g["i"] + 1, len(df))
        if any(bl[j] <= g["lo"] and bh[j] >= g["hi"] for j in later):
            continue
        g["skipped"] = any(h["dir"] == g["dir"] and h["i"] > g["i"] and
                           (h["lo"] >= g["hi"] if g["dir"] == "up" else h["hi"] <= g["lo"]) for h in gaps)
        g["tf"], g["formed"] = tf, idx[g["i"]]
        open_gaps.append(g)
    return open_gaps


def weekly_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Completed weeks only: the current week counts once its Friday close is in."""
    w = df.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if len(df) and df.index[-1].dayofweek != 4:
        w = w.iloc[:-1]
    return w


def td_setup(close: pd.Series) -> tuple[str, int]:
    """DeMark setup: consecutive closes above (sell) / below (buy) the close 4 bars earlier.
    Returns the current run; 9+ means a setup completed. (Basic count, no perfection rules.)"""
    c = close.dropna().to_numpy()
    sell = buy = 0
    for i in range(4, len(c)):
        if c[i] > c[i - 4]:
            sell, buy = sell + 1, 0
        elif c[i] < c[i - 4]:
            buy, sell = buy + 1, 0
        else:
            sell = buy = 0
    return ("sell", sell) if sell >= buy else ("buy", buy)


def td_text(td: tuple[str, int]) -> str:
    side, n = td
    if n >= 9:
        return f"TD {side} 9 ✓" + (f" ({n - 9}d ago)" if n > 9 else "")
    return f"TD {side} {n}/9" if n else "TD none"


def gap_and_week_info(df: pd.DataFrame) -> dict:
    """Unfilled daily (120d) and weekly (52w) gaps, last completed week's close,
    TD setup, and today's failed gap-up (gap open above the prior 20-day high
    whose body filled back down through the gap -> put trigger)."""
    w = weekly_bars(df)
    d = body_gaps(df, 120, "D") + body_gaps(w, 52, "W")
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    prev_bh = max(o[-2], c[-2])
    prior_hi20 = float(df["high"].iloc[-21:-1].max())
    failed_up = bool(o[-1] > prev_bh and o[-1] > prior_hi20 and c[-1] < o[-1] and c[-1] <= prev_bh)
    return {"gaps": d, "lw_close": float(w["close"].iloc[-1]) if len(w) else np.nan, "prev_body_hi": float(prev_bh),
            "td": td_setup(df["close"]), "failed_gap_up": failed_up}


# ---------------------------------------------------------------------------
# Put-setup detection (Taz's daily put confirmations)
# ---------------------------------------------------------------------------
def highs_compare(df: pd.DataFrame, rsi_s: pd.Series, macd_s: pd.Series, recent: int = 5, prior: int = 30) -> dict:
    """Compare the highest high of the last `recent` bars with the highest high
    of the bars before that (back to `prior`).
      higher high in price + lower RSI at that high  -> RSI bearish divergence
      higher high in price + lower MACD at that high -> MACD bearish divergence
      recent high below the prior high               -> lower high"""
    h = df["high"]
    rec, pri = h.iloc[-recent:], h.iloc[-prior:-recent]
    ri, pi = rec.idxmax(), pri.idxmax()
    hh = rec.max() > pri.max()
    return {
        "rsi_div": bool(hh and rsi_s[ri] < rsi_s[pi] - 1),
        "macd_div": bool(hh and macd_s[ri] < macd_s[pi]),
        "lower_high": bool(rec.max() < pri.max()),
        "prior_high": float(pri.max()),
    }


def resistance_levels(m: dict) -> list[tuple[float, str]]:
    """Levels that act as resistance from below (today's own high excluded)."""
    lv = [(m["prior_hi20"], "20d high"), (m["lw_close"], "last wk close")]
    lv += [(m[f"sma{n}"], f"{n} SMA") for n in (20, 50, 200)]
    span = m["hi60"] - m["lo60"]
    if span > 0:
        lv += [(m["hi60"] - r * span, f"fib {str(r)[1:]}") for r in FIB_RATIOS]
    for g in m.get("gaps", []):
        if g["lo"] > m["close"]:  # an open gap overhead: its bottom edge is resistance
            lv.append((g["lo"], f"{g['tf']} gap{' ⭐' if g['skipped'] else ''}"))
    return [(p, lab) for p, lab in lv if p == p]


def rejection(m: dict) -> tuple[float, str] | None:
    """Today's high reached a resistance level (within 0.2%) but the close
    finished below it. Returns the highest such level."""
    hits = [(p, lab) for p, lab in resistance_levels(m)
            if m["day_high"] >= p * 0.998 and m["close"] < p * 0.999]
    return max(hits) if hits else None

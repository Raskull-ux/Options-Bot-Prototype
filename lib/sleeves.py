"""
The five signal sleeves and the regime labels (research basis:
RESEARCH_RECORD_v2_OPUS.md). Pure functions over panels so each one can be
unit-tested against synthetic data with a known right answer.

Conventions
-----------
close, volume : DataFrame, index = trading dates (sorted), columns = symbols
ar            : market-adjusted daily return (stock return minus SPY return)
today         : last date in the panel (the bot runs after the close)
Every signal is a dict with the same keys (see SIGNAL_COLUMNS). Each sleeve
also returns a status string so a quiet day is explained, never silent.
"""
import json

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import config
from lib.signals import amihud_illiquidity, cross_sectional_dispersion, percentile_rank_of_latest

SIGNAL_COLUMNS = [
    "date", "entry_session", "sleeve", "symbol", "direction", "structure",
    "hold_days", "exit_rule", "evidence", "deviations", "inputs", "conflict", "alert",
]


def mark_conflicts(signals: list[dict]) -> int:
    """[eng] Same symbol, same day, opposite directions (e.g. reversal calls vs
    drift puts): both rows get conflict=True and the paper engine skips them.
    Novy-Marx's regressions show the two effects coexist and pull opposite ways,
    so neither row is a clean bet."""
    dirs: dict[str, set] = {}
    for s in signals:
        if s["direction"] in ("bullish", "bearish"):
            dirs.setdefault(s["symbol"], set()).add(s["direction"])
    clashing = {sym for sym, d in dirs.items() if len(d) > 1}
    for s in signals:
        s["conflict"] = s["symbol"] in clashing and s["direction"] in ("bullish", "bearish")
        # alert = goes to Discord. Conflicts never alert; shadow sleeves only log.
        s["alert"] = (not s["conflict"]) and s["sleeve"] not in config.SHADOW_SLEEVES
    return len(clashing)

BULL_STRUCT = "long_call_or_call_debit_spread"
BEAR_STRUCT = "long_put_or_put_debit_spread"


def simple_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close / close.shift(1) - 1


def market_adjusted(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.sub(returns["SPY"], axis=0)


def next_session(today: pd.Timestamp) -> pd.Timestamp:
    # Business-day approximation: does not know exchange holidays (disclosed).
    return (today + BDay(1)).normalize()


def _signal(today, sleeve, symbol, direction, structure, hold_days, exit_rule, evidence, deviations, inputs) -> dict:
    return {
        "date": today.strftime("%Y-%m-%d"),
        "entry_session": next_session(today).strftime("%Y-%m-%d"),
        "sleeve": sleeve,
        "symbol": symbol,
        "direction": direction,
        "structure": structure,
        "hold_days": hold_days,
        "exit_rule": exit_rule,
        "evidence": evidence,
        "deviations": deviations,
        "inputs": json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in inputs.items()}),
    }


def _hour(h) -> str:
    return str(h).strip().lower() if isinstance(h, str) else ""


# ---------------------------------------------------------------------------
# Sleeve 1: earnings reversal (Jansen & Nikiforov 2016)
# ---------------------------------------------------------------------------
def earnings_reversal(ar: pd.DataFrame, earnings: pd.DataFrame, universe: set, today: pd.Timestamp) -> tuple[list, str]:
    nxt = next_session(today)
    out, considered = [], 0
    if earnings.empty:
        return out, "no_earnings_data"
    e = earnings[earnings["symbol"].isin(universe)].copy()
    e["ed"] = pd.to_datetime(e["earnings_date"], errors="coerce").dt.normalize()
    for _, row in e.dropna(subset=["ed"]).iterrows():
        sym, d, hour = row["symbol"], row["ed"], _hour(row.get("hour"))
        if hour == "amc":
            entry, deviation = d, ""
        else:
            entry = (d - BDay(1)).normalize()
            deviation = f"{hour or 'unknown'}-time report: entered day -1, screen window days -6..-2"
        if entry != nxt or sym not in ar.columns:
            continue
        considered += 1
        window = ar[sym].loc[:today].dropna().iloc[-config.EARN_REV_WINDOW_DAYS:]
        if len(window) < config.EARN_REV_WINDOW_DAYS:
            continue
        car = float(window.sum())
        if abs(car) < config.EARN_REV_THRESHOLD:
            continue
        bearish = car > 0  # extreme winners reverse down, extreme losers reverse up
        out.append(_signal(
            today, "earnings_reversal", sym,
            "bearish" if bearish else "bullish", BEAR_STRUCT if bearish else BULL_STRUCT,
            int(len(pd.bdate_range(entry, d + BDay(1)))),
            "exit at close of day +1 after the announcement",
            "moderate: one study, equity-only; earnings IV crush untested",
            "; ".join(x for x in [deviation, "abnormal return = market-adjusted (minus SPY)"] if x),
            {"pre_announcement_abnormal_return": car, "earnings_date": d.strftime("%Y-%m-%d"), "hour": hour,
             "thresholds_hit": [t for t in config.EARN_REV_LOG_THRESHOLDS if abs(car) >= t]},
        ))
    return out, f"{considered} entering tomorrow screened, {len(out)} fired"


# ---------------------------------------------------------------------------
# Sleeve 2: post-earnings drift (Novy-Marx: CAR3 continuation, monthly)
# ---------------------------------------------------------------------------
def post_earnings_drift(ar: pd.DataFrame, earnings: pd.DataFrame, universe: set, today: pd.Timestamp) -> tuple[list, str]:
    if earnings.empty:
        return [], "no_earnings_data"
    idx = ar.index
    t_pos = len(idx) - 1
    lo, hi = config.PEAD_CAR_WINDOW
    pool, todays = [], []
    e = earnings[earnings["symbol"].isin(universe)].copy()
    e["ed"] = pd.to_datetime(e["earnings_date"], errors="coerce").dt.normalize()
    for _, row in e.dropna(subset=["ed"]).drop_duplicates(["symbol", "ed"]).iterrows():
        sym, d = row["symbol"], row["ed"]
        if sym not in ar.columns:
            continue
        pos = idx.searchsorted(d)  # first trading day on/after the announcement date
        if pos >= len(idx) or pos + hi > t_pos or pos + lo < 0:
            continue
        if t_pos - (pos + hi) > config.PEAD_POOL_LOOKBACK_DAYS:
            continue
        vals = ar[sym].iloc[pos + lo: pos + hi + 1]
        if vals.isna().any() or len(vals) != hi - lo + 1:
            continue
        car3 = float(vals.sum())
        pool.append(car3)
        if pos + hi == t_pos:  # window completes today
            todays.append((sym, d, car3))
    if len(pool) < config.PEAD_MIN_POOL:
        return [], f"insufficient_pool ({len(pool)} < {config.PEAD_MIN_POOL}); {len(todays)} completed today, none ranked"
    pool_arr = np.array(pool)
    out = []
    for sym, d, car3 in todays:
        pct = float((pool_arr <= car3).mean())
        if pct >= 1 - config.PEAD_TAIL_PERCENTILE:
            direction, struct = "bullish", BULL_STRUCT
        elif pct <= config.PEAD_TAIL_PERCENTILE:
            direction, struct = "bearish", BEAR_STRUCT
        else:
            continue
        out.append(_signal(
            today, "post_earnings_drift", sym, direction, struct, config.PEAD_HOLD_DAYS,
            f"exit after {config.PEAD_HOLD_DAYS} trading days",
            "strong statistically (monthly, Novy-Marx); one study finds the drift is mostly eaten by trading costs",
            "CAR3 market-adjusted (minus SPY); decile computed within the bot's own recent-announcer pool",
            {"car3": car3, "pool_percentile": pct, "pool_size": len(pool), "earnings_date": d.strftime("%Y-%m-%d")},
        ))
    return out, f"pool {len(pool)}, {len(todays)} completed today, {len(out)} fired"


# ---------------------------------------------------------------------------
# Sleeve 3: short-term reversal, calls only (Da-Liu-Schaumburg; Jacobs)
# ---------------------------------------------------------------------------
def short_term_reversal(close: pd.DataFrame, universe: set, sector_map: dict, today: pd.Timestamp,
                        vix_above_median, earnings: pd.DataFrame | None = None) -> tuple[list, str]:
    idx = close.index
    if len(idx) < config.STR_FORMATION_DAYS + 3:
        return [], "insufficient_history"
    if idx[-1].month == idx[-2].month:
        return [], "not_month_start (fires on the first trading day of each month)"
    if config.STR_REQUIRE_VIX_ABOVE_MEDIAN:
        if vix_above_median is None:
            return [], "no_vix_data (cannot apply the Jacobs high-VIX condition, so not firing)"
        if not vix_above_median:
            return [], "vix_below_median (Jacobs: monthly reversal earns ~0 here)"
    end, start = -2, -2 - config.STR_FORMATION_DAYS  # prior month, ending yesterday's close
    # [eng] DLS's strong (residual) version strips cash-flow news; the closest
    # available approximation is excluding names that reported in the window.
    reported = set()
    if earnings is not None and not earnings.empty:
        ed = pd.to_datetime(earnings["earnings_date"], errors="coerce")
        in_win = (ed >= idx[start]) & (ed <= idx[end])
        reported = set(earnings.loc[in_win, "symbol"])
    rows = []
    for sym in universe:
        if sym not in close.columns or sym in reported:
            continue
        a, b = close[sym].iloc[start], close[sym].iloc[end]
        if pd.isna(a) or pd.isna(b) or a <= 0:
            continue
        etf, real = sector_map.get(sym, ("SPY", False))
        if etf not in close.columns:
            etf, real = "SPY", False
        ea, eb = close[etf].iloc[start], close[etf].iloc[end]
        if pd.isna(ea) or pd.isna(eb):
            continue
        rows.append((sym, b / a - 1 - (eb / ea - 1), etf, real))
    if not rows:
        return [], "no_scorable_symbols"
    df = pd.DataFrame(rows, columns=["symbol", "adj_ret", "etf", "real_sector"])
    cutoff = df["adj_ret"].quantile(config.STR_TAIL_PERCENTILE)
    decile = df[df["adj_ret"] <= cutoff].sort_values("adj_ret")
    picked = decile.head(config.STR_MAX_SIGNALS)
    out = [
        _signal(
            today, "short_term_reversal", r.symbol, "bullish", BULL_STRUCT, config.STR_HOLD_DAYS,
            f"exit after {config.STR_HOLD_DAYS} trading days",
            "moderate: monthly reversal, strongest when VIX is elevated (Jacobs); loser side only (DLS Table 8)",
            "sector-adjusted; names that reported earnings in the formation month excluded as a stand-in for "
            "the paper's removal of cash-flow news (analyst revisions unavailable)"
            + ("" if r.real_sector else "; no real sector match, adjusted vs SPY"),
            {"sector_adjusted_prior_month_return": float(r.adj_ret), "sector_etf": r.etf,
             "decile_size": int(len(decile)), "decile_cutoff": float(cutoff)},
        )
        for r in picked.itertuples()
    ]
    return out, (f"{len(reported & universe)} excluded for earnings in window; decile {len(decile)} names, "
                 f"{len(out)} alerted (cap {config.STR_MAX_SIGNALS})")


# ---------------------------------------------------------------------------
# Sleeve 4: next-day volume reversal (Llorente et al. 2002) -- hypothesis
# ---------------------------------------------------------------------------
def volume_reversal(close: pd.DataFrame, volume: pd.DataFrame, ar: pd.DataFrame, universe: set,
                    today: pd.Timestamp) -> tuple[list, str]:
    syms = [s for s in universe if s in close.columns and s in volume.columns]
    dv = (close[syms] * volume[syms]).iloc[-20:].mean()
    dv = dv.dropna()
    if dv.empty:
        return [], "no_dollar_volume_data"
    large = set(dv[dv >= dv.quantile(config.VOLREV_LARGE_TERCILE)].index)
    base_n = config.VOLREV_VOLUME_BASELINE_DAYS
    cands = []
    for sym in large:
        v = volume[sym]
        if v.iloc[-base_n - 1:].isna().any() or len(v) < base_n + 1:
            continue
        ratio = float(v.iloc[-1] / v.iloc[-base_n - 1:-1].mean())
        sd = float(ar[sym].iloc[-21:-1].std())
        move = ar[sym].iloc[-1]
        if pd.isna(move) or not sd or np.isnan(sd):
            continue
        if ratio >= config.VOLREV_VOLUME_RATIO and abs(move) >= config.VOLREV_MOVE_SIGMA * sd:
            cands.append((sym, float(move), ratio, abs(move) / sd))
    cands.sort(key=lambda x: -x[3])
    out = []
    for sym, move, ratio, z in cands[: config.VOLREV_MAX_SIGNALS]:
        bearish = move > 0
        out.append(_signal(
            today, "volume_reversal", sym, "bearish" if bearish else "bullish",
            BEAR_STRUCT if bearish else BULL_STRUCT, config.VOLREV_HOLD_DAYS,
            "exit at next session's close",
            "hypothesis: mechanism only (1993-98 data), no P&L ever measured",
            "entry is next open, so part of the close-to-close effect the paper measures is missed; "
            "'large' = top third by dollar volume (proxy for size/spread/coverage)",
            {"abnormal_move": move, "volume_vs_200d": ratio, "move_in_sd": z},
        ))
    return out, f"{len(large)} large names checked, {len(cands)} qualified, {len(out)} alerted"


# ---------------------------------------------------------------------------
# Sleeve 5: SPY volatility straddle (Johnson 2017)
# ---------------------------------------------------------------------------
def index_volatility(vix: pd.DataFrame, today: pd.Timestamp) -> tuple[list, str]:
    if vix is None or vix.empty:
        return [], "no_vix_data"
    v = vix.loc[:today]
    if v.empty:
        return [], "no_vix_data_through_today"
    last_date = v.index[-1]
    last = v.iloc[-1]
    stale = (today - last_date).days > config.VIX_STALE_DAYS
    if stale:
        return [], f"vix_data_stale (latest {last_date.date()})"
    inverted = last["vix"] > last["vix3m"]
    if not inverted:
        return [], f"contango (VIX {last['vix']:.2f} < VIX3M {last['vix3m']:.2f})"
    newly = len(v) > 1 and not (v.iloc[-2]["vix"] > v.iloc[-2]["vix3m"])
    return [_signal(
        today, "index_volatility", "SPY", "long_volatility", "spy_atm_straddle", config.IDXVOL_MAX_HOLD_DAYS,
        f"exit when VIX curve returns to contango, or after {config.IDXVOL_MAX_HOLD_DAYS} trading days",
        "moderate: index-level (Johnson 2017, SPX 1996-2013); long vol loses on average outside inversion",
        "",
        {"vix": float(last["vix"]), "vix3m": float(last["vix3m"]), "slope": float(last["vix3m"] - last["vix"]),
         "newly_inverted": bool(newly), "vix_date": last_date.strftime("%Y-%m-%d")},
    )], "inverted"


# ---------------------------------------------------------------------------
# Regime labels (recorded on every signal; not entry triggers)
# ---------------------------------------------------------------------------
def regime_labels(close: pd.DataFrame, volume: pd.DataFrame, universe: list, vix: pd.DataFrame | None,
                  today: pd.Timestamp) -> dict:
    lab = {"date": today.strftime("%Y-%m-%d")}
    spy = close["SPY"].dropna() if "SPY" in close.columns else pd.Series(dtype=float)
    n = config.PANIC_BEAR_LOOKBACK_DAYS
    if len(spy) > n:
        r24 = float(spy.iloc[-1] / spy.iloc[-n - 1] - 1)
        lab["spy_return_24m"] = r24
        lab["bear_24m"] = r24 < 0
    else:
        lab["spy_return_24m"] = np.nan
        lab["bear_24m"] = None
    lr = np.log(spy).diff()
    rv = lr.rolling(config.PANIC_VAR_WINDOW_DAYS, min_periods=10).var()
    var_pct = percentile_rank_of_latest(rv, len(rv)) if len(rv.dropna()) else np.nan
    lab["market_var_percentile"] = var_pct
    high_var = (not np.isnan(var_pct)) and var_pct >= config.PANIC_VAR_PERCENTILE
    lab["panic_state"] = None if lab["bear_24m"] is None else bool(lab["bear_24m"] and high_var)

    syms = [s for s in universe if s in close.columns and s in volume.columns]
    rets = simple_returns(close[syms])
    amihud = pd.DataFrame({s: amihud_illiquidity(rets[s], close[s] * volume[s]) for s in syms})
    illiq_pct = percentile_rank_of_latest(amihud.mean(axis=1, skipna=True), config.ILLIQUIDITY_PERCENTILE_LOOKBACK_DAYS)
    disp_pct = percentile_rank_of_latest(cross_sectional_dispersion(rets), config.DISPERSION_PERCENTILE_LOOKBACK_DAYS)
    lab["illiquidity_percentile"] = illiq_pct
    lab["illiquidity_high"] = bool(not np.isnan(illiq_pct) and illiq_pct >= config.ILLIQUIDITY_PERCENTILE_THRESHOLD)
    lab["dispersion_percentile"] = disp_pct
    lab["dispersion_high"] = bool(not np.isnan(disp_pct) and disp_pct >= config.DISPERSION_PERCENTILE_THRESHOLD)

    if vix is not None and not vix.empty and not vix.loc[:today].empty:
        v = vix.loc[:today]
        last = v.iloc[-1]
        med = float(v["vix"].iloc[-config.VIX_MEDIAN_LOOKBACK_DAYS:].median())
        lab.update(vix=float(last["vix"]), vix3m=float(last["vix3m"]), vix_slope=float(last["vix3m"] - last["vix"]),
                   vix_inverted=bool(last["vix"] > last["vix3m"]), vix_10y_median=med,
                   vix_above_median=bool(last["vix"] > med), vix_date=v.index[-1].strftime("%Y-%m-%d"))
    else:
        lab.update(vix=np.nan, vix3m=np.nan, vix_slope=np.nan, vix_inverted=None, vix_10y_median=np.nan,
                   vix_above_median=None, vix_date=None)
    return lab

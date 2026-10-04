"""
Replay the five sleeves over the last N trading days, as if the bot had run
after each close -- each day sees ONLY data up to that day (no lookahead).
Then report what each signal's underlying actually did over its hold:
entry at the next session's OPEN, exit at the CLOSE of the last hold day.

This measures the STOCK move in the signal's direction, not option P&L.
Option P&L (premium, spread, IV change) comes with the backtest. A right-way
stock move is necessary for an option win but not sufficient.

Disclosed limits:
  - universe = today's membership list (a symbol that qualified later may
    appear on an earlier date: mild survivorship)
  - earnings dates/hours are as the calendar shows them now
Usage: python replay_signals.py [N_trading_days]   (default 10)
"""
import sys

import numpy as np
import pandas as pd

import config
from lib import sleeves
from lib.symbol_filter import load_excluded_symbols
import signal_engine as se
from lib.earnings_data import load_earnings


def outcome(sig: dict, p: dict) -> dict:
    close, opn = p["close"], p["open"]
    idx = close.index
    d = pd.Timestamp(sig["date"])
    after = idx[idx > d]
    if len(after) == 0:
        return {"status": "not_entered_yet"}
    entry = after[0]
    e_pos = idx.get_loc(entry)
    x_pos = e_pos + int(sig["hold_days"]) - 1
    status = "closed"
    if x_pos > len(idx) - 1:
        x_pos, status = len(idx) - 1, "still_open"
    sym = sig["symbol"]
    o, c = opn[sym].iloc[e_pos], close[sym].iloc[x_pos]
    if pd.isna(o) or pd.isna(c) or o <= 0:
        return {"status": "missing_prices"}
    ret = float(c / o - 1)
    res = {"status": status, "entry_date": entry.strftime("%Y-%m-%d"),
           "exit_date": idx[x_pos].strftime("%Y-%m-%d"), "stock_return": ret}
    if sig["direction"] == "bullish":
        res["return_in_signal_direction"] = ret
    elif sig["direction"] == "bearish":
        res["return_in_signal_direction"] = -ret
    else:  # straddle: profits from size of move vs what was paid; needs option prices
        res["abs_move"] = abs(ret)
    return res


def main(n_days: int) -> int:
    p = se.load_panels()
    bars = pd.read_csv(config.BARS_FILE, parse_dates=["date"])
    p["open"] = bars.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
    close = p["close"]
    membership = pd.read_csv(config.UNIVERSE_MEMBERSHIP_FILE)
    universe = [s for s in sorted(membership["symbol"].unique()) if s in close.columns]
    excluded, have_types = load_excluded_symbols()
    if have_types:
        universe = [s for s in universe if s not in excluded]
    uni = set(universe)
    sector_map = se.load_sector_map()
    vix = se.load_vix()
    earnings = load_earnings()

    days = close.index[-n_days:]
    print(f"Replaying {len(days)} trading days: {days[0].date()} to {days[-1].date()}, universe {len(uni)}\n")
    all_rows = []
    for d in days:
        sub = {k: v.loc[:d] for k, v in p.items()}
        sv = vix.loc[:d] if vix is not None else None
        regime = sleeves.regime_labels(sub["close"], sub["volume"], universe, sv, d)
        ar = sleeves.market_adjusted(sleeves.simple_returns(sub["close"]))
        results = {
            "earnings_reversal": sleeves.earnings_reversal(ar, earnings, uni, d),
            "post_earnings_drift": sleeves.post_earnings_drift(ar, earnings, uni, d),
            "short_term_reversal": sleeves.short_term_reversal(sub["close"], uni, sector_map, d,
                                                               regime.get("vix_above_median"), earnings),
            "volume_reversal": sleeves.volume_reversal(sub["close"], sub["volume"], ar, uni, d),
            "index_volatility": sleeves.index_volatility(sv, d),
        }
        sigs = [r for rows, _ in results.values() for r in rows]
        sleeves.mark_conflicts(sigs)
        print(f"=== {d.date()}  (VIX {regime.get('vix')}, inverted={regime.get('vix_inverted')}, "
              f"above 10y median={regime.get('vix_above_median')}, panic={regime.get('panic_state')})")
        for name, (_, status) in results.items():
            print(f"   [{name}] {status}")
        for s in sigs:
            o = outcome(s, p)
            row = {**s, **o}
            all_rows.append(row)
            if "return_in_signal_direction" in o:
                perf = f"stock {o['stock_return']:+.2%} -> {o['return_in_signal_direction']:+.2%} in signal direction"
            elif "abs_move" in o:
                perf = f"SPY moved {o['stock_return']:+.2%} (straddle needs option prices)"
            else:
                perf = o["status"]
            print(f"   -> {s['sleeve']:20s} {s['symbol']:6s} {s['direction']:15s} "
                  f"{'[CONFLICT-skip] ' if s['conflict'] else ''}{'[shadow] ' if s['sleeve'] in config.SHADOW_SLEEVES else ''}{perf} [{o.get('status')}]")
        print()

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{config.DATA_DIR}/replay_signals.csv", index=False)
    print("===== SUMMARY =====")
    if df.empty:
        print("No signals in the window.")
        return 0
    tradeable = df[(~df["conflict"]) & df.get("return_in_signal_direction", pd.Series(dtype=float)).notna()]
    print(f"{len(df)} signals total, {int(df['conflict'].sum())} conflicts")
    for name, g in df.groupby("sleeve"):
        t = tradeable[tradeable["sleeve"] == name]
        line = f"  {name:20s} fired {len(g):3d}"
        if len(t):
            line += (f" | right-way {(t['return_in_signal_direction'] > 0).mean():.0%} of {len(t)}"
                     f" | avg {t['return_in_signal_direction'].mean():+.2%} in signal direction")
        print(line)
    print("\nThis is a handful of days of stock moves: a sanity check that the sleeves fire on real data,")
    print("NOT evidence of an edge (that needs option P&L and far more trades).")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10))

# PULLBACK CONFIRMATION ON INDEX/SECTOR ETFs (data the first study never touched)
#
# The pullback study found the quick pullback at the hourly 50 tag on QQQ + SPY
# (E0 vs random shorts +0.115 ATR, t=2.14), but NOT on single stocks. QQQ+SPY was one
# slice out of several, so it could be luck. If it's a real index effect, it should
# show up on OTHER index and sector ETFs that were not in that study.
#
# Same events, same doors, same windows and same conservative scoring as backtest_pullback.py.
# Note on the scoring: a candle that touches both target and stop counts as the stop, and for
# entries made during a candle only its close (not its low) can hit the target. That makes
# every average negative -- even random shorts. So only the COMPARISONS (setup vs random,
# setup vs no-cross control) mean anything, not the raw averages.
import sys
import traceback

import numpy as np
import pandas as pd

import config
import backtest_crosses as bx
import backtest_feature_lab as fl
import backtest_pullback as pb
from lib.alpaca_client import AlpacaClient

ETFS = ["IWM", "DIA", "XLK", "SMH", "XLF", "XLE", "XLV", "XLY", "XLI", "XLC", "XLP", "XLU", "KRE", "XBI"]

PASS_RULES = """PRE-REGISTERED (written before this run; QQQ and SPY are NOT included)
  Universe: IWM DIA XLK SMH XLF XLE XLV XLY XLI XLC XLP XLU KRE XBI
  Primary: door E0 (short at the tag of the hourly 50 after a 20/50 bear cross), 0.5 ATR bracket, 2 hours.
  The QQQ/SPY pullback is CONFIRMED only if ALL hold on these ETFs:
    1. beats RANDOM shorts (R0, same mechanics) with t >= 2
    2. beats the no-cross CONTROL (the same tag of the 50 with the 20 above the 50) with t >= 1.5
       -- this is the apples-to-apples test: same entry mechanics, only the bear cross differs
    3. beats its own random on more than half of the ETFs with 5+ setups
    4. the hour-by-hour path is in your favor at the close of the tag candle and at +1h
  (The old rule 'average > 0' is dropped: under this scoring even random shorts average below 0,
   so it could never pass. That was my error in the first study, fixed here before running.)"""


def main():
    print(PASS_RULES)
    client = AlpacaClient()
    bx.log(f"hourly bars (24h) for {len(ETFS)} ETFs since {bx.HOURLY_FROM}...")
    day = bx.fetch_bars(client, ETFS, "1Hour", bx.HOURLY_FROM)
    day = day[(day["t"].dt.hour >= 4) & (day["t"].dt.hour < 20)]
    night = bx.fetch_bars(client, ETFS, "1Hour", bx.HOURLY_FROM, feed="boats")
    night = night[(night["t"].dt.hour >= 20) | (night["t"].dt.hour < 4)]
    hb = pd.concat([day, night]).drop_duplicates(["symbol", "t"]).sort_values(["symbol", "t"]).reset_index(drop=True)
    bx.log(f"  -> {len(hb)} hourly candles; symbols with data: {sorted(hb['symbol'].unique())}")
    test_from = pd.Timestamp(bx.TEST_FROM, tz=bx.ET)
    rows = []
    for s, g in hb.groupby("symbol"):
        rows += pb.symbol_rows(s, fl.prep_hourly(g), test_from)
    df = pd.DataFrame(rows)
    df.to_csv(f"{config.DATA_DIR}/pullback_rows_etf.csv.gz", index=False)
    R0 = df[df.door.str.startswith("R0")]
    R1 = df[df.door.str.startswith("R1")]
    pb.describe("ETF CONFIRMATION (no QQQ, no SPY)", df, R0, R1, df[df.kind == "control"])

    print("\n==================== PER ETF: E0 at 2h, setup minus its own random ====================")
    s = df[(df.kind == "setup") & (df.door == "E0 tag")]
    wins, counted = 0, 0
    for sym in sorted(df.symbol.unique()):
        ss, rr = s[s.symbol == sym], R0[R0.symbol == sym]
        if len(ss) < 5:
            print(f"  {sym:5s} n={len(ss)} (too few)")
            continue
        d = ss["brk2"].mean() - rr["brk2"].mean()
        counted += 1
        wins += d > 0
        print(f"  {sym:5s} n={len(ss):3d}  setup {ss['brk2'].mean():+.3f}  random {rr['brk2'].mean():+.3f}  diff {d:+.3f} ATR"
              f"  win {(ss['brk2'] > 0).mean():5.1%} vs {(rr['brk2'] > 0).mean():5.1%}")

    c = df[(df.kind == "control") & (df.door == "E0 tag")]
    dd, tt = bx.cdiff(s["brk2"], s["trade_day"], R0["brk2"], R0["trade_day"])
    dc, tc = bx.cdiff(s["brk2"], s["trade_day"], c["brk2"], c["trade_day"])
    p0, p1 = s["p0"].mean(), s["p1"].mean()
    checks = [tt == tt and tt >= 2, tc == tc and tc >= 1.5, counted > 0 and wins / counted > 0.5, p0 > 0 and p1 > 0]
    print(f"\n>>> PRIMARY (E0, {pb.BRACKET} ATR, 2h): n={len(s)} · vs random {dd:+.3f} ATR (t={tt:+.2f}) · "
          f"vs control {dc:+.3f} (t={tc:+.2f}) · ETFs beating their random {wins}/{counted} · path tag-close {p0:+.3f}, +1h {p1:+.3f}")
    print(f">>> {'CONFIRMED' if all(checks) else 'NOT CONFIRMED'}  [1 beats random t>=2: {checks[0]} · 2 beats control t>=1.5: {checks[1]} · "
          f"3 most ETFs: {checks[2]} · 4 path in favor: {checks[3]}]")
    print("\nAll rows: data/pullback_rows_etf.csv.gz")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

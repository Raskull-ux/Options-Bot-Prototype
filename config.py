"""
Central configuration for the day-to-5-day options bot.

Nothing here is a magic number pulled from nowhere -- every threshold traces
back to a specific paper in COMPLETE_RESEARCH_RECORD.md, or to a concrete
design reason noted inline.
"""

# ---------------------------------------------------------------------------
# Data feeds (locked in from the entitlements check on 2026-09-28)
# ---------------------------------------------------------------------------
STOCK_BARS_FEED = "sip"          # full consolidated volume; free for daily bars
                                   # (only the most recent 15 min is restricted
                                   # on the free Basic plan -- irrelevant for
                                   # end-of-day daily bars)
OPTION_QUOTE_FEED = "indicative"  # real bid/ask, but no real IV/greeks fields
                                   # (OPRA/Algo Trader Plus not subscribed --
                                   # revisit if that changes)
IV_IS_APPROXIMATE = True          # every IV this bot produces is derived by
                                   # inverting Black-Scholes on indicative
                                   # quotes, NOT real OPRA IV. This flag must
                                   # propagate into every log/alert/record.

RISK_FREE_RATE = 0.045            # approximation input for Black-Scholes;
                                   # update periodically from real T-bill yield
DIVIDEND_YIELD_ASSUMPTION = 0.0   # short-dated options on non-dividend-focused
                                   # names; flagged, not hidden

# ---------------------------------------------------------------------------
# Universe: NO hardcoded ticker list. The real optionable US equity universe
# is pulled fresh from Alpaca every run (see lib/alpaca_client.get_all_assets
# and daily_collect.build_eligible_universe). Tickers come and go on their
# own -- new listings and delistings are picked up automatically because the
# source is Alpaca's live asset list, not a list Claude wrote once.
#
# The only filter applied here is LIQUIDITY, not size or "boringness":
#   - price floor keeps out penny/junk stocks
#   - dollar-volume floor keeps out anything too thin to trade with tight
#     option spreads
# This says nothing about market cap. A mega-cap and a mid-cap that both
# clear this bar are both in the pool; which ones actually get selected for
# a trade is a downstream ranking decision (beta, idio vol, etc.) in the
# Phase 2 signal engine, not something baked into universe membership.
# ---------------------------------------------------------------------------
MIN_PRICE = 10.0                  # excludes penny/junk stocks
MIN_AVG_DOLLAR_VOLUME = 20_000_000  # 20-day avg price*volume; real liquidity
LIQUIDITY_LOOKBACK_DAYS = 35      # calendar days of bars pulled for the
                                    # liquidity/unusual-activity screen --
                                    # comfortably clears 20 trading days after
                                    # weekends/holidays are subtracted out

# Reference instruments for regime gates and beta/sector adjustment --
# NOT traded directly, kept separate from the tradable universe.
REFERENCE_ETFS = ["SPY", "XLK", "XLF", "XLE", "XLY", "XLV", "XLI", "XLB", "IWM", "VIXY"]

# ---------------------------------------------------------------------------
# Unusual-activity ("viral mover") detection.
#
# A move is flagged as unusual if EITHER:
#   (a) today's volume >= REL_VOLUME_SPIKE_THRESHOLD x its own 20-day average
#       volume, OR
#   (b) |today's return| >= max(MOVE_ABS_FLOOR, MOVE_VOL_MULTIPLE x that
#       stock's own trailing 20-day daily return volatility)
#
# (b) is deliberately volatility-NORMALIZED, not a flat percentage. A flat
# threshold treats every stock the same regardless of how much it normally
# moves, which either misses real outliers in quiet stocks or never fires
# for stocks that are volatile every day. MOVE_ABS_FLOOR exists only so a
# nearly-silent stock doesn't get flagged for a trivial move that happens to
# be "3x" its own near-zero volatility.
#
# A flagged name is promoted into the permanent working universe (see
# daily_collect.py) the same day it's flagged -- so a name having a moment
# doesn't need to already be on a list to get caught, and once caught it
# keeps accumulating real history going forward.
# ---------------------------------------------------------------------------
REL_VOLUME_SPIKE_THRESHOLD = 3.0
MOVE_ABS_FLOOR = 0.04
MOVE_VOL_MULTIPLE = 2.5
VOLATILITY_LOOKBACK_DAYS = 20

# ---------------------------------------------------------------------------
# Signal thresholds (from the research record)
# ---------------------------------------------------------------------------
TREND_STRENGTH_MIN_ABS = 0.80     # Dao et al. (2016) hard-capped breakeven
                                    # for a discrete single-contract bet;
                                    # |T| below this => expected P&L negative
HORIZON_CANDIDATES_DAYS = [1, 2, 3, 5, 10, 20]  # signature-plot horizons for
                                                  # this bot's own VR(T) calibration
MIN_HISTORY_DAYS_FOR_SCREEN = 260  # Lo-MacKinlay / panic-state percentile
                                     # calc wants ~260 trading days minimum

# ---------------------------------------------------------------------------
# Options selection for the daily IV-history snapshot (Phase 1 only builds
# the data layer -- this defines what we sample, not what we trade)
# ---------------------------------------------------------------------------
SNAPSHOT_MIN_DTE = 3        # Bhansali & Holdom tenor-matching: don't sample
SNAPSHOT_MAX_DTE = 10       # contracts far outside the 1-5 day signal horizon
SNAPSHOT_STRIKES_EACH_SIDE = 3  # strikes above and below spot to sample

# The full liquid universe can run 1,000-2,500+ names; pulling real options
# data (contracts + snapshots) on all of them daily burns API calls on names
# nobody's about to trade. Cap it, but ALWAYS prioritize anything flagged as
# unusual activity today -- that's exactly when real IV data on a mover is
# worth capturing. Remaining slots fill by highest relative volume, then
# highest realized volatility, until the cap is reached.
IV_SNAPSHOT_DAILY_CAP = 150

# ---------------------------------------------------------------------------
# Paths (all committed back to the repo by the daily workflow)
# ---------------------------------------------------------------------------
DATA_DIR = "data"
BARS_FILE = f"{DATA_DIR}/stock_bars.csv"
IV_SNAPSHOTS_FILE = f"{DATA_DIR}/iv_snapshots.csv"
EARNINGS_FILE = f"{DATA_DIR}/earnings_calendar.csv"
RUN_LOG_FILE = f"{DATA_DIR}/collector_run_log.csv"
UNUSUAL_ACTIVITY_FILE = f"{DATA_DIR}/unusual_activity.csv"
UNIVERSE_MEMBERSHIP_FILE = f"{DATA_DIR}/universe_membership.csv"

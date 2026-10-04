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
SPOT_PRICE_FEED = "iex"          # for a single "latest trade right now" lookup
                                   # (used for ATM strike selection and as the
                                   # Black-Scholes underlying price). SIP is
                                   # NOT usable here on the free plan -- a
                                   # real-time single-trade query hits the
                                   # exact restriction that daily historical
                                   # bars don't (confirmed 2026-09-28: HTTP 403
                                   # "subscription does not permit querying
                                   # recent SIP data"). Real-time IEX is free
                                   # and already confirmed working.
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
# NOT traded directly, kept separate from the tradable universe. Full 11
# GICS sector ETFs, since the liquid universe (unlike the old hardcoded seed
# list) naturally includes utilities, staples, communications, real estate.
REFERENCE_ETFS = ["SPY", "XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLI", "XLB", "XLU", "XLRE", "XLC", "VIXY"]

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
TREND_STRENGTH_MIN_ABS = 0.8805   # NOT the literal 0.80 from Dao et al. -- that
                                    # number was derived for a different, continuous
                                    # trend-signal formulation. Tested on THIS
                                    # implementation (2026-09-29): a literal 0.80
                                    # threshold let through ~47% of pure-noise trials
                                    # instead of the intended ~42% (a clean |T|>=0.80
                                    # normal-theory threshold's real meaning), because
                                    # normalizing by an ESTIMATED volatility on a small
                                    # sample produces fatter tails than a clean normal
                                    # reference. Widening the EWMA lambda did not fix
                                    # this (confirmed by testing, not assumed) -- the
                                    # excess variance is intrinsic to the small-sample
                                    # construction. This value is empirically calibrated
                                    # (lib/signals.calibrate_trend_threshold, 200k-trial
                                    # Monte Carlo, fixed seed, reproducible) to give the
                                    # SAME selectivity |T|=0.80 is supposed to represent,
                                    # while losing under 2 percentage points of power to
                                    # detect a real trend (88.6% vs 89.8% at a realistic
                                    # drift, tested). Re-run the calibration if
                                    # TREND_LOOKBACK_DAYS, TREND_SKIP_RECENT_DAYS, or
                                    # EWMA_VOL_LAMBDA ever change.
HORIZON_CANDIDATES_DAYS = [1, 2, 3, 5, 10, 20]  # signature-plot horizons for
                                                  # this bot's own VR(T) calibration
MIN_HISTORY_DAYS_FOR_SCREEN = 260  # Lo-MacKinlay / panic-state percentile
                                     # calc wants ~260 trading days minimum

# ---------------------------------------------------------------------------
# Options selection for the daily IV-history snapshot (Phase 1 only builds
# the data layer -- this defines what we sample, not what we trade)
# ---------------------------------------------------------------------------
SNAPSHOT_MIN_DTE = 7        # widened 2026-10-03: holds now run up to ~1 month
SNAPSHOT_MAX_DTE = 45
SNAPSHOT_TARGET_DTE = 30    # sample the nearest expiry >= MIN and the one closest to 30 DTE
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
SECTOR_PROFILES_FILE = f"{DATA_DIR}/sector_profiles.csv"
UNMAPPED_INDUSTRIES_FILE = f"{DATA_DIR}/unmapped_industries.csv"
SYMBOL_TYPES_FILE = f"{DATA_DIR}/symbol_types.csv"
SYMBOL_TYPE_MAX_AGE_DAYS = 30  # exchange listing composition changes slowly

# Real security types (from Finnhub's authoritative OpenFIGI-standard
# classification, confirmed via build_symbol_types.py on 2026-09-30) that
# are pooled/fund-like instruments, not individual companies with their own
# idiosyncratic risk -- the underlying-selection research (beta, idio vol,
# turnover, lottery demand) is about single companies and doesn't cleanly
# apply to these. REIT, ADR, MLP, Ltd Part, etc. are deliberately NOT
# excluded -- those are real individual entities, just with unusual
# corporate structures.
EXCLUDED_SECURITY_TYPES = {"ETP", "Closed-End Fund", "Open-End Fund"}

ATR_PERIOD = 14  # standard Wilder ATR lookback
DAILY_SIGNALS_FILE = f"{DATA_DIR}/daily_signals.csv"
REGIME_STATE_FILE = f"{DATA_DIR}/regime_state.csv"
SIGNALS_FILE = f"{DATA_DIR}/signals.csv"          # fired sleeve signals (what alerts are built from)
VIX_TERM_FILE = f"{DATA_DIR}/vix_term.csv"
EARNINGS_DAYS_BACK = 10           # daily pull window behind today (drift sleeve needs recent past)
EARNINGS_SEED_DAYS_BACK = 120     # one-time backfill so the drift pool isn't empty on day one
EDGAR_EARNINGS_FILE = f"{DATA_DIR}/edgar_earnings.csv"   # SEC 8-K Item 2.02 history (build_edgar_earnings.py)
EDGAR_SINCE = "2024-01-01"        # Alpaca's option history reaches back to about here

# Sleeves that keep running and logging but are NOT sent as alerts.
# volume_reversal: 286-trade replay (Jul-Oct 2026) hit 49.7%, +0.07% avg,
# t=0.38; no slice above t=0.87. Logged silently in case that changes.
# earnings_reversal: option backtest (123 priced trades, 2024-2026) lost in every
# year, direction and report timing: -16.1% avg gross, t(log)=-6.33; -24.1% at
# 10% costs, t=-7.32. Stock moved WITH the pre-earnings run (fade lost -2.6%,
# t=-2.74) and IV crush hit every position held through the report.
SHADOW_SLEEVES = {"volume_reversal", "earnings_reversal"}

# --- Earnings-sleeve option backtest (backtest_earnings.py) ---
BT_START = "2024-03-01"
BT_STOCK_LOOKBACK_DAYS = 700      # trading-day-ish lookback for the stock panel (covers BT_START + history)
BT_MIN_DAYS_AFTER_EXIT = {"earnings_reversal": 5, "post_earnings_drift": 7}  # [eng] don't hold into expiry week
BT_STRIKE_BAND = 0.10             # search strikes within +/-10% of spot, pick nearest to spot
BT_COST_LEVELS = [0.0, 0.05, 0.10, 0.20]  # round-trip cost as a fraction of premium (Pardo: stress at double)
BT_REQUEST_INTERVAL = 0.32        # seconds between Alpaca calls (~187/min, under the 200/min free limit)

# ---------------------------------------------------------------------------
# Signal engine (Phase 2)
# ---------------------------------------------------------------------------
# Trend signal construction
TREND_LOOKBACK_DAYS = 20          # window for the volatility-normalized trend score
TREND_SKIP_RECENT_DAYS = 2        # exclude the most recent 1-2 days (Goyal & Wahal, 2015 --
                                    # short-term reversal contamination right at the ranking edge)
EWMA_VOL_LAMBDA = 0.5             # RAMOM-style signal-construction vol (Dudler/Gmur/Malamud, 2015);
                                    # distinct from the 0.94 RiskMetrics lambda used for position sizing

# NOTE ON THE TREND-STRENGTH FORMULA: this is implemented from the research
# record's DESCRIPTION of Dao et al. (2016)'s construction, not from the
# paper's own equations (which aren't available here). It's a principled,
# volatility-normalized signal-to-noise construction consistent with the
# |T|~=0.80 breakeven the record describes -- treat it as a documented
# interpretation, not a verified line-for-line reproduction.

# ===========================================================================
# REVISION 2026-10-03: research basis is now RESEARCH_RECORD_v2_OPUS.md.
# The 1-5 day trend sleeve is REMOVED as an entry signal (v2: not supported;
# liquid stocks reverse at short horizons). VR/T are still computed and
# logged per symbol as diagnostics only. The old panic/crowding gates are
# replaced by regime LABELS (v2: monthly evidence = slow labels, not triggers).
# Every threshold below is tagged [paper] = taken from the cited study, or
# [eng] = an engineering choice of mine, disclosed, to be judged on paper data.
# ===========================================================================

# --- Regime labels --------------------------------------------------------
# Panic state, defined the way Daniel & Moskowitz (2016) define it:
# trailing 24-month market return < 0 AND high forecast market variance.
PANIC_BEAR_LOOKBACK_DAYS = 504          # [paper] 24 months of trading days
PANIC_VAR_WINDOW_DAYS = 21              # [eng] realized-variance proxy for D-M's GARCH forecast
PANIC_VAR_PERCENTILE = 0.50             # [eng] "high" = above median of available history
SPY_HISTORY_LOOKBACK_DAYS = 800         # [eng] SPY needs >504 trading days for the bear label

# Aggregate illiquidity + dispersion: recorded as labels on every signal.
ILLIQUIDITY_PERCENTILE_LOOKBACK_DAYS = 252
ILLIQUIDITY_PERCENTILE_THRESHOLD = 0.85
DISPERSION_PERCENTILE_LOOKBACK_DAYS = 252
DISPERSION_PERCENTILE_THRESHOLD = 0.85

# VIX term structure (Cboe primary, FRED backup -- both confirmed 2026-10-03)
VIX_MEDIAN_LOOKBACK_DAYS = 2520         # [eng] ~10y trailing median for Jacobs' high/low VIX split
VIX_STALE_DAYS = 4                      # flag if newest VIX row is older than this

# --- Sleeve 1: earnings reversal (Jansen & Nikiforov 2016, "Fear and Greed")
EARN_REV_WINDOW_DAYS = 5                # [paper] days -5..-1 before the announcement
EARN_REV_THRESHOLD = 0.10               # [paper] central |abnormal return| screen
EARN_REV_LOG_THRESHOLDS = [0.05, 0.10, 0.15]  # [paper] all three logged; only 10% fires
# Timing deviation (disclosed): the bot runs after the close. For after-close
# (amc) reports it enters day 0 and the screen is exactly days -5..-1. For
# before-open (bmo) or unknown-time reports it must enter on day -1 to be in
# before the release, so its screen is days -6..-2. Flagged on every row.

# --- Sleeve 2: post-earnings drift (Novy-Marx: CAR3 continuation, monthly)
PEAD_CAR_WINDOW = (-1, 1)               # [paper] CAR3 = abnormal return days -1..+1
PEAD_POOL_LOOKBACK_DAYS = 63            # [eng] cross-sectional pool of recent announcers
PEAD_MIN_POOL = 50                      # [eng] refuse to rank on a thinner pool
PEAD_TAIL_PERCENTILE = 0.10             # [paper-style] extreme deciles
PEAD_HOLD_DAYS = 21                     # [paper] monthly holding period

# --- Sleeve 3: short-term reversal, calls only (DLS 2014; Jacobs 2015)
STR_FORMATION_DAYS = 21                 # [paper] prior-month return
STR_HOLD_DAYS = 21                      # [paper] one-month hold
STR_TAIL_PERCENTILE = 0.10              # [paper-style] bottom decile, sector-adjusted
STR_MAX_SIGNALS = 10                    # [eng] a decile is ~190 names; alert the 10 most extreme
STR_REQUIRE_VIX_ABOVE_MEDIAN = True     # [paper] Jacobs: STR ~0 below median VIX

# --- Sleeve 4: next-day volume reversal (Llorente et al. 2002) -- HYPOTHESIS
VOLREV_LARGE_TERCILE = 2 / 3            # [eng] top third by dollar volume = "large, liquid" proxy
VOLREV_VOLUME_BASELINE_DAYS = 200       # [paper] 200-day volume baseline
VOLREV_VOLUME_RATIO = 2.0               # [eng] Llorente is a regression, no threshold given
VOLREV_MOVE_SIGMA = 2.5                 # [eng] move size vs 20-day abnormal-return SD
VOLREV_MAX_SIGNALS = 10                 # [eng]
VOLREV_HOLD_DAYS = 1                    # [paper] next-day effect

# --- Sleeve 5: SPY volatility straddle (Johnson 2017)
IDXVOL_MAX_HOLD_DAYS = 21               # [paper] next-day to next-month evidence

# IV rank: logged as a diagnostic only (v2: untested heuristic, not a gate)
IV_RANK_MIN_HISTORY_DAYS = 60

# Sector-profile cache: refreshed infrequently (industry classification
# rarely changes), NOT re-fetched every daily run
SECTOR_PROFILE_MAX_AGE_DAYS = 90

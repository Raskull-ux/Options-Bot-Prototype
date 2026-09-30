"""
Shared helper: which symbols are pooled/fund-type instruments (ETPs,
closed-end funds, etc.) rather than individual companies, per Finnhub's real
security-type data. Used at TWO points so the fix applies both going
forward and to what's already in the historical universe ledger:
  1. daily_collect.py -- so fund-type symbols never get newly admitted
  2. signal_engine.py -- so fund-type symbols already sitting in the
     append-only universe_membership.csv (from before this fix existed)
     don't get scored as candidates either

universe_membership.csv itself is intentionally NOT rewritten -- it's an
honest historical record of what qualified and when. The exclusion is
applied live at read-time in both places instead.
"""
import pandas as pd

import config


def load_excluded_symbols() -> tuple[set[str], bool]:
    """
    Returns (excluded_symbols, data_available). If symbol_types.csv doesn't
    exist yet, returns (empty set, False) -- callers should log this clearly
    rather than silently proceeding as if the check happened.
    """
    try:
        df = pd.read_csv(config.SYMBOL_TYPES_FILE)
    except FileNotFoundError:
        return set(), False
    excluded = set(df[df["security_type"].isin(config.EXCLUDED_SECURITY_TYPES)]["symbol"])
    return excluded, True

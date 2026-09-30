"""
Diagnostic, zero side effects: cross-references the real Finnhub security-
type data against data/universe_membership.csv to show exactly how many of
the current working universe are common stocks vs. ETFs/funds/other -- so
the exact filter rule gets decided from real numbers, not a guess.

Does NOT modify universe_membership.csv, daily_signals.csv, or anything
else. Run build_symbol_types.py first.
"""
import sys

import pandas as pd

import config


def main() -> int:
    try:
        types_df = pd.read_csv(config.SYMBOL_TYPES_FILE)
    except FileNotFoundError:
        print(f"FATAL: {config.SYMBOL_TYPES_FILE} doesn't exist -- run build_symbol_types.py first")
        return 1

    try:
        membership = pd.read_csv(config.UNIVERSE_MEMBERSHIP_FILE)
    except FileNotFoundError:
        print(f"FATAL: {config.UNIVERSE_MEMBERSHIP_FILE} doesn't exist -- run daily_collect.py first")
        return 1

    type_map = dict(zip(types_df["symbol"], types_df["security_type"]))
    membership = membership.copy()
    membership["security_type"] = membership["symbol"].map(type_map)
    n_unmatched = membership["security_type"].isna().sum()

    print(f"Working universe: {len(membership)} symbols")
    print(f"  -> {n_unmatched} not found in Finnhub's US symbol list at all (unusual -- worth a manual look)")
    print()
    print("Breakdown of the WORKING UNIVERSE by real security type:")
    counts = membership["security_type"].value_counts(dropna=False)
    for t, n in counts.items():
        pct = n / len(membership)
        print(f"  {str(t):30s} {n:5d}  ({pct:.1%})")

    print()
    print("=== Candidate filter rules, applied to the CURRENT universe ===")

    common_stock_only = membership[membership["security_type"] == "Common Stock"]
    print(f"Rule A -- 'Common Stock' only: {len(common_stock_only)} symbols kept, {len(membership) - len(common_stock_only)} excluded")

    lenient_types = {"Common Stock", "REIT", "ADR"}
    lenient = membership[membership["security_type"].isin(lenient_types)]
    print(f"Rule B -- Common Stock + REIT + ADR: {len(lenient)} symbols kept, {len(membership) - len(lenient)} excluded")

    excluded_under_a = membership[~membership.index.isin(common_stock_only.index)]
    print()
    print("Symbols Rule A would exclude, by type (first 40 shown):")
    print(excluded_under_a[["symbol", "security_type"]].head(40).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

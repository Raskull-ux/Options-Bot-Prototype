"""
Maps a company's real Finnhub industry classification to one of the 11 GICS
sector ETFs, for the sector-adjustment step in the trend signal
(Moskowitz & Grinblatt, 1999 -- subtract the stock's own sector return
before ranking on momentum).

IMPORTANT HONESTY NOTE: Finnhub does not publish a fixed, guaranteed-complete
list of the strings finnhubIndustry can return. The mapping below covers
industry strings that are well-documented and commonly seen in Finnhub's
data (verified against public API examples and third-party integrations),
but it is a STARTER set, not a verified-complete taxonomy. Anything that
doesn't match falls back to SPY (market-only adjustment) rather than being
guessed at -- and every fallback is logged to data/unmapped_industries.csv
so the mapping can be extended from real, observed values over time instead
of from assumptions.
"""

# industry string (as returned by Finnhub) -> sector ETF ticker
INDUSTRY_TO_ETF = {
    # Technology (XLK)
    "Technology": "XLK",
    "Semiconductors": "XLK",
    "Software": "XLK",
    "Computer Hardware": "XLK",
    "IT Services": "XLK",
    "Electronic Equipment": "XLK",
    # Financials (XLF)
    "Banking": "XLF",
    "Banks": "XLF",
    "Insurance": "XLF",
    "Capital Markets": "XLF",
    "Financial Services": "XLF",
    "Consumer Finance": "XLF",
    "Diversified Financial Services": "XLF",
    # Energy (XLE)
    "Oil & Gas": "XLE",
    "Oil, Gas & Consumable Fuels": "XLE",
    "Energy": "XLE",
    # Consumer Discretionary (XLY)
    "Auto Manufacturers": "XLY",
    "Automobiles": "XLY",
    "Retail": "XLY",
    "Specialty Retail": "XLY",
    "Hotels, Restaurants & Leisure": "XLY",
    "Leisure": "XLY",
    "Apparel": "XLY",
    "Homebuilding": "XLY",
    # Consumer Staples (XLP)
    "Beverages": "XLP",
    "Food Products": "XLP",
    "Packaged Foods": "XLP",
    "Household Products": "XLP",
    "Consumer products": "XLP",
    # Health Care (XLV)
    "Biotechnology": "XLV",
    "Pharmaceuticals": "XLV",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Medical Devices": "XLV",
    "Life Sciences Tools & Services": "XLV",
    # Industrials (XLI)
    "Aerospace & Defense": "XLI",
    "Airlines": "XLI",
    "Machinery": "XLI",
    "Industrial Conglomerates": "XLI",
    "Transportation": "XLI",
    "Construction": "XLI",
    # Materials (XLB)
    "Chemicals": "XLB",
    "Metals & Mining": "XLB",
    "Metal & Mining": "XLB",
    "Paper & Forest Products": "XLB",
    # Utilities (XLU)
    "Utilities": "XLU",
    "Electric Utilities": "XLU",
    # Real Estate (XLRE)
    "Real Estate": "XLRE",
    "REIT": "XLRE",
    # Communication Services (XLC)
    "Telecommunication": "XLC",
    "Media": "XLC",
    "Entertainment": "XLC",
    "Internet": "XLC",
}

FALLBACK_ETF = "SPY"  # market-only adjustment when the industry is unmapped


def map_industry_to_etf(industry: str | None) -> tuple[str, bool]:
    """
    Returns (etf_ticker, was_real_sector_match). If industry is missing or
    unrecognized, returns (FALLBACK_ETF, False) -- caller should log the raw
    industry string when was_real_sector_match is False, for future mapping.
    """
    if not industry:
        return FALLBACK_ETF, False
    etf = INDUSTRY_TO_ETF.get(industry.strip())
    if etf:
        return etf, True
    return FALLBACK_ETF, False

"""
Core signal math for the day-to-5-day options bot's signal engine.

Every function here is pure (numpy/pandas in, numbers out) so it can be
unit-tested against synthetic series with known statistical properties
before trusting it on real data.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lo & MacKinlay (1988) variance ratio, overlapping-return estimator
# ---------------------------------------------------------------------------
def variance_ratio(returns: np.ndarray, q: int) -> float:
    """
    VR(q) = Var(r_t(q)) / (q * Var(r_t(1))), using the standard overlapping
    q-period return estimator with Lo-MacKinlay's finite-sample normalizer m.
    Returns np.nan if there isn't enough data for the given q.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    N = len(returns)
    if N <= q or q < 1:
        return np.nan

    mu = returns.mean()
    var_a = np.sum((returns - mu) ** 2) / (N - 1)
    if var_a <= 0:
        return np.nan

    qret = pd.Series(returns).rolling(window=q).sum().dropna().values  # overlapping q-period returns
    m = q * (N - q + 1) * (1 - q / N)
    if m <= 0:
        return np.nan
    var_b = np.sum((qret - q * mu) ** 2) / m

    # NOTE: var_b, as normalized by m above, is ALREADY a per-single-period-
    # equivalent variance estimate at horizon q (the factor of q is baked
    # into m). VR(q) is therefore the direct ratio var_b/var_a -- NOT
    # var_b/(q*var_a). An earlier version of this function had an extra,
    # incorrect division by q here, which was caught and fixed by testing
    # against a synthetic random walk (VR should be ~1.0 at every horizon
    # for a true random walk -- it was showing ~1/q instead).
    return var_b / var_a


# ---------------------------------------------------------------------------
# RAMOM-style volatility normalization (Dudler, Gmur & Malamud, 2015)
# ---------------------------------------------------------------------------
def ewma_vol_normalize(returns: pd.Series, lam: float) -> pd.Series:
    """
    Normalizes each return by an EWMA estimate of that day's OWN volatility,
    using only PRIOR returns (no lookahead): a given-size move on a calm day
    counts more than the same move on a chaotic day.
    """
    r = returns.to_numpy(dtype=float)
    n = len(r)
    ewma_var = np.full(n, np.nan)
    normalized = np.full(n, np.nan)
    if n == 0:
        return pd.Series(normalized, index=returns.index)

    first_valid = next((i for i in range(n) if not np.isnan(r[i])), None)
    if first_valid is None:
        return pd.Series(normalized, index=returns.index)

    ewma_var[first_valid] = r[first_valid] ** 2
    for t in range(first_valid + 1, n):
        prev_var = ewma_var[t - 1]
        prev_ret = r[t - 1]
        if np.isnan(prev_var) or np.isnan(prev_ret):
            ewma_var[t] = ewma_var[t - 1] if not np.isnan(ewma_var[t - 1]) else np.nan
            continue
        ewma_var[t] = lam * prev_var + (1 - lam) * prev_ret ** 2
        vol = np.sqrt(ewma_var[t])
        normalized[t] = r[t] / vol if vol > 1e-12 else np.nan
    return pd.Series(normalized, index=returns.index)


# ---------------------------------------------------------------------------
# Trend-strength score T -- see config.py's note on this being a documented
# interpretation of the research record's description, not a verified
# reproduction of Dao et al.'s exact published equations.
# ---------------------------------------------------------------------------
def trend_strength(
    idio_returns: pd.Series,
    lookback: int,
    skip_recent: int,
    ewma_lambda: float,
) -> tuple[float, int]:
    """
    idio_returns: the stock's returns with sector return already subtracted,
    most recent last. Skips the most recent `skip_recent` observations, then
    uses the prior `lookback` observations.

    Returns (T, n_obs_used). T = mean(normalized returns) / standard error
    of that mean -- a signal-to-noise (t-statistic-style) construction,
    positive for an up-trend, negative for a down-trend, magnitude scaling
    with both consistency and length of the move.
    """
    if skip_recent > 0:
        usable = idio_returns.iloc[:-skip_recent]
    else:
        usable = idio_returns
    window = usable.iloc[-lookback:]
    normalized = ewma_vol_normalize(window, ewma_lambda).dropna()
    n = len(normalized)
    if n < max(5, lookback // 2):  # refuse to score on too little data
        return np.nan, n
    mean = normalized.mean()
    se = normalized.std(ddof=1) / np.sqrt(n)
    if se <= 1e-12:
        return np.nan, n
    return mean / se, n


# ---------------------------------------------------------------------------
# Amihud (2002) illiquidity measure -- used here for the AGGREGATE
# illiquidity-state gate (Avramov, Cheng & Hameed, 2016), not per-stock
# ---------------------------------------------------------------------------
def amihud_illiquidity(returns: pd.Series, dollar_volume: pd.Series) -> pd.Series:
    """Per-observation Amihud ratio: |return| / dollar volume."""
    dv = dollar_volume.replace(0, np.nan)
    return returns.abs() / dv


def percentile_rank_of_latest(series: pd.Series, lookback: int) -> float:
    """
    Where does the most recent value of `series` sit within its own trailing
    `lookback`-observation history? Returns a value in [0, 1], or np.nan if
    there isn't enough history.
    """
    s = series.dropna()
    if len(s) < max(10, lookback // 4):
        return np.nan
    window = s.iloc[-lookback:]
    latest = window.iloc[-1]
    return float((window <= latest).mean())


def cross_sectional_dispersion(returns_by_symbol: pd.DataFrame) -> pd.Series:
    """
    returns_by_symbol: rows = dates, columns = symbols, values = that day's
    return. Returns a Series indexed by date: the cross-sectional standard
    deviation of returns across all symbols that day.
    """
    return returns_by_symbol.std(axis=1, ddof=1)


def calibrate_trend_threshold(
    lookback: int,
    skip_recent: int,
    ewma_lambda: float,
    target_two_sided_tail_prob: float = 0.4237,  # what |Z|>=0.80 means under a clean normal
    n_trials: int = 100_000,
    daily_sigma: float = 0.015,
    seed: int = 20260928,
) -> float:
    """
    Empirically finds the |T| threshold that gives THIS specific trend_strength
    construction the same two-sided false-positive rate that a clean |T|>=0.80
    normal-theory threshold is supposed to represent. Needed because
    normalizing by an ESTIMATED (not true) volatility on a small sample
    produces fatter tails than a clean normal reference -- confirmed via
    testing on 2026-09-29: a literal |T|>=0.80 threshold let through ~47%
    of pure-noise trials instead of the intended ~42%, and widening the EWMA
    lambda did not meaningfully fix it (the excess variance is intrinsic to
    the small-sample estimated-vol normalization, not the EWMA's responsiveness).

    Re-run this (and update config.TREND_STRENGTH_MIN_ABS) if lookback,
    skip_recent, or ewma_lambda ever change.
    """
    n_obs = lookback + skip_recent
    Ts = []
    rng = np.random.RandomState(seed)
    for _ in range(n_trials):
        series = pd.Series(rng.normal(0.0, daily_sigma, n_obs))
        T, n = trend_strength(series, lookback=lookback, skip_recent=skip_recent, ewma_lambda=ewma_lambda)
        if not np.isnan(T):
            Ts.append(T)
    Ts = np.array(Ts)
    return float(np.quantile(np.abs(Ts), 1 - target_two_sided_tail_prob))

"""Risk and performance metrics. Each function is small and documents its formula.

All functions take a pandas Series of periodic (daily) simple returns unless noted.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

import config

PPY = config.PERIODS_PER_YEAR


# ---------- return & volatility ----------
def annualized_return(r: pd.Series, ppy: int = PPY) -> float:
    """Geometric annualised return: (prod(1+r))^(ppy/n) - 1."""
    return float((1 + r).prod() ** (ppy / len(r)) - 1)


def annualized_vol(r: pd.Series, ppy: int = PPY) -> float:
    """Std of periodic returns scaled by sqrt(ppy)."""
    return float(r.std(ddof=1) * np.sqrt(ppy))


def sharpe_ratio(r: pd.Series, rf: float = config.RISK_FREE_RATE, ppy: int = PPY) -> float:
    """(annualised return - rf) / annualised vol."""
    return (annualized_return(r, ppy) - rf) / annualized_vol(r, ppy)


def daily_risk_free(rf: float = config.RISK_FREE_RATE, ppy: int = PPY) -> float:
    """Convert an annual risk-free rate into a per-period (daily) rate: (1 + rf)^(1/ppy) - 1."""
    return (1 + rf) ** (1 / ppy) - 1


def downside_deviation(r: pd.Series, rf: float = config.RISK_FREE_RATE, ppy: int = PPY) -> float:
    """Annualised std of returns below the daily risk-free rate (the 'bad' volatility)."""
    target = daily_risk_free(rf, ppy)
    shortfall = np.minimum(r - target, 0)
    return float(np.sqrt((shortfall ** 2).mean()) * np.sqrt(ppy))


def sortino_ratio(r: pd.Series, rf: float = config.RISK_FREE_RATE, ppy: int = PPY) -> float:
    """(annualised return - rf) / downside deviation. Only penalises downside moves."""
    return (annualized_return(r, ppy) - rf) / downside_deviation(r, rf, ppy)


# ---------- drawdowns ----------
def wealth_index(r: pd.Series, start: float = 1.0) -> pd.Series:
    """Growth of 1 unit invested."""
    return start * (1 + r).cumprod()


def drawdown_series(r: pd.Series) -> pd.Series:
    """Drawdown_t = wealth_t / running_peak_t - 1 (always <= 0)."""
    w = wealth_index(r)
    return w / w.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough fall."""
    return float(drawdown_series(r).min())


def max_drawdown_duration(r: pd.Series) -> int:
    """Longest stretch (in periods) spent below a previous peak."""
    underwater = drawdown_series(r) < 0
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def calmar_ratio(r: pd.Series, ppy: int = PPY) -> float:
    """Annualised return / |max drawdown|."""
    return annualized_return(r, ppy) / abs(max_drawdown(r))


# ---------- distribution shape ----------
def skewness(r: pd.Series) -> float:
    """Third standardised moment. Negative = fatter left tail."""
    return float(((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3)


def kurtosis(r: pd.Series) -> float:
    """Fourth standardised moment (normal = 3). Higher = fatter tails."""
    return float(((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4)


# ---------- VaR & CVaR (returned as POSITIVE loss numbers) ----------
def var_historical(r: pd.Series, level: float = config.VAR_LEVEL) -> float:
    """Empirical percentile: loss exceeded only `level` of the time."""
    return float(-np.percentile(r, level * 100))


def var_gaussian(r: pd.Series, level: float = config.VAR_LEVEL) -> float:
    """Assumes normal returns: -(mu + z * sigma)."""
    z = norm.ppf(level)
    return float(-(r.mean() + z * r.std(ddof=0)))


def var_cornish_fisher(r: pd.Series, level: float = config.VAR_LEVEL) -> float:
    """Gaussian VaR with z adjusted for skewness (S) and excess kurtosis (K).

    z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36
    """
    z = norm.ppf(level)
    s, k = skewness(r), kurtosis(r) - 3
    z_cf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    return float(-(r.mean() + z_cf * r.std(ddof=0)))


def cvar_historical(r: pd.Series, level: float = config.VAR_LEVEL) -> float:
    """Expected Shortfall: average loss on the days worse than VaR."""
    cutoff = np.percentile(r, level * 100)
    return float(-r[r <= cutoff].mean())


# ---------- relative to a benchmark ----------
def beta_alpha(r: pd.Series, bench: pd.Series, rf: float = config.RISK_FREE_RATE, ppy: int = PPY):
    """CAPM regression of excess returns. Returns (beta, annualised alpha)."""
    rf_p = daily_risk_free(rf, ppy)
    x, y = bench - rf_p, r - rf_p
    beta = float(np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1))
    alpha = float((y.mean() - beta * x.mean()) * ppy)
    return beta, alpha


def tracking_error(r: pd.Series, bench: pd.Series, ppy: int = PPY) -> float:
    """Annualised std of active return (portfolio - benchmark)."""
    return float((r - bench).std(ddof=1) * np.sqrt(ppy))


def information_ratio(r: pd.Series, bench: pd.Series, ppy: int = PPY) -> float:
    """Annualised active return / tracking error."""
    active = annualized_return(r, ppy) - annualized_return(bench, ppy)
    return active / tracking_error(r, bench, ppy)


# ---------- portfolio-level (weights + covariance) ----------
def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Share of total portfolio variance from each asset: w_i * (Cov w)_i / (w' Cov w). Sums to 1."""
    w = np.asarray(weights)
    marginal = cov @ w
    return w * marginal / (w @ cov @ w)


def summary(r: pd.Series, bench: pd.Series | None = None) -> pd.Series:
    """One-stop table of headline numbers for a return series."""
    out = {
        "Ann. return": annualized_return(r),
        "Ann. volatility": annualized_vol(r),
        "Sharpe": sharpe_ratio(r),
        "Sortino": sortino_ratio(r),
        "Calmar": calmar_ratio(r),
        "Max drawdown": max_drawdown(r),
        "Max DD duration (days)": max_drawdown_duration(r),
        "Skewness": skewness(r),
        "Kurtosis": kurtosis(r),
        "VaR 95% (hist.)": var_historical(r),
        "VaR 95% (Gaussian)": var_gaussian(r),
        "VaR 95% (Cornish-Fisher)": var_cornish_fisher(r),
        "CVaR 95%": cvar_historical(r),
    }
    if bench is not None:
        beta, alpha = beta_alpha(r, bench)
        out.update({"Beta": beta, "Alpha (ann.)": alpha,
                    "Tracking error": tracking_error(r, bench),
                    "Information ratio": information_ratio(r, bench)})
    return pd.Series(out)

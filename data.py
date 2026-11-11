"""Download prices and turn them into returns."""
import os

import numpy as np
import pandas as pd

import config


def load_prices(use_cache=True):
    """Adjusted close prices for all tickers (saved to a CSV so reruns work offline)."""
    if use_cache and os.path.exists(config.CACHE_FILE):
        return pd.read_csv(config.CACHE_FILE, index_col=0, parse_dates=True)

    import yfinance as yf

    prices = yf.download(config.TICKERS, start=config.START, end=config.END,
                         auto_adjust=True, progress=False)["Close"]
    prices = prices[config.TICKERS].dropna()
    prices.to_csv(config.CACHE_FILE)
    return prices


def to_returns(prices):
    """Simple daily returns: r_t = P_t / P_(t-1) - 1."""
    return prices.pct_change().dropna()


def get_current_weights(columns):
    """Turn the WEIGHTS dict in config.py into an array in the same order as `columns`."""
    weights = []
    for ticker in columns:
        weights.append(config.WEIGHTS[ticker])
    return np.array(weights)


def portfolio_returns(returns, weights):
    """Daily portfolio return = weighted sum of asset returns (rebalanced daily).

    `weights` is an array in the same order as the columns of `returns`.
    """
    values = returns.values @ np.array(weights)
    return pd.Series(values, index=returns.index, name="portfolio")

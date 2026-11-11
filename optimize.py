"""Portfolio optimisation: GMV, MSR, ERC and the efficient frontier.

Rules for every portfolio: long-only (no weight below 0) and fully invested (weights sum to 1).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config
import metrics


def annual_stats(returns):
    """Annualised expected return of each asset and the annualised covariance matrix."""
    mu = returns.apply(metrics.annualized_return)
    cov = returns.cov() * config.PERIODS_PER_YEAR
    return mu.values, cov.values


def port_vol(w, cov):
    """Portfolio volatility = sqrt(w' * Cov * w)."""
    return float(np.sqrt(w @ cov @ w))


def port_ret(w, mu):
    """Portfolio expected return = w' * mu."""
    return float(w @ mu)


def solve_weights(objective, n_assets, extra_constraints=None, start_weights=None):
    """Find the weights that minimise `objective`.

    objective         -- function that takes weights and returns a number to minimise
    n_assets          -- how many assets we are allocating between
    extra_constraints -- optional list of extra scipy constraints (e.g. hit a target return)
    start_weights     -- where the optimiser starts searching (default: equal weight)
    """
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]   # fully invested
    if extra_constraints is not None:
        constraints = constraints + extra_constraints
    if start_weights is None:
        start_weights = np.full(n_assets, 1 / n_assets)
    bounds = [(0, 1)] * n_assets                                   # long-only

    result = minimize(objective, start_weights, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"ftol": 1e-12, "maxiter": 500})
    return result.x


def gmv(cov):
    """Global Minimum Variance: lowest volatility possible. Needs no return estimates."""
    def variance(w):
        return w @ cov @ w
    return solve_weights(variance, len(cov))


def msr(mu, cov, rf=config.RISK_FREE_RATE):
    """Maximum Sharpe Ratio portfolio (the tangency portfolio)."""
    def negative_sharpe(w):
        return -(port_ret(w, mu) - rf) / port_vol(w, cov)
    return solve_weights(negative_sharpe, len(mu))


def erc(cov):
    """Equal Risk Contribution (risk parity): every asset contributes the same share of risk."""
    n = len(cov)
    target = np.full(n, 1 / n)

    def distance_from_equal_risk(w):
        return ((metrics.risk_contributions(w, cov) - target) ** 2).sum()
    return solve_weights(distance_from_equal_risk, n)


def min_variance_for_target(mu, cov, target_return, start_weights):
    """Lowest-variance portfolio that earns exactly `target_return`."""
    def hits_target(w):
        return port_ret(w, mu) - target_return

    def variance(w):
        return w @ cov @ w

    constraint = [{"type": "eq", "fun": hits_target}]
    return solve_weights(variance, len(mu), constraint, start_weights)


def efficient_frontier(mu, cov, n_points=40):
    """Trace the frontier: for each target return, find the minimum-variance portfolio."""
    w_gmv = gmv(cov)
    targets = np.linspace(port_ret(w_gmv, mu), mu.max(), n_points)
    rows = []
    for target in targets:
        w = min_variance_for_target(mu, cov, target, w_gmv)
        rows.append({"ret": port_ret(w, mu), "vol": port_vol(w, cov)})
    return pd.DataFrame(rows)


def random_portfolios(mu, cov, n=5000, rf=config.RISK_FREE_RATE, seed=42):
    """Cloud of random long-only portfolios, used as the background of the frontier chart."""
    rng = np.random.default_rng(seed)
    all_weights = rng.dirichlet(np.ones(len(mu)), size=n)   # random weights that sum to 1
    rows = []
    for w in all_weights:
        ret, vol = port_ret(w, mu), port_vol(w, cov)
        rows.append({"ret": ret, "vol": vol, "sharpe": (ret - rf) / vol})
    return pd.DataFrame(rows)

"""Model validation: out-of-sample portfolio test and VaR exceedance backtest."""
import numpy as np
import pandas as pd
from scipy.stats import chi2

import config
import data
import metrics
import optimize


def out_of_sample(returns: pd.DataFrame, split: str = config.SPLIT_DATE) -> dict:
    """Fit GMV / MSR / ERC weights on data BEFORE `split`, then hold them AFTER.

    Lesson: in-sample optimal portfolios often disappoint out-of-sample (estimation error).
    """
    train = returns[returns.index < split]    # used to estimate mu and cov
    test = returns[returns.index >= split]    # never seen by the optimiser
    mu, cov = optimize.annual_stats(train)
    n = len(mu)

    weights = {
        "Equal weight": np.full(n, 1 / n),
        "Current": data.get_current_weights(returns.columns),
        "GMV": optimize.gmv(cov),
        "MSR": optimize.msr(mu, cov),
        "ERC": optimize.erc(cov),
    }

    test_returns = {}
    test_metrics = {}
    for name, w in weights.items():
        test_returns[name] = data.portfolio_returns(test, w)
        test_metrics[name] = metrics.summary(test_returns[name])

    return {"weights": pd.DataFrame(weights, index=returns.columns).T,
            "returns": pd.DataFrame(test_returns),
            "table": pd.DataFrame(test_metrics).T}


def var_backtest(r: pd.Series, window: int = 504, level: float = config.VAR_LEVEL) -> pd.DataFrame:
    """Rolling-window VaR backtest for the 3 methods.

    For each day, estimate VaR from the previous `window` days, then check whether
    today's loss exceeded it. A good 95% VaR is breached ~5% of days.
    Kupiec POF test p-value > 0.05 means we cannot reject correct coverage.
    """
    methods = {"Historical": metrics.var_historical, "Gaussian": metrics.var_gaussian,
               "Cornish-Fisher": metrics.var_cornish_fisher}
    n_obs = len(r) - window
    out = {}
    for name, fn in methods.items():
        breaches = 0
        for i in range(window, len(r)):
            if -r.iloc[i] > fn(r.iloc[i - window:i], level):
                breaches += 1
        out[name] = {"Breaches": breaches, "Observations": n_obs,
                     "Breach rate": breaches / n_obs, "Expected rate": level,
                     "Kupiec p-value": _kupiec_pvalue(breaches, n_obs, level)}
    return pd.DataFrame(out).T


def _kupiec_pvalue(x: int, n: int, p: float) -> float:
    """Kupiec likelihood-ratio test that the true breach probability equals p.

    x = number of breaches, n = number of days, p = expected breach rate (e.g. 0.05).
    Compares the likelihood of the data under p with the likelihood under the observed rate x/n.
    """
    if x == 0:  # avoid log(0) when there are no breaches at all
        lr = -2 * n * np.log(1 - p)
    else:
        observed = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - observed) - x * np.log(observed))
    return float(1 - chi2.cdf(lr, df=1))


def stress_test(returns: pd.DataFrame, weights) -> pd.DataFrame:
    """Portfolio and benchmark return over the named crisis windows."""
    port = data.portfolio_returns(returns, weights)
    bench = returns[config.BENCHMARK]
    rows = {}
    for name, (start, end) in config.STRESS_PERIODS.items():
        port_window = port.loc[start:end]
        bench_window = bench.loc[start:end]
        rows[name] = {"Portfolio return": (1 + port_window).prod() - 1,
                      "Benchmark return": (1 + bench_window).prod() - 1,
                      "Portfolio max DD": metrics.max_drawdown(port_window)}
    return pd.DataFrame(rows).T

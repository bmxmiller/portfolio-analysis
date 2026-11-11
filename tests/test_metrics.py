"""Known-answer tests: tiny examples we can verify by hand.

Run with:  pytest   (conftest.py in the project root lets pytest find our modules)
"""
import numpy as np
import pandas as pd
import pytest

import config
import data
import metrics
import optimize


def test_current_weights_follow_column_order():
    columns = list(reversed(config.TICKERS))
    w = data.get_current_weights(columns)
    assert w[0] == config.WEIGHTS[columns[0]]
    assert w.sum() == pytest.approx(1.0)


def test_max_drawdown_toy():
    # 100 -> 120 -> 60 -> 90: peak 120, trough 60 => -50%
    r = pd.Series([0.20, -0.50, 0.50])
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_drawdown_duration():
    r = pd.Series([0.1, -0.05, -0.02, 0.01, 0.2])  # underwater for 3 periods then new peak
    assert metrics.max_drawdown_duration(r) == 3


def test_annualized_return_constant():
    r = pd.Series([0.001] * 252)
    assert metrics.annualized_return(r) == pytest.approx(1.001 ** 252 - 1)


def test_historical_var_cvar_ordering():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 5000))
    assert metrics.cvar_historical(r) >= metrics.var_historical(r) > 0


def test_cornish_fisher_equals_gaussian_for_normal_moments():
    # With skew=0 and kurtosis=3 the CF adjustment vanishes; use a large normal sample
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200_000))
    assert metrics.var_cornish_fisher(r) == pytest.approx(metrics.var_gaussian(r), rel=0.02)


def test_beta_of_benchmark_is_one():
    b = pd.Series(np.random.default_rng(2).normal(0, 0.01, 1000))
    beta, alpha = metrics.beta_alpha(b, b)
    assert beta == pytest.approx(1.0)
    assert alpha == pytest.approx(0.0, abs=1e-9)


@pytest.fixture
def cov_mu():
    cov = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.01]])
    mu = np.array([0.08, 0.12, 0.03])
    return mu, cov


def test_weights_sum_to_one_and_long_only(cov_mu):
    mu, cov = cov_mu
    for w in (optimize.gmv(cov), optimize.msr(mu, cov, 0.02), optimize.erc(cov)):
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all()


def test_gmv_has_lowest_vol(cov_mu):
    mu, cov = cov_mu
    v_gmv = optimize.port_vol(optimize.gmv(cov), cov)
    assert v_gmv <= min(np.sqrt(np.diag(cov))) + 1e-9
    assert v_gmv <= optimize.port_vol(optimize.msr(mu, cov, 0.02), cov) + 1e-9


def test_erc_equalises_risk_contributions(cov_mu):
    _, cov = cov_mu
    rc = metrics.risk_contributions(optimize.erc(cov), cov)
    assert rc == pytest.approx(np.full(3, 1 / 3), abs=1e-3)
    assert rc.sum() == pytest.approx(1.0)

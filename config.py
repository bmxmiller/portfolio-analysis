"""Central settings. Change the tickers/weights here to analyse your own portfolio."""

# Multi-asset ETF universe: US equity, developed ex-US, emerging, bonds, gold, real estate
TICKERS = ["SPY", "EFA", "EEM", "AGG", "GLD", "VNQ"]

# "Current" portfolio we are reviewing (must sum to 1)
WEIGHTS = {"SPY": 0.40, "EFA": 0.15, "EEM": 0.05, "AGG": 0.25, "GLD": 0.10, "VNQ": 0.05}

BENCHMARK = "SPY"
START = "2012-01-01"
END = "2025-12-31"

RISK_FREE_RATE = 0.02      # annual
PERIODS_PER_YEAR = 252     # daily data
VAR_LEVEL = 0.05           # 95% VaR / CVaR

# Out-of-sample test: optimise on data before this date, evaluate after
SPLIT_DATE = "2020-01-01"

# Stress windows for the report
STRESS_PERIODS = {
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate hikes": ("2022-01-03", "2022-10-14"),
}

CACHE_FILE = "prices_cache.csv"

"""Interactive version of the portfolio review.

Run locally:  streamlit run app.py
All the maths lives in metrics.py / optimize.py; this file only draws things.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import data
import metrics
import optimize

st.set_page_config(page_title="Portfolio Risk Review", layout="wide")


@st.cache_data
def get_returns():
    return data.to_returns(data.load_prices())


all_returns = get_returns()

# ---------------- sidebar: user inputs ----------------
st.sidebar.header("Portfolio")
raw_weights = []
for ticker in config.TICKERS:
    default = int(config.WEIGHTS[ticker] * 100)
    raw_weights.append(st.sidebar.slider(ticker, 0, 100, default, step=5))

if sum(raw_weights) == 0:
    st.error("Give at least one asset a weight above 0.")
    st.stop()
weights = np.array(raw_weights) / sum(raw_weights)   # rescale so weights sum to 100%

st.sidebar.header("Settings")
years = sorted(set(all_returns.index.year))
start_year = st.sidebar.selectbox("Start year", years[:-2], index=0)
rf = st.sidebar.number_input("Risk-free rate (%)", 0.0, 10.0, config.RISK_FREE_RATE * 100, 0.25) / 100

# ---------------- calculations ----------------
rets = all_returns[all_returns.index.year >= start_year]
port = data.portfolio_returns(rets, weights)
bench = rets[config.BENCHMARK]
mu, cov = optimize.annual_stats(rets)

# ---------------- header + KPI tiles ----------------
st.title("Portfolio Risk Review")
st.caption(" Â· ".join(f"{t} {w:.0%}" for t, w in zip(config.TICKERS, weights))
           + f"  |  {rets.index[0]:%b %Y} â€“ {rets.index[-1]:%b %Y}  |  benchmark {config.BENCHMARK}")

tiles = st.columns(6)
tiles[0].metric("Ann. return", f"{metrics.annualized_return(port):.1%}")
tiles[1].metric("Ann. volatility", f"{metrics.annualized_vol(port):.1%}")
tiles[2].metric("Sharpe", f"{metrics.sharpe_ratio(port, rf):.2f}")
tiles[3].metric("VaR 95% (1-day)", f"{metrics.var_historical(port):.2%}")
tiles[4].metric("CVaR 95% (1-day)", f"{metrics.cvar_historical(port):.2%}")
tiles[5].metric("Max drawdown", f"{metrics.max_drawdown(port):.1%}")

# ---------------- performance charts ----------------
left, right = st.columns(2)
growth = pd.DataFrame({"Portfolio": metrics.wealth_index(port),
                       config.BENCHMARK: metrics.wealth_index(bench)})
left.plotly_chart(px.line(growth, title="Growth of $1", labels={"value": "", "variable": ""}),
                  width="stretch")
drawdowns = pd.DataFrame({"Portfolio": metrics.drawdown_series(port),
                          config.BENCHMARK: metrics.drawdown_series(bench)})
fig_dd = px.line(drawdowns, title="Drawdown", labels={"value": "", "variable": ""})
fig_dd.update_yaxes(tickformat=".0%")
right.plotly_chart(fig_dd, width="stretch")

# ---------------- efficient frontier ----------------
st.subheader("Efficient frontier")
cloud = optimize.random_portfolios(mu, cov, n=2000, rf=rf)
frontier = optimize.efficient_frontier(mu, cov, n_points=30)

fig = px.scatter(cloud, x="vol", y="ret", color="sharpe", opacity=0.4,
                 color_continuous_scale="Viridis", labels={"vol": "Volatility", "ret": "Return"})
fig.add_trace(go.Scatter(x=frontier["vol"], y=frontier["ret"], mode="lines",
                         name="Efficient frontier", line=dict(color="black", width=3)))

n = len(mu)
key_portfolios = {
    "GMV": optimize.gmv(cov),
    "MSR": optimize.msr(mu, cov, rf),
    "ERC": optimize.erc(cov),
    "Equal weight": np.full(n, 1 / n),
    "Your portfolio": weights,
}
for name, w in key_portfolios.items():
    fig.add_trace(go.Scatter(x=[optimize.port_vol(w, cov)], y=[optimize.port_ret(w, mu)],
                             mode="markers", name=name, marker=dict(size=14, line=dict(width=1, color="white"))))
fig.update_layout(height=550, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                  legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, width="stretch")

weights_table = pd.DataFrame(key_portfolios, index=config.TICKERS)
st.dataframe(weights_table.style.format("{:.0%}"), width="stretch")

# ---------------- risk tables ----------------
left, right = st.columns(2)
summary = pd.DataFrame({"Portfolio": metrics.summary(port, bench),
                        config.BENCHMARK: metrics.summary(bench)})
left.subheader("All metrics")
left.dataframe(summary.style.format("{:.4f}", na_rep="â€“"), width="stretch", height=600)

right.subheader("Weight vs share of risk")
risk = pd.DataFrame({"Weight": weights, "Risk contribution": metrics.risk_contributions(weights, cov)},
                    index=config.TICKERS)
fig_risk = px.bar(risk, barmode="group", labels={"value": "", "index": "", "variable": ""})
fig_risk.update_yaxes(tickformat=".0%")
right.plotly_chart(fig_risk, width="stretch")

right.subheader("Correlation")
right.plotly_chart(px.imshow(rets.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1), width="stretch")

st.caption("Educational project, not investment advice. Data: Yahoo Finance via yfinance.")

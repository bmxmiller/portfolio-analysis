"""Build the one-page portfolio review (report.png / report.pdf).

Run:  python report.py
"""
import matplotlib

matplotlib.use("Agg")  # draw to files only, no pop-up window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import backtest
import config
import data
import metrics
import optimize

COLORS = {"port": "#1f5fbf", "bench": "#8a8f98", "gmv": "#2a9d5c", "msr": "#d9822b",
          "erc": "#7b4fb8", "eq": "#555555", "bad": "#c0392b", "ink": "#1d2433"}

# Rows in the tables that should be shown as percentages
PERCENT_ROWS = ["Ann. return", "Ann. volatility", "Max drawdown", "VaR 95% (hist.)",
                "VaR 95% (Gaussian)", "VaR 95% (Cornish-Fisher)", "CVaR 95%",
                "Alpha (ann.)", "Tracking error", "Portfolio return", "Benchmark return",
                "Portfolio max DD"]


def format_value(name, value):
    """Turn a number into display text, depending on which metric it is."""
    if pd.isna(value):
        return "–"
    if name in PERCENT_ROWS:
        return f"{value:.2%}"
    if "days" in name:
        return f"{value:.0f}"
    return f"{value:.2f}"


def percent_axis(axis):
    """Show tick labels like 10% instead of 0.1."""
    axis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def load_everything():
    """Load data and compute everything the charts need."""
    rets = data.to_returns(data.load_prices())
    w_current = data.get_current_weights(rets.columns)
    mu, cov = optimize.annual_stats(rets)

    optimal = {
        "GMV": optimize.gmv(cov),
        "MSR": optimize.msr(mu, cov),
        "ERC": optimize.erc(cov),
    }
    return {
        "rets": rets,
        "port": data.portfolio_returns(rets, w_current),
        "bench": rets[config.BENCHMARK],
        "mu": mu,
        "cov": cov,
        "w_current": w_current,
        "optimal": optimal,
    }


def draw_kpi_tiles(fig, slots, port):
    tiles = [
        ("Ann. return", f"{metrics.annualized_return(port):.1%}", COLORS["ink"]),
        ("Ann. volatility", f"{metrics.annualized_vol(port):.1%}", COLORS["ink"]),
        ("Sharpe", f"{metrics.sharpe_ratio(port):.2f}", COLORS["ink"]),
        ("VaR 95% (1-day)", f"{metrics.var_historical(port):.2%}", COLORS["bad"]),
        ("CVaR 95% (1-day)", f"{metrics.cvar_historical(port):.2%}", COLORS["bad"]),
        ("Max drawdown", f"{metrics.max_drawdown(port):.1%}", COLORS["bad"]),
    ]
    for slot, (label, value, color) in zip(slots, tiles):
        ax = fig.add_subplot(slot)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#f1f4f9", zorder=0))
        ax.text(0.5, 0.60, value, ha="center", va="center", fontsize=19, weight="bold", color=color)
        ax.text(0.5, 0.22, label, ha="center", va="center", fontsize=9.5, color="#5b6475")


def plot_wealth(ax, port, bench):
    ax.plot(metrics.wealth_index(port), color=COLORS["port"], lw=1.6, label="Portfolio")
    ax.plot(metrics.wealth_index(bench), color=COLORS["bench"], lw=1.2,
            label=f"{config.BENCHMARK} (benchmark)")
    ax.set_title("Growth of $1", loc="left", weight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def plot_drawdown(ax, port, bench):
    ax.fill_between(port.index, metrics.drawdown_series(port), 0,
                    color=COLORS["port"], alpha=0.35, label="Portfolio")
    ax.plot(metrics.drawdown_series(bench), color=COLORS["bench"], lw=0.9, label=config.BENCHMARK)
    ax.set_title("Drawdown", loc="left", weight="bold", fontsize=11)
    percent_axis(ax.yaxis)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")


def plot_frontier(ax, ctx):
    mu, cov, rf = ctx["mu"], ctx["cov"], config.RISK_FREE_RATE

    # 1) cloud of random portfolios, coloured by Sharpe ratio
    cloud = optimize.random_portfolios(mu, cov)
    dots = ax.scatter(cloud["vol"], cloud["ret"], c=cloud["sharpe"], cmap="viridis", s=5, alpha=0.5)
    plt.colorbar(dots, ax=ax, label="Sharpe", pad=0.01, fraction=0.04)

    # 2) the efficient frontier line
    frontier = optimize.efficient_frontier(mu, cov)
    ax.plot(frontier["vol"], frontier["ret"], color=COLORS["ink"], lw=2, label="Efficient frontier")

    # 3) Capital Market Line: from the risk-free rate through the MSR portfolio
    w_msr = ctx["optimal"]["MSR"]
    msr_vol = optimize.port_vol(w_msr, cov)
    msr_ret = optimize.port_ret(w_msr, mu)
    x = np.linspace(0, frontier["vol"].max() * 1.05, 20)
    ax.plot(x, rf + (msr_ret - rf) / msr_vol * x, "--", color=COLORS["msr"], lw=1.2,
            label="Capital Market Line")

    # 4) mark the portfolios we care about
    n = len(mu)
    markers = [
        ("GMV", ctx["optimal"]["GMV"], COLORS["gmv"], "o", 80),
        ("MSR", w_msr, COLORS["msr"], "*", 170),
        ("ERC", ctx["optimal"]["ERC"], COLORS["erc"], "D", 80),
        ("Equal weight", np.full(n, 1 / n), COLORS["eq"], "s", 80),
        ("Current", ctx["w_current"], COLORS["port"], "P", 80),
    ]
    for name, w, color, shape, size in markers:
        ax.scatter(optimize.port_vol(w, cov), optimize.port_ret(w, mu), color=color, marker=shape,
                   s=size, edgecolor="white", zorder=5, label=name)

    ax.set_xlabel("Volatility (ann.)")
    ax.set_ylabel("Return (ann.)")
    percent_axis(ax.xaxis)
    percent_axis(ax.yaxis)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
    ax.set_title("Efficient frontier: GMV and MSR", loc="left", weight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def plot_weights(ax, ctx):
    tickers = list(ctx["rets"].columns)
    portfolios = [
        ("Current", ctx["w_current"], COLORS["port"]),
        ("GMV", ctx["optimal"]["GMV"], COLORS["gmv"]),
        ("MSR", ctx["optimal"]["MSR"], COLORS["msr"]),
        ("ERC", ctx["optimal"]["ERC"], COLORS["erc"]),
    ]
    x = np.arange(len(tickers))
    width = 0.2
    for i, (name, w, color) in enumerate(portfolios):
        ax.bar(x + (i - 1.5) * width, w, width, label=name, color=color)
    ax.set_xticks(x, tickers)
    percent_axis(ax.yaxis)
    ax.set_title("Weights by portfolio", loc="left", weight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=4)


def plot_risk_contributions(ax, ctx):
    tickers = list(ctx["rets"].columns)
    risk_share = metrics.risk_contributions(ctx["w_current"], ctx["cov"])
    x = np.arange(len(tickers))
    ax.bar(x - 0.2, ctx["w_current"], 0.4, label="Weight", color="#b8c7e0")
    ax.bar(x + 0.2, risk_share, 0.4, label="Risk contribution", color=COLORS["port"])
    ax.set_xticks(x, tickers)
    percent_axis(ax.yaxis)
    ax.set_title("Weight vs share of risk (current)", loc="left", weight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=8)


def plot_correlation(ax, rets):
    corr = rets.corr()
    ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=text_color)
    ax.set_title("Correlation", loc="left", weight="bold", fontsize=11)


def draw_table(ax, df, title):
    """Draw a DataFrame as a table: metric names down the left, one column per portfolio."""
    ax.axis("off")
    ax.set_title(title, loc="left", weight="bold", fontsize=11)

    # build the text for every cell first
    cells = []
    for metric_name, row in df.iterrows():
        line = [metric_name]
        for value in row:
            line.append(format_value(metric_name, value))
        cells.append(line)

    n_rows = len(df) + 1                      # +1 for the header row
    row_height = min(0.14, 0.88 / n_rows)     # shrink rows if the table is long
    table_height = row_height * n_rows
    table = ax.table(cellText=cells, colLabels=[""] + list(df.columns), cellLoc="center",
                     bbox=[0, 1 - table_height - 0.04, 1, table_height])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # light styling: grey borders, shaded bold header
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dde2ea")
        if row == 0:
            cell.set_facecolor("#f1f4f9")
            cell.set_text_props(weight="bold")


def build_report(path_png="report.png", path_pdf="report.pdf"):
    ctx = load_everything()
    port, bench, rets = ctx["port"], ctx["bench"], ctx["rets"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#c9cfda"})

    fig = plt.figure(figsize=(16, 25), facecolor="white")
    grid = GridSpec(7, 6, figure=fig, height_ratios=[0.4, 0.6, 2.3, 1.7, 2.4, 2.0, 1.4],
                    hspace=0.5, wspace=0.55, left=0.06, right=0.97, top=0.965, bottom=0.03)

    # --- header ---
    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    ax.text(0, 0.65, "Portfolio Risk Review", fontsize=26, weight="bold", color=COLORS["ink"])
    weights_text = " · ".join(f"{ticker} {w:.0%}" for ticker, w in config.WEIGHTS.items())
    ax.text(0, 0.05, f"{weights_text}   |   {rets.index[0]:%b %Y} – {rets.index[-1]:%b %Y}   |   "
                     f"benchmark {config.BENCHMARK}, rf {config.RISK_FREE_RATE:.0%}, daily data",
            fontsize=10.5, color="#5b6475")

    # --- KPI tiles ---
    draw_kpi_tiles(fig, [grid[1, i] for i in range(6)], port)

    # --- performance charts ---
    plot_wealth(fig.add_subplot(grid[2, :3]), port, bench)
    plot_drawdown(fig.add_subplot(grid[2, 3:]), port, bench)

    # --- metric tables ---
    summary = pd.DataFrame({"Portfolio": metrics.summary(port, bench),
                            config.BENCHMARK: metrics.summary(bench)})
    var_rows = ["VaR 95% (hist.)", "VaR 95% (Gaussian)", "VaR 95% (Cornish-Fisher)", "CVaR 95%"]
    other_rows = ["Sortino", "Calmar", "Skewness", "Kurtosis", "Beta", "Alpha (ann.)",
                  "Tracking error", "Information ratio", "Max DD duration (days)"]
    draw_table(fig.add_subplot(grid[3, :3]), summary.loc[var_rows],
               "1-day loss at 95%: three VaR methods")
    draw_table(fig.add_subplot(grid[3, 3:]), summary.loc[other_rows],
               "More risk-adjusted & relative metrics")

    # --- optimisation ---
    plot_frontier(fig.add_subplot(grid[4, :4]), ctx)
    plot_correlation(fig.add_subplot(grid[4, 4:]), rets)
    plot_weights(fig.add_subplot(grid[5, :3]), ctx)
    plot_risk_contributions(fig.add_subplot(grid[5, 3:]), ctx)

    # --- validation: out-of-sample and stress tests ---
    oos = backtest.out_of_sample(rets)
    oos_table = oos["table"][["Ann. return", "Ann. volatility", "Sharpe", "Max drawdown"]].T
    draw_table(fig.add_subplot(grid[6, :4]), oos_table,
               f"Out-of-sample: weights fitted before {config.SPLIT_DATE[:4]}, held after")
    stress = backtest.stress_test(rets, ctx["w_current"]).T
    draw_table(fig.add_subplot(grid[6, 4:]), stress, "Stress periods")

    fig.savefig(path_png, dpi=130)
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f"Saved {path_png} and {path_pdf}")


if __name__ == "__main__":
    build_report()

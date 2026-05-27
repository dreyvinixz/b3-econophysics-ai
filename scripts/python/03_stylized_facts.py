from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

RETURNS_LONG_PATH = TABLES_DIR / "demo_assets_returns_long_1998_2025.csv"
RETURNS_WIDE_PATH = TABLES_DIR / "demo_assets_returns_wide_1998_2025.csv"

ASSET_ORDER = ["PETR4", "VALE3", "BBDC4"]
ASSET_COLORS = {
    "PETR4": "#1f77b4",
    "VALE3": "#2ca02c",
    "BBDC4": "#d62728",
}
RETURN_OFFSETS = {
    "PETR4": 0.20,
    "VALE3": 0.00,
    "BBDC4": -0.20,
}


@dataclass(frozen=True)
class FigureSpec:
    slug: str
    start_date: str
    end_date: str
    caption: str


FIGURE_SPECS = [
    FigureSpec(
        slug="figure_1a_stylized_facts_demo_assets_2006_2021",
        start_date="2006-01-01",
        end_date="2021-12-31",
        caption="2006-2021",
    ),
    FigureSpec(
        slug="figure_1b_stylized_facts_demo_assets_1998_2025",
        start_date="1998-03-16",
        end_date="2025-12-31",
        caption="1998-2025",
    ),
]
JOURNAL_SPEC = FigureSpec(
    slug="figure_1c_stylized_facts_demo_assets_journal_style_2006_2021",
    start_date="2006-01-01",
    end_date="2021-12-31",
    caption="2006-2021",
)
CLEAN_SPEC = FigureSpec(
    slug="figure_1d_stylized_facts_demo_assets_clean_2006_2021",
    start_date="2006-01-01",
    end_date="2021-12-31",
    caption="2006-2021",
)


def load_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    returns_long = pd.read_csv(RETURNS_LONG_PATH, parse_dates=["date"])
    returns_wide = pd.read_csv(RETURNS_WIDE_PATH, parse_dates=["date"])
    returns_wide = returns_wide.set_index("date").sort_index()

    symbols = [symbol for symbol in ASSET_ORDER if symbol in returns_wide.columns]
    returns_wide = returns_wide[symbols]
    returns_long = returns_long[returns_long["symbol"].isin(symbols)].copy()

    return returns_long, returns_wide


def filter_period(
    returns_long: pd.DataFrame,
    returns_wide: pd.DataFrame,
    spec: FigureSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_date = pd.Timestamp(spec.start_date)
    end_date = pd.Timestamp(spec.end_date)

    period_wide = returns_wide.loc[start_date:end_date].copy()
    period_long = returns_long[
        (returns_long["date"] >= start_date) & (returns_long["date"] <= end_date)
    ].copy()

    return period_long, period_wide


def normalized_price_index(returns_wide: pd.DataFrame) -> pd.DataFrame:
    gross_log_return = returns_wide.fillna(0).cumsum()
    return np.exp(gross_log_return)


def normalized_price_from_prices(returns_long: pd.DataFrame) -> pd.DataFrame:
    prices = returns_long.pivot(index="date", columns="symbol", values="price")
    prices = prices[[symbol for symbol in ASSET_ORDER if symbol in prices.columns]].sort_index()
    return prices / prices.iloc[0]


def empirical_ccdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    clean = values.dropna().to_numpy()
    clean = clean[clean >= 1e-4]
    clean.sort()

    n_obs = len(clean)
    ccdf = (n_obs - np.arange(n_obs)) / n_obs

    return clean, ccdf


def tail_reference_line(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    tail_mask = (x_values >= np.quantile(x_values, 0.95)) & (y_values > 0)
    tail_x = x_values[tail_mask]
    tail_y = y_values[tail_mask]

    if len(tail_x) < 5:
        return np.array([]), np.array([]), np.nan

    slope, intercept = np.polyfit(np.log(tail_x), np.log(tail_y), 1)
    reference_x = np.linspace(tail_x.min(), tail_x.max(), 80)
    reference_y = np.exp(intercept) * reference_x**slope
    alpha = -slope

    return reference_x, reference_y, alpha


def absolute_return_acf(series: pd.Series, nlags: int = 300) -> np.ndarray:
    clean = series.abs().dropna()
    if len(clean) <= nlags + 1:
        return np.full(nlags + 1, np.nan)
    return acf(clean, nlags=nlags, fft=True, missing="drop")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.markersize": 2.2,
        }
    )


def configure_journal_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        }
    )


def style_journal_axes(axes: np.ndarray) -> None:
    for ax in axes.ravel():
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(width=0.7, length=3.5)


def configure_clean_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        }
    )


def plot_stylized_facts(
    returns_long: pd.DataFrame,
    returns_wide: pd.DataFrame,
    spec: FigureSpec,
) -> plt.Figure:
    configure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    price_index = normalized_price_index(returns_wide)

    ax = axes[0, 0]
    for symbol in returns_wide.columns:
        ax.plot(price_index.index, price_index[symbol], label=symbol, color=ASSET_COLORS[symbol], linewidth=1.1)
    ax.set_yscale("log")
    ax.set_title("normalized price")
    ax.set_ylabel("normalized price")
    ax.legend(loc="upper left", ncols=3, frameon=False)

    ax = axes[0, 1]
    for symbol in returns_wide.columns:
        shifted_returns = returns_wide[symbol] + RETURN_OFFSETS[symbol]
        ax.plot(returns_wide.index, shifted_returns, color=ASSET_COLORS[symbol], linewidth=0.55, alpha=0.85)
        ax.axhline(RETURN_OFFSETS[symbol], color=ASSET_COLORS[symbol], linewidth=0.6, alpha=0.5)
        ax.text(
            returns_wide.index[-1],
            RETURN_OFFSETS[symbol],
            f" {symbol}",
            color=ASSET_COLORS[symbol],
            va="center",
            ha="left",
            fontsize=8,
            clip_on=False,
        )
    ax.set_title("returns")
    ax.set_ylabel("log return + offset")
    ax.margins(x=0.03)

    ax = axes[1, 0]
    for symbol in returns_wide.columns:
        x_values, y_values = empirical_ccdf(returns_wide[symbol].abs())
        ax.loglog(x_values, y_values, label=symbol, color=ASSET_COLORS[symbol], linewidth=1.05)

    pooled_x, pooled_y = empirical_ccdf(returns_long["log_return"].abs())
    ref_x, ref_y, alpha = tail_reference_line(pooled_x, pooled_y)
    ax.loglog(pooled_x, pooled_y, label="all", color="#222222", linewidth=1.3, linestyle="--")
    if len(ref_x):
        ax.loglog(ref_x, ref_y, color="#666666", linewidth=1.0, linestyle=":", label=fr"$x^{{-{alpha:.2f}}}$")
    ax.set_title("complementary CDF of returns")
    ax.set_xlabel("|log return|")
    ax.set_ylabel("P(|r| > x)")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1, 1]
    lags = np.arange(1, 301)
    acf_by_asset = []
    for symbol in returns_wide.columns:
        acf_values = absolute_return_acf(returns_wide[symbol], nlags=300)[1:]
        acf_by_asset.append(acf_values)
        ax.plot(
            lags,
            acf_values,
            marker="o",
            markevery=8,
            linewidth=0.6,
            alpha=0.75,
            label=symbol,
            color=ASSET_COLORS[symbol],
        )

    all_acf = np.nanmean(np.vstack(acf_by_asset), axis=0)
    ax.plot(lags, all_acf, color="#222222", linewidth=1.4, label="all")
    ax.axhline(0.0, color="#222222", linewidth=0.7)
    ax.set_title("autocorrelations")
    ax.set_xlabel("lag, trading days")
    ax.set_ylabel("ACF of |r|")
    ax.legend(loc="upper right", frameon=False)

    return fig


def plot_clean_stylized_facts(
    returns_long: pd.DataFrame,
    returns_wide: pd.DataFrame,
) -> plt.Figure:
    configure_clean_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.3), constrained_layout=True)
    style_journal_axes(axes)
    price_index = normalized_price_from_prices(returns_long)

    ax = axes[0, 0]
    for symbol in price_index.columns:
        ax.plot(
            price_index.index,
            price_index[symbol],
            label=symbol,
            color=ASSET_COLORS[symbol],
            linewidth=0.8,
        )
    ax.set_title("normalized price")
    ax.set_ylabel(r"$P_t / P_0$")
    ax.legend(loc="upper left", ncols=3, handlelength=2.0, columnspacing=1.3)

    ax = axes[0, 1]
    for symbol in returns_wide.columns:
        ax.plot(
            returns_wide.index,
            returns_wide[symbol] + RETURN_OFFSETS[symbol],
            label=symbol,
            color=ASSET_COLORS[symbol],
            linewidth=0.35,
            alpha=0.85,
        )
        ax.axhline(RETURN_OFFSETS[symbol], color="black", linewidth=0.3, alpha=0.3)
    ax.set_title("returns")
    ax.set_ylabel(r"$r_t$ + offset")
    ax.legend(loc="upper left", handlelength=2.0)

    ax = axes[1, 0]
    for symbol in returns_wide.columns:
        x_values, y_values = empirical_ccdf(returns_wide[symbol].abs())
        ax.loglog(
            x_values,
            y_values,
            label=symbol,
            color=ASSET_COLORS[symbol],
            linewidth=0.9,
        )

    pooled_x, pooled_y = empirical_ccdf(returns_long["log_return"].abs())
    ref_x, ref_y, alpha = tail_reference_line(pooled_x, pooled_y)
    ax.loglog(pooled_x, pooled_y, label="all", color="black", linewidth=1.0)
    if len(ref_x):
        ax.loglog(ref_x, ref_y, color="black", linewidth=0.8, linestyle="--", label=rf"$x^{{-{alpha:.2f}}}$")
    ax.set_title("complementary CDF of returns")
    ax.set_xlabel("|log return|")
    ax.set_ylabel("P(|r| > x)")
    ax.legend(loc="lower left", handlelength=2.0)

    ax = axes[1, 1]
    max_lag = 150
    lags = np.arange(1, max_lag + 1)
    acf_by_asset = []
    for symbol in returns_wide.columns:
        acf_values = absolute_return_acf(returns_wide[symbol], nlags=max_lag)[1:]
        acf_by_asset.append(acf_values)
        ax.plot(
            lags,
            acf_values,
            label=symbol,
            color=ASSET_COLORS[symbol],
            linewidth=0.55,
            alpha=0.55,
        )

    all_acf = np.nanmean(np.vstack(acf_by_asset), axis=0)
    ax.plot(lags, all_acf, label="all", color="black", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlim(0, max_lag)
    ax.set_title("autocorrelations")
    ax.set_xlabel("lag, trading days")
    ax.set_ylabel(r"ACF$(|r|)$")
    ax.legend(loc="upper right", handlelength=2.0)

    return fig


def plot_journal_stylized_facts(
    returns_long: pd.DataFrame,
    returns_wide: pd.DataFrame,
) -> plt.Figure:
    configure_journal_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), constrained_layout=True)
    style_journal_axes(axes)
    price_index = normalized_price_index(returns_wide)

    ax = axes[0, 0]
    for symbol in returns_wide.columns:
        ax.plot(
            price_index.index,
            price_index[symbol],
            label=symbol,
            color=ASSET_COLORS[symbol],
            linewidth=0.8,
        )
    ax.set_yscale("log")
    ax.set_title("normalized price")
    ax.set_ylabel("normalized price")
    ax.legend(loc="upper left", ncols=3, handlelength=2.2, columnspacing=1.5)

    ax = axes[0, 1]
    for symbol in returns_wide.columns:
        shifted_returns = returns_wide[symbol] + RETURN_OFFSETS[symbol]
        ax.plot(
            returns_wide.index,
            shifted_returns,
            color=ASSET_COLORS[symbol],
            linewidth=0.35,
            alpha=0.85,
        )
        ax.axhline(RETURN_OFFSETS[symbol], color="black", linewidth=0.35, alpha=0.35)
        ax.text(
            returns_wide.index[-1],
            RETURN_OFFSETS[symbol],
            f" {symbol}",
            color=ASSET_COLORS[symbol],
            va="center",
            ha="left",
            fontsize=9,
            clip_on=False,
        )
    ax.set_title("returns")
    ax.set_ylabel("log return + offset")
    ax.margins(x=0.04)

    ax = axes[1, 0]
    for symbol in returns_wide.columns:
        x_values, y_values = empirical_ccdf(returns_wide[symbol].abs())
        ax.loglog(
            x_values,
            y_values,
            label=symbol,
            color=ASSET_COLORS[symbol],
            linewidth=0.9,
        )

    pooled_x, pooled_y = empirical_ccdf(returns_long["log_return"].abs())
    ref_x, ref_y, alpha = tail_reference_line(pooled_x, pooled_y)
    ax.loglog(pooled_x, pooled_y, label="all", color="black", linewidth=1.0)
    if len(ref_x):
        ax.loglog(ref_x, ref_y, color="black", linewidth=0.8, linestyle="--", label=fr"$x^{{-{alpha:.2f}}}$")
    ax.set_title("complementary CDF of returns")
    ax.set_xlabel("|log return|")
    ax.set_ylabel("P(|r| > x)")
    ax.legend(loc="lower left", handlelength=2.2)

    ax = axes[1, 1]
    lags = np.arange(1, 301)
    acf_by_asset = []
    for symbol in returns_wide.columns:
        acf_values = absolute_return_acf(returns_wide[symbol], nlags=300)[1:]
        acf_by_asset.append(acf_values)
        ax.scatter(
            lags,
            acf_values,
            s=5,
            alpha=0.65,
            label=symbol,
            color=ASSET_COLORS[symbol],
            linewidths=0,
        )

    all_acf = np.nanmean(np.vstack(acf_by_asset), axis=0)
    ax.plot(lags, all_acf, color="black", linewidth=1.1, label="all")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title("autocorrelations")
    ax.set_xlabel("lag, trading days")
    ax.set_ylabel("ACF of |r|")
    ax.legend(loc="upper right", handlelength=2.2)

    return fig


def save_figure(fig: plt.Figure, slug: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for extension in ("pdf", "svg", "png"):
        output_path = FIGURES_DIR / f"{slug}.{extension}"
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path}")


def main() -> None:
    returns_long, returns_wide = load_returns()

    for spec in FIGURE_SPECS:
        period_long, period_wide = filter_period(returns_long, returns_wide, spec)
        fig = plot_stylized_facts(period_long, period_wide, spec)
        save_figure(fig, spec.slug)
        plt.close(fig)

    journal_long, journal_wide = filter_period(returns_long, returns_wide, JOURNAL_SPEC)
    fig = plot_journal_stylized_facts(journal_long, journal_wide)
    save_figure(fig, JOURNAL_SPEC.slug)
    plt.close(fig)

    clean_long, clean_wide = filter_period(returns_long, returns_wide, CLEAN_SPEC)
    fig = plot_clean_stylized_facts(clean_long, clean_wide)
    save_figure(fig, CLEAN_SPEC.slug)
    plt.close(fig)


if __name__ == "__main__":
    main()

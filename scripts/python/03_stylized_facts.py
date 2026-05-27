from __future__ import annotations

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

FIGURE_BASE_NAME = "figure_1_stylized_facts_demo_assets"
ASSET_ORDER = ["PETR4", "VALE3", "BBDC4"]
ASSET_COLORS = {
    "PETR4": "#1f77b4",
    "VALE3": "#2ca02c",
    "BBDC4": "#d62728",
}


def load_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    returns_long = pd.read_csv(RETURNS_LONG_PATH, parse_dates=["date"])
    returns_wide = pd.read_csv(RETURNS_WIDE_PATH, parse_dates=["date"])
    returns_wide = returns_wide.set_index("date").sort_index()

    symbols = [symbol for symbol in ASSET_ORDER if symbol in returns_wide.columns]
    returns_wide = returns_wide[symbols]
    returns_long = returns_long[returns_long["symbol"].isin(symbols)].copy()

    return returns_long, returns_wide


def normalized_price_index(returns_wide: pd.DataFrame) -> pd.DataFrame:
    gross_log_return = returns_wide.fillna(0).cumsum()
    return np.exp(gross_log_return) * 100.0


def empirical_ccdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    clean = values.dropna().to_numpy()
    clean = clean[clean >= 1e-4]
    clean.sort()

    n_obs = len(clean)
    ccdf = (n_obs - np.arange(n_obs)) / n_obs

    return clean, ccdf


def absolute_return_acf(series: pd.Series, nlags: int = 60) -> np.ndarray:
    clean = series.abs().dropna()
    if len(clean) <= nlags + 1:
        return np.full(nlags + 1, np.nan)
    return acf(clean, nlags=nlags, fft=True, missing="drop")


def plot_stylized_facts(returns_long: pd.DataFrame, returns_wide: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    price_index = normalized_price_index(returns_wide)

    ax = axes[0, 0]
    for symbol in returns_wide.columns:
        ax.plot(price_index.index, price_index[symbol], label=symbol, color=ASSET_COLORS[symbol], linewidth=1.2)
    ax.set_title("Normalized price index")
    ax.set_ylabel("Index, first observation = 100")
    ax.legend(loc="upper left", ncols=3, frameon=False)

    ax = axes[0, 1]
    for symbol in returns_wide.columns:
        ax.plot(returns_wide.index, returns_wide[symbol], label=symbol, color=ASSET_COLORS[symbol], linewidth=0.7, alpha=0.8)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_title("Daily log returns")
    ax.set_ylabel("Log return")

    ax = axes[1, 0]
    for symbol in returns_wide.columns:
        x_values, y_values = empirical_ccdf(returns_wide[symbol].abs())
        ax.loglog(x_values, y_values, label=symbol, color=ASSET_COLORS[symbol], linewidth=1.2)
    pooled_x, pooled_y = empirical_ccdf(returns_long["log_return"].abs())
    ax.loglog(pooled_x, pooled_y, label="Pooled", color="#222222", linewidth=1.5, linestyle="--")
    ax.set_title("CCDF of absolute returns")
    ax.set_xlabel("|log return|")
    ax.set_ylabel("P(|r| > x)")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1, 1]
    lags = np.arange(1, 61)
    for symbol in returns_wide.columns:
        acf_values = absolute_return_acf(returns_wide[symbol], nlags=60)[1:]
        ax.plot(lags, acf_values, label=symbol, color=ASSET_COLORS[symbol], linewidth=1.2)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_title("Autocorrelation of absolute returns")
    ax.set_xlabel("Lag, trading days")
    ax.set_ylabel("ACF")
    ax.legend(loc="upper right", frameon=False)

    fig.suptitle("Stylized facts for B3 demo assets: PETR4, VALE3, BBDC4", fontsize=14)
    return fig


def save_figure(fig: plt.Figure) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for extension in ("pdf", "svg", "png"):
        output_path = FIGURES_DIR / f"{FIGURE_BASE_NAME}.{extension}"
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path}")


def main() -> None:
    returns_long, returns_wide = load_returns()
    fig = plot_stylized_facts(returns_long, returns_wide)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()

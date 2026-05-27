from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns by symbol using the `price` column.

    Expected columns:
    - date
    - symbol
    - price
    """
    required_columns = {"date", "symbol", "price"}
    missing = required_columns - set(prices_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = prices_df.copy()
    df = df.sort_values(["symbol", "date"])

    df["log_price"] = np.log(df["price"])
    df["log_return"] = df.groupby("symbol")["log_price"].diff()

    df = df.dropna(subset=["log_return"]).reset_index(drop=True)

    return df


def returns_to_wide(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long returns DataFrame to wide matrix: date x symbol.
    """
    required_columns = {"date", "symbol", "log_return"}
    missing = required_columns - set(returns_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    wide = (
        returns_df
        .pivot(index="date", columns="symbol", values="log_return")
        .sort_index()
    )

    return wide


def summarize_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics by symbol.
    """
    summary = (
        returns_df
        .groupby("symbol")["log_return"]
        .agg(
            n_obs="count",
            mean="mean",
            std="std",
            min="min",
            max="max",
            skew="skew",
        )
        .reset_index()
    )

    summary["abs_mean"] = returns_df.groupby("symbol")["log_return"].apply(
        lambda x: x.abs().mean()
    ).values

    return summary.sort_values("n_obs", ascending=False)

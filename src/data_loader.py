from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.db import get_client, get_daily_table


def format_symbols_for_sql(symbols: Iterable[str]) -> str:
    quoted = [f"'{symbol}'" for symbol in symbols]
    return ", ".join(quoted)


def load_daily_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
    price_column: str = "adj_close",
) -> pd.DataFrame:
    """
    Load adjusted daily prices from ClickHouse for a selected list of assets.
    """
    if not symbols:
        raise ValueError("symbols cannot be empty")

    if price_column not in {"adj_close", "close"}:
        raise ValueError("price_column must be either 'adj_close' or 'close'")

    client = get_client()
    table = get_daily_table()
    symbols_sql = format_symbols_for_sql(symbols)

    query = f"""
    SELECT
        date,
        symbol,
        company_name,
        {price_column} AS price,
        adj_close,
        close,
        volume,
        financial_volume
    FROM {table}
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
      AND symbol IN ({symbols_sql})
      AND {price_column} > 0
    ORDER BY
        symbol,
        date
    """

    df = client.query_df(query)

    if df.empty:
        raise RuntimeError("No price data returned from ClickHouse.")

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)

    return df

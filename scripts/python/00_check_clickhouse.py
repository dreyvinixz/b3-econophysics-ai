from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.db import get_client, get_daily_table


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    client = get_client()
    table = get_daily_table()

    print_section("ClickHouse connection test")

    result = client.query("SELECT 1")
    print(f"SELECT 1 result: {result.result_rows}")

    print_section("Configured table")
    print(table)

    print_section("DESCRIBE TABLE")
    describe_df = client.query_df(f"DESCRIBE TABLE {table}")
    print(describe_df.to_string(index=False))

    print_section("Table summary")
    summary_df = client.query_df(
        f"""
        SELECT
            min(date) AS first_date,
            max(date) AS last_date,
            count() AS rows,
            uniq(symbol) AS n_symbols
        FROM {table}
        """
    )
    print(summary_df.to_string(index=False))

    print_section("Duplicate check by symbol/date")
    duplicates_df = client.query_df(
        f"""
        SELECT
            count() AS duplicated_keys
        FROM
        (
            SELECT
                symbol,
                date,
                count() AS n
            FROM {table}
            GROUP BY
                symbol,
                date
            HAVING n > 1
        )
        """
    )
    print(duplicates_df.to_string(index=False))

    print_section("Null/invalid adjusted close check")
    adj_close_df = client.query_df(
        f"""
        SELECT
            countIf(adj_close IS NULL) AS null_adj_close,
            countIf(adj_close <= 0) AS non_positive_adj_close,
            countIf(close IS NULL) AS null_close,
            countIf(close <= 0) AS non_positive_close
        FROM {table}
        """
    )
    print(adj_close_df.to_string(index=False))

    print_section("Top 30 assets by average financial volume")
    top_assets_df = client.query_df(
        f"""
        SELECT
            symbol,
            any(company_name) AS company_name,
            min(date) AS first_date,
            max(date) AS last_date,
            count() AS n_days,
            round(avg(volume), 2) AS avg_volume,
            round(avg(financial_volume), 2) AS avg_financial_volume
        FROM {table}
        WHERE adj_close > 0
        GROUP BY symbol
        ORDER BY avg_financial_volume DESC
        LIMIT 30
        """
    )
    print(top_assets_df.to_string(index=False))


if __name__ == "__main__":
    main()
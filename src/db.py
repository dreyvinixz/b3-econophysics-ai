from __future__ import annotations

from pathlib import Path
from typing import Any

import clickhouse_connect

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "clickhouse.toml"


def load_clickhouse_config() -> dict[str, Any]:
    """
    Load ClickHouse configuration from config/clickhouse.toml.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"ClickHouse config file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("rb") as file:
        return tomllib.load(file)


def get_client():
    """
    Create a ClickHouse client using the local research configuration.

    This assumes the QuantBase backend Docker stack exposes ClickHouse HTTP
    on localhost:8123.
    """
    config = load_clickhouse_config()
    ch_config = config["clickhouse"]

    return clickhouse_connect.get_client(
        host=ch_config["host"],
        port=int(ch_config["port"]),
        username=ch_config["username"],
        password=ch_config["password"],
        database=ch_config["database"],
    )


def get_daily_table() -> str:
    """
    Return the configured daily candles table name.
    """
    config = load_clickhouse_config()
    return config["tables"]["daily"]


def query_df(sql: str):
    """
    Convenience wrapper to execute a SQL query and return a pandas DataFrame.
    """
    client = get_client()
    return client.query_df(sql)
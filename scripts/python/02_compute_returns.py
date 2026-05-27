from __future__ import annotations

from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_daily_prices
from src.returns import compute_log_returns, returns_to_wide, summarize_returns


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
ASSETS_CONFIG_PATH = PROJECT_ROOT / "config" / "assets_universe.yaml"


def load_assets_config() -> dict:
    with ASSETS_CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_symbols(config: dict, key: str) -> list[str]:
    section = config[key]

    if isinstance(section, dict) and "symbols" in section:
        return section["symbols"]

    raise KeyError(f"Could not find symbols for section: {key}")


def save_returns_for_universe(
    universe_name: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    price_column: str = "adj_close",
) -> None:
    print(f"\nProcessing universe: {universe_name}")
    print(f"Symbols: {len(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Price column: {price_column}")

    prices = load_daily_prices(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        price_column=price_column,
    )

    returns_long = compute_log_returns(prices)
    returns_wide = returns_to_wide(returns_long)
    summary = summarize_returns(returns_long)

    long_path = OUTPUT_DIR / f"{universe_name}_returns_long_1998_2025.csv"
    wide_path = OUTPUT_DIR / f"{universe_name}_returns_wide_1998_2025.csv"
    summary_path = OUTPUT_DIR / f"{universe_name}_returns_summary_1998_2025.csv"

    returns_long.to_csv(long_path, index=False)
    returns_wide.to_csv(wide_path)
    summary.to_csv(summary_path, index=False)

    print(f"Saved long returns: {long_path}")
    print(f"Saved wide returns: {wide_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Long shape: {returns_long.shape}")
    print(f"Wide shape: {returns_wide.shape}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_assets_config()

    start_date = "1998-03-16"
    end_date = "2025-12-31"
    price_column = "adj_close"

    demo_assets = extract_symbols(config, "demo_assets")
    core_historical = extract_symbols(config, "core_historical")
    modern_top_50 = extract_symbols(config, "modern_liquid_top_50")

    save_returns_for_universe(
        universe_name="demo_assets",
        symbols=demo_assets,
        start_date=start_date,
        end_date=end_date,
        price_column=price_column,
    )

    save_returns_for_universe(
        universe_name="core_historical",
        symbols=core_historical,
        start_date=start_date,
        end_date=end_date,
        price_column=price_column,
    )

    save_returns_for_universe(
        universe_name="modern_liquid_top_50",
        symbols=modern_top_50,
        start_date=start_date,
        end_date=end_date,
        price_column=price_column,
    )


if __name__ == "__main__":
    main()

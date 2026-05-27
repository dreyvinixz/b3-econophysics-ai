import subprocess
import os

# Create outputs/tables if it doesn't exist
os.makedirs("outputs/tables", exist_ok=True)

query_core = """SELECT
    symbol,
    any(company_name) AS company_name,
    any(tipo_mercado) AS tipo_mercado,
    any(cod_bdi) AS cod_bdi,
    any(specification) AS specification,
    min(date) AS first_date,
    max(date) AS last_date,
    count() AS n_days,
    round(avg(financial_volume), 2) AS avg_financial_volume
FROM (
    SELECT symbol, company_name, tipo_mercado, cod_bdi, specification, date, financial_volume
    FROM quantbase.candles_1d
    WHERE date BETWEEN '1998-03-16' AND '2025-12-31'
      AND adj_close > 0
      AND cod_bdi = '02'
      AND specification NOT LIKE '%DR%'
)
GROUP BY symbol
HAVING n_days >= 5000
ORDER BY avg_financial_volume DESC;"""

query_modern = """SELECT
    symbol,
    any(company_name) AS company_name,
    any(tipo_mercado) AS tipo_mercado,
    any(cod_bdi) AS cod_bdi,
    any(specification) AS specification,
    min(date) AS first_date,
    max(date) AS last_date,
    count() AS n_days,
    round(avg(financial_volume), 2) AS avg_financial_volume
FROM (
    SELECT symbol, company_name, tipo_mercado, cod_bdi, specification, date, financial_volume
    FROM quantbase.candles_1d
    WHERE date BETWEEN '1998-03-16' AND '2025-12-31'
      AND adj_close > 0
      AND cod_bdi = '02'
      AND specification NOT LIKE '%DR%'
)
GROUP BY symbol
HAVING n_days >= 1000
ORDER BY avg_financial_volume DESC
LIMIT 150;"""

query_modern_50 = """SELECT
    symbol,
    any(company_name) AS company_name,
    any(tipo_mercado) AS tipo_mercado,
    any(cod_bdi) AS cod_bdi,
    any(specification) AS specification,
    min(date) AS first_date,
    max(date) AS last_date,
    count() AS n_days,
    round(avg(financial_volume), 2) AS avg_financial_volume
FROM (
    SELECT symbol, company_name, tipo_mercado, cod_bdi, specification, date, financial_volume
    FROM quantbase.candles_1d
    WHERE date BETWEEN '1998-03-16' AND '2025-12-31'
      AND adj_close > 0
      AND cod_bdi = '02'
      AND specification NOT LIKE '%DR%'
)
GROUP BY symbol
HAVING n_days >= 1000
ORDER BY avg_financial_volume DESC
LIMIT 50;"""

def run_query_and_save(query, output_file):
    cmd = [
        "docker", "exec", "quantbase-clickhouse-dev",
        "clickhouse-client", "--query", query, "--format", "CSVWithNames"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode == 0:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(res.stdout)
        
        # Count lines (subtract 1 for header)
        lines = len([line for line in res.stdout.strip().split('\n') if line]) - 1
        return max(0, lines)
    else:
        print(f"Error running query for {output_file}: {res.stderr}")
        return 0

print("Generating universe tables...")

core_count = run_query_and_save(query_core, "outputs/tables/core_historical_assets_1998_2025.csv")
modern_count = run_query_and_save(query_modern, "outputs/tables/modern_liquid_assets_1998_2025.csv")
modern_50_count = run_query_and_save(query_modern_50, "outputs/tables/modern_liquid_top_50_1998_2025.csv")

print()
print(f"Core historical assets: {core_count}")
print(f"Modern liquid assets: {modern_count}")
print(f"Modern top 50 assets: {modern_50_count}")

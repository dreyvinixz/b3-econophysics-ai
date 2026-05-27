from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"


def compute_statistics(returns_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    df = returns_df[
        (returns_df["date"] >= start_date) &
        (returns_df["date"] <= end_date)
    ].copy()

    records = []
    
    for symbol, group in df.groupby("symbol"):
        r = group["log_return"].dropna().values
        abs_r = np.abs(r)
        
        n_obs = len(r)
        if n_obs < 60:
            continue
            
        r_mean = np.mean(r)
        r_std = np.std(r, ddof=1)
        r_skew = skew(r, bias=False)
        r_kurt = kurtosis(r, fisher=True, bias=False)
        r_min = np.min(r)
        r_max = np.max(r)
        var_1pct = np.percentile(r, 1)
        var_5pct = np.percentile(r, 5)
        
        gt_10 = np.sum(abs_r > 0.10)
        gt_20 = np.sum(abs_r > 0.20)
        
        acf_vals = acf(abs_r, nlags=60, fft=True)
        acf_1 = acf_vals[1]
        acf_5 = acf_vals[5]
        acf_20 = acf_vals[20]
        acf_60 = acf_vals[60]
        
        records.append({
            "symbol": symbol,
            "n_obs": n_obs,
            "mean": r_mean,
            "std": r_std,
            "skewness": r_skew,
            "excess_kurtosis": r_kurt,
            "min": r_min,
            "max": r_max,
            "var_1pct": var_1pct,
            "var_5pct": var_5pct,
            "n_abs_return_gt_10pct": gt_10,
            "n_abs_return_gt_20pct": gt_20,
            "acf_abs_lag_1": acf_1,
            "acf_abs_lag_5": acf_5,
            "acf_abs_lag_20": acf_20,
            "acf_abs_lag_60": acf_60,
        })
        
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    input_path = OUTPUT_DIR / "demo_assets_returns_long_1998_2025.csv"
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        sys.exit(1)
        
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Configuration 1: 2006 - 2025
    stats_2006 = compute_statistics(df, "2006-01-01", "2025-12-31")
    out_csv_2006 = OUTPUT_DIR / "table_1_descriptive_stats_2006_2025.csv"
    out_tex_2006 = OUTPUT_DIR / "table_1_descriptive_stats_2006_2025.tex"
    
    stats_2006.to_csv(out_csv_2006, index=False)
    
    # Format for LaTeX
    formatters = {
        "mean": "{:.5f}".format,
        "std": "{:.5f}".format,
        "skewness": "{:.2f}".format,
        "excess_kurtosis": "{:.2f}".format,
        "min": "{:.4f}".format,
        "max": "{:.4f}".format,
        "var_1pct": "{:.4f}".format,
        "var_5pct": "{:.4f}".format,
        "acf_abs_lag_1": "{:.3f}".format,
        "acf_abs_lag_5": "{:.3f}".format,
        "acf_abs_lag_20": "{:.3f}".format,
        "acf_abs_lag_60": "{:.3f}".format,
    }
    
    with open(out_tex_2006, "w") as f:
        f.write(stats_2006.to_latex(index=False, float_format="%.4f", formatters=formatters))
    
    print(f"Saved {out_csv_2006}")
    print(f"Saved {out_tex_2006}")
    
    # Configuration 2: 1998 - 2025
    stats_1998 = compute_statistics(df, "1998-03-16", "2025-12-31")
    out_csv_1998 = OUTPUT_DIR / "table_1_descriptive_stats_1998_2025.csv"
    
    stats_1998.to_csv(out_csv_1998, index=False)
    print(f"Saved {out_csv_1998}")


if __name__ == "__main__":
    main()

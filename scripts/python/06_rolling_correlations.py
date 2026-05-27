from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress warnings from all-NaN slices in rolling corr
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def compute_market_rolling_correlation(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Computes rolling average pairwise correlation for the entire market.
    """
    dates = df["date"].values
    data = df.drop(columns=["date"])
    
    records = []
    
    # We iterate starting from 'window' to the end
    for i in range(window, len(df)):
        window_data = data.iloc[i - window : i]
        
        # Drop columns with all NaNs in this window
        valid_cols = window_data.columns[window_data.notna().any()]
        if len(valid_cols) < 2:
            continue
            
        corr_mat = window_data[valid_cols].corr(method="pearson").values
        
        # Extract upper triangle without diagonal
        n = corr_mat.shape[0]
        upper_tri_idx = np.triu_indices(n, k=1)
        corrs = corr_mat[upper_tri_idx]
        
        # Drop NaNs
        corrs = corrs[~np.isnan(corrs)]
        
        if len(corrs) > 0:
            records.append({
                "date": dates[i],
                "avg_correlation": np.mean(corrs),
                "median_correlation": np.median(corrs),
                "std_correlation": np.std(corrs),
                "p25_correlation": np.percentile(corrs, 25),
                "p75_correlation": np.percentile(corrs, 75),
                "n_assets": len(valid_cols),
                "n_pairs": len(corrs)
            })
            
    return pd.DataFrame(records)


def compute_pairwise_dynamics(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Computes P.Corr 20d, P.Corr 60d, and EWMA (halflife 20) for given pairs.
    """
    data = df.copy()
    data.set_index("date", inplace=True)
    
    results = []
    
    for s1, s2 in pairs:
        pair_df = data[[s1, s2]].dropna()
        if pair_df.empty:
            continue
            
        corr_20d = pair_df[s1].rolling(20).corr(pair_df[s2])
        corr_60d = pair_df[s1].rolling(60).corr(pair_df[s2])
        
        # EWMA covariance and variances
        ewma_cov = pair_df[s1].ewm(halflife=20).cov(pair_df[s2])
        ewma_var1 = pair_df[s1].ewm(halflife=20).var()
        ewma_var2 = pair_df[s2].ewm(halflife=20).var()
        
        ewma_corr = ewma_cov / np.sqrt(ewma_var1 * ewma_var2)
        
        res = pd.DataFrame({
            "date": pair_df.index,
            "pair": f"{s1} x {s2}",
            "pearson_20d": corr_20d,
            "pearson_60d": corr_60d,
            "ewma": ewma_corr
        })
        results.append(res)
        
    return pd.concat(results).reset_index(drop=True)


def plot_rolling_average(df_252: pd.DataFrame, df_504: pd.DataFrame, pdf_path: Path, png_path: Path):
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df_252["date"], df_252["avg_correlation"], label="252-day rolling mean", linewidth=1.2, color="steelblue")
    ax.plot(df_504["date"], df_504["avg_correlation"], label="504-day rolling mean", linewidth=1.5, color="darkred")
    
    # Crisis markers
    crises = [
        ("2008-09-01", "2009-03-01", "2008 Crisis"),
        ("2015-01-01", "2016-12-31", "2015-16 Recession"),
        ("2020-03-01", "2022-12-31", "2020 COVID Shock")
    ]
    
    for start, end, name in crises:
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="grey", alpha=0.1)
        ax.text(pd.to_datetime(start), ax.get_ylim()[1] * 0.95, name, rotation=90, 
                verticalalignment='top', fontsize=10, color="black",
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    ax.set_ylabel("Mean pairwise correlation", fontsize=12)
    ax.set_xlabel("time", fontsize=12)
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.legend(loc="upper left", frameon=True, edgecolor="lightgrey")
    
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dynamic_pairwise(df_pairs: pd.DataFrame, pdf_path: Path, png_path: Path):
    plt.rcParams["font.family"] = "serif"
    
    pairs = df_pairs["pair"].unique()
    n_pairs = len(pairs)
    
    fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 2.5 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]
        
    for ax, pair in zip(axes, pairs):
        data = df_pairs[df_pairs["pair"] == pair]
        
        ax.plot(data["date"], data["pearson_20d"], color="lightblue", linewidth=0.45, alpha=0.25, label="P.Corr. 20d")
        ax.plot(data["date"], data["pearson_60d"], color="steelblue", linewidth=0.80, alpha=0.95, label="P.Corr. 60d")
        ax.plot(data["date"], data["ewma"], color="darkred", linewidth=0.95, linestyle="--", alpha=1.00, label="EWMA")
        
        ax.set_ylim(-0.5, 1.0)
        ax.set_ylabel("Correlation", fontsize=10)
        ax.set_title(pair, fontsize=12, pad=5)
        ax.grid(True, alpha=0.15, linestyle="--")
        
    axes[-1].set_xlabel("time", fontsize=12)
    
    # Legend in the last panel
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=True, edgecolor="lightgrey")
    
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    (FIGURES_DIR / "vector").mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "preview").mkdir(parents=True, exist_ok=True)
    
    core_path = OUTPUT_DIR / "core_historical_returns_wide_1998_2025.csv"
    print(f"Loading {core_path}...")
    df = pd.read_csv(core_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded returns matrix shape: {df.shape}")
    
    # --- Phase 6.3: Rolling Average Market Correlation ---
    print("\nComputing rolling market correlation (252 and 504 days)...")
    res_252 = compute_market_rolling_correlation(df, 252)
    res_504 = compute_market_rolling_correlation(df, 504)
    
    out_csv_252 = OUTPUT_DIR / "core_historical_rolling_average_correlation_1998_2025.csv"
    res_252.to_csv(out_csv_252, index=False)
    
    pdf_path_4 = FIGURES_DIR / "vector" / "figure_4_rolling_average_correlation.pdf"
    png_path_4 = FIGURES_DIR / "preview" / "figure_4_rolling_average_correlation.png"
    
    plot_rolling_average(res_252, res_504, pdf_path_4, png_path_4)
    
    print(f"Rolling window sizes: 252, 504")
    print(f"Number of rolling windows (252): {len(res_252)}")
    print(f"Average correlation summary:\n{res_252['avg_correlation'].describe()}")
    
    top_dates = res_252.sort_values("avg_correlation", ascending=False).head(5)
    low_dates = res_252.sort_values("avg_correlation").head(5)
    print(f"\nTop high-correlation dates:\n{top_dates[['date', 'avg_correlation']].to_string(index=False)}")
    print(f"\nLowest correlation dates:\n{low_dates[['date', 'avg_correlation']].to_string(index=False)}")
    print(f"Saved output paths: {out_csv_252}, {pdf_path_4}, {png_path_4}")
    
    # --- Phase 6.4: Dynamic Pairwise Correlations ---
    print("\nComputing dynamic pairwise correlations (2006-2025)...")
    df_2006 = df[(df["date"] >= "2006-01-01") & (df["date"] <= "2025-12-31")]
    
    pairs_list = [
        ("PETR4", "VALE3"),
        ("BBDC4", "BBAS3"),
        ("GGBR4", "USIM5"),
        ("ELET3", "CMIG4")
    ]
    
    res_pairs = compute_pairwise_dynamics(df_2006, pairs_list)
    
    out_csv_pairs = OUTPUT_DIR / "pairwise_dynamic_correlations_2006_2025.csv"
    res_pairs.to_csv(out_csv_pairs, index=False)
    
    pdf_path_5 = FIGURES_DIR / "vector" / "figure_5_dynamic_pairwise_correlations.pdf"
    png_path_5 = FIGURES_DIR / "preview" / "figure_5_dynamic_pairwise_correlations.png"
    
    plot_dynamic_pairwise(res_pairs, pdf_path_5, png_path_5)
    
    print(f"Saved output paths: {out_csv_pairs}, {pdf_path_5}, {png_path_5}")


if __name__ == "__main__":
    main()

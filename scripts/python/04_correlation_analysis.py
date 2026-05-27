from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def compute_correlation_matrices(df: pd.DataFrame):
    """
    Computes pairwise Pearson correlation and pairwise observation counts.
    Returns (corr_matrix, nobs_matrix, pairs_df)
    """
    # Exclude date column
    data = df.drop(columns=["date"]) if "date" in df.columns else df
    
    # Pairwise correlation
    corr_matrix = data.corr(method="pearson", min_periods=10)
    
    # Pairwise observation counts
    # A bit trickier: count non-null overlaps
    notnull = data.notnull().astype(int)
    nobs_matrix = notnull.T.dot(notnull)
    
    # Flatten to pairs
    symbols = corr_matrix.columns
    records = []
    
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            s1 = symbols[i]
            s2 = symbols[j]
            records.append({
                "symbol1": s1,
                "symbol2": s2,
                "correlation": corr_matrix.loc[s1, s2],
                "n_obs": nobs_matrix.loc[s1, s2]
            })
            
    pairs_df = pd.DataFrame(records).dropna(subset=["correlation"])
    pairs_df = pairs_df.sort_values("correlation", ascending=False)
    
    return corr_matrix, nobs_matrix, pairs_df


def plot_correlation_histogram(pairs_df: pd.DataFrame, num_assets: int, pdf_path: Path, png_path: Path):
    corrs = pairs_df["correlation"].dropna().values
    mean_corr = corrs.mean()
    median_corr = np.median(corrs)
    n_pairs = len(corrs)
    
    # Configure serif font
    plt.rcParams["font.family"] = "serif"
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot histogram with discrete styling
    ax.hist(corrs, bins=60, density=True, alpha=0.5, color="steelblue", edgecolor="white", linewidth=0.5)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_corr, color="black", linestyle="--", linewidth=1.2, label=f"Mean: {mean_corr:.2f}")
    ax.axvline(median_corr, color="black", linestyle=":", linewidth=1.2, label=f"Median: {median_corr:.2f}")
    
    # Textbox with N and N_pairs
    textstr = f"$N$ = {num_assets}\n$N_{{pairs}}$ = {n_pairs}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgrey")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=props)
    
    ax.set_xlabel(r"Correlation Coefficient $\rho_{ij}$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    
    # Faint grid
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.legend(loc="upper right", frameon=True, edgecolor="lightgrey")
    
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    (FIGURES_DIR / "vector").mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "preview").mkdir(parents=True, exist_ok=True)
    
    # --- DEMO ASSETS ---
    demo_path = OUTPUT_DIR / "demo_assets_returns_wide_1998_2025.csv"
    print(f"Loading {demo_path}...")
    df_demo = pd.read_csv(demo_path)
    df_demo["date"] = pd.to_datetime(df_demo["date"])
    
    # Filter 2006-2025
    df_demo_filt = df_demo[
        (df_demo["date"] >= "2006-01-01") & 
        (df_demo["date"] <= "2025-12-31")
    ].copy()
    
    corr_demo, _, _ = compute_correlation_matrices(df_demo_filt)
    out_demo_corr = OUTPUT_DIR / "demo_assets_correlation_matrix_2006_2025.csv"
    corr_demo.to_csv(out_demo_corr)
    print(f"Saved {out_demo_corr}")
    
    # --- CORE HISTORICAL ---
    core_path = OUTPUT_DIR / "core_historical_returns_wide_1998_2025.csv"
    print(f"Loading {core_path}...")
    df_core = pd.read_csv(core_path)
    df_core["date"] = pd.to_datetime(df_core["date"])
    
    # Filter 1998-2025 (all data)
    df_core_filt = df_core[
        (df_core["date"] >= "1998-03-16") & 
        (df_core["date"] <= "2025-12-31")
    ].copy()
    
    corr_core, nobs_core, pairs_core = compute_correlation_matrices(df_core_filt)
    
    out_core_corr = OUTPUT_DIR / "core_historical_correlation_matrix_1998_2025.csv"
    out_core_nobs = OUTPUT_DIR / "core_historical_correlation_nobs_1998_2025.csv"
    out_core_pairs = OUTPUT_DIR / "core_historical_correlation_pairs_1998_2025.csv"
    
    corr_core.to_csv(out_core_corr)
    nobs_core.to_csv(out_core_nobs)
    pairs_core.to_csv(out_core_pairs, index=False)
    
    print(f"Saved {out_core_corr}")
    print(f"Saved {out_core_nobs}")
    print(f"Saved {out_core_pairs}")
    
    # --- PLOT HISTOGRAM ---
    pdf_path = FIGURES_DIR / "vector" / "figure_2_correlation_histogram.pdf"
    png_path = FIGURES_DIR / "preview" / "figure_2_correlation_histogram.png"
    
    plot_correlation_histogram(pairs_core, len(corr_core.columns), pdf_path, png_path)
    print(f"Saved Figure 2 (Correlation Histogram) to {pdf_path} and {png_path}")

if __name__ == "__main__":
    main()

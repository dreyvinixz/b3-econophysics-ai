from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def marcenko_pastur_pdf(x: np.ndarray, Q: float, sigma2: float = 1.0) -> np.ndarray:
    """
    Computes the Marcenko-Pastur probability density function.
    """
    lambda_min = sigma2 * (1 + 1/Q - 2*np.sqrt(1/Q))
    lambda_max = sigma2 * (1 + 1/Q + 2*np.sqrt(1/Q))
    
    # Values outside [lambda_min, lambda_max] have density 0
    pdf = np.zeros_like(x)
    valid_idx = (x >= lambda_min) & (x <= lambda_max)
    
    x_val = x[valid_idx]
    pdf[valid_idx] = (Q / (2 * np.pi * sigma2 * x_val)) * np.sqrt((lambda_max - x_val) * (x_val - lambda_min))
    return pdf


def main() -> None:
    (FIGURES_DIR / "vector").mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "preview").mkdir(parents=True, exist_ok=True)
    
    core_path = OUTPUT_DIR / "core_historical_returns_wide_1998_2025.csv"
    print(f"Loading {core_path}...")
    df = pd.read_csv(core_path)
    
    # Load sector map
    sector_map_path = OUTPUT_DIR / "assets_sector_map.csv"
    sector_df = pd.read_csv(sector_map_path)
    
    # Complete-case filtering
    data = df.drop(columns=["date"]) if "date" in df.columns else df
    data = data.dropna(how="any")
    
    T, N = data.shape
    Q = T / N
    
    if T <= N:
        print(f"Error: Not enough observations after dropna. T={T}, N={N}")
        sys.exit(1)
        
    # Calculate empirical correlation matrix
    corr_matrix = data.corr(method="pearson").values
    symbols = data.columns.tolist()
    
    # Calculate eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    # Sort descending
    idx = eigvals.argsort()[::-1]
    eigvals_desc = eigvals[idx]
    eigvecs_desc = eigvecs[:, idx]
    
    # Marcenko-Pastur bounds
    sigma2 = 1.0
    lambda_min = sigma2 * (1 + 1/Q - 2*np.sqrt(1/Q))
    lambda_max = sigma2 * (1 + 1/Q + 2*np.sqrt(1/Q))
    
    largest_eigenvalue = eigvals_desc[0]
    n_above_lambda_max = int(np.sum(eigvals_desc > lambda_max))
    
    # Output terminal summary
    print("\n--- RMT Summary ---")
    print(f"N assets: {N}")
    print(f"T observations after complete-case filtering: {T}")
    print(f"Q = T / N: {Q:.4f}")
    print(f"lambda_min: {lambda_min:.4f}")
    print(f"lambda_max: {lambda_max:.4f}")
    print(f"largest eigenvalue: {largest_eigenvalue:.4f}")
    print(f"number of eigenvalues above lambda_max: {n_above_lambda_max}")
    print("-------------------\n")
    
    # Save CSVs for eigenvalues
    eig_df = pd.DataFrame({
        "rank": np.arange(1, N + 1),
        "eigenvalue": eigvals_desc
    })
    out_eig = OUTPUT_DIR / "rmt_eigenvalues_core_historical_1998_2025.csv"
    eig_df.to_csv(out_eig, index=False)
    
    summary_df = pd.DataFrame([{
        "N": N, "T": T, "Q": Q,
        "lambda_min": lambda_min, "lambda_max": lambda_max,
        "max_empirical_eigenvalue": largest_eigenvalue,
        "n_above_lambda_max": n_above_lambda_max
    }])
    out_summary = OUTPUT_DIR / "rmt_summary_core_historical_1998_2025.csv"
    summary_df.to_csv(out_summary, index=False)
    
    # --- Eigenvector Loading Analysis (Phase 7.1) ---
    loading_records = []
    for k in range(n_above_lambda_max):
        vec = eigvecs_desc[:, k]
        # Sign convention
        if vec.sum() < 0:
            vec = -vec
            
        rank = k + 1
        eigenvalue = eigvals_desc[k]
        
        # Sort loadings by absolute value to determine rank
        abs_loadings = np.abs(vec)
        sort_indices = np.argsort(abs_loadings)[::-1]
        
        loading_ranks = np.zeros(N, dtype=int)
        loading_ranks[sort_indices] = np.arange(1, N + 1)
        
        for i, sym in enumerate(symbols):
            meta = sector_df[sector_df["symbol"] == sym]
            c_name = meta["company_name"].values[0] if len(meta) else "Unknown"
            sector = meta["sector"].values[0] if len(meta) else "Unknown"
            subsect = meta["subsector"].values[0] if len(meta) else "Unknown"
            segment = meta["segment"].values[0] if len(meta) else "Unknown"
            
            loading_records.append({
                "eigen_rank": rank,
                "eigenvalue": eigenvalue,
                "above_mp_lambda_max": True,
                "symbol": sym,
                "company_name": c_name,
                "sector": sector,
                "subsector": subsect,
                "segment": segment,
                "loading": vec[i],
                "abs_loading": abs_loadings[i],
                "loading_rank": loading_ranks[i]
            })
            
    loadings_df = pd.DataFrame(loading_records)
    out_loadings = OUTPUT_DIR / "rmt_top_eigenvector_loadings_core_historical_1998_2025.csv"
    loadings_df.to_csv(out_loadings, index=False)
    print(f"Saved {out_loadings}")
    
    # Sector summary
    sector_summary_records = []
    for rank in range(1, n_above_lambda_max + 1):
        df_rank = loadings_df[loadings_df["eigen_rank"] == rank]
        sectors = df_rank["sector"].unique()
        eigenvalue = df_rank["eigenvalue"].iloc[0]
        
        for sec in sectors:
            df_sec = df_rank[df_rank["sector"] == sec]
            n_assets_sec = len(df_sec)
            sum_abs = df_sec["abs_loading"].sum()
            mean_abs = df_sec["abs_loading"].mean()
            max_abs = df_sec["abs_loading"].max()
            
            dominant_row = df_sec.loc[df_sec["abs_loading"].idxmax()]
            dominant_sym = dominant_row["symbol"]
            dominant_load = dominant_row["loading"]
            
            sector_summary_records.append({
                "eigen_rank": rank,
                "eigenvalue": eigenvalue,
                "sector": sec,
                "n_assets": n_assets_sec,
                "sum_abs_loading": sum_abs,
                "mean_abs_loading": mean_abs,
                "max_abs_loading": max_abs,
                "dominant_symbol": dominant_sym,
                "dominant_loading": dominant_load
            })
            
    sector_summary_df = pd.DataFrame(sector_summary_records)
    out_sector_sum = OUTPUT_DIR / "rmt_eigenvector_sector_summary_core_historical_1998_2025.csv"
    sector_summary_df.to_csv(out_sector_sum, index=False)
    print(f"Saved {out_sector_sum}")
    
    # Plot Figure 6
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(max(0.01, eigvals_desc[-1])), np.log10(eigvals_desc[0]), 60)
    ax.hist(eigvals_desc, bins=bins, density=True, color="steelblue", alpha=0.6, edgecolor="white", label="Empirical Eigenvalues")
    x_mp = np.linspace(lambda_min, lambda_max, 500)
    y_mp = marcenko_pastur_pdf(x_mp, Q, sigma2)
    ax.plot(x_mp, y_mp, color="darkred", linewidth=2, label="Marčenko-Pastur distribution")
    ax.axvline(lambda_max, color="black", linestyle="--", linewidth=1, label=f"$\lambda_{{max}} = {lambda_max:.2f}$")
    ax.axvline(largest_eigenvalue, color="dimgrey", linestyle=":", linewidth=1, label=f"Market Mode = {largest_eigenvalue:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Eigenvalue $\lambda$ (log scale)", fontsize=12)
    ax.set_ylabel("Density (log scale)", fontsize=12)
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.legend(loc="upper right", frameon=True, edgecolor="lightgrey")
    pdf_path_6 = FIGURES_DIR / "vector" / "figure_6_rmt_eigenvalue_spectrum.pdf"
    png_path_6 = FIGURES_DIR / "preview" / "figure_6_rmt_eigenvalue_spectrum.png"
    plt.tight_layout()
    plt.savefig(pdf_path_6, format="pdf", bbox_inches="tight")
    plt.savefig(png_path_6, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Top 8 assets table
    top_8_records = []
    for rank in range(1, n_above_lambda_max + 1):
        df_rank = loadings_df[loadings_df["eigen_rank"] == rank]
        top_8 = df_rank.sort_values("abs_loading", ascending=False).head(8)
        top_8_records.append(top_8)
        
    top_8_df = pd.concat(top_8_records, ignore_index=True)
    out_top_8 = OUTPUT_DIR / "rmt_top_8_assets_per_eigenvector_core_historical_1998_2025.csv"
    top_8_df.to_csv(out_top_8, index=False)
    print(f"Saved {out_top_8}")
    
    # Helper to plot Figure 7
    def plot_figure_7(df_data, is_top15=False):
        suffix = "top_loadings" if is_top15 else "all_assets"
        fig, axes = plt.subplots(n_above_lambda_max, 1, figsize=(10, 2.5 * n_above_lambda_max), sharex=False)
        if n_above_lambda_max == 1:
            axes = [axes]
            
        global_sort_order = loadings_df[loadings_df["eigen_rank"] == 1].sort_values("loading", ascending=False)["symbol"].tolist()
            
        for k, ax in enumerate(axes):
            rank = k + 1
            df_rank = df_data[df_data["eigen_rank"] == rank].copy()
            
            if is_top15:
                # Sort by abs_loading to get top 15, but preserve original sign
                df_rank = df_rank.sort_values("abs_loading", ascending=False).head(15)
                # Sort these top 15 by loading for better visual
                df_rank = df_rank.sort_values("loading", ascending=False)
            else:
                df_rank["symbol"] = pd.Categorical(df_rank["symbol"], categories=global_sort_order, ordered=True)
                df_rank = df_rank.sort_values("symbol")
            
            bars = ax.bar(df_rank["symbol"].astype(str), df_rank["loading"], color="darkgray", edgecolor="none")
            
            # Title logic
            if rank == 1:
                title_str = f"Eigenvector 1 ($\lambda = {df_rank['eigenvalue'].iloc[0]:.2f}$) | Market Mode"
            else:
                top_sec = sector_summary_df[sector_summary_df["eigen_rank"] == rank].sort_values("sum_abs_loading", ascending=False).iloc[0]["sector"]
                title_str = f"Eigenvector {rank} ($\lambda = {df_rank['eigenvalue'].iloc[0]:.2f}$) | Dominant Sector: {top_sec}"
                
            ax.set_title(title_str, fontsize=11, pad=5)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(True, alpha=0.15, linestyle="--", axis="y")
            ax.tick_params(axis="x", labelrotation=90, labelsize=8 if is_top15 else 7)
            ax.set_ylabel("Loading", fontsize=10)
            
        plt.tight_layout()
        pdf_path = FIGURES_DIR / "vector" / f"figure_7a_rmt_top_eigenvectors_{suffix}.pdf" if not is_top15 else FIGURES_DIR / "vector" / f"figure_7b_rmt_top_eigenvectors_{suffix}.pdf"
        png_path = FIGURES_DIR / "preview" / f"figure_7a_rmt_top_eigenvectors_{suffix}.png" if not is_top15 else FIGURES_DIR / "preview" / f"figure_7b_rmt_top_eigenvectors_{suffix}.png"
        
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {pdf_path}")

    # Generate Figure 7a and 7b
    plot_figure_7(loadings_df, is_top15=False)
    plot_figure_7(loadings_df, is_top15=True)
    
    # --- Phase 7.2 Matrix Reconstruction ---
    print("\nReconstructing correlation matrices...")
    
    C_market = np.zeros((N, N))
    C_group = np.zeros((N, N))
    C_noise = np.zeros((N, N))
    
    for k in range(N):
        lam = eigvals_desc[k]
        u = eigvecs_desc[:, k:k+1] # column vector
        outer_product = lam * (u @ u.T)
        
        if k == 0:
            C_market += outer_product
        elif k < n_above_lambda_max:
            C_group += outer_product
        else:
            C_noise += outer_product
            
    C_filtered = C_market + C_group
    np.fill_diagonal(C_filtered, 1.0)
    
    # Export Matrices
    def save_matrix(matrix, filename):
        df_mat = pd.DataFrame(matrix, index=symbols, columns=symbols)
        path = OUTPUT_DIR / filename
        df_mat.to_csv(path)
        print(f"Saved {path}")
        
    save_matrix(C_market, "correlation_market_mode_core_historical_1998_2025.csv")
    save_matrix(C_group, "correlation_group_mode_core_historical_1998_2025.csv")
    save_matrix(C_noise, "correlation_noise_mode_core_historical_1998_2025.csv")
    save_matrix(C_filtered, "correlation_filtered_core_historical_1998_2025.csv")
    
    # Plot Figure 8 Heatmaps
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    matrices = [corr_matrix, C_market, C_group, C_noise, C_filtered]
    titles = ["Original", "Market Mode", "Group/Sector Mode", "Noise Component", "Filtered (Market + Group)"]
    
    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1 if title != "Group/Sector Mode" and title != "Noise Component" else -0.5, 
                       vmax=1 if title != "Group/Sector Mode" and title != "Noise Component" else 0.5)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    pdf_path_8 = FIGURES_DIR / "vector" / "figure_8_rmt_filtered_matrices.pdf"
    png_path_8 = FIGURES_DIR / "preview" / "figure_8_rmt_filtered_matrices.png"
    plt.savefig(pdf_path_8, format="pdf", bbox_inches="tight")
    plt.savefig(png_path_8, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved Figure 8 to {pdf_path_8} and {png_path_8}")


if __name__ == "__main__":
    main()

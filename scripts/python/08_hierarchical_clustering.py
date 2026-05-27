from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def squareform_condensed(D: np.ndarray) -> np.ndarray:
    """
    Manually convert a symmetric distance matrix to a condensed 1D array.
    """
    N = D.shape[0]
    idx = np.triu_indices(N, k=1)
    return D[idx]


def main() -> None:
    (FIGURES_DIR / "vector").mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "preview").mkdir(parents=True, exist_ok=True)

    print("Loading matrices...")
    
    # 1. Load Original Matrix
    core_path = OUTPUT_DIR / "core_historical_returns_wide_1998_2025.csv"
    df_returns = pd.read_csv(core_path)
    data = df_returns.drop(columns=["date"]) if "date" in df_returns.columns else df_returns
    data = data.dropna(how="any")
    C_orig_df = data.corr(method="pearson")
    symbols = C_orig_df.columns.tolist()
    N_assets = len(symbols)
    
    # 2. Load Filtered Matrix (Market + Group)
    path_filtered = OUTPUT_DIR / "correlation_filtered_core_historical_1998_2025.csv"
    C_filt_df = pd.read_csv(path_filtered, index_col=0)
    
    # 3. Load Group/Sector Matrix
    path_group = OUTPUT_DIR / "correlation_group_mode_core_historical_1998_2025.csv"
    C_grp_df = pd.read_csv(path_group, index_col=0)
    
    matrices = {
        "Original": C_orig_df,
        "Filtered": C_filt_df,
        "Group_Mode": C_grp_df
    }
    
    # Sector mapping for colors
    sector_map_path = OUTPUT_DIR / "assets_sector_map.csv"
    sector_df = pd.read_csv(sector_map_path)
    
    unique_sectors = sector_df["sector"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sectors)))
    sector_color_dict = {sec: colors[i] for i, sec in enumerate(unique_sectors)}
    
    sym_to_color = {}
    for sym in symbols:
        meta = sector_df[sector_df["symbol"] == sym]
        if len(meta) > 0:
            sym_to_color[sym] = sector_color_dict.get(meta["sector"].values[0], "black")
        else:
            sym_to_color[sym] = "black"

    leaf_orders = {}
    summary_records = []
    
    # --- Figure 9: Dendrograms ---
    plt.rcParams["font.family"] = "serif"
    fig9, axes9 = plt.subplots(3, 1, figsize=(10, 24)) # Stacked vertically, taller figure
    
    print("\n--- Clustering Verification Log ---")
    
    for i, (name, C_df) in enumerate(matrices.items()):
        C = C_df.values
        
        # Logging before clipping
        min_corr = np.nanmin(C)
        max_corr = np.nanmax(C)
        n_clipped = np.sum((C < -1.0) | (C > 1.0))
        
        # Cophenetic and distance
        C_for_distance = np.clip(C, -1.0, 1.0)
        np.fill_diagonal(C_for_distance, 1.0) # Force diagonal 1 for distance
        
        D = np.sqrt(2 * (1 - C_for_distance))
        D_diag_max_abs = np.max(np.abs(np.diag(D)))
        
        D_condensed = squareform_condensed(D)
        
        # Linkage
        Z = linkage(D_condensed, method='average')
        
        # Cophenetic Correlation
        c, coph_dists = cophenet(Z, D_condensed)
        
        ax = axes9[i]
        dendro = dendrogram(
            Z,
            labels=symbols,
            ax=ax,
            orientation='left', # 'left' puts root on the right and leaves on the left, but we want root on left, leaves on right so 'right'
            # Wait, scipy documentation for orientation: 'right' means root is plotted at the right.
            # Usually people use 'left' to mean tree grows left-to-right (root on left). Let's use 'left' to have root on left. Wait, 'right' places root at right. 'left' places root at left. Let's use 'left'.
            # Wait, no. scipy says: 'left' = root on left. But if I use 'right', root is on right, labels are on the left.
            # Let's use 'right' as user explicitly suggested, which usually looks good.
            leaf_font_size=8,
            color_threshold=0, # Force default color for links
            above_threshold_color="darkgray"
        )
        
        # Note on orientation: when orientation='right', the leaves are on the left Y-axis.
        # when orientation='left', the leaves are on the right Y-axis.
        
        ax.set_title(f"{name} Correlation Matrix Dendrogram (UPGMA)", fontsize=13)
        ax.set_xlabel("Mantegna Distance", fontsize=11)
        
        # Color the y-axis labels by sector
        ylbls = ax.get_ymajorticklabels()
        for lbl in ylbls:
            sym = lbl.get_text()
            lbl.set_color(sym_to_color.get(sym, "black"))
            
        leaf_orders[name] = dendro['leaves']
        first_10 = [symbols[idx] for idx in dendro['leaves'][:10]]
        
        print(f"\nMatrix: {name}")
        print(f"  n_assets: {N_assets}")
        print(f"  min corr: {min_corr:.6f}")
        print(f"  max corr: {max_corr:.6f}")
        print(f"  clipped values: {n_clipped}")
        print(f"  distance diagonal max abs: {D_diag_max_abs:.6e}")
        print(f"  cophenetic correlation: {c:.6f}")
        print(f"  first 10 leaves: {first_10}")
        
        summary_records.append({
            "matrix_name": name,
            "method": "average",
            "n_assets": N_assets,
            "cophenetic_correlation": c,
            "min_corr_before_clip": min_corr,
            "max_corr_before_clip": max_corr,
            "n_values_clipped": n_clipped
        })
        
    plt.tight_layout()
    pdf_path_9 = FIGURES_DIR / "vector" / "figure_9_dendrograms_comparison.pdf"
    png_path_9 = FIGURES_DIR / "preview" / "figure_9_dendrograms_comparison.png"
    plt.savefig(pdf_path_9, format="pdf", bbox_inches="tight")
    plt.savefig(png_path_9, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Export Leaf Orders
    order_records = {"rank": np.arange(1, len(symbols) + 1)}
    for name, leaves in leaf_orders.items():
        ordered_symbols = [symbols[idx] for idx in leaves]
        order_records[f"{name}_symbol"] = ordered_symbols
        
    order_df = pd.DataFrame(order_records)
    out_order = OUTPUT_DIR / "clustering_leaf_order_core_historical.csv"
    order_df.to_csv(out_order, index=False)
    
    # Export Summary
    sum_df = pd.DataFrame(summary_records)
    out_sum = OUTPUT_DIR / "clustering_cophenetic_summary_core_historical.csv"
    sum_df.to_csv(out_sum, index=False)
    
    # --- Figure 10: Ordered Heatmaps ---
    fig10, axes10 = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, C_df) in enumerate(matrices.items()):
        ax = axes10[i]
        
        # Reorder C based on dendrogram
        leaves = leaf_orders[name]
        ordered_syms = [symbols[idx] for idx in leaves]
        C_ordered = C_df.loc[ordered_syms, ordered_syms]
        
        vmin = -1 if name != "Group_Mode" else -0.5
        vmax = 1 if name != "Group_Mode" else 0.5
        
        sns.heatmap(
            C_ordered, 
            cmap="coolwarm", 
            vmin=vmin, 
            vmax=vmax, 
            ax=ax, 
            cbar=(i == 2), # Only show colorbar on last plot
            xticklabels=False, 
            yticklabels=False
        )
        ax.set_title(f"{name} (Ordered)", fontsize=13)
        
    plt.tight_layout()
    pdf_path_10 = FIGURES_DIR / "vector" / "figure_10_ordered_heatmaps.pdf"
    png_path_10 = FIGURES_DIR / "preview" / "figure_10_ordered_heatmaps.png"
    plt.savefig(pdf_path_10, format="pdf", bbox_inches="tight")
    plt.savefig(png_path_10, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

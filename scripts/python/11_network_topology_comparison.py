from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

def load_centrality(net_type: str, mat_type: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"{net_type}_centrality_{mat_type}_core_historical_1998_2025.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Ensure degree and betweenness are present
    cols_to_keep = ["symbol", "degree", "betweenness_centrality"]
    
    # MST might have degree_centrality but not degree count saved directly in the original script
    if "degree" not in df.columns and "degree_centrality" in df.columns:
        # Reconstruct degree count: degree_centrality = degree / (N-1)
        N = len(df)
        df["degree"] = np.round(df["degree_centrality"] * (N - 1)).astype(int)
        
    df = df[cols_to_keep].copy()
    df = df.rename(columns={
        "degree": f"degree_{net_type}_{mat_type}",
        "betweenness_centrality": f"betweenness_{net_type}_{mat_type}"
    })
    return df

def main() -> None:
    (FIGURES_DIR / "vector").mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "preview").mkdir(parents=True, exist_ok=True)
    
    print("Running Phase 9.3: MST vs PMFG Topology Comparison...")
    
    # ---------------------------------------------------------
    # Phase 9.3A: Topology Summary Table
    # ---------------------------------------------------------
    try:
        mst_summary = pd.read_csv(OUTPUT_DIR / "mst_summary_core_historical_1998_2025.csv")
        mst_summary["network_type"] = "MST"
    except FileNotFoundError:
        mst_summary = pd.DataFrame()
        
    try:
        pmfg_summary = pd.read_csv(OUTPUT_DIR / "pmfg_summary_core_historical_1998_2025.csv")
        pmfg_summary["network_type"] = "PMFG"
    except FileNotFoundError:
        pmfg_summary = pd.DataFrame()
        
    topo_df = pd.concat([mst_summary, pmfg_summary], ignore_index=True)
    
    # Add clique counts
    topo_df["n_triangles"] = 0
    topo_df["n_4_cliques"] = 0
    
    for idx, row in topo_df.iterrows():
        if row["network_type"] == "PMFG":
            mat_type = row["matrix_type"]
            clique_path = OUTPUT_DIR / f"pmfg_cliques_{mat_type}_core_historical_1998_2025.csv"
            if clique_path.exists():
                clique_df = pd.read_csv(clique_path)
                n_3 = len(clique_df[clique_df["clique_size"] == 3])
                n_4 = len(clique_df[clique_df["clique_size"] == 4])
                topo_df.at[idx, "n_triangles"] = n_3
                topo_df.at[idx, "n_4_cliques"] = n_4
                
    # Fill average_clustering for MST if missing
    if "average_clustering" not in topo_df.columns:
        topo_df["average_clustering"] = 0.0
    topo_df.loc[topo_df["network_type"] == "MST", "average_clustering"] = 0.0
    
    cols = [
        "network_type", "matrix_type", "n_nodes", "n_edges",
        "mean_edge_correlation", "median_edge_correlation", 
        "min_edge_correlation", "max_edge_correlation",
        "mean_edge_distance", "median_edge_distance",
        "same_sector_edge_ratio", "same_subsector_edge_ratio",
        "average_clustering", "density",
        "top_degree_node", "top_betweenness_node", "top_closeness_node",
        "n_triangles", "n_4_cliques"
    ]
    # Filter columns that exist
    cols = [c for c in cols if c in topo_df.columns]
    topo_df = topo_df[cols]
    
    topo_path = OUTPUT_DIR / "network_topology_comparison_core_historical_1998_2025.csv"
    topo_df.to_csv(topo_path, index=False)
    
    print("\nNetwork topology comparison:")
    print(topo_df.to_string(index=False))
    
    # ---------------------------------------------------------
    # Phase 9.3B: Hub Stability and Rank Comparison
    # ---------------------------------------------------------
    # Load metadata to get symbol, company_name, sector, subsector
    universe_path = OUTPUT_DIR / "core_historical_universe_1998_2025.csv"
    sector_path = OUTPUT_DIR / "assets_sector_map.csv"
    
    if sector_path.exists():
        metadata = pd.read_csv(sector_path)
    else:
        # Fallback if we only have the centrality files
        sample = pd.read_csv(OUTPUT_DIR / "pmfg_centrality_original_core_historical_1998_2025.csv")
        metadata = sample[["symbol", "company_name", "sector", "subsector"]].copy()
        
    hub_df = metadata[["symbol", "company_name", "sector", "subsector"]].copy()
    
    combinations = [
        ("mst", "original"),
        ("mst", "group_mode"),
        ("pmfg", "original"),
        ("pmfg", "group_mode")
    ]
    
    print("\nTop betweenness hubs:")
    for net_type, mat_type in combinations:
        df_c = load_centrality(net_type, mat_type)
        if not df_c.empty:
            hub_df = hub_df.merge(df_c, on="symbol", how="left")
            
            # Print top hubs
            top_hubs = df_c.sort_values(f"betweenness_{net_type}_{mat_type}", ascending=False).head(5)["symbol"].tolist()
            print(f"{net_type.upper()} {mat_type.capitalize()}: {', '.join(top_hubs)}")
            
            # Compute ranks
            # Rank 1 is highest degree/betweenness
            hub_df[f"degree_rank_{net_type}_{mat_type}"] = hub_df[f"degree_{net_type}_{mat_type}"].rank(method="min", ascending=False)
            hub_df[f"betweenness_rank_{net_type}_{mat_type}"] = hub_df[f"betweenness_{net_type}_{mat_type}"].rank(method="min", ascending=False)
            
    # Compute rank shifts (Original to Group Mode)
    # A positive shift means the rank numerically decreased (i.e., became a better rank, like 10 -> 2 is +8 shift)
    # shift = rank(original) - rank(group)
    if "degree_rank_mst_original" in hub_df.columns and "degree_rank_mst_group_mode" in hub_df.columns:
        hub_df["degree_rank_shift_mst_original_to_group"] = hub_df["degree_rank_mst_original"] - hub_df["degree_rank_mst_group_mode"]
        hub_df["betweenness_rank_shift_mst_original_to_group"] = hub_df["betweenness_rank_mst_original"] - hub_df["betweenness_rank_mst_group_mode"]
        
    if "degree_rank_pmfg_original" in hub_df.columns and "degree_rank_pmfg_group_mode" in hub_df.columns:
        hub_df["degree_rank_shift_pmfg_original_to_group"] = hub_df["degree_rank_pmfg_original"] - hub_df["degree_rank_pmfg_group_mode"]
        hub_df["betweenness_rank_shift_pmfg_original_to_group"] = hub_df["betweenness_rank_pmfg_original"] - hub_df["betweenness_rank_pmfg_group_mode"]
        
    hub_path = OUTPUT_DIR / "network_hub_rank_comparison_core_historical_1998_2025.csv"
    hub_df.to_csv(hub_path, index=False)
    
    print("\nHub rank shifts (PMFG Betweenness, positive means gained importance):")
    if "betweenness_rank_shift_pmfg_original_to_group" in hub_df.columns:
        top_gainers = hub_df.sort_values("betweenness_rank_shift_pmfg_original_to_group", ascending=False).head(5)
        top_losers = hub_df.sort_values("betweenness_rank_shift_pmfg_original_to_group", ascending=True).head(5)
        print("Top Gainers (from Original to Group Mode):")
        for _, row in top_gainers.iterrows():
            print(f"  {row['symbol']}: +{row['betweenness_rank_shift_pmfg_original_to_group']} positions")
        print("Top Losers (from Original to Group Mode):")
        for _, row in top_losers.iterrows():
            print(f"  {row['symbol']}: {row['betweenness_rank_shift_pmfg_original_to_group']} positions")
            
    # ---------------------------------------------------------
    # Phase 9.3C: Figures
    # ---------------------------------------------------------
    plt.rcParams["font.family"] = "serif"
    
    # Figure 13 - MST vs PMFG Topology Metrics
    fig13, axes13 = plt.subplots(2, 2, figsize=(12, 10), facecolor="white")
    axes13 = axes13.flatten()
    
    # We want to plot: mean_edge_correlation, same_sector_edge_ratio, same_subsector_edge_ratio, average_clustering
    # Only for 'original' and 'group_mode'
    plot_df = topo_df[topo_df["matrix_type"].isin(["original", "group_mode"])].copy()
    plot_df["Group"] = plot_df["network_type"] + " " + plot_df["matrix_type"].str.capitalize()
    
    metrics = [
        ("mean_edge_correlation", "Mean Edge Correlation"),
        ("same_sector_edge_ratio", "Same Sector Edge Ratio"),
        ("same_subsector_edge_ratio", "Same Subsector Edge Ratio"),
        ("average_clustering", "Average Clustering Coef.")
    ]
    
    # Custom colors
    palette_13 = {"MST Original": "#1f77b4", "MST Group_mode": "#aec7e8", 
                  "PMFG Original": "#d62728", "PMFG Group_mode": "#ff9896"}
    
    for i, (col, title) in enumerate(metrics):
        ax = axes13[i]
        sns.barplot(data=plot_df, x="Group", y=col, hue="Group", ax=ax, palette=palette_13, legend=False)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Rotate x labels slightly
        ax.tick_params(axis='x', labelrotation=15)
        
    plt.suptitle("Topology Metrics Comparison: MST vs PMFG", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    f13_pdf = FIGURES_DIR / "vector" / "figure_13_network_topology_comparison.pdf"
    f13_png = FIGURES_DIR / "preview" / "figure_13_network_topology_comparison.png"
    plt.savefig(f13_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(f13_png, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Figure 14 - Hub Rank Comparison (Horizontal Bar Chart)
    # PMFG Original vs Group Mode Top 10 Betweenness
    fig14, axes14 = plt.subplots(1, 2, figsize=(16, 8), facecolor="white")
    
    # PMFG Original Top 10
    pmfg_orig_top = hub_df.sort_values("betweenness_pmfg_original", ascending=False).head(10)
    pmfg_orig_top = pmfg_orig_top.sort_values("betweenness_pmfg_original", ascending=True) # Reverse for hbar
    
    # PMFG Group Top 10
    pmfg_grp_top = hub_df.sort_values("betweenness_pmfg_group_mode", ascending=False).head(10)
    pmfg_grp_top = pmfg_grp_top.sort_values("betweenness_pmfg_group_mode", ascending=True)
    
    def plot_hbar(ax, df_top, val_col, title):
        colors = plt.cm.tab10(np.linspace(0, 1, len(df_top)))
        ax.barh(df_top["symbol"], df_top[val_col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Betweenness Centrality")
        for i, (val, sec) in enumerate(zip(df_top[val_col], df_top["sector"])):
            ax.text(val + 0.005, i, f" {sec}", va='center', fontsize=9, fontstyle='italic', alpha=0.7)
            
    plot_hbar(axes14[0], pmfg_orig_top, "betweenness_pmfg_original", "PMFG Original - Top 10 Betweenness")
    plot_hbar(axes14[1], pmfg_grp_top, "betweenness_pmfg_group_mode", "PMFG Group Mode - Top 10 Betweenness")
    
    plt.suptitle("Hub Shift: Impact of RMT Filtering on Network Centrality", fontsize=18, fontweight="bold")
    plt.tight_layout()
    
    f14_pdf = FIGURES_DIR / "vector" / "figure_14_network_hub_rank_comparison.pdf"
    f14_png = FIGURES_DIR / "preview" / "figure_14_network_hub_rank_comparison.png"
    plt.savefig(f14_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(f14_png, dpi=300, bbox_inches="tight")
    plt.close()
    
    print("\nSaved:")
    print(f"  {topo_path}")
    print(f"  {hub_path}")
    print(f"  {f13_png}")
    print(f"  {f14_png}")

if __name__ == "__main__":
    main()

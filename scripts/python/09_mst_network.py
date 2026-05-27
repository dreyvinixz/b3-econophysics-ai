from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
NETWORKS_DIR = PROJECT_ROOT / "outputs" / "networks" / "graphml"

def main() -> None:
    (FIGURES_DIR / "vector").mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "preview").mkdir(parents=True, exist_ok=True)
    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    
    # 1. Load Original Matrix
    core_path = OUTPUT_DIR / "core_historical_returns_wide_1998_2025.csv"
    df_returns = pd.read_csv(core_path)
    data = df_returns.drop(columns=["date"]) if "date" in df_returns.columns else df_returns
    data = data.dropna(how="any")
    C_orig_df = data.corr(method="pearson")
    symbols = C_orig_df.columns.tolist()
    
    # 2. Load Filtered Matrices
    path_filtered = OUTPUT_DIR / "correlation_filtered_core_historical_1998_2025.csv"
    C_filt_df = pd.read_csv(path_filtered, index_col=0)
    
    path_group = OUTPUT_DIR / "correlation_group_mode_core_historical_1998_2025.csv"
    C_grp_df = pd.read_csv(path_group, index_col=0)
    
    matrices = {
        "original": C_orig_df,
        "filtered": C_filt_df,
        "group_mode": C_grp_df
    }
    
    # 3. Load Metadata
    sector_df = pd.read_csv(OUTPUT_DIR / "assets_sector_map.csv")
    sector_df = sector_df.set_index("symbol")
    
    try:
        universe_df = pd.read_csv(OUTPUT_DIR / "core_historical_universe_1998_2025.csv")
        universe_df = universe_df.set_index("symbol")
    except FileNotFoundError:
        universe_df = pd.DataFrame(index=symbols) # Fallback if missing
        
    unique_sectors = sector_df["sector"].unique() if "sector" in sector_df.columns else []
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_sectors))))
    sector_color_dict = {sec: colors[i] for i, sec in enumerate(unique_sectors)}
    
    summary_records = []
    graphs = {}
    
    print("\n--- MST Extraction ---")
    
    for mat_type, C_df in matrices.items():
        C = C_df.values
        N = len(symbols)
        
        # Cophenetic and distance prep
        C_for_distance = np.clip(C, -1.0, 1.0)
        np.fill_diagonal(C_for_distance, 1.0)
        D = np.sqrt(2 * (1 - C_for_distance))
        
        # Build full graph
        G_full = nx.Graph()
        
        # Add nodes with rich attributes
        for sym in symbols:
            attrs = {"symbol": sym}
            if sym in sector_df.index:
                row = sector_df.loc[sym]
                for col in ["company_name", "sector", "subsector", "segment"]:
                    if col in sector_df.columns:
                        attrs[col] = str(row[col])
                        
            if sym in universe_df.index:
                row = universe_df.loc[sym]
                for col in ["n_days", "first_date", "last_date", "specification"]:
                    if col in universe_df.columns:
                        attrs[col] = str(row[col])
                # Special handle for numeric vs string
                if "avg_financial_volume" in universe_df.columns:
                    val = row["avg_financial_volume"]
                    attrs["avg_financial_volume"] = float(val) if pd.notnull(val) else 0.0
                    
            G_full.add_node(sym, **attrs)
            
        # Add edges
        for i in range(N):
            for j in range(i + 1, N):
                sym_i = symbols[i]
                sym_j = symbols[j]
                
                dist = float(D[i, j])
                corr = float(C[i, j])
                
                # Check same sector
                same_sec = False
                same_subsec = False
                if sym_i in sector_df.index and sym_j in sector_df.index:
                    if "sector" in sector_df.columns:
                        same_sec = (sector_df.loc[sym_i, "sector"] == sector_df.loc[sym_j, "sector"])
                    if "subsector" in sector_df.columns:
                        same_subsec = (sector_df.loc[sym_i, "subsector"] == sector_df.loc[sym_j, "subsector"])
                
                G_full.add_edge(sym_i, sym_j, 
                                weight=dist, 
                                distance=dist, 
                                correlation=corr,
                                same_sector=int(same_sec),
                                same_subsector=int(same_subsec),
                                matrix_type=mat_type)
                                
        # MST Extraction
        T = nx.minimum_spanning_tree(G_full, weight='weight')
        graphs[mat_type] = T
        
        # Centralities
        deg_cent = nx.degree_centrality(T)
        bet_cent = nx.betweenness_centrality(T, weight='weight')
        clo_cent = nx.closeness_centrality(T, distance='weight')
        try:
            eig_cent = nx.eigenvector_centrality(T, weight='weight', max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            eig_cent = {node: 0.0 for node in T.nodes()}
            
        # Add centralities to nodes
        for node in T.nodes():
            T.nodes[node]["degree_centrality"] = deg_cent[node]
            T.nodes[node]["betweenness_centrality"] = bet_cent[node]
            T.nodes[node]["closeness_centrality"] = clo_cent[node]
            T.nodes[node]["eigenvector_centrality"] = eig_cent[node]
            
        # Extract Tables
        # Centrality
        cent_df = pd.DataFrame({
            "symbol": list(T.nodes()),
            "degree_centrality": list(deg_cent.values()),
            "betweenness_centrality": list(bet_cent.values()),
            "closeness_centrality": list(clo_cent.values()),
            "eigenvector_centrality": list(eig_cent.values())
        })
        cent_df = cent_df.sort_values("degree_centrality", ascending=False)
        cent_df.to_csv(OUTPUT_DIR / f"mst_centrality_{mat_type}_core_historical_1998_2025.csv", index=False)
        
        # Edges
        edges_data = []
        same_sec_count = 0
        same_subsec_count = 0
        corrs = []
        dists = []
        for u, v, dat in T.edges(data=True):
            edges_data.append({
                "source": u,
                "target": v,
                "weight": dat["weight"],
                "distance": dat["distance"],
                "correlation": dat["correlation"],
                "same_sector": dat["same_sector"],
                "same_subsector": dat["same_subsector"],
                "matrix_type": dat["matrix_type"]
            })
            if dat["same_sector"]: same_sec_count += 1
            if dat["same_subsector"]: same_subsec_count += 1
            corrs.append(dat["correlation"])
            dists.append(dat["distance"])
            
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(OUTPUT_DIR / f"mst_edges_{mat_type}_core_historical_1998_2025.csv", index=False)
        
        # GraphML Export
        nx.write_graphml(T, NETWORKS_DIR / f"mst_{mat_type}_core_historical_1998_2025.graphml")
        
        # Summary
        top_deg = cent_df.iloc[0]["symbol"]
        top_bet = cent_df.sort_values("betweenness_centrality", ascending=False).iloc[0]["symbol"]
        
        n_edges = T.number_of_edges()
        
        summary_records.append({
            "matrix_type": mat_type,
            "n_nodes": T.number_of_nodes(),
            "n_edges": n_edges,
            "mean_edge_correlation": np.mean(corrs),
            "median_edge_correlation": np.median(corrs),
            "mean_edge_distance": np.mean(dists),
            "median_edge_distance": np.median(dists),
            "same_sector_edges": same_sec_count,
            "same_sector_edge_ratio": same_sec_count / n_edges if n_edges > 0 else 0,
            "same_subsector_edges": same_subsec_count,
            "same_subsector_edge_ratio": same_subsec_count / n_edges if n_edges > 0 else 0,
            "top_degree_node": top_deg,
            "top_betweenness_node": top_bet
        })
        
        print(f"\nMatrix: {mat_type}")
        print(f"  n_nodes: {T.number_of_nodes()}")
        print(f"  n_edges: {n_edges}")
        print(f"  top 10 degree: {cent_df['symbol'].head(10).tolist()}")
        print(f"  top 10 betweenness: {cent_df.sort_values('betweenness_centrality', ascending=False)['symbol'].head(10).tolist()}")
        print(f"  same-sector edge ratio: {same_sec_count / n_edges:.4f}")
        print(f"  mean/median edge correlation: {np.mean(corrs):.4f} / {np.median(corrs):.4f}")

    sum_df = pd.DataFrame(summary_records)
    sum_df.to_csv(OUTPUT_DIR / "mst_summary_core_historical_1998_2025.csv", index=False)
    
    # --- Plot Figure 11: Original vs Group Mode MST ---
    print("\nPlotting Figure 11...")
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for ax, mat_type, title in zip(axes, ["original", "group_mode"], ["MST - Original Correlation", "MST - Group/Sector Mode"]):
        T = graphs[mat_type]
        
        # Kamada-Kawai layout
        pos = nx.kamada_kawai_layout(T, weight="weight")
        
        # Colors
        node_colors = []
        for n in T.nodes():
            sec = T.nodes[n].get("sector", "Unknown")
            node_colors.append(sector_color_dict.get(sec, "gray"))
            
        # Sizes by degree (rescaled for visibility)
        degrees = dict(T.degree())
        node_sizes = [degrees[n] * 100 + 50 for n in T.nodes()]
        
        # Edge widths by correlation magnitude
        corrs = [abs(dat['correlation']) for u, v, dat in T.edges(data=True)]
        if len(corrs) > 0:
            max_c = max(corrs)
            min_c = min(corrs)
            # Rescale to 0.5 - 3.0
            range_c = max_c - min_c if max_c > min_c else 1.0
            edge_widths = [0.5 + 2.5 * ((c - min_c) / range_c) for c in corrs]
        else:
            edge_widths = [1.0] * T.number_of_edges()
            
        nx.draw_networkx_edges(T, pos, ax=ax, width=edge_widths, edge_color="darkgray", alpha=0.7)
        nx.draw_networkx_nodes(T, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="white", linewidths=0.5)
        
        # Labels only for top 10 hubs (by degree or betweenness, let's use betweenness)
        bet = nx.betweenness_centrality(T, weight='weight')
        top_10 = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:10]
        labels = {n: n for n, _ in top_10}
        
        nx.draw_networkx_labels(T, pos, labels, ax=ax, font_size=9, font_family="serif", font_weight="bold",
                                bbox=dict(facecolor="white", edgecolor='none', alpha=0.7, pad=0.5))
        
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        
    # Add a legend for sectors
    handles = []
    for sec, col in sector_color_dict.items():
        if sec != "Unknown":
            patch = mpatches.Patch(color=col, label=sec)
            handles.append(patch)
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=11)
        
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    pdf_path = FIGURES_DIR / "vector" / "figure_11_mst_original_vs_group_mode.pdf"
    png_path = FIGURES_DIR / "preview" / "figure_11_mst_original_vs_group_mode.png"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved Figure 11 to {pdf_path}")

if __name__ == "__main__":
    main()

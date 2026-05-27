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
    
    # 1. Load Matrices
    core_path = OUTPUT_DIR / "core_historical_returns_wide_1998_2025.csv"
    df_returns = pd.read_csv(core_path)
    data = df_returns.drop(columns=["date"]) if "date" in df_returns.columns else df_returns
    data = data.dropna(how="any")
    C_orig_df = data.corr(method="pearson")
    symbols = C_orig_df.columns.tolist()
    N = len(symbols)
    EXPECTED_EDGES = 3 * N - 6
    
    path_filtered = OUTPUT_DIR / "correlation_filtered_core_historical_1998_2025.csv"
    C_filt_df = pd.read_csv(path_filtered, index_col=0)
    
    path_group = OUTPUT_DIR / "correlation_group_mode_core_historical_1998_2025.csv"
    C_grp_df = pd.read_csv(path_group, index_col=0)
    
    matrices = {
        "original": C_orig_df,
        "filtered": C_filt_df,
        "group_mode": C_grp_df
    }
    
    # 2. Load Metadata
    sector_df = pd.read_csv(OUTPUT_DIR / "assets_sector_map.csv")
    sector_df = sector_df.set_index("symbol")
    
    try:
        universe_df = pd.read_csv(OUTPUT_DIR / "core_historical_universe_1998_2025.csv")
        universe_df = universe_df.set_index("symbol")
    except FileNotFoundError:
        universe_df = pd.DataFrame(index=symbols)
        
    unique_sectors = sector_df["sector"].unique() if "sector" in sector_df.columns else []
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_sectors))))
    sector_color_dict = {sec: colors[i] for i, sec in enumerate(unique_sectors)}
    
    summary_records = []
    graphs = {}
    
    print("\n--- PMFG Extraction ---")
    
    for mat_type, C_df in matrices.items():
        C = C_df.values
        
        # Logging before clipping
        min_corr = np.nanmin(C)
        max_corr = np.nanmax(C)
        n_clipped = np.sum((C < -1.0) | (C > 1.0))
        
        # Preprocessing
        C_for_distance = np.clip(C, -1.0, 1.0)
        np.fill_diagonal(C_for_distance, 1.0)
        D = np.sqrt(2 * (1 - C_for_distance))
        
        # Candidate Edges
        candidates = []
        for i in range(N):
            for j in range(i + 1, N):
                sym_i = symbols[i]
                sym_j = symbols[j]
                
                dist = float(D[i, j])
                corr = float(C[i, j])
                
                same_sec = False
                same_subsec = False
                sec_i, sec_j = "", ""
                subsec_i, subsec_j = "", ""
                
                if sym_i in sector_df.index and sym_j in sector_df.index:
                    sec_i = sector_df.loc[sym_i, "sector"] if "sector" in sector_df.columns else ""
                    sec_j = sector_df.loc[sym_j, "sector"] if "sector" in sector_df.columns else ""
                    subsec_i = sector_df.loc[sym_i, "subsector"] if "subsector" in sector_df.columns else ""
                    subsec_j = sector_df.loc[sym_j, "subsector"] if "subsector" in sector_df.columns else ""
                    same_sec = (sec_i == sec_j) and sec_i != ""
                    same_subsec = (subsec_i == subsec_j) and subsec_i != ""
                
                candidates.append({
                    "source": sym_i,
                    "target": sym_j,
                    "distance": dist,
                    "correlation": corr,
                    "weight": dist,
                    "same_sector": same_sec,
                    "same_subsector": same_subsec,
                    "sector_source": sec_i,
                    "sector_target": sec_j,
                    "subsector_source": subsec_i,
                    "subsector_target": subsec_j,
                    "matrix_type": mat_type
                })
                
        # Sort by distance ascending
        candidates.sort(key=lambda x: x["distance"])
        
        # PMFG Algorithm
        G = nx.Graph()
        
        # Add nodes
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
                if "avg_financial_volume" in universe_df.columns:
                    val = row["avg_financial_volume"]
                    attrs["avg_financial_volume"] = float(val) if pd.notnull(val) else 0.0
            G.add_node(sym, **attrs)
            
        for edge in candidates:
            G.add_edge(edge["source"], edge["target"], **edge)
            is_planar, _ = nx.check_planarity(G)
            if not is_planar:
                G.remove_edge(edge["source"], edge["target"])
                
            if G.number_of_edges() == EXPECTED_EDGES:
                break
                
        graphs[mat_type] = G
        
        # Validate PMFG
        is_planar, _ = nx.check_planarity(G)
        actual_edges = G.number_of_edges()
        
        # Centralities
        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(G, weight='weight')
        clo_cent = nx.closeness_centrality(G, distance='weight')
        clustering = nx.clustering(G, weight='weight')
        try:
            eig_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            eig_cent = {node: float('nan') for node in G.nodes()}
            
        degrees = dict(G.degree())
            
        for node in G.nodes():
            G.nodes[node]["degree"] = degrees[node]
            G.nodes[node]["degree_centrality"] = deg_cent[node]
            G.nodes[node]["betweenness_centrality"] = bet_cent[node]
            G.nodes[node]["closeness_centrality"] = clo_cent[node]
            G.nodes[node]["eigenvector_centrality"] = eig_cent[node]
            G.nodes[node]["clustering_coefficient"] = clustering[node]
            
        # Export Centrality
        cent_data = []
        for n in G.nodes(data=True):
            dat = n[1]
            cent_data.append({
                "symbol": n[0],
                "company_name": dat.get("company_name", ""),
                "sector": dat.get("sector", ""),
                "subsector": dat.get("subsector", ""),
                "segment": dat.get("segment", ""),
                "degree": dat.get("degree", 0),
                "degree_centrality": dat.get("degree_centrality", 0.0),
                "betweenness_centrality": dat.get("betweenness_centrality", 0.0),
                "closeness_centrality": dat.get("closeness_centrality", 0.0),
                "eigenvector_centrality": dat.get("eigenvector_centrality", 0.0),
                "clustering_coefficient": dat.get("clustering_coefficient", 0.0),
                "avg_financial_volume": dat.get("avg_financial_volume", 0.0),
                "n_days": dat.get("n_days", ""),
                "first_date": dat.get("first_date", ""),
                "last_date": dat.get("last_date", ""),
                "specification": dat.get("specification", "")
            })
        cent_df = pd.DataFrame(cent_data).sort_values("degree", ascending=False)
        cent_df.to_csv(OUTPUT_DIR / f"pmfg_centrality_{mat_type}_core_historical_1998_2025.csv", index=False)
        
        # Export Edges
        edges_data = []
        same_sec_count = 0
        same_subsec_count = 0
        corrs = []
        dists = []
        for u, v, dat in G.edges(data=True):
            edges_data.append(dat)
            if dat["same_sector"]: same_sec_count += 1
            if dat["same_subsector"]: same_subsec_count += 1
            corrs.append(dat["correlation"])
            dists.append(dat["distance"])
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(OUTPUT_DIR / f"pmfg_edges_{mat_type}_core_historical_1998_2025.csv", index=False)
        
        # Extract Cliques
        clique_data = []
        all_cliques = list(nx.enumerate_all_cliques(G))
        for c in all_cliques:
            if len(c) in [3, 4]:
                # Collect attributes
                c_sectors = [G.nodes[n].get("sector", "") for n in c]
                c_subsectors = [G.nodes[n].get("subsector", "") for n in c]
                unique_sec = len(set([s for s in c_sectors if s]))
                
                same_sec = (unique_sec == 1) if c_sectors[0] else False
                unique_subsec = len(set([s for s in c_subsectors if s]))
                same_subsec = (unique_subsec == 1) if c_subsectors[0] else False
                
                # Internal correlation
                internal_corrs = []
                for idx_i in range(len(c)):
                    for idx_j in range(idx_i + 1, len(c)):
                        # Look up in edges_data or C matrix
                        node_i, node_j = c[idx_i], c[idx_j]
                        i_idx = symbols.index(node_i)
                        j_idx = symbols.index(node_j)
                        internal_corrs.append(C[i_idx, j_idx])
                        
                clique_data.append({
                    "matrix_type": mat_type,
                    "clique_size": len(c),
                    "nodes": "|".join(c),
                    "sectors": "|".join(c_sectors),
                    "subsectors": "|".join(c_subsectors),
                    "n_unique_sectors": unique_sec,
                    "same_sector_clique": same_sec,
                    "same_subsector_clique": same_subsec,
                    "mean_internal_correlation": np.mean(internal_corrs),
                    "min_internal_correlation": np.min(internal_corrs),
                    "max_internal_correlation": np.max(internal_corrs)
                })
        clique_df = pd.DataFrame(clique_data)
        clique_df.to_csv(OUTPUT_DIR / f"pmfg_cliques_{mat_type}_core_historical_1998_2025.csv", index=False)
        
        # Export GraphML
        nx.write_graphml(G, NETWORKS_DIR / f"pmfg_{mat_type}_core_historical_1998_2025.graphml")
        
        # Summary
        top_deg = cent_df.iloc[0]["symbol"]
        top_bet = cent_df.sort_values("betweenness_centrality", ascending=False).iloc[0]["symbol"]
        top_clo = cent_df.sort_values("closeness_centrality", ascending=False).iloc[0]["symbol"]
        avg_clustering = np.mean(list(clustering.values()))
        density = nx.density(G)
        
        num_3_cliques = len([c for c in all_cliques if len(c) == 3])
        num_4_cliques = len([c for c in all_cliques if len(c) == 4])
        
        summary_records.append({
            "matrix_type": mat_type,
            "n_nodes": G.number_of_nodes(),
            "n_edges": actual_edges,
            "expected_pmfg_edges": EXPECTED_EDGES,
            "is_planar": is_planar,
            "mean_edge_correlation": np.mean(corrs),
            "median_edge_correlation": np.median(corrs),
            "min_edge_correlation": np.min(corrs),
            "max_edge_correlation": np.max(corrs),
            "mean_edge_distance": np.mean(dists),
            "median_edge_distance": np.median(dists),
            "same_sector_edges": same_sec_count,
            "same_sector_edge_ratio": same_sec_count / actual_edges,
            "same_subsector_edges": same_subsec_count,
            "same_subsector_edge_ratio": same_subsec_count / actual_edges,
            "average_clustering": avg_clustering,
            "density": density,
            "top_degree_node": top_deg,
            "top_betweenness_node": top_bet,
            "top_closeness_node": top_clo
        })
        
        print(f"\nMatrix: {mat_type}")
        print(f"  n_nodes: {G.number_of_nodes()}")
        print(f"  n_edges: {actual_edges}")
        print(f"  expected_edges: {EXPECTED_EDGES}")
        print(f"  is_planar: {is_planar}")
        print(f"  mean_edge_correlation: {np.mean(corrs):.4f}")
        print(f"  median_edge_correlation: {np.median(corrs):.4f}")
        print(f"  same_sector_edge_ratio: {same_sec_count / actual_edges:.4f}")
        print(f"  same_subsector_edge_ratio: {same_subsec_count / actual_edges:.4f}")
        print(f"  average_clustering: {avg_clustering:.4f}")
        print(f"  top degree nodes: {cent_df['symbol'].head(10).tolist()}")
        print(f"  top betweenness nodes: {cent_df.sort_values('betweenness_centrality', ascending=False)['symbol'].head(10).tolist()}")
        print(f"  number of 3-cliques: {num_3_cliques}")
        print(f"  number of 4-cliques: {num_4_cliques}")

    sum_df = pd.DataFrame(summary_records)
    sum_df.to_csv(OUTPUT_DIR / "pmfg_summary_core_historical_1998_2025.csv", index=False)
    
    # --- Plot Figure 12: Original vs Group Mode PMFG ---
    print("\nPlotting Figure 12...")
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for ax, mat_type, title in zip(axes, ["original", "group_mode"], ["PMFG - Original Correlation", "PMFG - Group/Sector Mode"]):
        G = graphs[mat_type]
        
        # Spring layout
        pos = nx.spring_layout(G, seed=42, weight=None, iterations=500)
        
        # Colors
        node_colors = []
        for n in G.nodes():
            sec = G.nodes[n].get("sector", "Unknown")
            node_colors.append(sector_color_dict.get(sec, "gray"))
            
        # Sizes by betweenness centrality (rescaled for visibility)
        bet = nx.betweenness_centrality(G, weight='weight')
        # scale bet between 50 and 800
        bet_vals = list(bet.values())
        if max(bet_vals) > min(bet_vals):
            node_sizes = [50 + 750 * ((bet[n] - min(bet_vals)) / (max(bet_vals) - min(bet_vals))) for n in G.nodes()]
        else:
            node_sizes = [200] * G.number_of_nodes()
        
        # Edge widths by correlation magnitude
        corrs = [abs(dat['correlation']) for u, v, dat in G.edges(data=True)]
        if len(corrs) > 0:
            max_c = max(corrs)
            min_c = min(corrs)
            range_c = max_c - min_c if max_c > min_c else 1.0
            edge_widths = [0.5 + 2.5 * ((c - min_c) / range_c) for c in corrs]
        else:
            edge_widths = [1.0] * G.number_of_edges()
            
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color="darkgray", alpha=0.4)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="white", linewidths=0.5)
        
        # Labels for top 10 hubs (by betweenness centrality)
        top_10 = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:10]
        # Ensure PETR4, VALE3, BBDC4 are labeled if present
        labels = {n: n for n, _ in top_10}
        for important in ["PETR4", "VALE3", "BBDC4"]:
            if important in G.nodes() and important not in labels:
                labels[important] = important
        
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_family="serif", font_weight="bold",
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
    pdf_path = FIGURES_DIR / "vector" / "figure_12_pmfg_original_vs_group_mode.pdf"
    png_path = FIGURES_DIR / "preview" / "figure_12_pmfg_original_vs_group_mode.png"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved outputs:")
    print("  edge tables")
    print("  centrality tables")
    print("  summary table")
    print("  clique tables")
    print("  graphml files")
    print("  figure paths")

if __name__ == "__main__":
    main()

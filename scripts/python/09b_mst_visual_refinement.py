from __future__ import annotations

import sys
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
NETWORKS_DIR = PROJECT_ROOT / "outputs" / "networks" / "graphml"

# Enhanced Palette for B3 Sectors (Same as PMFG refinement)
CUSTOM_PALETTE = {
    "Financials": "#2ca02c", # Elegant Green
    "Basic Materials": "#ff7f0e", # Strong Orange
    "Oil Gas and Biofuels": "#1f77b4", # Deep Blue
    "Utilities": "#e377c2", # Soft Pink/Purple
    "Consumer Cyclical": "#8c564b", # Brown/Earth
    "Consumer Non-Cyclical": "#17becf", # Cyan
    "Industrials": "#d62728", # Red
    "Real Estate": "#7f7f7f", # Gray
    "Telecommunications": "#bcbd22", # Olive
    "Unknown": "#aaaaaa"
}

def main() -> None:
    (FIGURES_DIR / "vector").mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "preview").mkdir(parents=True, exist_ok=True)
    
    print("Loading GraphML MST networks...")
    
    # We will plot Original vs Group Mode
    path_orig = NETWORKS_DIR / "mst_original_core_historical_1998_2025.graphml"
    path_group = NETWORKS_DIR / "mst_group_mode_core_historical_1998_2025.graphml"
    
    if not path_orig.exists() or not path_group.exists():
        print("Error: GraphML files not found. Run 09_mst_network.py first.")
        sys.exit(1)
        
    G_orig = nx.read_graphml(path_orig)
    G_group = nx.read_graphml(path_group)
    
    graphs = {
        "Original": G_orig,
        "Group/Sector Mode": G_group
    }
    
    print("Generating Figure 11B (MST Visual Refinement)...")
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(22, 11), facecolor="white")
    
    for ax, (title, G) in zip(axes, graphs.items()):
        
        # 1. Layout
        # User requested Kamada-Kawai for MST. It usually preserves tree topology well.
        # We ensure weight="weight" is used as it expects distance for the kamada_kawai_layout.
        pos = nx.kamada_kawai_layout(G, weight="weight")
        
        # 2. Node Processing
        node_colors = []
        node_sizes = []
        labels = {}
        
        # Extract betweenness to scale nodes
        bet_vals = []
        for n, d in G.nodes(data=True):
            bet = float(d.get("betweenness_centrality", 0.0))
            bet_vals.append(bet)
            
        min_bet = min(bet_vals) if bet_vals else 0
        max_bet = max(bet_vals) if bet_vals else 1
        range_bet = max_bet - min_bet if max_bet > min_bet else 1.0
        
        # Identify Top Hubs for Labeling
        top_n = sorted(G.nodes(data=True), key=lambda x: float(x[1].get("betweenness_centrality", 0.0)), reverse=True)
        top_hubs = [x[0] for x in top_n[:8]] # Top 8 hubs
        # Ensure mandatory assets are labeled if they exist
        for mandatory in ["PETR4", "VALE3", "BBDC4", "ITSA4"]:
            if mandatory in G.nodes() and mandatory not in top_hubs:
                top_hubs.append(mandatory)
                
        for n, d in G.nodes(data=True):
            sec = d.get("sector", "Unknown")
            color = CUSTOM_PALETTE.get(sec, "#aaaaaa")
            node_colors.append(color)
            
            # Non-linear size scaling for organic feel
            bet = float(d.get("betweenness_centrality", 0.0))
            normalized_bet = (bet - min_bet) / range_bet
            size = 30 + 1000 * (normalized_bet ** 0.75) # Base 30, max 1030
            node_sizes.append(size)
            
            if n in top_hubs:
                labels[n] = n
                
        # 3. Edge Processing
        for u, v, d in G.edges(data=True):
            corr = abs(float(d.get("correlation", 0.0)))
            same_sec_str = str(d.get("same_sector", "0"))
            same_sector = same_sec_str.lower() in ['true', '1']
            
            # Width scales with correlation
            width = 0.5 + 2.5 * corr # Slightly thicker for MST since there are fewer edges
            
            # Alpha scales with correlation
            alpha = 0.30 + 0.50 * corr # Higher alpha for MST to make it visible
            
            # Color: if same sector, inherit the source node's sector color. Else, light gray.
            if same_sector:
                sec_u = G.nodes[u].get("sector", "Unknown")
                base_color = CUSTOM_PALETTE.get(sec_u, "#aaaaaa")
            else:
                base_color = "#888888"
                
            edge_color = to_rgba(base_color, alpha=alpha)
            
            # Draw individual edge with curve
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                ax=ax,
                width=width,
                edge_color=[edge_color],
                connectionstyle="arc3,rad=0.15", # Gentle organic curve
                arrows=True,
                arrowstyle="-" 
            )
            
        # Draw Nodes
        nx.draw_networkx_nodes(
            G, pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="white",
            linewidths=0.6
        )
        
        # Draw Labels
        for n, label in labels.items():
            x, y = pos[n]
            # Offset slightly upwards
            ax.text(
                x, y + 0.04,
                label,
                fontsize=11,
                fontfamily="serif",
                fontweight="bold",
                color="black",
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.3, boxstyle="round,pad=0.2")
            )
            
        ax.set_title(f"MST - {title} Correlation", fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")
        
    # 4. Legend
    all_sectors = set()
    for G in graphs.values():
        for _, d in G.nodes(data=True):
            all_sectors.add(d.get("sector", "Unknown"))
            
    handles = []
    for sec in sorted(list(all_sectors)):
        if sec != "Unknown":
            col = CUSTOM_PALETTE.get(sec, "#aaaaaa")
            patch = mpatches.Patch(color=col, label=sec)
            handles.append(patch)
            
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles) // 2 + 1 if len(handles) > 5 else len(handles),
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=12
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    pdf_path = FIGURES_DIR / "vector" / "figure_11b_mst_refined_comparison.pdf"
    png_path = FIGURES_DIR / "preview" / "figure_11b_mst_refined_comparison.png"
    
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved highly refined MST visualization to:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")

if __name__ == "__main__":
    main()

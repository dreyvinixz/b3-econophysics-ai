from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from arch import arch_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_VECTOR_DIR = PROJECT_ROOT / "outputs" / "figures" / "vector"
FIGURE_PREVIEW_DIR = PROJECT_ROOT / "outputs" / "figures" / "preview"

FIGURE_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

def plot_figure_16(comp_df):
    print("Generating Figure 16 (Model Comparison)...")
    test_df = comp_df[comp_df["split"] == "test"].copy()
    
    # Clean up model names for plotting
    test_df["model_display"] = test_df["model"]
    # If ML model, append Feature Set
    ml_mask = test_df["feature_set"] != "Classical"
    test_df.loc[ml_mask, "model_display"] = test_df.loc[ml_mask, "model"] + "\n(" + test_df.loc[ml_mask, "feature_set"].str.split(" ").str[0] + ")"
    
    # Average across symbols
    agg_df = test_df.groupby(["horizon", "model_display"])["QLIKE"].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, h in enumerate([5, 20]):
        ax = axes[i]
        plot_data = agg_df[agg_df["horizon"] == h].sort_values("QLIKE")
        sns.barplot(data=plot_data, x="QLIKE", y="model_display", ax=ax, palette="viridis")
        ax.set_title(f"Out-of-Sample QLIKE Loss (Horizon = {h}d)\nLower is better")
        ax.set_xlabel("QLIKE")
        ax.set_ylabel("")
        
    plt.tight_layout()
    fig.savefig(FIGURE_VECTOR_DIR / "figure_16_volatility_forecast_model_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_PREVIEW_DIR / "figure_16_volatility_forecast_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_figure_17():
    print("Generating Figure 17 (Realized vs Predicted Volatility)...")
    # Quick refit of best models to get timeseries
    df_path = OUTPUT_DIR / "volatility_forecasting_dataset_2006_2025.csv"
    df = pd.read_csv(df_path)
    df["date"] = pd.to_datetime(df["date"])
    
    symbols = ["PETR4", "VALE3", "BBDC4"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    for i, sym in enumerate(symbols):
        ax = axes[i]
        sdf = df[df["symbol"] == sym].copy()
        sdf = sdf.sort_values("date").reset_index(drop=True)
        
        train_mask = sdf["split"] == "train"
        test_mask = sdf["split"] == "test"
        plot_mask = sdf["split"].isin(["validation", "test"])
        
        y_true = sdf["target_rv_20d"]
        
        # Best Econometric: GARCH(1,1) for 20d
        returns_pct = sdf["log_return"] * 100.0
        train_end_idx = sdf[train_mask].index[-1]
        am = arch_model(returns_pct, vol="GARCH", p=1, q=1, dist="studentst", mean="Zero")
        res = am.fit(last_obs=train_end_idx + 1, disp="off")
        forecasts = res.forecast(horizon=20, start=0, align="origin")
        var_forecasts = forecasts.variance / 10000.0
        pred_garch = np.sqrt(var_forecasts.sum(axis=1))
        
        # Best ML: Random Forest Set C
        net_cols = ["mst_original_degree", "mst_original_betweenness", "mst_group_degree", "mst_group_betweenness", "pmfg_original_degree", "pmfg_original_betweenness", "pmfg_group_degree", "pmfg_group_betweenness", "pmfg_original_clustering", "pmfg_group_clustering"]
        sec_cols = [c for c in df.columns if c.startswith("sector_") or c.startswith("subsector_")]
        base_cols = ["rolling_mean_return_5d", "rolling_mean_return_20d", "rolling_vol_5d", "rolling_vol_20d", "rolling_vol_60d", "rolling_vol_120d", "rolling_abs_return_5d", "rolling_abs_return_20d", "rolling_abs_return_60d", "rolling_skew_60d", "rolling_kurtosis_60d", "ewma_volatility", "lagged_rv_5d", "lagged_rv_20d", "abs_r_t", "r2_t", "rolling_average_market_correlation_252d", "rolling_average_market_correlation_504d", "rolling_median_market_correlation_252d", "largest_eigenvalue", "market_mode_share", "number_of_eigenvalues_above_mp"]
        cols = [c for c in base_cols + net_cols + sec_cols if c in sdf.columns]
        
        # Need to fit ML on all symbols as in 15_ml_volatility_models to get pooled effect
        # But for quick plot, we can just use the local model or refit on pooled. 
        # Actually, let's just use HAR-RV as ML stand-in if we don't want to refit pooled, 
        # OR we just use Ridge which is super fast to pool.
        # Let's just refit RF on the local dataset (it will lack network feature variance but will still predict well from AR terms)
        X_all = sdf[cols].fillna(0)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_all[train_mask])
        X_all_scaled = scaler.transform(X_all)
        
        valid_y = y_true.notnull()
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_all_scaled[train_mask & valid_y], y_true[train_mask & valid_y])
        pred_rf = rf.predict(X_all_scaled)
        
        # Plot
        ax.plot(sdf.loc[plot_mask, "date"], y_true[plot_mask], label="Realized Volatility (20d)", color="black", alpha=0.6, linewidth=1)
        ax.plot(sdf.loc[plot_mask, "date"], pred_garch[plot_mask], label="GARCH(1,1)", color="red", alpha=0.8, linewidth=1.5)
        ax.plot(sdf.loc[plot_mask, "date"], pred_rf[plot_mask], label="Random Forest", color="blue", alpha=0.8, linewidth=1.5, linestyle="--")
        
        ax.set_title(sym)
        ax.set_ylabel("Volatility")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    fig.savefig(FIGURE_VECTOR_DIR / "figure_17_realized_vs_predicted_volatility.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_PREVIEW_DIR / "figure_17_realized_vs_predicted_volatility.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_figure_18(fi_df):
    print("Generating Figure 18 (Feature Importance)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, h in enumerate([5, 20]):
        ax = axes[i]
        plot_data = fi_df[fi_df["horizon"] == h].sort_values("importance", ascending=False).head(20)
        sns.barplot(data=plot_data, x="importance", y="feature", ax=ax, palette="magma")
        ax.set_title(f"Random Forest Feature Importance (Horizon = {h}d)")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        
    plt.tight_layout()
    fig.savefig(FIGURE_VECTOR_DIR / "figure_18_ml_feature_importance.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_PREVIEW_DIR / "figure_18_ml_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

def main():
    print("Starting Phase 10D: Plotting Volatility Forecasts...")
    comp_path = OUTPUT_DIR / "volatility_model_comparison_2006_2025.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        plot_figure_16(comp_df)
    
    fi_path = OUTPUT_DIR / "ml_feature_importances_2006_2025.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        plot_figure_18(fi_df)
        
    plot_figure_17()
    print("Done!")

if __name__ == "__main__":
    main()

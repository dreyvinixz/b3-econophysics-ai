from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

def prepare_dataset():
    print("Starting Phase 10 Volatility Forecasting Dataset Preparation...")
    
    # 1. Load returns
    ret_path = OUTPUT_DIR / "demo_assets_returns_long_1998_2025.csv"
    df = pd.read_csv(ret_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    
    # Base features
    df["r_t"] = df["log_return"]
    df["abs_r_t"] = df["r_t"].abs()
    df["r2_t"] = df["r_t"] ** 2
    
    symbols = df["symbol"].unique()
    
    # 2. Compute rolling asset features and targets
    print("Computing rolling volatility and return features...")
    processed_dfs = []
    
    for sym in symbols:
        sdf = df[df["symbol"] == sym].copy()
        
        # Rolling returns
        sdf["rolling_mean_return_5d"] = sdf["r_t"].rolling(5, min_periods=3).mean()
        sdf["rolling_mean_return_20d"] = sdf["r_t"].rolling(20, min_periods=10).mean()
        
        # Rolling standard deviations (volatility)
        sdf["rolling_vol_5d"] = sdf["r_t"].rolling(5, min_periods=3).std()
        sdf["rolling_vol_20d"] = sdf["r_t"].rolling(20, min_periods=10).std()
        sdf["rolling_vol_60d"] = sdf["r_t"].rolling(60, min_periods=30).std()
        sdf["rolling_vol_120d"] = sdf["r_t"].rolling(120, min_periods=60).std()
        
        # Rolling absolute returns
        sdf["rolling_abs_return_5d"] = sdf["abs_r_t"].rolling(5, min_periods=3).mean()
        sdf["rolling_abs_return_20d"] = sdf["abs_r_t"].rolling(20, min_periods=10).mean()
        sdf["rolling_abs_return_60d"] = sdf["abs_r_t"].rolling(60, min_periods=30).mean()
        
        # Skew and Kurtosis
        sdf["rolling_skew_60d"] = sdf["r_t"].rolling(60, min_periods=30).skew()
        sdf["rolling_kurtosis_60d"] = sdf["r_t"].rolling(60, min_periods=30).kurt()
        
        # EWMA Volatility
        sdf["ewma_volatility"] = sdf["r_t"].ewm(span=60).std()
        
        # Lagged Realized Volatility (backward-looking sums of r2)
        sdf["lagged_rv_5d"] = np.sqrt(sdf["r2_t"].rolling(5, min_periods=3).sum())
        sdf["lagged_rv_20d"] = np.sqrt(sdf["r2_t"].rolling(20, min_periods=10).sum())
        
        # Target Realized Volatility (forward-looking sums of r2)
        # RV_{t,h} = sqrt(sum_{k=1}^h r_{t+k}^2)
        sdf["r2_t_forward_5d"] = sdf["r2_t"].rolling(window=5, min_periods=3).sum().shift(-5)
        sdf["r2_t_forward_20d"] = sdf["r2_t"].rolling(window=20, min_periods=10).sum().shift(-20)
        
        sdf["target_rv_5d"] = np.sqrt(sdf["r2_t_forward_5d"])
        sdf["target_rv_20d"] = np.sqrt(sdf["r2_t_forward_20d"])
        
        sdf = sdf.drop(columns=["r2_t_forward_5d", "r2_t_forward_20d"])
        
        processed_dfs.append(sdf)
        
    df = pd.concat(processed_dfs, ignore_index=True)
    
    # 3. Market-level Econophysics Features
    print("Merging market-level features...")
    
    # 3a. Rolling Market Correlation
    corr_path = OUTPUT_DIR / "core_historical_rolling_average_correlation_1998_2025.csv"
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
        corr_df["date"] = pd.to_datetime(corr_df["date"])
        # Map fields
        # Note: Phase 6 outputs might have 'avg_correlation_252d' instead of 'rolling_average_market_correlation_252d'
        # We rename to match Phase 10 specs if possible, else keep as is
        rename_map = {
            "avg_correlation_252d": "rolling_average_market_correlation_252d",
            "avg_correlation_504d": "rolling_average_market_correlation_504d",
            "median_correlation_252d": "rolling_median_market_correlation_252d"
        }
        corr_df = corr_df.rename(columns=rename_map)
        cols_to_keep = ["date"] + [c for c in corr_df.columns if "rolling" in c]
        if len(cols_to_keep) > 1:
            df = pd.merge(df, corr_df[cols_to_keep], on="date", how="left")
            
    # 3b. RMT Eigenvalues (Static over the whole period conceptually for the initial structural plan, or we load the rolling ones if they existed. But the plan says they are static/period-level covariates from Phase 7. So we just broadcast them or use rolling if available)
    # The plan says "If some features are static rather than daily, use the static value as an asset-level or period-level covariate."
    rmt_summ_path = OUTPUT_DIR / "rmt_summary_core_historical_1998_2025.csv"
    if rmt_summ_path.exists():
        rmt_summ = pd.read_csv(rmt_summ_path).iloc[0]
        # Columns in our CSV: N, T, Q, lambda_min, lambda_max, max_empirical_eigenvalue, n_above_lambda_max
        df["largest_eigenvalue"] = rmt_summ.get("max_empirical_eigenvalue", np.nan)
        
        # market mode share = max_empirical_eigenvalue / N
        N_val = rmt_summ.get("N", 58)
        max_eig = rmt_summ.get("max_empirical_eigenvalue", np.nan)
        df["market_mode_share"] = max_eig / N_val if pd.notnull(max_eig) else np.nan
        
        df["number_of_eigenvalues_above_mp"] = rmt_summ.get("n_above_lambda_max", np.nan)
        
    # 4. Network Features
    print("Merging network features...")
    # Load centralities and merge by symbol
    network_files = {
        "mst_original": "mst_centrality_original_core_historical_1998_2025.csv",
        "mst_group": "mst_centrality_group_mode_core_historical_1998_2025.csv",
        "pmfg_original": "pmfg_centrality_original_core_historical_1998_2025.csv",
        "pmfg_group": "pmfg_centrality_group_mode_core_historical_1998_2025.csv"
    }
    
    for prefix, fname in network_files.items():
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            ndf = pd.read_csv(fpath)
            ndf = ndf.rename(columns={
                "degree": f"{prefix}_degree",
                "degree_centrality": f"{prefix}_degree_centrality", 
                "betweenness_centrality": f"{prefix}_betweenness",
                "clustering_coefficient": f"{prefix}_clustering"
            })
            # Pick available columns
            cols = ["symbol"] + [c for c in ndf.columns if prefix in c]
            df = pd.merge(df, ndf[cols], on="symbol", how="left")
            
            # If `degree` wasn't explicitly saved for MST but degree_centrality was:
            if f"{prefix}_degree" not in df.columns and f"{prefix}_degree_centrality" in df.columns:
                # N=58 from core universe
                df[f"{prefix}_degree"] = np.round(df[f"{prefix}_degree_centrality"] * 57).astype(int)
                
    # 5. Sector Features
    print("Merging sector features...")
    sec_path = OUTPUT_DIR / "assets_sector_map.csv"
    if sec_path.exists():
        sec_df = pd.read_csv(sec_path)[["symbol", "sector", "subsector"]]
        df = pd.merge(df, sec_df, on="symbol", how="left")
        
        # Dummies
        df = pd.get_dummies(df, columns=["sector", "subsector"], prefix=["sector", "subsector"], dummy_na=False)

    # 6. Time Window Filtering
    print("Applying temporal filters and splits...")
    df = df[(df["date"] >= "2006-01-01") & (df["date"] <= "2025-12-31")].copy()
    
    # 7. Split mapping
    def assign_split(date_val):
        if date_val < pd.Timestamp("2018-01-01"):
            return "train"
        elif date_val < pd.Timestamp("2021-01-01"):
            return "validation"
        else:
            return "test"
            
    df["split"] = df["date"].apply(assign_split)
    
    # Reorder columns slightly for clarity
    front_cols = ["date", "symbol", "log_return", "target_rv_5d", "target_rv_20d", "split"]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]
    
    # 8. Output Datasets
    out_path = OUTPUT_DIR / "volatility_forecasting_dataset_2006_2025.csv"
    df.to_csv(out_path, index=False)
    
    # Feature summary
    summary_path = OUTPUT_DIR / "volatility_forecasting_feature_summary_2006_2025.csv"
    summ_df = df.describe().T
    summ_df["missing_pct"] = df.isna().mean()
    summ_df.to_csv(summary_path)
    
    print("\nDataset generation complete!")
    print(f"  n_rows: {len(df)}")
    print(f"  n_assets: {df['symbol'].nunique()}")
    print(f"  date_min: {df['date'].min().strftime('%Y-%m-%d')}")
    print(f"  date_max: {df['date'].max().strftime('%Y-%m-%d')}")
    print("\nMissing values by column (top 15):")
    print(df.isna().mean().sort_values(ascending=False).head(15).to_string())
    
    print("\nTarget summary by asset:")
    print(df.groupby("symbol")[["target_rv_5d", "target_rv_20d"]].describe().T)

if __name__ == "__main__":
    prepare_dataset()

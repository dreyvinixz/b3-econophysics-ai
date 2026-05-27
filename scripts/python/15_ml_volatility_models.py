from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

def qlike_loss(y_true, y_pred, epsilon=1e-12):
    var_true = np.maximum(y_true**2, epsilon)
    var_pred = np.maximum(y_pred**2, epsilon)
    qlike = np.log(var_pred) + var_true / var_pred
    return np.mean(qlike)

def evaluate_model(y_true, y_pred, y_train_true):
    mask = pd.notnull(y_true) & pd.notnull(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan, np.nan
        
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    qlike = qlike_loss(y_true, y_pred)
    
    mse_pred = mean_squared_error(y_true, y_pred)
    mse_baseline = np.mean((y_true - np.mean(y_train_true))**2)
    r2_oos = 1 - (mse_pred / mse_baseline) if mse_baseline > 0 else np.nan
    
    return mae, rmse, qlike, r2_oos

def get_feature_sets(df):
    set_A_cols = [
        "rolling_mean_return_5d", "rolling_mean_return_20d", "rolling_vol_5d", 
        "rolling_vol_20d", "rolling_vol_60d", "rolling_vol_120d", "rolling_abs_return_5d", 
        "rolling_abs_return_20d", "rolling_abs_return_60d", "rolling_skew_60d", 
        "rolling_kurtosis_60d", "ewma_volatility", "lagged_rv_5d", "lagged_rv_20d", 
        "abs_r_t", "r2_t"
    ]
    set_A_cols = [c for c in set_A_cols if c in df.columns]
    
    set_B_cols = set_A_cols + [
        "rolling_average_market_correlation_252d", "rolling_average_market_correlation_504d", 
        "rolling_median_market_correlation_252d", "largest_eigenvalue", 
        "market_mode_share", "number_of_eigenvalues_above_mp"
    ]
    set_B_cols = [c for c in set_B_cols if c in df.columns]
    
    net_cols = [
        "mst_original_degree", "mst_original_betweenness", "mst_group_degree", "mst_group_betweenness", 
        "pmfg_original_degree", "pmfg_original_betweenness", "pmfg_group_degree", "pmfg_group_betweenness", 
        "pmfg_original_clustering", "pmfg_group_clustering"
    ]
    sec_cols = [c for c in df.columns if c.startswith("sector_") or c.startswith("subsector_")]
    
    set_C_cols = set_B_cols + net_cols + sec_cols
    set_C_cols = [c for c in set_C_cols if c in df.columns]
    
    return {"Set A (Classical)": set_A_cols, "Set B (+Market/RMT)": set_B_cols, "Set C (+Network)": set_C_cols}

def main():
    print("Starting Phase 10C: Machine Learning Volatility Models...")
    
    df_path = OUTPUT_DIR / "volatility_forecasting_dataset_2006_2025.csv"
    df = pd.read_csv(df_path)
    df["date"] = pd.to_datetime(df["date"])
    
    feature_sets = get_feature_sets(df)
    symbols = df["symbol"].unique()
    horizons = [5, 20]
    
    models_to_train = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)
    }
    
    results = []
    feature_importances = []
    
    # Train Global Models across all symbols to leverage cross-sectional and static network features
    print(f"\nTraining global models across {len(symbols)} symbols...")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"
    
    for h in horizons:
        target_col = f"target_rv_{h}d"
        y_all = df[target_col]
        valid_mask = pd.notnull(y_all)
        y_train = y_all[train_mask & valid_mask]
        
        if len(y_train) == 0:
            continue
            
        for set_name, cols in feature_sets.items():
            print(f"  Horizon {h}d | {set_name} | cols: {len(cols)}")
            X_all = df[cols].fillna(0)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_all[train_mask])
            X_all_scaled = scaler.transform(X_all)
            
            X_train_final = X_all_scaled[train_mask & valid_mask]
            
            for model_name, model in models_to_train.items():
                model.fit(X_train_final, y_train)
                
                preds_all = model.predict(X_all_scaled)
                preds_all = np.maximum(preds_all, 1e-6)
                preds_series = pd.Series(preds_all, index=df.index)
                
                # Evaluate per symbol
                for sym in symbols:
                    sym_mask = df["symbol"] == sym
                    for split_name, split_mask in [("train", train_mask), ("validation", val_mask), ("test", test_mask)]:
                        eval_mask = split_mask & valid_mask & sym_mask
                        if eval_mask.sum() > 0:
                            # Use y_train from this specific symbol for R2 OOS baseline
                            y_train_sym = y_all[train_mask & valid_mask & sym_mask]
                            mae, rmse, qlike, r2oos = evaluate_model(y_all[eval_mask], preds_series[eval_mask], y_train_sym)
                            results.append({
                                "model": model_name, "feature_set": set_name, "symbol": sym, 
                                "horizon": h, "split": split_name, 
                                "MAE": mae, "RMSE": rmse, "QLIKE": qlike, "R2_oos": r2oos
                            })
                            
                # Feature importances
                if model_name == "Random Forest" and set_name == "Set C (+Network)":
                    importances = model.feature_importances_
                    for i, col in enumerate(cols):
                        feature_importances.append({
                            "horizon": h, "feature": col, "importance": importances[i]
                        })
                            
    comp_df = pd.DataFrame(results)
    fi_df = pd.DataFrame(feature_importances)
    
    # Save ML comparisons
    ml_comp_path = OUTPUT_DIR / "ml_model_comparison_2006_2025.csv"
    comp_df.to_csv(ml_comp_path, index=False)
    
    # Merge with econometric baselines if available
    eco_path = OUTPUT_DIR / "econometric_model_comparison_2006_2025.csv"
    if eco_path.exists():
        eco_df = pd.read_csv(eco_path)
        full_comp_df = pd.concat([eco_df, comp_df], ignore_index=True)
        full_comp_path = OUTPUT_DIR / "volatility_model_comparison_2006_2025.csv"
        full_comp_df.to_csv(full_comp_path, index=False)
        print(f"Merged ML and Econometric results into {full_comp_path.name}")
    
    fi_path = OUTPUT_DIR / "ml_feature_importances_2006_2025.csv"
    fi_df.to_csv(fi_path, index=False)
    
    print("\nMachine Learning modeling complete!")

if __name__ == "__main__":
    main()

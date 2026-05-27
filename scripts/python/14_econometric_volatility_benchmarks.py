from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

def qlike_loss(y_true, y_pred, epsilon=1e-12):
    # Ensure strictly positive variance
    var_true = np.maximum(y_true**2, epsilon)
    var_pred = np.maximum(y_pred**2, epsilon)
    # QLIKE = log(var_pred) + var_true / var_pred
    qlike = np.log(var_pred) + var_true / var_pred
    return np.mean(qlike)

def directional_accuracy(y_true, y_pred):
    # direction of change vs previous day?
    # actually, usually it's comparing if it goes up or down relative to today's volatility or just correlation
    pass

def evaluate_model(y_true, y_pred, y_train_true):
    # Filter NaNs
    mask = pd.notnull(y_true) & pd.notnull(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan, np.nan
        
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    qlike = qlike_loss(y_true, y_pred)
    
    # R2 OOS = 1 - MSE(pred) / MSE(mean_train)
    mse_pred = mean_squared_error(y_true, y_pred)
    mse_baseline = np.mean((y_true - np.mean(y_train_true))**2)
    r2_oos = 1 - (mse_pred / mse_baseline) if mse_baseline > 0 else np.nan
    
    return mae, rmse, qlike, r2_oos

def main():
    print("Starting Phase 10B: Econometric Volatility Benchmarks...")
    
    # Load Dataset
    df_path = OUTPUT_DIR / "volatility_forecasting_dataset_2006_2025.csv"
    df = pd.read_csv(df_path)
    df["date"] = pd.to_datetime(df["date"])
    
    symbols = df["symbol"].unique()
    horizons = [5, 20]
    
    results = []
    garch_params_list = []
    
    all_garch_forecasts = []
    all_har_forecasts = []
    all_hv_forecasts = []
    all_ewma_forecasts = []
    
    for sym in symbols:
        print(f"\nProcessing {sym}...")
        sdf = df[df["symbol"] == sym].copy()
        sdf = sdf.sort_values("date").reset_index(drop=True)
        
        train_mask = sdf["split"] == "train"
        val_mask = sdf["split"] == "validation"
        test_mask = sdf["split"] == "test"
        
        # We need the index of the last train observation for GARCH splitting
        train_end_idx = sdf[train_mask].index[-1]
        
        # 1. Historical Volatility & EWMA
        # HV is daily vol. To forecast h days forward, we multiply by sqrt(h).
        hv_20d_daily = sdf["rolling_vol_20d"]
        hv_60d_daily = sdf["rolling_vol_60d"]
        ewma_daily = sdf["ewma_volatility"]
        
        for h in horizons:
            target_col = f"target_rv_{h}d"
            y_true = sdf[target_col]
            y_train_true = sdf.loc[train_mask, target_col].dropna()
            
            # Forecasts
            pred_hv20 = hv_20d_daily * np.sqrt(h)
            pred_hv60 = hv_60d_daily * np.sqrt(h)
            pred_ewma = ewma_daily * np.sqrt(h)
            
            # Store forecasts
            for split_name, split_mask in [("train", train_mask), ("validation", val_mask), ("test", test_mask)]:
                for model_name, preds in [("HV_20d", pred_hv20), ("HV_60d", pred_hv60), ("EWMA", pred_ewma)]:
                    mae, rmse, qlike, r2oos = evaluate_model(y_true[split_mask], preds[split_mask], y_train_true)
                    results.append({
                        "model": model_name, "feature_set": "Classical", "symbol": sym, 
                        "horizon": h, "split": split_name, 
                        "MAE": mae, "RMSE": rmse, "QLIKE": qlike, "R2_oos": r2oos
                    })
                    
        # 2. GARCH(1,1) Student-t
        print(f"Fitting GARCH(1,1) Student-t for {sym}...")
        returns_pct = sdf["log_return"] * 100.0
        
        # Fit on train data only
        am = arch_model(returns_pct, vol="GARCH", p=1, q=1, dist="studentst", mean="Zero")
        res = am.fit(last_obs=train_end_idx + 1, disp="off")
        
        # Save params
        garch_params_list.append({
            "symbol": sym,
            "omega": res.params.get("omega", np.nan),
            "alpha": res.params.get("alpha[1]", np.nan),
            "beta": res.params.get("beta[1]", np.nan),
            "alpha_plus_beta": res.params.get("alpha[1]", 0) + res.params.get("beta[1]", 0),
            "nu": res.params.get("nu", np.nan),
            "loglikelihood": res.loglikelihood,
            "aic": res.aic,
            "bic": res.bic
        })
        
        # Generate full-sample forecasts iteratively (align='origin')
        forecasts = res.forecast(horizon=20, start=0, align="origin")
        var_forecasts_pct = forecasts.variance # DataFrame where cols are h.01 ... h.20
        var_forecasts = var_forecasts_pct / 10000.0 # scale back
        
        # Cumulative variance for h=5 and h=20
        pred_garch_5d = np.sqrt(var_forecasts[[f"h.{i:02d}" for i in range(1, 6)]].sum(axis=1))
        pred_garch_20d = np.sqrt(var_forecasts.sum(axis=1))
        
        for h, preds in [(5, pred_garch_5d), (20, pred_garch_20d)]:
            target_col = f"target_rv_{h}d"
            y_true = sdf[target_col]
            y_train_true = sdf.loc[train_mask, target_col].dropna()
            
            for split_name, split_mask in [("train", train_mask), ("validation", val_mask), ("test", test_mask)]:
                mae, rmse, qlike, r2oos = evaluate_model(y_true[split_mask], preds[split_mask], y_train_true)
                results.append({
                    "model": "GARCH(1,1)", "feature_set": "Classical", "symbol": sym, 
                    "horizon": h, "split": split_name, 
                    "MAE": mae, "RMSE": rmse, "QLIKE": qlike, "R2_oos": r2oos
                })
                
        # 3. HAR-RV
        print(f"Fitting HAR-RV for {sym}...")
        # Features: daily (abs_r_t), weekly (lagged_rv_5d), monthly (lagged_rv_20d)
        har_features = ["abs_r_t", "lagged_rv_5d", "lagged_rv_20d"]
        X_all = sdf[har_features].fillna(0)
        
        # Fit a separate Ridge model for h=5 and h=20
        for h in horizons:
            target_col = f"target_rv_{h}d"
            y_all = sdf[target_col]
            
            # Mask for valid training data
            valid_train = train_mask & y_all.notnull()
            X_train = X_all[valid_train]
            y_train = y_all[valid_train]
            
            # Use a tiny ridge penalty
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            preds = pd.Series(model.predict(X_all), index=X_all.index)
            # Clip predictions to 0 to avoid negative RV
            preds = np.maximum(preds, 1e-6)
            
            for split_name, split_mask in [("train", train_mask), ("validation", val_mask), ("test", test_mask)]:
                mae, rmse, qlike, r2oos = evaluate_model(y_all[split_mask], preds[split_mask], y_train)
                results.append({
                    "model": "HAR-RV", "feature_set": "Classical", "symbol": sym, 
                    "horizon": h, "split": split_name, 
                    "MAE": mae, "RMSE": rmse, "QLIKE": qlike, "R2_oos": r2oos
                })
                
    # Compile Results
    comp_df = pd.DataFrame(results)
    garch_params_df = pd.DataFrame(garch_params_list)
    
    comp_path = OUTPUT_DIR / "econometric_model_comparison_2006_2025.csv"
    comp_df.to_csv(comp_path, index=False)
    
    garch_params_path = OUTPUT_DIR / "garch_parameters_2006_2025.csv"
    garch_params_df.to_csv(garch_params_path, index=False)
    
    print("\nEconometric modeling complete!")
    print(f"Outputs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

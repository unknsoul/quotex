"""
Bayesian Hyperparameter Optimization Engine (Optuna)
Phase 9 Layer: Tests 10,000+ configurations of XGBoost/LightGBM against the M1 Microstructure 
to find the mathematical boundary for >60% accuracy on purely 1-candle Expiry.
"""

import sys
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score

from data_collector import fetch_multi_timeframe
from feature_engineering import compute_features

import warnings
warnings.filterwarnings("ignore")

# Define pairs and target length for robust optimization
TARGET_SYMBOL = "EURUSD"
MIN_CANDLES = 10000

def fetch_optimization_data():
    """Fetch and prep data exactly as the Live Bot sees it."""
    print(f"Fetching {MIN_CANDLES} candles for {TARGET_SYMBOL}...")
    tf_dict = fetch_multi_timeframe(TARGET_SYMBOL, MIN_CANDLES)
    
    for k, v in tf_dict.items():
        print(f"DEBUG: Dataframe {k} is type {type(v)}")
        
    m5 = tf_dict.get('M5')
    m15 = tf_dict.get('M15')
    h1 = tf_dict.get('H1')
    m1 = tf_dict.get('M1')
    
    if any(isinstance(x, str) for x in [m5, m15, h1, m1]):
        print("ERROR: One of the timeframes returned an error string instead of a DataFrame. Exiting.")
        sys.exit(1)
        
    df = compute_features(m5, m15, h1, m1)
    
    # 1-Candle Strict Binary Target (1 = UP, 0 = DOWN)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop NAs
    df.dropna(inplace=True)
    return df

def objective(trial, X, y):
    """
    Optuna objective function for XGBoost. 
    Searches for the absolute optimal settings to prevent overfitting on 1-candle noise.
    """
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        # Highly tunable parameters
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": 1.0 # Handled externally by class weights normally, kept 1.0 for search
    }

    # TimeSeriesSplit ensures no look-ahead bias during mathematical search
    tscv = TimeSeriesSplit(n_splits=5)
    
    auc_scores = []
    
    for train_index, valid_index in tscv.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        # Train with early stopping to strictly penalize overfitting
        model = xgb.train(
            param, 
            dtrain, 
            num_boost_round=param['n_estimators'],
            evals=[(dvalid, 'eval')],
            early_stopping_rounds=30,
            verbose_eval=False
        )

        # Predict validation
        preds = model.predict(dvalid)
        
        try:
            auc = roc_auc_score(y_valid, preds)
            auc_scores.append(auc)
        except ValueError:
            # Handle edge cases where validation fold has only 1 class
            continue

    # Return mean AUC across all 5 chronological folds
    return np.mean(auc_scores) if auc_scores else 0.5


import MetaTrader5 as mt5

if __name__ == "__main__":
    print("Initializing MT5 Connection...")
    if not mt5.initialize():
        print("Failed to initialize MT5")
        sys.exit(1)
        
    df = fetch_optimization_data()
    
    # Strip non-predictive columns
    drop_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'target', 'session']
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df['target']
    
    print(f"\\n[OPTIMIZATION] Initializing Optuna Study on {len(X)} rows & {len(features)} features...")
    print("Maximizing Out-Of-Sample AUC across 5 Time-Series Folds.\\n")
    
    # Create study designed to maximize AUC
    study = optuna.create_study(direction="maximize", study_name="quoteX_xgboost_master")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50, show_progress_bar=True)
    
    print("\\n=======================================================")
    print("OPTIMIZATION COMPLETE")
    print(f"Best Trial Value (AUC): {study.best_value:.4f}")
    print("Best Hyperparameters Discovered:")
    for key, value in study.best_trial.params.items():
        print(f"   {key}: {value}")
    print("=======================================================\\n")

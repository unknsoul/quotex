"""
Central configuration v4 — Quant Stability & Precision.
Ensemble models, continuous regime, dynamic weights.
"""

import os

# --- Project Paths -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CHART_DIR = os.path.join(BASE_DIR, "charts")

MODEL_PATH = os.path.join(MODEL_DIR, "primary_model.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "primary_features.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_models.pkl")
ENSEMBLE_TRENDING_PATH = os.path.join(MODEL_DIR, "ensemble_trending.pkl")
ENSEMBLE_RANGING_PATH = os.path.join(MODEL_DIR, "ensemble_ranging.pkl")
META_MODEL_PATH = os.path.join(MODEL_DIR, "meta_model.pkl")
META_FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "meta_features.pkl")
OOF_PREDICTIONS_PATH = os.path.join(MODEL_DIR, "oof_predictions.pkl")
WEIGHT_MODEL_PATH = os.path.join(MODEL_DIR, "weight_model.pkl")

PREDICTION_LOG_CSV = os.path.join(LOG_DIR, "predictions.csv")
PREDICTION_LOG_JSON = os.path.join(LOG_DIR, "predictions.json")

# --- MetaTrader5 -------------------------------------------------------------
MT5_PATH = os.getenv("MT5_PATH", "")
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

# --- Data Settings -----------------------------------------------------------
DEFAULT_SYMBOL = "EURUSD"
CANDLES_TO_FETCH = 15000
MTF_TIMEFRAMES = ["M5", "M15", "H1"]

# --- EMA Periods -------------------------------------------------------------
EMA_20 = 20
EMA_50 = 50
EMA_100 = 100
EMA_200 = 200
EMA_SLOPE_WINDOW = 10

# --- Indicator Periods -------------------------------------------------------
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
ADX_PERIOD = 14
ROLLING_STD_PERIOD = 20
VOLATILITY_ROLLING = 10
MOMENTUM_ROLLING = 5
RETURN_LOOKBACK = 5

# --- v2/v3/v4 Feature Params ------------------------------------------------
RANGE_POSITION_WINDOW = 20
LIQUIDITY_SWEEP_WINDOW = 10
VOLATILITY_ZSCORE_WINDOW = 50
ATR_PERCENTILE_WINDOW = 100  # v4: for ATR percentile rank

# --- v5 Target Threshold -----------------------------------------------------
TARGET_ATR_THRESHOLD = 0.3  # only train on moves > 0.3 × ATR

# --- Regime Detection --------------------------------------------------------
ADX_TRENDING_THRESHOLD = 25
ADX_RANGING_THRESHOLD = 20
ATR_HIGH_VOL_MULTIPLIER = 1.5
ATR_LOW_VOL_MULTIPLIER = 0.7
ATR_ROLLING_WINDOW = 100

# --- Regime Thresholds (trade suggestions) -----------------------------------
REGIME_THRESHOLDS = {
    "Trending":        {"primary": 0.62, "meta": 0.58},
    "Ranging":         {"primary": 0.58, "meta": 0.62},
    "High_Volatility": {"primary": 0.70, "meta": 0.65},
    "Low_Volatility":  {"primary": 0.75, "meta": 0.75},
}
PRIMARY_BASE_THRESHOLD = 0.60
META_BASE_THRESHOLD = 0.60

# --- Confidence Levels (%) --------------------------------------------------
CONFIDENCE_HIGH_MIN = 75.0
CONFIDENCE_MEDIUM_MIN = 60.0

# --- Risk Warnings -----------------------------------------------------------
MIN_ATR_CLOSE_RATIO = 0.0003
SPREAD_PERCENTILE = 90
ATR_SPIKE_MULTIPLIER = 3.0

# --- Ensemble (v4) ----------------------------------------------------------
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1024]
ENSEMBLE_SIZE = len(ENSEMBLE_SEEDS)

# --- XGBoost Primary Hyperparams --------------------------------------------
XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
TIMESERIES_SPLITS = 5

# --- Meta Model Hyperparams -------------------------------------------------
META_N_ESTIMATORS = 200
META_MAX_DEPTH = 4
META_LEARNING_RATE = 0.05
META_SUBSAMPLE = 0.8

# --- Stability ---------------------------------------------------------------
PROBABILITY_SMOOTHING_SPAN = 5
ROLLING_ACCURACY_WINDOW = 50
RETRAINING_ACCURACY_TRIGGER = 0.48
META_ROLLING_WINDOW = 50
WIN_STREAK_CAP = 10
DRIFT_COSINE_THRESHOLD = 0.85  # v4: alert if cosine sim < this

# --- Session Hours (UTC) ----------------------------------------------------
SESSION_ASIA = (0, 8)
SESSION_LONDON = (8, 16)
SESSION_NEW_YORK = (13, 21)

# --- Logging -----------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

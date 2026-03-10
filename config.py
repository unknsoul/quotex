"""
Central configuration — QUOTEX LORD v10 Production Engine.

14-layer pipeline: 5-model ensemble + HistGBM meta + logistic weight learner.
Triple barrier labels (TP=1.5×ATR, SL=1.0×ATR, max_bars=4).
"""

import os

# Load .env file if present (python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# --- Project Paths -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CHART_DIR = os.path.join(BASE_DIR, "charts")

MODEL_PATH = os.path.join(MODEL_DIR, "primary_model.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "primary_features.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_models.pkl")
META_MODEL_PATH = os.path.join(MODEL_DIR, "meta_model.pkl")
META_FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "meta_features.pkl")
OOF_PREDICTIONS_PATH = os.path.join(MODEL_DIR, "oof_predictions.pkl")
WEIGHT_MODEL_PATH = os.path.join(MODEL_DIR, "weight_model.pkl")
CONFIDENCE_THRESHOLDS_PATH = os.path.join(MODEL_DIR, "confidence_thresholds.pkl")
SELECTED_FEATURES_PATH = os.path.join(MODEL_DIR, "selected_features.pkl")

PREDICTION_LOG_CSV = os.path.join(LOG_DIR, "predictions.csv")
PREDICTION_LOG_JSON = os.path.join(LOG_DIR, "predictions.json")

# --- MetaTrader5 -------------------------------------------------------------
MT5_PATH = os.getenv("MT5_PATH", "")
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "")

# --- Gemini API --------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Data Settings -----------------------------------------------------------
DEFAULT_SYMBOL = "EURUSD"
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
CANDLES_TO_FETCH = 50000
MTF_TIMEFRAMES = ["M5", "M15", "H1"]
TRAINING_SYMBOLS = ["EURUSD"]  # multi-symbol joint training pool

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
ATR_PERCENTILE_WINDOW = 100

# --- v5 Target Threshold -----------------------------------------------------
TARGET_ATR_THRESHOLD = 0.3  # only train on moves > 0.3 * ATR

# --- Target Generation (Phase 10: Master Architecture) ----------------------
TARGET_LOOKAHEAD = 3        # Predict smoothed 3 candles
TRIPLE_BARRIER_ENABLED = True
TRIPLE_BARRIER_TP = 1.5   # Take Profit = 1.5 × ATR
TRIPLE_BARRIER_SL = 1.0   # Stop Loss  = 1.0 × ATR (asymmetric: tighter SL)
TRIPLE_BARRIER_MAX_BARS = 4  # max 4 candles (20 min at M5)

# --- Regime Detection --------------------------------------------------------
ADX_TRENDING_THRESHOLD = 25
ADX_RANGING_THRESHOLD = 20
ATR_HIGH_VOL_MULTIPLIER = 1.5
ATR_LOW_VOL_MULTIPLIER = 0.7
ATR_ROLLING_WINDOW = 100

# --- Regime Thresholds (trade suggestions) -----------------------------------
REGIME_THRESHOLDS = {
    "Trending":        {"primary": 0.58, "meta": 0.55},
    "Ranging":         {"primary": 0.59, "meta": 0.55},
    "High_Volatility": {"primary": 0.60, "meta": 0.56},
    "Low_Volatility":  {"primary": 0.59, "meta": 0.55},
}
PRIMARY_BASE_THRESHOLD = 0.58
META_BASE_THRESHOLD = 0.55

# --- Adaptive Regime Filter (Phase 3) ---------------------------------------
REGIME_FILTER_ENABLED = True
REGIME_SKIP_ACCURACY_THRESHOLD = 0.55  # skip if rolling accuracy < 55%
REGIME_COOLDOWN_BARS = 10              # bars to stay in cooldown after accuracy drop
REGIME_ROLLING_ACCURACY_WINDOW = 20    # rolling window for accuracy check
SESSION_FILTER_ENABLED = True
# Session confidence multipliers (learned: London best, Asian worst)
SESSION_CONFIDENCE_MULT = {
    "London":   1.0,    # best session
    "New_York": 0.95,   # good, but slightly worse
    "Overlap":  1.0,    # London+NY overlap = best liquidity
    "Asian":    0.85,   # lowest predictability
    "Off":      0.80,   # off-hours
}

# --- Ensemble Variance Hard Filter (Phase 4) --------------------------------
ENSEMBLE_VAR_SKIP_THRESHOLD = 0.02  # skip if ensemble variance > this
ENSEMBLE_VAR_FILTER_ENABLED = True

# --- Slippage Modeling (Phase 4) --------------------------------------------
SLIPPAGE_SPREAD_COST = 0.00035    # 3.5 pips EURUSD spread cost
SLIPPAGE_TICKS = 1                 # 1-tick random slippage

# --- Confidence Levels (%) --------------------------------------------------
CONFIDENCE_HIGH_MIN = 75.0
CONFIDENCE_MEDIUM_MIN = 60.0

# --- Production Decision Gate -----------------------------------------------
PRODUCTION_SIGNAL_GATING_ENABLED = True
PRODUCTION_MIN_CONFIDENCE = 48.0
PRODUCTION_MIN_META_RELIABILITY = 48.0
PRODUCTION_MIN_UNANIMITY = 0.60
PRODUCTION_MAX_UNCERTAINTY = 8.0
PRODUCTION_MIN_QUALITY_SCORE = 35.0
PRODUCTION_CONFIDENCE_ALERT_PENALTY = 0.95
PRODUCTION_REQUIRE_TREND_ALIGNMENT = False
PRODUCTION_BLOCKED_REGIMES = set()  # v10: no blanket regime blocks; risk filter handles edge cases

# --- Risk Warnings -----------------------------------------------------------
MIN_ATR_CLOSE_RATIO = 0.00005   # lowered: M5 candles have smaller ATR
SPREAD_PERCENTILE = 90
ATR_SPIKE_MULTIPLIER = 3.0

# --- Ensemble (5 seeded XGBoost) --------------------------------------------
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1024]
ENSEMBLE_SIZE = len(ENSEMBLE_SEEDS)

# --- XGBoost Primary Hyperparams (Optuna Phase 9) ---------------------------
XGB_N_ESTIMATORS = 318
XGB_MAX_DEPTH = 4  
XGB_LEARNING_RATE = 0.005022
XGB_SUBSAMPLE = 0.747
XGB_COLSAMPLE_BYTREE = 0.734
XGB_MIN_CHILD_WEIGHT = 12
XGB_GAMMA = 2.822
XGB_REG_ALPHA = 0.266
XGB_REG_LAMBDA = 0.0036
TIMESERIES_SPLITS = 5

# --- Calibration Split -------------------------------------------------------
CALIBRATION_SPLIT_RATIO = 0.8  # 80% train_main, 20% calibration

# --- OOF Settings ------------------------------------------------------------
OOF_INTERNAL_SPLITS = 3  # For generating OOF predictions within train_main
PURGE_EMBARGO_BARS = 50  # purge gap between train/test to prevent leakage
ROLLING_TRAIN_WINDOW = 0  # 0=expanding, >0=rolling window (bars). Try 5000 (~17 days M5)

# --- Meta Model (LightGBM) --------------------------------------------------
META_N_ESTIMATORS = 100
META_MAX_DEPTH = 3
META_LEARNING_RATE = 0.05
META_SUBSAMPLE = 0.8
META_NUM_LEAVES = 15

# --- Confidence Correlation Monitoring ---------------------------------------
CONFIDENCE_CORRELATION_WINDOW = 200
CONFIDENCE_CORRELATION_ALERT = 0.1
ROLLING_CONFIDENCE_WINDOW = 100

# --- Stability ---------------------------------------------------------------
PROBABILITY_SMOOTHING_SPAN = 5
ROLLING_ACCURACY_WINDOW = 50
RETRAINING_ACCURACY_TRIGGER = 0.48
META_ROLLING_WINDOW = 50
WIN_STREAK_CAP = 10
DRIFT_COSINE_THRESHOLD = 0.85

# --- Session Hours (UTC) ----------------------------------------------------
SESSION_ASIA = (0, 8)
SESSION_LONDON = (8, 16)
SESSION_NEW_YORK = (13, 21)

# --- Logging -----------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

# --- Telegram Bot ------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_SEND_BEFORE_CLOSE_SEC = 15  # start processing 15s early, signal arrives ~5s before close

# --- v10 Validation Quality Gates -------------------------------------------
VALIDATION_MAX_BRIER = 0.22
VALIDATION_MIN_WF_ACCURACY = 0.60
VALIDATION_MIN_HIGH_CONF_ACC = 0.75
VALIDATION_MIN_PROFIT_FACTOR = 1.4
VALIDATION_MAX_DRAWDOWN = 0.20

# --- Auto-Learning -----------------------------------------------------------
AUTO_RETRAIN_ACCURACY_TRIGGER = 0.52
AUTO_RETRAIN_CORRELATION_TRIGGER = 0.1
DATA_BUFFER_SIZE = 20000
MODEL_BACKUP_DIR = os.path.join(BASE_DIR, "models", "backup")

# --- Signal Validator --------------------------------------------------------
SIGNAL_MIN_CONFIDENCE = 52.0          # realistic minimum for current model accuracy
SIGNAL_DUPLICATE_WINDOW_SEC = 300     # 5 min duplicate guard
SIGNAL_MAX_PER_HOUR = 6              # max signals per hour
SIGNAL_REQUIRE_MTF_ALIGNMENT = True   # block if multi-TF opposes direction

# --- Session Filter ----------------------------------------------------------
SESSION_HOURS = {
    "Asian":    list(range(0, 8)),
    "London":   list(range(8, 16)),
    "New_York": list(range(13, 21)),
    "Overlap":  list(range(13, 16)),
    "Off":      list(range(21, 24)),
}
SESSION_QUALITY_SCORES = {
    "Overlap":  1.0,
    "London":   0.95,
    "New_York": 0.90,
    "Asian":    0.65,
    "Off":      0.40,
}
# Sessions to block entirely per symbol (data-driven)
SESSION_BLOCK_MAP = {
    "EURUSD": ["Off"],
    "GBPUSD": ["Asian", "Off"],
    "USDJPY": ["Off"],
    "AUDUSD": ["Off"],
    "USDCAD": ["Off"],
    "USDCHF": ["Asian", "Off"],
}

# --- Performance Dashboard ---------------------------------------------------
DASHBOARD_DIR = os.path.join(LOG_DIR, "dashboards")
DASHBOARD_ROLLING_WINDOW = 50    # trades for rolling metrics

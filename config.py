"""
Central configuration — all constants, thresholds, and paths.
No magic numbers anywhere else in the codebase.
"""

import os

# ─── Project Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.pkl")
PREDICTION_LOG_CSV = os.path.join(LOG_DIR, "predictions.csv")
PREDICTION_LOG_JSON = os.path.join(LOG_DIR, "predictions.json")

# ─── MetaTrader5 ────────────────────────────────────────────────────────────
MT5_PATH = os.getenv("MT5_PATH", "")   # path to terminal64.exe (optional)
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

# ─── Data Settings ──────────────────────────────────────────────────────────
DEFAULT_SYMBOL = "EURUSD"
CANDLES_TO_FETCH = 10000            # ~1 year of M5 data

# ─── EMA Periods ────────────────────────────────────────────────────────────
EMA_20 = 20
EMA_50 = 50
EMA_100 = 100
EMA_200 = 200
EMA_SLOPE_WINDOW = 10               # bars to measure slope over

# ─── Indicator Periods ─────────────────────────────────────────────────────
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

# ─── Regime Detection ──────────────────────────────────────────────────────
ADX_TRENDING_THRESHOLD = 25
ADX_RANGING_THRESHOLD = 20
ATR_HIGH_VOL_MULTIPLIER = 1.5
ATR_LOW_VOL_MULTIPLIER = 0.7
ATR_ROLLING_WINDOW = 100

# ─── Decision Thresholds (per regime) ──────────────────────────────────────
BASE_THRESHOLD = 0.60
REGIME_THRESHOLDS = {
    "Trending":        0.62,
    "Ranging":         0.58,
    "High_Volatility": 0.70,
    "Low_Volatility":  0.75,
}

# ─── Confidence Levels ─────────────────────────────────────────────────────
CONFIDENCE_LOW_MAX = 0.60            # < 0.60 → Low
CONFIDENCE_MED_MAX = 0.75            # 0.60–0.74 → Medium  |  ≥ 0.75 → High

# ─── Risk Filters ──────────────────────────────────────────────────────────
MIN_ATR_CLOSE_RATIO = 0.0003        # minimum ATR/close for acceptable volatility
MAX_SPREAD_MULTIPLIER = 3.0         # max spread vs rolling avg spread
ATR_SPIKE_MULTIPLIER = 3.0          # skip if ATR > mean * this

# ─── XGBoost Hyperparameters ───────────────────────────────────────────────
XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
TIMESERIES_SPLITS = 5

# ─── Session Hours (UTC) ───────────────────────────────────────────────────
SESSION_ASIA = (0, 8)
SESSION_LONDON = (8, 16)
SESSION_NEW_YORK = (13, 21)

# ─── Logging ───────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

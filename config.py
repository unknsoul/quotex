# -*- coding: utf-8 -*-
"""Central configuration - QUOTEX LORD v11 Advanced Production Engine.

v11 upgrades:
  - 9 Strategy Signal Engine (parallel evaluation)
  - Same-Candle / Stale Candle Detector (SC-1 to SC-5)
  - Staleness Tracker (crossover signal decay)
  - 12 Candlestick Pattern Overlay
  - CHOPPY regime detection + signal suspension
  - Anti-streak engine (prevents directional collapse)
  - Candle quality filter (detects stale/repetitive candles)
  - Momentum exhaustion detector
  - Symmetric triple barrier labeling (fixes bullish bias)
  - Hour-strategy confidence scaling (no blocking)
  - Symbol-specific confidence adjustments
  - Martingale money management (capped at step 3)
  - Data-driven from 418 outcome analysis
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
TRAINING_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]  # v14: multi-symbol training

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
SESSION_FILTER_ENABLED = False   # v13: All session restrictions removed — pure technical analysis
# Session confidence multipliers — all set to 1.0 (no session penalty)
SESSION_CONFIDENCE_MULT = {
    "London":   1.0,
    "New_York": 1.0,
    "Overlap":  1.0,
    "Asian":    1.0,
    "Off":      1.0,
}

# --- Ensemble Variance Hard Filter (Phase 4) --------------------------------
ENSEMBLE_VAR_SKIP_THRESHOLD = 0.045  # v14.1: relaxed for 7-model diverse ensemble
ENSEMBLE_VAR_FILTER_ENABLED = True

# --- Slippage Modeling (Phase 4) --------------------------------------------
SLIPPAGE_SPREAD_COST = 0.00035    # 3.5 pips EURUSD spread cost
SLIPPAGE_TICKS = 1                 # 1-tick random slippage

# --- Confidence Levels (%) --------------------------------------------------
CONFIDENCE_HIGH_MIN = 75.0
CONFIDENCE_MEDIUM_MIN = 60.0

# --- Production Decision Gate -----------------------------------------------
PRODUCTION_SIGNAL_GATING_ENABLED = True
PRODUCTION_MIN_CONFIDENCE = 52.0       # v14.1: balanced (was 58, blocked too many borderline signals)
PRODUCTION_MIN_META_RELIABILITY = 48.0  # v14: raised from 45
PRODUCTION_MIN_UNANIMITY = 0.667        # v14: at least 4/6 models must agree
PRODUCTION_MAX_UNCERTAINTY = 8.0         # v14: tightened from 10
PRODUCTION_MIN_QUALITY_SCORE = 40.0      # v14: raised from 30
PRODUCTION_CONFIDENCE_ALERT_PENALTY = 0.95
PRODUCTION_REQUIRE_TREND_ALIGNMENT = False
PRODUCTION_BLOCKED_REGIMES = {"CHOPPY", "Ranging"}  # v14: Ranging=41% accuracy from 405 outcomes

# --- Risk Warnings -----------------------------------------------------------
MIN_ATR_CLOSE_RATIO = 0.00005   # lowered: M5 candles have smaller ATR
SPREAD_PERCENTILE = 90
ATR_SPIKE_MULTIPLIER = 3.0

# --- Ensemble (5 seeded XGBoost) --------------------------------------------
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192]  # v14: 8 seeds for up to 8 ensemble members
ENSEMBLE_SIZE = len(ENSEMBLE_SEEDS)

# --- XGBoost Primary Hyperparams (Optuna Phase 9) ---------------------------
# v14: Improved hyperparameters — more estimators, deeper trees, stronger regularization
XGB_N_ESTIMATORS = 450
XGB_MAX_DEPTH = 5  
XGB_LEARNING_RATE = 0.004
XGB_SUBSAMPLE = 0.75
XGB_COLSAMPLE_BYTREE = 0.70
XGB_MIN_CHILD_WEIGHT = 18
XGB_GAMMA = 3.0
XGB_REG_ALPHA = 0.5
XGB_REG_LAMBDA = 2.0
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
SIGNAL_MIN_CONFIDENCE = 58.0          # v14: aligned with production gate (raised from 42)
SIGNAL_DUPLICATE_WINDOW_SEC = 300     # 5 min duplicate guard
SIGNAL_MAX_PER_HOUR = 8              # raised from 6 to allow more signals
SIGNAL_REQUIRE_MTF_ALIGNMENT = True   # block if multi-TF opposes direction
SIGNAL_SOFT_CONFLUENCE_OVERRIDE_CONFIDENCE = 55.0
SIGNAL_SOFT_CONFLUENCE_OVERRIDE_QUALITY = 65.0
SIGNAL_SOFT_CONFLUENCE_OVERRIDE_UNANIMITY = 0.80

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
    "London":   1.0,
    "New_York": 1.0,
    "Asian":    1.0,
    "Off":      1.0,
}
# Sessions to block entirely per symbol — DISABLED (trade all sessions)
SESSION_BLOCK_MAP = {}

# --- Performance Dashboard ---------------------------------------------------
DASHBOARD_DIR = os.path.join(LOG_DIR, "dashboards")
DASHBOARD_ROLLING_WINDOW = 50    # trades for rolling metrics

# --- v11: Anti-Streak Engine (data-driven from 418 outcomes) -----------------
# Model predicted DOWN 96% of last 100 signals. Long streaks (>10) had 49.8% WR
# vs short streaks (<=3) at 55% WR. This fixes directional collapse.
V11_ANTI_STREAK_ENABLED = True
V11_STREAK_PENALTY_START = 8         # Allow sustained trend periods before confidence suppression starts
V11_STREAK_PENALTY_RATE = 0.02       # Softer penalty to avoid long low-confidence droughts
V11_STREAK_MAX_PENALTY = 0.16        # Cap reduction so valid trend signals still survive
V11_STREAK_FORCE_HOLD = 16           # Only hard-stop extremely one-sided runs

# --- v11: Candle Quality Filter ----------------------------------------------
# Detects stale/repetitive candles in trending markets (user's specific concern)
V11_CANDLE_QUALITY_ENABLED = True
V11_CANDLE_FRESHNESS_MIN = 0.12      # Skip if freshness < 12% (very stale)
V11_CANDLE_QUALITY_MIN = 25.0        # Skip if quality score < 25/100

# --- v11: Momentum Exhaustion ------------------------------------------------
V11_EXHAUSTION_ENABLED = True
V11_EXHAUSTION_SKIP_THRESHOLD = 75   # Skip if exhaustion > 75%
V11_EXHAUSTION_PENALTY_START = 50    # Start penalizing at 50%

# --- v11: Hour-Strategy Confidence Scaling (replaces hour blocking) -----------
# No longer blocks hours — uses per-hour confidence multipliers in session_filter.py
# Weak hours (9,10 UTC) get 0.80-0.85x penalty but can still trade with strong signals
# See session_filter.HOUR_CONFIDENCE_MULT for full hour mapping
V11_HOUR_STRATEGY_ENABLED = False     # v13: Disabled — no hour-based restrictions
V11_WEAK_HOUR_MIN_UNANIMITY = 0.50   # v13: Relaxed — no weak-hour extra unanimity needed

# --- v11: Symbol-Specific Confidence Adjustments (data-driven) ---------------
# EURUSD (42.1%) and AUDUSD (42.3%) need higher thresholds
# v14: Updated from 405 live outcomes analysis
V11_SYMBOL_CONFIDENCE_ADJUSTMENTS = {
    "USDJPY": 1.08,     # 58.8% WR — best performer, boost
    "XAUUSD": 1.05,     # 56.8% WR — strong performer
    "GBPUSD": 0.95,     # 47.1% WR — slight penalty
    "GBPJPY": 0.92,     # 43.2% WR — penalty
    "EURUSD": 0.85,     # 39.3% WR — heavy penalty
    "AUDUSD": 0.82,     # 40.0% WR — heaviest penalty
    "USDCAD": 0.95,     # moderate
    "USDCHF": 0.95,     # moderate
}

# --- v11: Symmetric Triple Barrier -------------------------------------------
V11_TRIPLE_BARRIER_SYMMETRIC = True  # Use symmetric TP/SL checking (fixes bias)

# --- v11: Same-Candle Detector (SC-1 to SC-5) --------------------------------
V11_SAME_CANDLE_ENABLED = True
V11_SAME_CANDLE_ACTION = "SKIP_SIGNAL"  # SKIP_SIGNAL or REDUCE_CONFIDENCE

# --- v11: Strategy Engine (9 parallel strategies) ----------------------------
V11_STRATEGY_ENGINE_ENABLED = True
V11_MIN_STRATEGIES_AGREEING = 4   # at least 4 strategies must agree
V11_MIN_COMPOSITE_SCORE = 62      # minimum composite strategy score
V11_CHOPPY_REGIME_SUSPEND = True  # suspend all signals in CHOPPY regime

# --- v11: Staleness Tracker (crossover decay) --------------------------------
V11_STALENESS_ENABLED = True
V11_STALENESS_FRESH_BARS = 1      # crossover within 1 bar = fresh
V11_STALENESS_STALE_BARS = 3      # beyond 3 bars = noise

# --- v11: Martingale Money Management ----------------------------------------
MARTINGALE_ENABLED = False         # disabled by default for safety
MARTINGALE_MAX_STEP = 3
MARTINGALE_MIN_SIGNAL_SCORE = 70   # only on high-confidence signals
MARTINGALE_DAILY_DD_PAUSE = 8.0    # pause if daily drawdown > 8%

# --- v11: Candlestick Pattern Overlay ----------------------------------------
V11_PATTERN_OVERLAY_ENABLED = True
V11_PATTERN_CONFIRM_BONUS = 10     # +10 to composite when pattern confirms
V11_PATTERN_CONTRADICT_PENALTY = 15  # -15 when pattern contradicts
V11_PATTERN_MAX_BONUS = 20          # cap at ±20 for multiple patterns

# --- v12: Accuracy Upgrades --------------------------------------------------
# Dispatch model (learns from live outcomes which signals actually win)
V12_DISPATCH_MODEL_ENABLED = True
V12_DISPATCH_MIN_WIN_PROB = 0.46       # min predicted win probability to dispatch

# Bayesian thresholds (per symbol/regime/hour adaptive confidence floors)
V12_BAYESIAN_THRESHOLD_ENABLED = True

# Outcome feedback loop (auto-tunes dispatch parameters)
V12_OUTCOME_FEEDBACK_ENABLED = True

# Feature importance drift monitoring
V12_FEATURE_IMPORTANCE_MONITOR_ENABLED = True

# Adaptive session multipliers (learns from hour-level outcomes)
V12_ADAPTIVE_SESSION_ENABLED = True

# Confidence recalibration (isotonic-style in predict_engine)
V12_CONFIDENCE_RECALIBRATION_ENABLED = True

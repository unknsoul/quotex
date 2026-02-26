# Regime-Aware Next Candle Probability Engine

Production-ready backend that predicts the probability of the next 5-minute candle being bullish or bearish using MetaTrader5, XGBoost, and regime-aware dynamic thresholds.

## Features

- **30 Technical Features** — EMAs, RSI, MACD, Stochastic, ADX, ATR, Bollinger Bands, candlestick patterns
- **4 Market Regimes** — Trending, Ranging, High Volatility, Low Volatility
- **Dynamic Thresholds** — probability thresholds adjust per regime (0.58–0.75)
- **Risk Filters** — volatility, spread, ATR spike guards
- **Walk-Forward Backtest** — per-regime win rate, Sharpe ratio, max drawdown, profit factor
- **FastAPI Server** — `POST /predict` returns full JSON with regime, probabilities, confidence, and reason summary
- **Prediction Logging** — every prediction logged to CSV + JSON

## Architecture

```
quotex/
├── config.py               # All constants and thresholds
├── data_collector.py        # MT5 -> validate -> CSV
├── feature_engineering.py   # 30 features, zero data leakage
├── regime_detection.py      # 4-regime classifier (ADX/ATR/EMA)
├── train_model.py           # XGBoost + 5-fold TimeSeriesSplit
├── predict_engine.py        # Load model, predict, log
├── risk_filter.py           # Pre-prediction safety guards
├── backtest.py              # Per-regime performance analysis
├── api_server.py            # FastAPI POST /predict endpoint
├── requirements.txt         # Dependencies
├── data/                    # Historical OHLC CSVs (gitignored)
├── models/                  # Trained model files (gitignored)
└── logs/                    # Prediction logs (gitignored)
```

## Quick Start

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Collect data (MT5 must be running)
python data_collector.py --symbol EURUSD

# 3. Train model
python train_model.py --symbol EURUSD

# 4. Backtest
python backtest.py --symbol EURUSD

# 5. Start API
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "EURUSD"}'
```

Response:
```json
{
  "symbol": "EURUSD",
  "timestamp": "2026-02-26T05:23:01+00:00",
  "market_regime": "Trending",
  "green_probability": 0.6431,
  "red_probability": 0.3569,
  "threshold_used": 0.62,
  "confidence_level": "Medium",
  "suggested_trade": "BUY",
  "volatility_status": "Normal",
  "reason_summary": "Regime: Trending; EMA bullish (20>50>200); RSI 58; MACD+; Moderate trend (ADX 29)"
}
```

## Requirements

- Windows 10/11
- Python 3.11+
- MetaTrader5 terminal (installed and running)

## License

MIT

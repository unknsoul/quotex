---
description: How to train, test, and deploy the prediction model
---
// turbo-all

# Training & Testing Workflow

## 1. Full Retrain (from scratch)
```powershell
cd c:\Users\hardi\Downloads\quotex

# Step 1: Train primary ensemble (5 XGBoost with temporal windows)
python train_model.py --symbol EURUSD

# Step 2: Train meta model (GBM + sigmoid calibration)
python meta_model.py --symbol EURUSD

# Step 3: Train weight learner (logistic regression)
python weight_learner.py --symbol EURUSD

# Step 4: Run walk-forward backtest
python backtest.py --symbol EURUSD
```

## 2. Lite Retrain (fast, meta + weight only)
Use when you want to quickly adapt to recent market changes without retraining the full ensemble:
```powershell
python auto_learner.py --symbol EURUSD --lite
```

## 3. Forward Test (live paper trading)
```powershell
python forward_test.py --symbol EURUSD
```
This runs predictions on live data without placing trades, logging results for analysis.

## 4. Start Telegram Bot
```powershell
# Set token (or use .env file)
$env:TELEGRAM_BOT_TOKEN='your_token_here'
python telegram_bot.py
```

## 5. Check Results
- **CSV outcomes**: `logs/signal_outcomes.csv`
- **Production state**: `logs/production_state.json`
- **Prediction logs**: `logs/predictions.csv`
- **Charts**: `charts/` folder

## 6. Key Metrics to Watch
| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Brier Score | <0.22 | 0.22-0.24 | >0.24 |
| Spearman | >0.15 | 0.10-0.15 | <0.10 |
| Live Accuracy | >60% | 55-60% | <55% |
| ≥80% Conf Acc | >80% | 75-80% | <75% |

## 7. When to Retrain
- Rolling accuracy drops below 55% for 50+ predictions
- Spearman correlation drops below 0.10
- After major market events (rate decisions, NFP)
- Weekly lite retrain recommended (Sunday 00:00 UTC)

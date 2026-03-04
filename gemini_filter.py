"""
Layer 13: Gemini AI Master Trader Verification Layer.

This module packages the raw technical context of a trade signal
and sends it to Google's Gemini LLM. The LLM acts as an expert
quantitative analyst and provides a final WIN/LOSE/UNCERTAIN verdict.
"""

import os
import logging
import json
import asyncio
import google.generativeai as genai
from config import GEMINI_API_KEY

log = logging.getLogger("gemini_filter")

# Initialize Gemini if an API key is provided
_gemini_enabled = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        system_instruction = (
            "You are a world-class binary options trader with 20+ years of experience, specializing in 5-minute forex trades. "
            "Your edge is **extreme selectivity** \u2014 you only trade when the setup is overwhelmingly clear and all indicators align. "
            "You never force a trade. Your goal is maximum accuracy, not frequency. Use all available data: price action, technicals, "
            "order flow, news sentiment, and intermarket correlations. Emulate the rigour of top proprietary trading firms. "
            "Output only valid JSON."
        )
        
        # Using gemini-2.5-flash with JSON mode
        _model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
                response_mime_type="application/json"
            )
        )
        _gemini_enabled = True
        log.info("Gemini API Verification Layer ENABLED (Advanced Prop-Trader Prompt).")
    except Exception as e:
        log.warning("Failed to initialize Gemini API: %s", e)
else:
    log.warning("GEMINI_API_KEY not found in config. Layer 13 will be skipped.")

async def verify_signal(
    symbol: str, 
    timestamp: str,
    direction: str,
    green_prob: float,
    confidence: float,
    meta_trust: float,
    uncertainty: float,
    regime: str,
    session: str,
    kelly: float,
    recent_candles: str,
    support1: str,
    resistance1: str,
    ema20: float,
    ema50: float,
    rsi: float,
    atr: float,
    # Phase 7 & 8 Microstructure
    dist_to_res: float,
    dist_to_sup: float,
    bull_div: float,
    bear_div: float,
    vol_accel: float,
    cumulative_delta: float = 0.0,
    m1_absorption: float = 0.0,
    trade_intensity: float = 0.0,
    news_sentiment: str = "N/A",
    dxy_price: float = 0.0,
    gold_price: float = 0.0,
    eurgbp_price: float = 0.0
) -> dict:
    """
    Sends the structural trade context to Gemini for final verification asynchronously.
    """
    if not _gemini_enabled:
        return {"approved": True, "reason": "Gemini disabled (No API Key)", "gemini_confidence": 0, "verdict": "SKIPPED"}
    
    ml_prob_display = (green_prob * 100) if direction == "UP" else ((1 - green_prob) * 100)
    
    prompt = f"""Analyse the following market context and decide whether to trade the next M5 candle. If the evidence strongly supports a clear direction, output a BUY or SELL signal. Otherwise, output NO_TRADE.

**Symbol:** {symbol}
**Current UTC time:** {timestamp}
**Market session:** {session}
**Regime:** {regime}

**Recent M5 candles (OHLCV):**
{recent_candles}

**Technical levels:**
- 20 EMA: {ema20:.5f}
- 50 EMA: {ema50:.5f}
- RSI(14): {rsi:.1f}
- ATR(14): {atr:.5f}
- Daily pivot: N/A
- Support 1: {dist_to_sup:.2f} ATRs away
- Resistance 1: {dist_to_res:.2f} ATRs away
- Bullish Divergence: {"YES" if bull_div > 0 else "NO"}
- Bearish Divergence: {"YES" if bear_div > 0 else "NO"}

**Order flow (if available):**
- Cumulative delta (last 5 min): {cumulative_delta:.2f}
- Bid-ask spread (pips): N/A
- Volume profile (Intensity / Absorption): {trade_intensity:.2f} / {m1_absorption:.2f}

**Recent news headlines (last 30 min):**
{news_sentiment}

**Correlated assets:**
- USD Index: {dxy_price}
- Gold (XAU/USD): {gold_price}
- EUR/GBP: {eurgbp_price}

**Your ML model's prediction (optional):** {direction} with {ml_prob_display:.1f}% probability.

Now, apply your expertise:
- Identify if there is a strong, non-conflicting signal.
- Consider candlestick patterns, support/resistance tests, momentum divergences, order flow imbalances, and news impact.
- If the setup is ambiguous or risks are high, choose NO_TRADE.

Return a JSON object with exactly these fields:
- action: string, one of ["BUY", "SELL", "NO_TRADE"]
- confidence: integer 0-100 (your certainty in the action; if NO_TRADE, set to 0)
- reasoning: string (max 300 chars, explaining your decision)
- key_factors: list of strings (top 3 reasons for your decision)

Ensure the JSON is valid and parsable.
"""
    
    def _call_gemini():
        import time
        for attempt in range(3):
            try:
                response = _model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e):
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
                        continue
                raise e

    try:
        # Prevent blocking the main Telegram asyncio loop
        text = await asyncio.to_thread(_call_gemini)
        
        # Strip markdown formatting by extracting just the JSON object
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]
            
        parsed = json.loads(text)
        
        action = parsed.get("action", "NO_TRADE").upper()
        reason = parsed.get("reasoning", "No reasoning provided.")
        gemini_conf = parsed.get("confidence", 0)
        
        if action == "BUY":
            direction_pred = "UP"
            approved = True
        elif action == "SELL":
            direction_pred = "DOWN"
            approved = True
        else:
            direction_pred = "NO_TRADE"
            approved = False
            
        return {
            "approved": approved,
            "reason": f"Gemini {action} ({gemini_conf}%): {reason}",
            "gemini_confidence": gemini_conf,
            "verdict": direction_pred
        }
            
    except Exception as e:
        log.warning("Gemini API call failed for %s: %s", symbol, e)
        # Fail open by defaulting to original direction if API goes down
        return {"approved": True, "reason": f"Gemini Error (Auto-Passed): {str(e)}", "gemini_confidence": 0, "verdict": direction}

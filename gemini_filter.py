"""
Layer 13: Gemini AI Master Trader Verification Layer.

This module packages the raw technical context of a trade signal
(Asset, Direction, EMA stack, ATR, Key Levels, ADX) and sends it
to Google's Gemini LLM. The LLM acts as a Master Trader and provides
a final APPROVED or REJECTED verdict based on the technical structure,
preventing the bot from taking trades that look mathematically fine
but structurally poor.
"""

import os
import logging
import json
import google.generativeai as genai
from config import GEMINI_API_KEY

log = logging.getLogger("gemini_filter")

# Initialize Gemini if an API key is provided
_gemini_enabled = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using gemini-2.5-flash which is fast and structurally sound.
        _model = genai.GenerativeModel('gemini-2.5-flash')
        _gemini_enabled = True
        log.info("Gemini API Verification Layer is ENABLED.")
    except Exception as e:
        log.warning("Failed to initialize Gemini API: %s", e)
else:
    log.warning("GEMINI_API_KEY not found in config. Layer 13 will be skipped.")

def verify_signal(symbol: str, direction: str, current_price: float, 
                  ema_20: float, ema_50: float, ema_100: float, ema_200: float,
                  atr: float, adx: float, rsi: float, volume_imbalance: float, 
                  regime: str) -> dict:
    """
    Sends the structural trade context to Gemini for final verification.
    
    Returns:
        dict: {
            "approved": bool,
            "reason": str
        }
    """
    if not _gemini_enabled:
        return {"approved": True, "reason": "Gemini disabled (No API Key)"}
    
    prompt = f"""You are a Master Forex/Options Trader evaluating an automated algorithmic signal.
    
    The algorithm wants to place a trade with the following parameters:
    - Asset: {symbol}
    - Suggested Direction: {direction}
    - Current Price: {current_price:.5f}
    - Market Regime: {regime}
    
    Technical Context:
    - EMA 20: {ema_20:.5f}
    - EMA 50: {ema_50:.5f}
    - EMA 100: {ema_100:.5f}
    - EMA 200: {ema_200:.5f}
    - ATR: {atr:.5f} (Volatility/Movement size)
    - ADX: {adx:.2f} (Trend strength, >25 is strong)
    - RSI: {rsi:.2f} (Momentum, >70 Overbought, <30 Oversold)
    - Volume Imbalance: {volume_imbalance:.2f}
    
    Your task: Look at the exact position of the price vs the EMAs, and check the ADX/RSI conditions.
    If the requested {direction} direction fundamentally contradicts the EMA stack structure, or if 
    going {direction} here is foolish given the RSI/ADX, you must reject it.
    If the setup looks structurally sound to a professional trader, approve it.
    
    You must output ONLY a JSON object in this exact format:
    {{"verdict": "APPROVED", "reason": "Short explanation of why"}} or
    {{"verdict": "REJECTED", "reason": "Short explanation of why"}}
    """
    
    try:
        response = _model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean up possible markdown wrappers from the LLM
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        parsed = json.loads(text.strip())
        
        if parsed.get("verdict", "").upper() == "APPROVED":
            return {"approved": True, "reason": parsed.get("reason", "Gemini approved")}
        else:
            return {"approved": False, "reason": parsed.get("reason", "Gemini rejected structure")}
            
    except Exception as e:
        log.warning("Gemini API call failed for %s: %s", symbol, e)
        # Fail open if the API goes down so trading continues, but log it.
        return {"approved": True, "reason": f"Gemini Error (Auto-Passed): {str(e)}"}

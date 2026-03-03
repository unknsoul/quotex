"""
Multi-Timeframe Confluence Filter — Strategy 1.

Only allows signals when M5, M15, and H1 agree on direction.
If lower timeframe says UP but higher says DOWN, the signal
is fighting the macro trend and should be skipped.

Returns:
    confluence_score  : 0–3 (how many TFs agree)
    confluence_pass   : True if score >= threshold
    tf_directions     : dict of directions per TF
"""

import logging
import numpy as np

log = logging.getLogger("confluence_filter")


def _tf_direction(df, lookback=3):
    """
    Determine direction from a dataframe using multiple confirmations:
    - EMA slope (trend direction)
    - Last N candles momentum
    - Current price vs EMA-20
    Returns: 'UP', 'DOWN', or 'NEUTRAL'
    """
    if df is None or len(df) < max(lookback, 20):
        return "NEUTRAL"

    try:
        close = df["close"].values
        # EMA-20 slope check
        ema20 = df["close"].ewm(span=20, min_periods=10).mean().values
        ema_slope = (ema20[-1] - ema20[-3]) / max(abs(ema20[-3]), 1e-10)

        # Last N candles: more green or red?
        recent = df.tail(lookback)
        green_count = (recent["close"] > recent["open"]).sum()
        red_count = (recent["close"] < recent["open"]).sum()

        # Price vs EMA-20
        above_ema = close[-1] > ema20[-1]

        # Scoring: each factor votes
        up_votes = 0
        down_votes = 0

        if ema_slope > 0.0001:
            up_votes += 1
        elif ema_slope < -0.0001:
            down_votes += 1

        if green_count > red_count:
            up_votes += 1
        elif red_count > green_count:
            down_votes += 1

        if above_ema:
            up_votes += 1
        else:
            down_votes += 1

        if up_votes >= 2:
            return "UP"
        elif down_votes >= 2:
            return "DOWN"
        else:
            return "NEUTRAL"

    except Exception as e:
        log.debug("TF direction calc failed: %s", e)
        return "NEUTRAL"


def check_confluence(m5_df, m15_df=None, h1_df=None,
                     predicted_direction="UP", min_score=2):
    """
    Check if multiple timeframes agree with the predicted direction.

    Args:
        m5_df:  M5 dataframe
        m15_df: M15 dataframe (optional)
        h1_df:  H1 dataframe (optional)
        predicted_direction: 'UP' or 'DOWN' from the model
        min_score: minimum agreeing TFs to pass (default 2 of 3)

    Returns:
        dict with:
            confluence_score: 0–3
            confluence_pass: bool
            tf_directions: {M5: ..., M15: ..., H1: ...}
            reason: human-readable explanation
    """
    directions = {}
    score = 0

    # M5 direction
    m5_dir = _tf_direction(m5_df, lookback=3)
    directions["M5"] = m5_dir
    if m5_dir == predicted_direction:
        score += 1

    # M15 direction
    if m15_df is not None and len(m15_df) >= 20:
        m15_dir = _tf_direction(m15_df, lookback=3)
        directions["M15"] = m15_dir
        if m15_dir == predicted_direction:
            score += 1
    else:
        directions["M15"] = "N/A"
        score += 1  # benefit of the doubt if data missing

    # H1 direction
    if h1_df is not None and len(h1_df) >= 20:
        h1_dir = _tf_direction(h1_df, lookback=2)
        directions["H1"] = h1_dir
        if h1_dir == predicted_direction:
            score += 1
    else:
        directions["H1"] = "N/A"
        score += 1  # benefit of the doubt if data missing

    passed = score >= min_score

    if not passed:
        opposing = [f"{tf}={d}" for tf, d in directions.items()
                    if d != predicted_direction and d != "N/A" and d != "NEUTRAL"]
        reason = f"TF conflict: {', '.join(opposing)} vs pred={predicted_direction}"
    else:
        reason = f"Confluence {score}/3 OK"

    return {
        "confluence_score": score,
        "confluence_pass": passed,
        "tf_directions": directions,
        "reason": reason,
    }

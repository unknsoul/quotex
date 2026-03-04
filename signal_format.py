"""
Simple plain-text signal message formatting — guaranteed to display on Telegram.
"""
from datetime import datetime, timezone, timedelta


def format_signal(predictions: dict) -> str:
    """Simple plain-text signal message — guaranteed to display on Telegram."""
    now_utc = datetime.now(timezone.utc)
    ist = now_utc + timedelta(hours=5, minutes=30)

    # Candle close time (next M5 boundary)
    minute = now_utc.minute
    next_close_min = ((minute // 5) + 1) * 5
    close_str = f"{now_utc.hour:02d}:{next_close_min:02d}"

    lines = [
        "\u26a1\u26a1\u26a1 LIVE SIGNAL \u26a1\u26a1\u26a1",
        "",
        "\U0001f451 QUOTEX LORD PRO",
        f"\u23f0 {now_utc.strftime('%H:%M:%S')} UTC | {ist.strftime('%H:%M')} IST",
        f"\U0001f4ca Candle closes: {close_str} UTC",
        "\u2501" * 24,
    ]

    for sym, pred in sorted(predictions.items()):
        direction = pred["suggested_direction"]
        arrow = "\U0001f7e2 \u2b06" if direction == "UP" else "\U0001f534 \u2b07"
        conf = pred["final_confidence_percent"]
        trade = pred["suggested_trade"]
        agree = pred.get("ensemble_agree_count", "?")
        total_m = pred.get("ensemble_total", 7)
        score = pred.get("signal_score", 0)

        lines.append("")
        lines.append(f"{arrow} {sym} -> {direction}")
        lines.append(f"   Confidence: {conf:.0f}%")
        lines.append(f"   Models: {agree}/{total_m} agree")
        lines.append(f"   Action: {trade}")
        lines.append(f"   Score: {score}/100")

    lines.append("")
    lines.append("\u2501" * 24)
    lines.append("\U0001f6e1 12-Layer Filter | 7 Models")
    lines.append(f"\u23f1 Expires: {close_str} UTC")

    return "\n".join(lines)


def format_result(outcomes: list) -> str:
    """Simple plain-text result message."""
    lines = [
        "\U0001f4cb SIGNAL RESULTS",
        "\u2501" * 24,
        "",
    ]
    for o in outcomes:
        emoji = "\u2705" if o["correct"] else "\u274c"
        lines.append(
            f"{emoji} {o['symbol']}: Predicted {o['predicted']} "
            f"-> Actual {o['actual']} ({o['conf']:.0f}% conf)"
        )

    return "\n".join(lines)

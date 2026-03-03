"""
News Calendar Filter — Upgrade 4.

Avoids sending signals during high-impact economic events
where OHLC-based models typically fail (NFP, rate decisions,
CPI, GDP, etc.).

Uses free ForexFactory-style calendar data via API.
Falls back to a hardcoded schedule of major recurring events.
"""

import logging
import json
import os
from datetime import datetime, timezone, timedelta

log = logging.getLogger("news_filter")

CACHE_FILE = os.path.join(os.path.dirname(__file__), "logs", "news_cache.json")
CACHE_TTL_HOURS = 6  # refresh cache every 6 hours

# Hardcoded recurring high-impact events (UTC times)
# These are approximate windows — real events shift monthly
RECURRING_HIGH_IMPACT = {
    # Day of week: [(hour_start, hour_end, description)]
    "Friday": [
        (12, 14, "US NFP / Employment"),
    ],
    "Wednesday": [
        (18, 20, "FOMC / Fed Rate Decision"),
    ],
    "Tuesday": [
        (12, 14, "US CPI / PPI"),
    ],
    "Thursday": [
        (12, 14, "US GDP / Jobless Claims"),
    ],
}

# Symbols affected by USD news
USD_AFFECTED = {"EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD"}
JPY_AFFECTED = {"USDJPY", "GBPJPY"}
GBP_AFFECTED = {"GBPUSD", "GBPJPY"}


def _fetch_calendar():
    """
    Try to fetch economic calendar from a free API.
    Returns list of events or empty list on failure.
    """
    try:
        import urllib.request
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        events = []
        for item in data:
            if item.get("impact", "").lower() in ("high", "holiday"):
                events.append({
                    "title": item.get("title", ""),
                    "country": item.get("country", ""),
                    "date": item.get("date", ""),
                    "impact": item.get("impact", ""),
                    "forecast": item.get("forecast", ""),
                    "previous": item.get("previous", ""),
                })

        # Cache it
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump({
                "fetched": datetime.now(timezone.utc).isoformat(),
                "events": events,
            }, f, indent=2)

        log.info("News calendar: fetched %d high-impact events", len(events))
        return events

    except Exception as e:
        log.debug("Calendar fetch failed: %s", e)
        return []


def _load_cached_events():
    """Load cached events if fresh enough."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        fetched = datetime.fromisoformat(data["fetched"])
        if datetime.now(timezone.utc) - fetched < timedelta(hours=CACHE_TTL_HOURS):
            return data.get("events", [])
    except Exception:
        pass
    return None


def _check_recurring(symbol: str, now: datetime) -> dict:
    """Check against hardcoded recurring schedule."""
    day_name = now.strftime("%A")
    hour = now.hour

    if day_name in RECURRING_HIGH_IMPACT:
        for start_h, end_h, desc in RECURRING_HIGH_IMPACT[day_name]:
            if start_h <= hour <= end_h:
                # Check if this symbol is affected
                if symbol in USD_AFFECTED:
                    return {
                        "blocked": True,
                        "reason": f"High-impact news window: {desc} ({start_h}:00-{end_h}:00 UTC)",
                    }

    return {"blocked": False, "reason": ""}


def check_news_filter(symbol: str, now: datetime = None) -> dict:
    """
    Check if a signal should be blocked due to upcoming news.

    Args:
        symbol: currency pair
        now: current UTC time (default: now)

    Returns:
        dict with:
            blocked: bool
            reason: str
            events: list of nearby events
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # 1. Check recurring schedule
    recurring = _check_recurring(symbol, now)
    if recurring["blocked"]:
        return {
            "blocked": True,
            "reason": recurring["reason"],
            "events": [],
        }

    # 2. Check live calendar
    events = _load_cached_events()
    if events is None:
        events = _fetch_calendar()

    if not events:
        return {"blocked": False, "reason": "No news events", "events": []}

    # Check if any high-impact event is within ±30 min
    nearby = []
    for evt in events:
        try:
            evt_time = datetime.fromisoformat(
                evt["date"].replace("Z", "+00:00")
            )
            delta = abs((evt_time - now).total_seconds())
            if delta < 1800:  # within 30 minutes
                nearby.append(evt)
        except Exception:
            continue

    if nearby:
        titles = [e["title"] for e in nearby[:3]]
        return {
            "blocked": True,
            "reason": f"High-impact news: {', '.join(titles)}",
            "events": nearby,
        }

    return {"blocked": False, "reason": "No nearby news", "events": []}

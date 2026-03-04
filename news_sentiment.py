import requests
import xml.etree.ElementTree as ET
import logging

log = logging.getLogger("news_sentiment")

def fetch_live_news() -> str:
    """
    Fetches the top 3 latest Forex news headlines to inject into the Gemini context.
    Using ForexLive RSS Feed as a highly-liquid primary source.
    """
    url = "https://www.forexlive.com/feed/news"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            headlines = []
            # Find the first 3 items
            for item in root.findall('.//item')[:3]:
                title = item.find('title')
                if title is not None and title.text:
                    headlines.append(f"- {title.text}")
            
            if headlines:
                return "\n".join(headlines)
    except Exception as e:
        log.warning("Failed to fetch live news: %s", e)
        
    return "No recent news available."

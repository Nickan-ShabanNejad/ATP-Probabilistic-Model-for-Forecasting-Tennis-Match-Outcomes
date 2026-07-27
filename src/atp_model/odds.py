
import requests
from .config import ODDS_API_KEY, ODDS_BOOKMAKER

SPORT_KEYS = [
    "tennis_atp_australian_open","tennis_atp_french_open",
    "tennis_atp_wimbledon","tennis_atp_us_open"
]

def fetch_current_odds():
    if not ODDS_API_KEY:
        return []
    all_events=[]
    for sport in SPORT_KEYS:
        url=f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params={"apiKey":ODDS_API_KEY,"regions":"eu","markets":"h2h","oddsFormat":"decimal"}
        r=requests.get(url,params=params,timeout=30)
        if r.status_code==404:
            continue
        r.raise_for_status()
        for event in r.json():
            for bookmaker in event.get("bookmakers",[]):
                if bookmaker.get("key")==ODDS_BOOKMAKER:
                    market=next((m for m in bookmaker.get("markets",[]) if m.get("key")=="h2h"),None)
                    if market:
                        prices={o["name"]:o["price"] for o in market.get("outcomes",[])}
                        all_events.append({
                            "event_id":event["id"],"commence_time":event["commence_time"],
                            "home_team":event["home_team"],"away_team":event["away_team"],
                            "prices":prices,"bookmaker":bookmaker.get("title",ODDS_BOOKMAKER)
                        })
    return all_events

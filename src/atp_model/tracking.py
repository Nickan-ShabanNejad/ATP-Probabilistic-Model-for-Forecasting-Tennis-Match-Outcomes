from pathlib import Path
from datetime import datetime, timezone
import json

from .config import ROOT


TRACKING_FILE = ROOT / "data" / "generated" / "predictions.jsonl"


def save_prediction(result, odds_a, odds_b, stake=0.0):
    TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "player_a": result.get("player_a"),
        "player_b": result.get("player_b"),
        "surface": result.get("surface"),
        "probability_a": result.get("probability_a"),
        "market_probability_a": result.get("market_probability_a"),
        "edge": result.get("edge"),
        "ev": result.get("ev"),
        "fair_odds_a": result.get("fair_odds_a"),
        "quarter_kelly": result.get("quarter_kelly"),
        "odds_a": float(odds_a),
        "odds_b": float(odds_b),
        "stake": float(stake),
    }

    with TRACKING_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return sum(1 for _ in TRACKING_FILE.open("r", encoding="utf-8"))


from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{ROOT / 'atp_model.db'}"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BOOKMAKER = os.getenv("ODDS_BOOKMAKER", "pinnacle")
MIN_EV = float(os.getenv("MIN_EV", "0.02"))

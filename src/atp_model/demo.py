"""Offline demo feed (ATP_DEMO=1).

Lets the app run without API keys (screenshots, UI work, tests). Every price here
is SYNTHETIC: it is generated from a rating curve plus noise and bookmaker margin.
The app shows a DEMO badge whenever this module is in use. Never stake on it.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


class DemoMatchstatClient:
    last_upcoming_source = "demo"

    def __init__(self, state: pd.DataFrame, n_matches: int = 12):
        hard = state[state.surface == "Hard"].copy()
        hard["rank"] = pd.to_numeric(hard["rank"], errors="coerce")
        hard = hard[hard["rank"] < 120].sort_values("rank")
        self._players = hard["player"].tolist()[: n_matches * 2]
        self._elo = dict(zip(hard["player"], pd.to_numeric(hard.get("dyn_elo", hard["overall_elo"]), errors="coerce")))
        self.n_matches = n_matches

    def upcoming_events(self, tour: str = "atp", max_events: int = 80) -> list[dict]:
        now = time.time()
        base = now - (now % 3600) + 3 * 3600
        events = []
        rng = np.random.default_rng(_seed(time.strftime("%Y-%m-%d")))
        players = list(self._players)
        rng.shuffle(players)
        for i in range(min(self.n_matches, len(players) // 2)):
            a, b = players[2 * i], players[2 * i + 1]
            league = "China Open - Beijing" if i % 2 == 0 else "Japan Open - Tokyo"
            events.append({
                "id": f"demo-{i}",
                "participant1": a,
                "participant2": b,
                "league": league,
                "status": "Not started",
                "startTimestamp": base + (i // 2) * 5400,
                "tourType": "ATP",
                "round": "Second",
            })
        return events[:max_events]

    def pre_match_odds(self, *_args, **_kwargs):
        raise RuntimeError("demo feed has no Matchstat odds")

    compared_odds = pre_match_odds
    recent_odds = pre_match_odds


class DemoPinnacleClient:
    enabled = True
    last_error = None
    last_total35_error = None

    def __init__(self, elo: dict[str, float]):
        self._elo = elo

    def invalidate_prices(self):
        return None

    def find_moneyline(self, p1: str, p2: str, start_timestamp=None, force=False):
        ea, eb = float(self._elo.get(p1, 1600)), float(self._elo.get(p2, 1600))
        p = 1 / (1 + 10 ** ((eb - ea) / 400))
        rng = np.random.default_rng(_seed(p1 + p2))
        p = float(np.clip(p + rng.normal(0, 0.04), 0.03, 0.97))
        margin = 1.025
        return {
            "moneyline": (round(1 / (p * margin), 3), round(1 / ((1 - p) * margin), 3)),
            "source": "DEMO (synthetic)",
        }

    def find_total_sets_35(self, *_args, **_kwargs):
        return None

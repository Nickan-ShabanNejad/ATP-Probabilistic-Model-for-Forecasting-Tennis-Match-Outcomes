from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any

import requests

from .matchstat import normalize_name




def _same_player(left: str, right: str) -> bool:
    a, b = normalize_name(left), normalize_name(right)
    if not a or not b:
        return False
    if a == b:
        return True
    ap, bp = a.split(), b.split()
    # Providers sometimes abbreviate a first name ("A Zverev") while the model
    # state uses the full name. Require surname equality plus first initial equality.
    return len(ap) >= 2 and len(bp) >= 2 and ap[-1] == bp[-1] and ap[0][0] == bp[0][0]

class PinnOddsClient:
    """Small client for the independent Pinnacle-only prematch feed at pinnodds.com.

    Matchstat remains the event/statistics source.  This client is used only as a
    sharp-price source when Matchstat does not expose Pinnacle for an event.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://pinnodds.com",
        timeout_seconds: float = 20.0,
        cache_seconds: float = 20.0,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "https://pinnodds.com").rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.cache_seconds = float(cache_seconds)
        self._session = requests.Session()
        self._cached_at = 0.0
        self._cached_events: list[dict] = []
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        if not self.enabled:
            raise RuntimeError("PINNODDS_API_KEY is not configured")
        response = self._session.get(
            f"{self.base_url}{path}",
            params=params or {},
            headers={"x-portal-apikey": self.api_key, "accept": "application/json"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected Pinnodds response shape")
        return payload

    def prematch_events(self, *, force: bool = False) -> list[dict]:
        now = time.time()
        if self._cached_events and not force and (now - self._cached_at) < self.cache_seconds:
            return self._cached_events
        try:
            payload = self._get("/kit/v1/prematch/fixtures", params={"sport_id": 2})
            events = payload.get("events") or []
            events = [x for x in events if isinstance(x, dict)]
            self._cached_events = events
            self._cached_at = now
            self.last_error = None
            return events
        except Exception as exc:
            self.last_error = str(exc)
            # A short provider hiccup should not blank a live board that already has
            # a recent Pinnacle snapshot in memory.
            if self._cached_events and now - self._cached_at < 300:
                return self._cached_events
            raise

    @staticmethod
    def _event_start_ts(row: dict) -> float | None:
        raw = row.get("starts") or row.get("start_ts")
        if raw is None:
            return None
        try:
            if isinstance(raw, (int, float)):
                value = float(raw)
                return value / 1000.0 if value > 10_000_000_000 else value
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        except Exception:
            return None

    @staticmethod
    def _moneyline(row: dict) -> tuple[float, float] | None:
        periods = row.get("periods") or {}
        game = periods.get("num_0") if isinstance(periods, dict) else None
        if not isinstance(game, dict):
            return None
        ml = game.get("money_line") or {}
        if not isinstance(ml, dict):
            return None
        try:
            home = float(ml.get("home"))
            away = float(ml.get("away"))
        except Exception:
            return None
        if home <= 1.0 or away <= 1.0:
            return None
        return home, away

    def find_moneyline(
        self,
        player_a: str,
        player_b: str,
        start_timestamp: float | None = None,
        *,
        force: bool = False,
    ) -> dict | None:
        """Find a Pinnacle prematch quote and orient it to player_a/player_b."""
        na, nb = normalize_name(player_a), normalize_name(player_b)
        if not na or not nb:
            return None
        best: tuple[float, dict, bool] | None = None
        for row in self.prematch_events(force=force):
            home = str(row.get("home") or "").strip()
            away = str(row.get("away") or "").strip()
            direct = _same_player(home, player_a) and _same_player(away, player_b)
            reverse = _same_player(home, player_b) and _same_player(away, player_a)
            if not (direct or reverse):
                continue
            pair = self._moneyline(row)
            if pair is None:
                continue
            event_ts = self._event_start_ts(row)
            if start_timestamp and event_ts:
                delta = abs(float(event_ts) - float(start_timestamp))
                # Same players can meet more than once over a season; date/time is
                # used as a strong disambiguator but provider timezone drift gets room.
                if delta > 36 * 3600:
                    continue
            else:
                delta = 0.0
            candidate = (delta, row, reverse)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            return None
        _, row, reverse = best
        home_odds, away_odds = self._moneyline(row)  # already validated above
        if reverse:
            odds_a, odds_b = away_odds, home_odds
        else:
            odds_a, odds_b = home_odds, away_odds
        return {
            "moneyline": (float(odds_a), float(odds_b)),
            "source": "pinnodds-prematch",
            "provider_event_id": str(row.get("event_id") or row.get("id") or ""),
            "provider_league": str(row.get("league_name") or ""),
            "provider_start": row.get("starts") or row.get("start_ts"),
        }

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
        self._cached_special_at = 0.0
        self._cached_special_events: list[dict] = []
        self.last_error: str | None = None
        self.last_total35_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        if not self.enabled:
            raise RuntimeError("PINNODDS_API_KEY is not configured")
        response = self._session.get(
            f"{self.base_url}{path}",
            params=params or {},
            headers={
                "x-portal-apikey": self.api_key,
                "x-api-key": self.api_key,
                "accept": "application/json",
            },
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


    def prematch_events_with_specials(self, *, force: bool = False) -> list[dict]:
        """Prematch tennis fixtures with specials nested under their parent match.

        This is intentionally cached longer than the moneyline board because the
        special-market catalogue is much larger.  Individual prices are still
        refreshed from the single-event endpoints once a Total Sets market is found.
        """
        now = time.time()
        if (
            self._cached_special_events
            and not force
            and (now - self._cached_special_at) < max(120.0, self.cache_seconds)
        ):
            return self._cached_special_events
        payload = self._get(
            "/kit/v1/prematch/fixtures",
            params={"sport_id": 2, "include_specials": "nested"},
        )
        events = payload.get("events") or []
        events = [x for x in events if isinstance(x, dict)]
        self._cached_special_events = events
        self._cached_special_at = now
        return events

    @staticmethod
    def _total35_from_periods(payload: Any) -> tuple[float, float] | None:
        """Read a full-match 3.5 total when Pinnodds exposes it in periods.num_0.

        On a BO5 tennis match a full-match 3.5 line cannot be a total-games line,
        so a 3.5 full-match total is interpreted as Total Sets.
        """
        rows = payload.get("events") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            rows = [payload] if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            periods = row.get("periods") or {}
            game = periods.get("num_0") if isinstance(periods, dict) else None
            if not isinstance(game, dict):
                continue
            totals = game.get("totals") or {}
            if not isinstance(totals, dict):
                continue
            candidates = []
            if "3.5" in totals:
                candidates.append(totals.get("3.5"))
            for value in totals.values():
                if isinstance(value, dict):
                    try:
                        if abs(float(value.get("points")) - 3.5) < 1e-9:
                            candidates.append(value)
                    except Exception:
                        pass
            for value in candidates:
                if not isinstance(value, dict):
                    continue
                try:
                    over = float(value.get("over"))
                    under = float(value.get("under"))
                except Exception:
                    continue
                if over > 1.0 and under > 1.0:
                    return over, under
        return None

    @staticmethod
    def _total35_from_special_tree(payload: Any) -> tuple[float, float] | None:
        """Find a Total Sets 3.5 special in nested/flat Pinnodds special payloads."""
        found: dict[str, float] = {}

        def walk(node: Any, inherited: str = "") -> None:
            if len(found) == 2:
                return
            if isinstance(node, list):
                for item in node:
                    walk(item, inherited)
                return
            if not isinstance(node, dict):
                return

            here = " ".join(
                str(node.get(k) or "")
                for k in (
                    "special", "special_category", "special_units", "type", "key",
                    "side", "name", "description", "market", "market_name",
                )
            ).casefold()
            context = (inherited + " " + here).strip()
            is_set_context = "set" in context

            prices = node.get("prices")
            if is_set_context and isinstance(prices, list):
                for price_row in prices:
                    if not isinstance(price_row, dict):
                        continue
                    label = str(price_row.get("name") or price_row.get("side") or "").casefold()
                    try:
                        points = float(price_row.get("points")) if price_row.get("points") not in (None, "") else None
                    except Exception:
                        points = None
                    has_35 = (points is not None and abs(points - 3.5) < 1e-9) or "3.5" in context or "3.5" in label
                    if not has_35:
                        continue
                    try:
                        price = float(price_row.get("price"))
                    except Exception:
                        continue
                    if price <= 1.0:
                        continue
                    if "over" in label or label in {"o", "over 3.5"}:
                        found["over"] = price
                    elif "under" in label or label in {"u", "under 3.5"}:
                        found["under"] = price

            # Some providers put the line directly on a dict rather than prices[].
            if is_set_context:
                try:
                    points = float(node.get("points")) if node.get("points") not in (None, "") else None
                except Exception:
                    points = None
                if (points is not None and abs(points - 3.5) < 1e-9) or "3.5" in context:
                    for side in ("over", "under"):
                        try:
                            price = float(node.get(side))
                        except Exception:
                            continue
                        if price > 1.0:
                            found[side] = price

            for key, value in node.items():
                if key == "prices":
                    continue
                if isinstance(value, (dict, list)):
                    walk(value, context)

        walk(payload)
        if "over" in found and "under" in found:
            return float(found["over"]), float(found["under"])
        return None

    def _find_fixture(
        self,
        player_a: str,
        player_b: str,
        start_timestamp: float | None,
        events: list[dict],
    ) -> tuple[dict, bool] | None:
        best: tuple[float, dict, bool] | None = None
        for row in events:
            home = str(row.get("home") or "").strip()
            away = str(row.get("away") or "").strip()
            direct = _same_player(home, player_a) and _same_player(away, player_b)
            reverse = _same_player(home, player_b) and _same_player(away, player_a)
            if not (direct or reverse):
                continue
            event_ts = self._event_start_ts(row)
            if start_timestamp and event_ts:
                delta = abs(float(event_ts) - float(start_timestamp))
                if delta > 36 * 3600:
                    continue
            else:
                delta = 0.0
            candidate = (delta, row, reverse)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            return None
        return best[1], best[2]

    def find_total_sets_35(
        self,
        player_a: str,
        player_b: str,
        start_timestamp: float | None = None,
        *,
        force: bool = False,
    ) -> dict | None:
        """Return Pinnacle Over/Under 3.5 SETS for a BO5 tennis match when offered."""
        try:
            match = self._find_fixture(
                player_a, player_b, start_timestamp, self.prematch_events(force=force)
            )
            if match is None:
                self.last_total35_error = "Pinnodds fixture not matched"
                return None
            row, _reverse = match
            event_id = row.get("event_id") or row.get("id")
            if event_id is None:
                self.last_total35_error = "Pinnodds fixture has no event_id"
                return None

            # First try the lightweight/full standard market payloads.
            for path, params in (
                ("/kit/v1/prematch/lines", {"event_id": event_id, "market_type": "totals"}),
                ("/kit/v1/prematch/markets", {"event_id": event_id}),
            ):
                payload = self._get(path, params=params)
                pair = self._total35_from_periods(payload) or self._total35_from_special_tree(payload)
                if pair:
                    self.last_total35_error = None
                    return {
                        "sets35": pair,
                        "source": "pinnodds-total-sets",
                        "provider_event_id": str(event_id),
                    }

            # Total Sets can also arrive as a Pinnacle special.  Request specials
            # nested under each parent match, then inspect only this matched parent.
            specials_match = self._find_fixture(
                player_a,
                player_b,
                start_timestamp,
                self.prematch_events_with_specials(force=force),
            )
            if specials_match is not None:
                special_parent, _ = specials_match
                pair = self._total35_from_special_tree(special_parent.get("specials") or special_parent)
                if pair:
                    self.last_total35_error = None
                    return {
                        "sets35": pair,
                        "source": "pinnodds-total-sets-special",
                        "provider_event_id": str(event_id),
                    }

            self.last_total35_error = "Pinnacle Total Sets 3.5 is not currently offered in the Pinnodds payload"
            return None
        except Exception as exc:
            self.last_total35_error = str(exc)
            return None

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
        matched = self._find_fixture(player_a, player_b, start_timestamp, self.prematch_events(force=force))
        if matched is None:
            return None
        row, reverse = matched
        if self._moneyline(row) is None:
            return None
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

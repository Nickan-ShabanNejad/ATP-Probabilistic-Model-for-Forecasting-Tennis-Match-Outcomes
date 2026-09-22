from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
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
    return len(ap) >= 2 and len(bp) >= 2 and ap[-1] == bp[-1] and ap[0][0] == bp[0][0]


class PinnOddsClient:
    """Quota-aware Pinnacle prematch client.

    The board refresh can run every 30 seconds, but the PinnOdds REST API should
    not be polled on every Streamlit rerun. A single tennis fixture snapshot is
    therefore reused for a much longer interval, and a 429 response activates a
    provider-wide cooldown. Recent successful data is kept as a stale fallback so
    one rate-limit response does not blank the board.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://pinnodds.com",
        timeout_seconds: float = 20.0,
        cache_seconds: float = 1200.0,
        stale_seconds: float = 21600.0,
        single_event_cache_seconds: float = 300.0,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "https://pinnodds.com").rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.cache_seconds = max(30.0, float(cache_seconds))
        self.stale_seconds = max(self.cache_seconds, float(stale_seconds))
        self.single_event_cache_seconds = max(60.0, float(single_event_cache_seconds))

        self._session = requests.Session()

        self._cached_at = 0.0
        self._cached_events: list[dict] = []

        self._cached_special_at = 0.0
        self._cached_special_events: list[dict] = []

        self._payload_cache: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            tuple[float, dict],
        ] = {}

        self._cooldown_until = 0.0

        self.last_error: str | None = None
        self.last_total35_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def cooldown_seconds(self) -> int:
        return max(0, int(math.ceil(self._cooldown_until - time.time())))

    def invalidate_prices(self) -> None:
        """Force the next board pass to attempt one fresh PinnOdds snapshot.

        Existing successful data is deliberately retained so it can still be used
        if the manual refresh is rate-limited or the provider is temporarily down.
        """

        self._cached_at = 0.0
        self._cached_special_at = 0.0
        self._payload_cache.clear()

    @staticmethod
    def _cache_key(
        path: str,
        params: dict[str, Any] | None,
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        normalized = tuple(
            sorted(
                (str(key), str(value))
                for key, value in (params or {}).items()
            )
        )
        return path, normalized

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float:
        raw = str(response.headers.get("Retry-After") or "").strip()
        if not raw:
            return 60.0

        try:
            return max(1.0, float(raw))
        except Exception:
            pass

        try:
            retry_at = parsedate_to_datetime(raw)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(
                1.0,
                retry_at.astimezone(timezone.utc).timestamp() - time.time(),
            )
        except Exception:
            return 60.0

    def _request_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict:
        if not self.enabled:
            raise RuntimeError("PINNODDS_API_KEY is not configured")

        now = time.time()
        if now < self._cooldown_until:
            raise RuntimeError(
                f"PinnOdds rate-limit cooldown active for {self.cooldown_seconds}s"
            )

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

        if response.status_code == 429:
            retry_after = self._retry_after_seconds(response)
            self._cooldown_until = max(
                self._cooldown_until,
                time.time() + retry_after,
            )
            raise RuntimeError(
                "PinnOdds rate limited (429). "
                f"Retry-After={int(math.ceil(retry_after))}s"
            )

        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected Pinnodds response shape")

        return payload

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        cache_seconds: float | None = None,
        stale_seconds: float | None = None,
    ) -> dict:
        """GET a PinnOdds payload with optional endpoint-level caching."""

        ttl = 0.0 if cache_seconds is None else max(0.0, float(cache_seconds))
        stale_ttl = (
            self.stale_seconds
            if stale_seconds is None
            else max(ttl, float(stale_seconds))
        )

        key = self._cache_key(path, params)
        cached = self._payload_cache.get(key)
        now = time.time()

        if cached is not None:
            cached_at, cached_payload = cached
            age = now - cached_at
            if ttl > 0 and age < ttl:
                return cached_payload

        try:
            payload = self._request_json(path, params=params)
        except Exception:
            if cached is not None:
                cached_at, cached_payload = cached
                if now - cached_at < stale_ttl:
                    return cached_payload
            raise

        self._payload_cache[key] = (now, payload)
        return payload

    def prematch_events(self, *, force: bool = False) -> list[dict]:
        """Return one cached tennis prematch snapshot.

        This is the only endpoint needed for normal moneyline pricing. PinnOdds'
        prematch fixture response already contains the full periods/money_line
        structure, so the live board does not make one REST request per match.
        """

        now = time.time()

        if (
            self._cached_events
            and not force
            and (now - self._cached_at) < self.cache_seconds
        ):
            return self._cached_events

        try:
            payload = self._request_json(
                "/kit/v1/prematch/fixtures",
                params={"sport_id": 2},
            )
            events = payload.get("events") or []
            events = [row for row in events if isinstance(row, dict)]

            self._cached_events = events
            self._cached_at = now
            self.last_error = None
            return events

        except Exception as exc:
            self.last_error = str(exc)

            if self._cached_events and (now - self._cached_at) < self.stale_seconds:
                return self._cached_events

            # Do not raise here. build_slate calls find_moneyline once per event;
            # returning an empty list prevents a single 429 from being retried for
            # every match on the board during the same Streamlit render.
            return []

    def prematch_events_with_specials(self, *, force: bool = False) -> list[dict]:
        now = time.time()
        special_cache_seconds = max(1800.0, self.cache_seconds)

        if (
            self._cached_special_events
            and not force
            and (now - self._cached_special_at) < special_cache_seconds
        ):
            return self._cached_special_events

        try:
            payload = self._request_json(
                "/kit/v1/prematch/fixtures",
                params={"sport_id": 2, "include_specials": "nested"},
            )
            events = payload.get("events") or []
            events = [row for row in events if isinstance(row, dict)]

            self._cached_special_events = events
            self._cached_special_at = now
            return events

        except Exception as exc:
            self.last_total35_error = str(exc)

            if (
                self._cached_special_events
                and (now - self._cached_special_at) < self.stale_seconds
            ):
                return self._cached_special_events

            return []

    @staticmethod
    def _total35_from_periods(payload: Any) -> tuple[float, float] | None:
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
                str(node.get(key) or "")
                for key in (
                    "special",
                    "special_category",
                    "special_units",
                    "type",
                    "key",
                    "side",
                    "name",
                    "description",
                    "market",
                    "market_name",
                )
            ).casefold()

            context = (inherited + " " + here).strip()
            is_set_context = "set" in context

            prices = node.get("prices")
            if is_set_context and isinstance(prices, list):
                for price_row in prices:
                    if not isinstance(price_row, dict):
                        continue

                    label = str(
                        price_row.get("name") or price_row.get("side") or ""
                    ).casefold()

                    try:
                        points = (
                            float(price_row.get("points"))
                            if price_row.get("points") not in (None, "")
                            else None
                        )
                    except Exception:
                        points = None

                    has_35 = (
                        (points is not None and abs(points - 3.5) < 1e-9)
                        or "3.5" in context
                        or "3.5" in label
                    )
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

            if is_set_context:
                try:
                    points = (
                        float(node.get("points"))
                        if node.get("points") not in (None, "")
                        else None
                    )
                except Exception:
                    points = None

                if (
                    (points is not None and abs(points - 3.5) < 1e-9)
                    or "3.5" in context
                ):
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
        try:
            match = self._find_fixture(
                player_a,
                player_b,
                start_timestamp,
                self.prematch_events(force=force),
            )

            if match is None:
                self.last_total35_error = self.last_error or "Pinnodds fixture not matched"
                return None

            row, _reverse = match
            event_id = row.get("event_id") or row.get("id")

            if event_id is None:
                self.last_total35_error = "Pinnodds fixture has no event_id"
                return None

            # First inspect the already-cached bulk fixture row. This costs no API call.
            pair = self._total35_from_periods(row) or self._total35_from_special_tree(row)
            if pair:
                self.last_total35_error = None
                return {
                    "sets35": pair,
                    "source": "pinnodds-prematch",
                    "provider_event_id": str(event_id),
                }

            # Grand Slam totals are queried only when necessary, and the response is
            # cached per event so Streamlit reruns do not repeatedly spend quota.
            for path, params in (
                (
                    "/kit/v1/prematch/lines",
                    {"event_id": event_id, "market_type": "totals"},
                ),
                (
                    "/kit/v1/prematch/markets",
                    {"event_id": event_id},
                ),
            ):
                try:
                    payload = self._get(
                        path,
                        params=params,
                        cache_seconds=self.single_event_cache_seconds,
                        stale_seconds=self.stale_seconds,
                    )
                except Exception as exc:
                    self.last_total35_error = str(exc)
                    continue

                pair = self._total35_from_periods(payload) or self._total35_from_special_tree(payload)
                if pair:
                    self.last_total35_error = None
                    return {
                        "sets35": pair,
                        "source": "pinnodds-total-sets",
                        "provider_event_id": str(event_id),
                    }

            specials_match = self._find_fixture(
                player_a,
                player_b,
                start_timestamp,
                self.prematch_events_with_specials(force=force),
            )

            if specials_match is not None:
                special_parent, _ = specials_match
                pair = self._total35_from_special_tree(
                    special_parent.get("specials") or special_parent
                )
                if pair:
                    self.last_total35_error = None
                    return {
                        "sets35": pair,
                        "source": "pinnodds-total-sets-special",
                        "provider_event_id": str(event_id),
                    }

            if not self.last_total35_error:
                self.last_total35_error = (
                    "Pinnacle Total Sets 3.5 is not currently offered in the Pinnodds payload"
                )
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

            return (
                datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                .astimezone(timezone.utc)
                .timestamp()
            )
        except Exception:
            return None

    @staticmethod
    def _moneyline(row: dict) -> tuple[float, float] | None:
        periods = row.get("periods") or {}
        game = periods.get("num_0") if isinstance(periods, dict) else None
        if not isinstance(game, dict):
            return None

        ml = game.get("money_line") or game.get("moneyline") or {}
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
        """Find the full-match Pinnacle moneyline from the cached bulk snapshot.

        Importantly, this does NOT make a separate /lines request for every match.
        PinnOdds documents /prematch/fixtures as already carrying each fixture's
        periods/money_line data, so one bulk request can price the whole slate.
        """

        na = normalize_name(player_a)
        nb = normalize_name(player_b)
        if not na or not nb:
            return None

        events = self.prematch_events(force=force)
        if not events:
            return None

        matched = self._find_fixture(
            player_a,
            player_b,
            start_timestamp,
            events,
        )

        if matched is None:
            self.last_error = f"Pinnodds fixture not matched: {player_a} vs {player_b}"
            return None

        row, reverse = matched
        pair = self._moneyline(row)

        if pair is None:
            event_id = row.get("event_id") or row.get("id") or "?"
            self.last_error = (
                f"Matched Pinnodds event {event_id} has no usable full-match moneyline "
                "in the cached prematch fixture snapshot"
            )
            return None

        home_odds, away_odds = pair

        if reverse:
            odds_a, odds_b = away_odds, home_odds
        else:
            odds_a, odds_b = home_odds, away_odds

        self.last_error = None

        return {
            "moneyline": (float(odds_a), float(odds_b)),
            "source": "pinnodds-prematch",
            "provider_event_id": str(row.get("event_id") or row.get("id") or ""),
            "provider_league": str(row.get("league_name") or ""),
            "provider_start": row.get("starts") or row.get("start_ts"),
        }

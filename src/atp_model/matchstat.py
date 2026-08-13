from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
import time
import unicodedata
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

DEFAULT_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
DEFAULT_BASE_URL = f"https://{DEFAULT_HOST}"


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return " ".join(text.split())


def _payload_rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "singles", "results", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def _has_next(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("hasNextPage"), bool):
        return bool(payload["hasNextPage"])
    pagination = payload.get("pagination")
    if isinstance(pagination, dict):
        return bool(pagination.get("hasNext"))
    return False


@dataclass
class MatchstatClient:
    api_key: str
    host: str = DEFAULT_HOST
    base_url: str = DEFAULT_BASE_URL
    min_interval_seconds: float = 0.66
    timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        self.api_key = str(self.api_key or "").strip()
        if not self.api_key:
            raise ValueError("MATCHSTAT_API_KEY is required")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": self.host,
                "User-Agent": "ATP-Pro-Value-Model/4.0",
            }
        )
        self._last_request = 0.0

    @classmethod
    def from_env(cls) -> "MatchstatClient":
        return cls(
            api_key=os.getenv("MATCHSTAT_API_KEY", ""),
            host=os.getenv("MATCHSTAT_API_HOST", DEFAULT_HOST),
            base_url=os.getenv("MATCHSTAT_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            min_interval_seconds=float(os.getenv("MATCHSTAT_MIN_INTERVAL", "0.66")),
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(5):
            self._throttle()
            try:
                response = self.session.get(url, params=params or {}, timeout=self.timeout_seconds)
                self._last_request = time.monotonic()
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 2 + attempt * 2))
                    time.sleep(min(max(retry_after, 1.0), 30.0))
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # network/API failures should be retried, then surfaced
                last_error = exc
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Matchstat request failed: {url}: {last_error}")

    def get_first_working(self, paths: Iterable[str], params: dict[str, Any] | None = None) -> Any:
        errors = []
        for path in paths:
            try:
                return self.get(path, params=params)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        raise RuntimeError("No Matchstat endpoint variant succeeded. " + " | ".join(errors))

    def rankings(self, tour: str = "atp", max_players: int = 350) -> list[dict]:
        rows: list[dict] = []
        page = 1
        while len(rows) < max_players:
            payload = self.get(
                f"/tennis/v2/{tour}/ranking/singles",
                params={"pageSize": min(100, max_players - len(rows)), "pageNo": page},
            )
            batch = _payload_rows(payload)
            if not batch:
                break
            rows.extend(batch)
            if not _has_next(payload) or len(rows) >= max_players:
                break
            page += 1
        return rows[:max_players]

    def upcoming_events(self, tour: str = "atp", max_events: int = 80) -> list[dict]:
        """Return upcoming events from Matchstat's Live/Extend API, paginated."""
        rows: list[dict] = []
        page = 1
        while len(rows) < max_events:
            payload = self.get(
                f"/tennis/v2/extend/api/events/upcoming/{tour}",
                params={"page": page, "limit": min(50, max_events - len(rows))},
            )
            batch = _payload_rows(payload)
            if not batch:
                break
            rows.extend(batch)
            pagination = payload.get("pagination", {}) if isinstance(payload, dict) else {}
            if not bool(pagination.get("hasNext")):
                break
            page += 1
            if page > 10:
                break
        return rows[:max_events]

    def event_information(self, player1: str, player2: str, date_only: str) -> dict:
        return self.get(
            f"/tennis/v2/extend/api/event/get/{player1}/{player2}/{date_only}"
        )

    def compared_odds(self, event_id: int | str, market_id: int = 1) -> dict:
        return self.get(
            f"/tennis/v2/extend/api/odds/compare/{event_id}",
            params={"market_id": int(market_id)},
        )

    def recent_odds(self, event_id: int | str) -> dict:
        return self.get(f"/tennis/v2/extend/api/event/recent-odds/get/{event_id}")

    def last_ten_odds_movements(self, event_id: int | str) -> dict:
        return self.get(f"/tennis/v2/extend/api/odds/summary/movements/last-10/{event_id}")

    def player_past_matches(
        self,
        player_id: int | str,
        year: int,
        tour: str = "atp",
        page_size: int = 100,
    ) -> list[dict]:
        rows: list[dict] = []
        page = 1
        params_base = {
            "include": "round,tournament,tournament.court,tournament.rank,stat",
            "filter": f"GameYear:{int(year)}",
            "pageSize": int(page_size),
        }
        while True:
            params = {**params_base, "pageNo": page}
            payload = self.get_first_working(
                [
                    f"/tennis/v2/{tour}/player/past-matches/{player_id}",
                    f"/tennis/v2/ms-api/{tour}/player/past-matches/{player_id}",
                ],
                params=params,
            )
            batch = _payload_rows(payload)
            if not batch:
                break
            rows.extend(batch)
            if not _has_next(payload):
                break
            page += 1
            if page > 20:
                break
        return rows


def rankings_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for item in records:
        player = item.get("player") if isinstance(item.get("player"), dict) else item
        if not isinstance(player, dict):
            continue
        pid = player.get("id", item.get("playerId"))
        name = player.get("name", item.get("name"))
        position = item.get("position", player.get("currentRank", item.get("currentRank")))
        points = item.get(
            "rankingPoints",
            item.get("point", player.get("points", item.get("points"))),
        )
        date = item.get("date") or datetime.now(timezone.utc).date().isoformat()
        if pid is None or not name or position is None:
            continue
        rows.append(
            {
                "matchstat_player_id": str(pid),
                "player": str(name),
                "normalized_name": normalize_name(name),
                "ranking": pd.to_numeric(position, errors="coerce"),
                "ranking_points": pd.to_numeric(points, errors="coerce"),
                "ranking_date": pd.to_datetime(date, errors="coerce"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.dropna(subset=["ranking", "player"]).copy()
    frame["ranking"] = frame["ranking"].astype(int)
    frame["ranking_date"] = frame["ranking_date"].dt.strftime("%Y%m%d")
    return frame.sort_values("ranking").drop_duplicates("normalized_name", keep="first")


def _surface_from_tournament(tournament: dict) -> tuple[str | None, float]:
    court = tournament.get("court") if isinstance(tournament, dict) else None
    name = ""
    if isinstance(court, dict):
        name = str(court.get("name") or "")
    if not name and isinstance(tournament, dict):
        name = str(tournament.get("surface") or "")
    compact = name.casefold().replace(" ", "")
    if compact in {"i.hard", "ihard", "indoorhard", "hardindoor"}:
        return "Hard", 1.0
    if "hard" in compact:
        return "Hard", 0.0
    if "clay" in compact:
        return "Clay", 0.0
    if "grass" in compact:
        return "Grass", 0.0
    if "carpet" in compact:
        return "Hard", 1.0
    return None, 0.0


def _level_from_tournament(tournament: dict) -> str:
    tier = str(tournament.get("tier") or "").casefold()
    rank_id = pd.to_numeric(tournament.get("rankId"), errors="coerce")
    if "grand slam" in tier or rank_id == 4:
        return "G"
    if "masters" in tier or rank_id == 3:
        return "M"
    if "500" in tier:
        return "500"
    if "250" in tier:
        return "250"
    if "final" in tier:
        return "F"
    if "chall" in tier or (pd.notna(rank_id) and rank_id <= 1):
        return "C"
    return "A"


def _canonical_player_id(name: str, ms_id: Any, canonical_ids: dict[str, str]) -> str:
    key = normalize_name(name)
    if key in canonical_ids:
        return str(canonical_ids[key])
    return f"ms:{ms_id}" if ms_id is not None else f"name:{key}"


def _stat_number(stats: dict, key: str) -> float | None:
    value = stats.get(key) if isinstance(stats, dict) else None
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except Exception:
        return None


def _age_on_date(birthday: Any, match_date: pd.Timestamp) -> float:
    born = pd.to_datetime(birthday, errors="coerce", utc=True)
    if pd.isna(born) or pd.isna(match_date):
        return np.nan
    return float((match_date - born).days / 365.2425)


def past_matches_to_master(records: list[dict], canonical_ids: dict[str, str]) -> pd.DataFrame:
    """Convert Matchstat historical matches to the project's winner/loser schema.

    Matchstat documents that historical Game records use player1=winner and
    player2=loser. The conversion intentionally preserves missing advanced
    statistics as NaN rather than turning them into zeros.
    """
    rows = []
    for item in records:
        p1 = item.get("player1") or {}
        p2 = item.get("player2") or {}
        tournament = item.get("tournament") or {}
        if not isinstance(p1, dict) or not isinstance(p2, dict) or not isinstance(tournament, dict):
            continue
        winner_name = str(p1.get("name") or "").strip()
        loser_name = str(p2.get("name") or "").strip()
        if not winner_name or not loser_name:
            continue
        date = pd.to_datetime(item.get("date"), errors="coerce", utc=True)
        if pd.isna(date):
            continue
        surface, indoor = _surface_from_tournament(tournament)
        if surface is None:
            continue
        wstats = p1.get("stats") if isinstance(p1.get("stats"), dict) else {}
        lstats = p2.get("stats") if isinstance(p2.get("stats"), dict) else {}
        w_svpt = _stat_number(wstats, "firstServeOf")
        l_svpt = _stat_number(lstats, "firstServeOf")
        w_bp_won = _stat_number(wstats, "breakPointsConverted")
        w_bp_chances = _stat_number(wstats, "breakPointsConvertedOf")
        l_bp_won = _stat_number(lstats, "breakPointsConverted")
        l_bp_chances = _stat_number(lstats, "breakPointsConvertedOf")
        level = _level_from_tournament(tournament)
        best_of = item.get("best_of")
        if pd.isna(pd.to_numeric(best_of, errors="coerce")):
            best_of = 5 if level == "G" else 3
        round_obj = item.get("round") if isinstance(item.get("round"), dict) else {}
        round_name = round_obj.get("name") or item.get("roundName") or item.get("roundId")
        row = {
            "tourney_id": str(item.get("tournamentId") or tournament.get("id") or ""),
            "tourney_name": tournament.get("name"),
            "surface": surface,
            "draw_size": np.nan,
            "tourney_level": level,
            "indoor": indoor,
            "tourney_date": int(date.strftime("%Y%m%d")),
            "match_num": pd.to_numeric(item.get("id", item.get("draw", 0)), errors="coerce"),
            "winner_id": _canonical_player_id(winner_name, item.get("player1Id", p1.get("id")), canonical_ids),
            "winner_seed": p1.get("seed"),
            "winner_entry": p1.get("winner_entry"),
            "winner_name": winner_name,
            "winner_hand": p1.get("hand"),
            "winner_ht": p1.get("height"),
            "winner_ioc": p1.get("countryAcr"),
            "winner_age": _age_on_date(p1.get("birthday"), date),
            # currentRank/points in the player object are profile snapshots, not
            # the historical ranking on this match date. Do not leak them into
            # historical training; archival duplicate rows can fill match-time
            # ranks during deduplication.
            "winner_rank": np.nan,
            "winner_rank_points": np.nan,
            "loser_id": _canonical_player_id(loser_name, item.get("player2Id", p2.get("id")), canonical_ids),
            "loser_seed": p2.get("seed"),
            "loser_entry": p2.get("loser_entry"),
            "loser_name": loser_name,
            "loser_hand": p2.get("hand"),
            "loser_ht": p2.get("height"),
            "loser_ioc": p2.get("countryAcr"),
            "loser_age": _age_on_date(p2.get("birthday"), date),
            "loser_rank": np.nan,
            "loser_rank_points": np.nan,
            "score": item.get("result"),
            "best_of": best_of,
            "round": round_name,
            "minutes": np.nan,
            "w_ace": _stat_number(wstats, "aces"),
            "w_df": _stat_number(wstats, "doubleFaults"),
            "w_svpt": w_svpt,
            "w_1stIn": _stat_number(wstats, "firstServe"),
            "w_1stWon": _stat_number(wstats, "winningOnFirstServe"),
            "w_2ndWon": _stat_number(wstats, "winningOnSecondServe"),
            "w_SvGms": np.nan,
            "w_bpSaved": (l_bp_chances - l_bp_won) if l_bp_chances is not None and l_bp_won is not None else np.nan,
            "w_bpFaced": l_bp_chances,
            "l_ace": _stat_number(lstats, "aces"),
            "l_df": _stat_number(lstats, "doubleFaults"),
            "l_svpt": l_svpt,
            "l_1stIn": _stat_number(lstats, "firstServe"),
            "l_1stWon": _stat_number(lstats, "winningOnFirstServe"),
            "l_2ndWon": _stat_number(lstats, "winningOnSecondServe"),
            "l_SvGms": np.nan,
            "l_bpSaved": (w_bp_chances - w_bp_won) if w_bp_chances is not None and w_bp_won is not None else np.nan,
            "l_bpFaced": w_bp_chances,
            "w_winners": _stat_number(wstats, "winners"),
            "w_unforced_errors": _stat_number(wstats, "unforcedErrors"),
            "w_net_won": _stat_number(wstats, "netApproaches"),
            "w_net_total": _stat_number(wstats, "netApproachesOf"),
            "w_total_points_won": _stat_number(wstats, "totalPointsWon"),
            "w_fastest_serve": _stat_number(wstats, "fastestServe"),
            "w_avg_first_serve_speed": _stat_number(wstats, "averageFirstServeSpeed"),
            "w_avg_second_serve_speed": _stat_number(wstats, "averageSecondServeSpeed"),
            "l_winners": _stat_number(lstats, "winners"),
            "l_unforced_errors": _stat_number(lstats, "unforcedErrors"),
            "l_net_won": _stat_number(lstats, "netApproaches"),
            "l_net_total": _stat_number(lstats, "netApproachesOf"),
            "l_total_points_won": _stat_number(lstats, "totalPointsWon"),
            "l_fastest_serve": _stat_number(lstats, "fastestServe"),
            "l_avg_first_serve_speed": _stat_number(lstats, "averageFirstServeSpeed"),
            "l_avg_second_serve_speed": _stat_number(lstats, "averageSecondServeSpeed"),
            "data_source": "Matchstat",
            "matchstat_match_id": item.get("id"),
        }
        rows.append(row)
    return pd.DataFrame(rows)

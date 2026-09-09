from __future__ import annotations

from datetime import datetime, timezone
import re
import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .market_store import record_snapshot
from .matchstat import MatchstatClient, normalize_name
from .pinnodds import PinnOddsClient
from .model_service import automatic_best_of, predict_match
from .odds import available_bookmakers, pinnacle_moneyline_from_payload, pinnacle_total_sets_35_from_payload
from .sets_service import predict_over35
from .tournament_features import canonical_tournament, encode_tournament_level

_ODDS_CACHE: dict[str, dict[str, Any]] = {}


def _pinnodds_same_player(left: str, right: str) -> bool:
    a, b = normalize_name(left), normalize_name(right)
    if not a or not b:
        return False
    if a == b:
        return True
    ap, bp = a.split(), b.split()
    return len(ap) >= 2 and len(bp) >= 2 and ap[-1] == bp[-1] and ap[0][0] == bp[0][0]


def _pinnodds_event_ts(row: dict) -> float | None:
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


def _pinnodds_match_fixture(events: list[dict], player_a: str, player_b: str, start_timestamp: float | None):
    best = None
    for row in events or []:
        if not isinstance(row, dict):
            continue
        home = str(row.get("home") or "").strip()
        away = str(row.get("away") or "").strip()
        if not ((_pinnodds_same_player(home, player_a) and _pinnodds_same_player(away, player_b))
                or (_pinnodds_same_player(home, player_b) and _pinnodds_same_player(away, player_a))):
            continue
        event_ts = _pinnodds_event_ts(row)
        if start_timestamp and event_ts:
            delta = abs(float(event_ts) - float(start_timestamp))
            if delta > 36 * 3600:
                continue
        else:
            delta = 0.0
        if best is None or delta < best[0]:
            best = (delta, row)
    return None if best is None else best[1]


def _pinnodds_total35_from_periods(payload: Any) -> tuple[float, float] | None:
    rows = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        rows = [payload] if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        periods = row.get("periods") or {}
        game = periods.get("num_0") if isinstance(periods, dict) else None
        totals = game.get("totals") if isinstance(game, dict) else None
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
                over, under = float(value.get("over")), float(value.get("under"))
            except Exception:
                continue
            if over > 1.0 and under > 1.0:
                return over, under
    return None


def _pinnodds_total35_from_special_tree(payload: Any) -> tuple[float, float] | None:
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
        here = " ".join(str(node.get(k) or "") for k in (
            "special", "special_category", "special_units", "type", "key",
            "side", "name", "description", "market", "market_name",
        )).casefold()
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
                if not ((points is not None and abs(points - 3.5) < 1e-9) or "3.5" in context or "3.5" in label):
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


def _compat_find_total_sets_35(self, player_a: str, player_b: str, start_timestamp: float | None = None, *, force: bool = False):
    """Compatibility shim so v0.3.2 works even if Streamlit still has the v0.3.1 PinnOddsClient loaded."""
    try:
        events = self.prematch_events(force=force)
        row = _pinnodds_match_fixture(events, player_a, player_b, start_timestamp)
        if row is None:
            self.last_total35_error = "Pinnodds fixture not matched"
            return None
        event_id = row.get("event_id") or row.get("id")
        if event_id is None:
            self.last_total35_error = "Pinnodds fixture has no event_id"
            return None
        for path, params in (
            ("/kit/v1/prematch/lines", {"event_id": event_id, "market_type": "totals"}),
            ("/kit/v1/prematch/markets", {"event_id": event_id}),
        ):
            payload = self._get(path, params=params)
            pair = _pinnodds_total35_from_periods(payload) or _pinnodds_total35_from_special_tree(payload)
            if pair:
                self.last_total35_error = None
                return {"sets35": pair, "source": "pinnodds-total-sets", "provider_event_id": str(event_id)}
        special_payload = self._get(
            "/kit/v1/prematch/fixtures",
            params={"sport_id": 2, "include_specials": "nested"},
        )
        special_events = [x for x in (special_payload.get("events") or []) if isinstance(x, dict)]
        special_parent = _pinnodds_match_fixture(special_events, player_a, player_b, start_timestamp)
        if special_parent is not None:
            pair = _pinnodds_total35_from_special_tree(special_parent.get("specials") or special_parent)
            if pair:
                self.last_total35_error = None
                return {"sets35": pair, "source": "pinnodds-total-sets-special", "provider_event_id": str(event_id)}
        self.last_total35_error = "Pinnacle Total Sets 3.5 is not currently offered in the Pinnodds payload"
        return None
    except Exception as exc:
        self.last_total35_error = str(exc)
        return None


# Streamlit can occasionally keep the previous class definition alive across a hot reload.
# Patch the class at import time so the totals feature still works even in that case.
if not hasattr(PinnOddsClient, "find_total_sets_35"):
    PinnOddsClient.find_total_sets_35 = _compat_find_total_sets_35  # type: ignore[attr-defined]



def tournament_context(master_path) -> dict[str, dict]:
    try:
        matches = pd.read_csv(
            master_path,
            usecols=lambda c: c in {"tourney_name", "surface", "tourney_level", "indoor", "tourney_date"},
            low_memory=False,
        )
    except Exception:
        return {}
    if matches.empty:
        return {}
    matches = matches.dropna(subset=["tourney_name", "surface"]).copy()
    matches["tourney_date"] = pd.to_numeric(matches.tourney_date, errors="coerce")
    matches["key"] = matches.tourney_name.map(canonical_tournament)
    latest = matches.sort_values("tourney_date").drop_duplicates("key", keep="last")
    out: dict[str, dict] = {}
    for _, row in latest.iterrows():
        level = encode_tournament_level(row.get("tourney_level", "A"))
        out[str(row["key"])] = {
            "surface": str(row["surface"]).title(),
            "level": float(level),
            "indoor": str(row.get("indoor", "")).strip().lower() in {"i", "1", "1.0", "true", "yes", "indoor"},
            "tournament": str(row["tourney_name"]),
            # Never infer a future Grand-Slam format from the most recent historical
            # row, because that row could be qualifying. Live event metadata wins.
            "best_of": automatic_best_of(level),
        }
    return out


def _resolve_context(league: str, context: dict[str, dict]) -> dict | None:
    key = canonical_tournament(league)
    if key in context:
        return context[key]
    # Conservative fallback for provider naming variants such as "Shanghai - China".
    if len(key) < 4:
        return None
    candidates = []
    for candidate_key, value in context.items():
        if len(candidate_key) >= 4 and (candidate_key in key or key in candidate_key):
            candidates.append(value)
    return candidates[0] if len(candidates) == 1 else None


def _looks_lower_tier(event: dict) -> bool:
    text = " ".join(str(event.get(k) or "") for k in ("league", "tourType", "name", "competition")).casefold()
    return any(token in text for token in ("challenger", "itf", "future"))


def _looks_qualifying(event: dict) -> bool:
    text = " ".join(str(event.get(k) or "") for k in ("name", "round", "roundName", "league")).casefold()
    return any(token in text for token in ("qualifying", "qualification", "qualifier round"))


def _event_best_of(event: dict, level: float) -> float:
    for key in ("bestOf", "best_of", "bestOfSets", "format", "matchFormat"):
        value = event.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
            if parsed in {3.0, 5.0}:
                return parsed
        except Exception:
            found = re.search(r"\b([35])\b", str(value))
            if found:
                return float(found.group(1))
    return automatic_best_of(level)


def eligible_events(
    events: list[dict],
    context: dict,
    *,
    include_qualifying: bool = False,
    horizon_hours: int = 36,
    today_only: bool = True,
    timezone_name: str = "America/Toronto",
) -> tuple[list[dict], dict]:
    now = time.time()
    max_ts = now + horizon_hours * 3600
    tz = ZoneInfo(timezone_name)
    local_today = datetime.now(tz).date()
    kept: list[dict] = []
    diag = {"lower_tier": 0, "unknown_context": 0, "qualifying": 0, "bad_status": 0, "bad_time": 0, "not_today": 0}
    for event in events:
        if not isinstance(event, dict):
            continue
        status = str(event.get("status") or "").casefold()
        if status not in {"not started", "upcoming", "scheduled", ""}:
            diag["bad_status"] += 1
            continue
        try:
            start = float(event.get("startTimestamp") or 0)
        except Exception:
            start = 0
        if start and (start < now - 300 or start > max_ts):
            diag["bad_time"] += 1
            continue
        if today_only and start and datetime.fromtimestamp(start, tz=timezone.utc).astimezone(tz).date() != local_today:
            diag["not_today"] += 1
            continue
        if _looks_lower_tier(event):
            diag["lower_tier"] += 1
            continue
        if not include_qualifying and _looks_qualifying(event):
            diag["qualifying"] += 1
            continue
        league = str(event.get("league") or "").strip()
        ctx = _resolve_context(league, context)
        if not ctx:
            diag["unknown_context"] += 1
            continue
        if float(ctx["level"]) < 2.0:
            diag["lower_tier"] += 1
            continue
        event = dict(event)
        event["_ctx"] = {**ctx, "best_of": _event_best_of(event, float(ctx["level"]))}
        kept.append(event)
    return kept, diag


def _ttl_seconds(start_ts: float) -> int:
    delta = max(0.0, float(start_ts or 0) - time.time())
    if delta <= 3600:
        return 30
    if delta <= 6 * 3600:
        return 60
    return 180


def _event_odds(client: MatchstatClient, event: dict, pinnacle_client: PinnOddsClient | None = None, force: bool = False) -> dict:
    eid = str(event.get("id"))
    start = float(event.get("startTimestamp") or 0)
    now = time.time()
    ttl = _ttl_seconds(start)
    cached = _ODDS_CACHE.get(eid)
    if cached and not force and now - float(cached.get("fetched_at", 0)) < ttl:
        return cached

    # Preferred sharp-price source: direct Pinnacle-only prematch feed. Matchstat
    # remains the schedule/stats API, but it does not expose Pinnacle for every ATP
    # event. One board-level Pinnodds snapshot is cached and matched by players/time.
    direct_sets = None
    direct_sets_source = None
    direct_sets_error = None
    ctx = event.get("_ctx") or {}
    wants_sets35 = float(ctx.get("level", 0) or 0) >= 5.0 and float(ctx.get("best_of", 0) or 0) >= 5.0
    if pinnacle_client is not None and pinnacle_client.enabled:
        try:
            direct = pinnacle_client.find_moneyline(
                str(event.get("participant1") or ""),
                str(event.get("participant2") or ""),
                start_timestamp=start or None,
                force=force,
            )
        except Exception as exc:
            direct = None
            direct_error = str(exc)
        else:
            direct_error = None

        if wants_sets35:
            try:
                direct_sets_quote = pinnacle_client.find_total_sets_35(
                    str(event.get("participant1") or ""),
                    str(event.get("participant2") or ""),
                    start_timestamp=start or None,
                    force=force,
                )
            except Exception as exc:
                direct_sets_quote = None
                direct_sets_error = str(exc)
            else:
                direct_sets_error = pinnacle_client.last_total35_error
            if direct_sets_quote and direct_sets_quote.get("sets35"):
                direct_sets = direct_sets_quote["sets35"]
                direct_sets_source = direct_sets_quote.get("source", "pinnodds-total-sets")

        if direct and direct.get("moneyline"):
            out = {
                "fetched_at": now,
                "moneyline": direct["moneyline"],
                "sets35": direct_sets,
                "sets35_source": direct_sets_source,
                "sets35_error": direct_sets_error,
                "error": None,
                "source": direct.get("source", "pinnodds-prematch"),
                "available_bookmakers": ["Pinnacle"],
                "resolved_event_id": eid,
                "provider_event_id": direct.get("provider_event_id"),
                "provider_league": direct.get("provider_league"),
            }
            _ODDS_CACHE[eid] = out
            return out
    else:
        direct_error = "PINNODDS_API_KEY not configured"

    def fetch_for_id(target_id: str) -> dict:
        payload = None
        compared = None
        recent = None
        ml = None
        sets35 = None
        error = None
        source = None
        books: set[str] = set()

        try:
            payload = client.pre_match_odds(target_id)
            books.update(available_bookmakers(payload))
            ml = pinnacle_moneyline_from_payload(payload)
            sets35 = pinnacle_total_sets_35_from_payload(payload)
            if ml is not None:
                source = "pre-match"
        except Exception as exc:
            error = str(exc)

        if ml is None:
            try:
                compared = client.compared_odds(target_id, market_id=1)
                books.update(available_bookmakers(compared))
                ml = pinnacle_moneyline_from_payload(compared)
                if ml is not None:
                    source = "compare"
            except Exception as exc:
                error = error or str(exc)

        if ml is None:
            try:
                recent = client.recent_odds(target_id)
                books.update(available_bookmakers(recent))
                ml = pinnacle_moneyline_from_payload(recent)
                if sets35 is None:
                    sets35 = pinnacle_total_sets_35_from_payload(recent)
                if ml is not None:
                    source = "recent-odds"
            except Exception as exc:
                error = error or str(exc)

        return {
            "moneyline": ml, "sets35": sets35, "error": error, "source": source,
            "available_bookmakers": sorted(books, key=str.casefold),
            "payload": payload, "compare_payload": compared, "recent_payload": recent,
            "resolved_event_id": str(target_id),
        }

    result = fetch_for_id(eid)
    if result.get("sets35") is None and direct_sets is not None:
        result["sets35"] = direct_sets
        result["sets35_source"] = direct_sets_source
    if result.get("sets35") is None and direct_sets_error:
        result["sets35_error"] = direct_sets_error
    if result.get("moneyline") is None and direct_error:
        result["direct_pinnacle_error"] = direct_error

    # Some Matchstat upcoming surfaces expose a Core match ID while odds endpoints
    # require the Live/Extend event ID. If prices are missing, resolve the canonical
    # event by player names + date and retry once using result.id.
    if result.get("moneyline") is None and start:
        p1 = str(event.get("participant1") or "").strip()
        p2 = str(event.get("participant2") or "").strip()
        if p1 and p2:
            try:
                date_only = datetime.fromtimestamp(start, tz=timezone.utc).date().isoformat()
                info = client.event_information(p1, p2, date_only)
                info_result = info.get("result", info) if isinstance(info, dict) else {}
                live_id = info_result.get("id") if isinstance(info_result, dict) else None
                if live_id is not None and str(live_id) != eid:
                    retry = fetch_for_id(str(live_id))
                    # Keep all discovered bookmakers for diagnostics even if Pinnacle
                    # is absent from the canonical Live event too.
                    retry["available_bookmakers"] = sorted(
                        set(result.get("available_bookmakers", [])) | set(retry.get("available_bookmakers", [])),
                        key=str.casefold,
                    )
                    if retry.get("sets35") is None and direct_sets is not None:
                        retry["sets35"] = direct_sets
                        retry["sets35_source"] = direct_sets_source
                    if retry.get("sets35") is None and direct_sets_error:
                        retry["sets35_error"] = direct_sets_error
                    if retry.get("moneyline") is not None or retry.get("available_bookmakers"):
                        result = retry
                    result["live_event_resolved"] = True
            except Exception as exc:
                result["resolve_error"] = str(exc)

    out = {"fetched_at": now, **result}
    _ODDS_CACHE[eid] = out
    return out


def build_slate(
    client: MatchstatClient,
    events: list[dict],
    state,
    bundle,
    *,
    pinnacle_client: PinnOddsClient | None = None,
    bankroll: float = 0.0,
    min_ev: float = .02,
    min_edge: float = .02,
    force_odds: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows: list[dict] = []
    detail: dict[str, dict] = {}
    now = time.time()
    player_lookup = {}
    for player_name in state.player.dropna().astype(str).unique():
        player_lookup.setdefault(normalize_name(player_name), player_name)

    for event in events:
        eid = str(event.get("id"))
        a_display = str(event.get("participant1") or "").strip()
        b_display = str(event.get("participant2") or "").strip()
        league = str(event.get("league") or "").strip()
        ctx = event["_ctx"]
        if not a_display or not b_display:
            continue
        # Matchstat and the training data often differ only by capitalization,
        # accents, punctuation or spacing. Resolve to the canonical player name
        # stored in player_state instead of requiring an exact provider string.
        a = player_lookup.get(normalize_name(a_display))
        b = player_lookup.get(normalize_name(b_display))
        surface = ctx["surface"]
        if not a or not b:
            start = float(event.get("startTimestamp") or 0)
            start_text = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if start else ""
            rows.append({
                "Event ID": eid, "Start UTC": start_text, "Match": f"{a_display} vs {b_display}",
                "Tournament": league, "Level": ctx["level"], "Surface": surface, "BO": int(ctx["best_of"]),
                "Status": "Player state not matched", "Pinnacle available": False,
                "ML pick": "No bet", "ML pick P": np.nan, "ML Pinnacle": np.nan, "ML Edge": np.nan,
                "ML EV": np.nan, "ML Kelly %": 0.0, "ML stake CA$": 0.0,
                "Best market": "No bet", "Best selection": "No bet", "Best price": np.nan,
                "Best Edge": np.nan, "Best EV": np.nan, "Best Kelly %": 0.0, "Best stake CA$": 0.0,
                "O3.5 P": np.nan, "O3.5 odds": np.nan, "U3.5 odds": np.nan, "Odds age s": np.nan,
            })
            continue
        ar = state[(state.player == a) & (state.surface == surface)]
        br = state[(state.player == b) & (state.surface == surface)]
        if ar.empty or br.empty:
            continue
        arank = pd.to_numeric(ar.iloc[0].get("rank"), errors="coerce")
        brank = pd.to_numeric(br.iloc[0].get("rank"), errors="coerce")
        rank_a = int(arank) if pd.notna(arank) else 999
        rank_b = int(brank) if pd.notna(brank) else 999

        quote = _event_odds(client, event, pinnacle_client=pinnacle_client, force=force_odds)
        ml = quote.get("moneyline")
        oa, ob = ml if ml else (np.nan, np.nan)
        sets_quote = quote.get("sets35")
        start = float(event.get("startTimestamp") or 0)
        year = datetime.fromtimestamp(start, tz=timezone.utc).year if start else datetime.now(timezone.utc).year

        result = predict_match(
            state,
            bundle,
            a,
            b,
            surface,
            rank_a,
            rank_b,
            oa,
            ob,
            tournament_level=ctx["level"],
            best_of=ctx["best_of"],
            tournament=league or ctx["tournament"],
            prediction_year=year,
            indoor=ctx["indoor"],
        )

        if ml or sets_quote:
            record_snapshot(
                eid,
                start,
                a,
                b,
                oa,
                ob,
                league,
                sets_over35=sets_quote[0] if sets_quote else None,
                sets_under35=sets_quote[1] if sets_quote else None,
            )

        rec_ok = (
            bool(ml)
            and result["recommended_side"] != "NO BET"
            and result["recommended_ev"] >= min_ev
            and result["recommended_edge"] >= min_edge
        )
        ml_pick = result["recommended_pick"] if rec_ok else "No bet"
        ml_kelly = float(result["recommended_quarter_kelly"] if rec_ok else 0.0)
        ml_stake = float(bankroll) * ml_kelly if bankroll and rec_ok else 0.0

        gs_sets = None
        if float(ctx["level"]) >= 5.0 and float(result["best_of"]) >= 5.0:
            gs_sets = predict_over35(
                state,
                a,
                b,
                surface,
                league or ctx["tournament"],
                result["court_speed"],
                *(sets_quote or (None, None)),
            )

        best_market = "No bet"
        best_selection = "No bet"
        best_price = np.nan
        best_ev = 0.0
        best_edge = 0.0
        best_kelly = 0.0
        if rec_ok:
            best_market = "Moneyline"
            best_selection = result["recommended_pick"]
            best_price = float(result["recommended_odds"])
            best_ev = float(result["recommended_ev"])
            best_edge = float(result["recommended_edge"])
            best_kelly = ml_kelly

        if (
            gs_sets
            and gs_sets.get("available")
            and gs_sets.get("recommended_market") != "No bet"
            and float(gs_sets.get("recommended_ev", -1)) >= min_ev
            and float(gs_sets.get("recommended_edge", -1)) >= min_edge
            and float(gs_sets.get("recommended_ev", -1)) > best_ev
        ):
            is_over = str(gs_sets["recommended_market"]).startswith("Over")
            best_market = "Total sets 3.5"
            best_selection = str(gs_sets["recommended_market"])
            best_price = float(gs_sets.get("odds_over35") if is_over else gs_sets.get("odds_under35"))
            best_ev = float(gs_sets["recommended_ev"])
            best_edge = float(gs_sets["recommended_edge"])
            best_kelly = float(gs_sets.get("recommended_quarter_kelly", 0.0))

        best_stake = float(bankroll) * best_kelly if bankroll and best_market != "No bet" else 0.0
        start_text = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if start else ""
        model_pick_p = max(float(result["probability_a"]), float(result["probability_b"]))
        model_favorite = a if float(result["probability_a"]) >= float(result["probability_b"]) else b
        row = {
            "Event ID": eid,
            "Start UTC": start_text,
            "Match": f"{a} vs {b}",
            "Tournament": league,
            "Level": ctx["level"],
            "Surface": surface,
            "BO": int(result["best_of"]),
            "Status": "Ready" if ml else (
                "Pinnacle unavailable"
                + (f" · books: {', '.join(quote.get('available_bookmakers', [])[:4])}" if quote.get('available_bookmakers') else "")
                + (f" · API: {quote.get('error')}" if quote.get('error') else "")
            ),
            "Pinnacle available": bool(ml),
            "ML pick": ml_pick if ml else f"Model: {model_favorite}",
            "ML pick P": model_pick_p,
            "ML Pinnacle": result["recommended_odds"] if rec_ok else np.nan,
            "ML Edge": result["recommended_edge"] if rec_ok else (max(result["edge_a"], result["edge_b"]) if ml else np.nan),
            "ML EV": result["recommended_ev"] if ml else np.nan,
            "ML Kelly %": ml_kelly,
            "ML stake CA$": ml_stake,
            "Best market": best_market,
            "Best selection": best_selection,
            "Best price": best_price,
            "Best Edge": best_edge,
            "Best EV": best_ev,
            "Best Kelly %": best_kelly,
            "Best stake CA$": best_stake,
            "O3.5 P": gs_sets.get("probability_over35") if gs_sets and gs_sets.get("available") else np.nan,
            "O3.5 odds": gs_sets.get("odds_over35") if gs_sets else np.nan,
            "U3.5 odds": gs_sets.get("odds_under35") if gs_sets else np.nan,
            "Odds age s": int(max(0, now - quote["fetched_at"])),
        }
        rows.append(row)
        detail[eid] = {
            "event": event,
            "context": ctx,
            "quote": quote,
            "result": result,
            "sets": gs_sets,
            "row": row,
            "rank_a": rank_a,
            "rank_b": rank_b,
        }

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["Start UTC", "Best EV"], ascending=[True, False]).reset_index(drop=True)
    return frame, detail

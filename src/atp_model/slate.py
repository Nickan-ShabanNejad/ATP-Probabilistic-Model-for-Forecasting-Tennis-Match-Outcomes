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
from .model_service import automatic_best_of, predict_match
from .odds import available_bookmakers, pinnacle_moneyline_from_payload, pinnacle_total_sets_35_from_payload
from .sets_service import predict_over35
from .tournament_features import canonical_tournament, encode_tournament_level

_ODDS_CACHE: dict[str, dict[str, Any]] = {}


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


def _event_odds(client: MatchstatClient, event: dict, force: bool = False) -> dict:
    eid = str(event.get("id"))
    start = float(event.get("startTimestamp") or 0)
    now = time.time()
    ttl = _ttl_seconds(start)
    cached = _ODDS_CACHE.get(eid)
    if cached and not force and now - float(cached.get("fetched_at", 0)) < ttl:
        return cached

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

        quote = _event_odds(client, event, force=force_odds)
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

        if ml:
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

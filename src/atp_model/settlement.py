from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any

from .matchstat import MatchstatClient, normalize_name
from .pinnodds import PinnOddsClient
from . import supabase_store


FINAL_STATUSES = {"ended", "finished", "completed", "complete", "final"}
ABNORMAL_STATUSES = {"retired", "retirement", "walkover", "walk over", "wo", "w/o", "cancelled", "canceled", "abandoned"}


@dataclass
class MatchOutcome:
    status: str
    winner: str | None
    sets_played: int | None
    score: str
    abnormal: bool = False


def _as_utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _same_name(a: str | None, b: str | None) -> bool:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    aa, bb = na.split(), nb.split()
    return len(aa) >= 2 and len(bb) >= 2 and aa[-1] == bb[-1] and aa[0][0] == bb[0][0]


def _score_sets(score: str) -> list[tuple[int, int]]:
    """Extract completed set scores from common tennis score strings."""
    text = str(score or "")
    # Handles 6-4, 7-6(5), 6:3 and whitespace/comma separated score strings.
    pairs = re.findall(r"(?<!\d)([0-7])\s*[-:]\s*([0-7])(?:\s*\([^)]*\))?", text)
    out: list[tuple[int, int]] = []
    for left, right in pairs:
        a, b = int(left), int(right)
        # Ignore obvious point/game fragments such as 0-0; a completed tennis set
        # has at least six games for one side, except a final-set match tiebreak that
        # providers may encode separately (we do not use those fragments here).
        if max(a, b) >= 6 and a != b:
            out.append((a, b))
    return out


def _explicit_winner(result: dict[str, Any], p1: str, p2: str) -> str | None:
    for key in ("winnerName", "winner_name", "winner", "winnerPlayer", "winningPlayer"):
        value = result.get(key)
        if isinstance(value, dict):
            value = value.get("name") or value.get("fullName") or value.get("displayName")
        if isinstance(value, str):
            if _same_name(value, p1):
                return p1
            if _same_name(value, p2):
                return p2
        try:
            idx = int(value)
            if idx == 1:
                return p1
            if idx == 2:
                return p2
        except Exception:
            pass
    return None


def outcome_from_event(payload: Any, stored_player_a: str, stored_player_b: str) -> MatchOutcome | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        return None

    status = str(result.get("status") or "").strip().casefold()
    p1 = str(result.get("participant1") or stored_player_a or "").strip()
    p2 = str(result.get("participant2") or stored_player_b or "").strip()
    score = str(result.get("score") or result.get("result") or "").strip()
    abnormal = status in ABNORMAL_STATUSES or any(token in status for token in ("retir", "walkover", "abandon", "cancel"))

    if status not in FINAL_STATUSES and not abnormal:
        return None

    sets = _score_sets(score)
    winner = _explicit_winner(result, p1, p2)
    if winner is None and sets:
        p1_sets = sum(a > b for a, b in sets)
        p2_sets = sum(b > a for a, b in sets)
        if p1_sets > p2_sets:
            winner = p1
        elif p2_sets > p1_sets:
            winner = p2

    return MatchOutcome(
        status=status,
        winner=winner,
        sets_played=len(sets) if sets else None,
        score=score,
        abnormal=abnormal,
    )


def _fetch_event(client: MatchstatClient, player_a: str, player_b: str, match_date: Any) -> dict | None:
    dt = _as_utc(match_date)
    if dt is None:
        return None
    # Tournament-local midnight can differ from UTC. Try the stored UTC date and
    # adjacent dates; the endpoint is keyed by player names + date only.
    dates = [dt.date(), (dt - timedelta(days=1)).date(), (dt + timedelta(days=1)).date()]
    seen = set()
    for day in dates:
        text = day.isoformat()
        if text in seen:
            continue
        seen.add(text)
        try:
            payload = client.event_information(player_a, player_b, text)
        except Exception:
            continue
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict) and result:
            return payload
    return None


def _price_for_selection(
    pinn: PinnOddsClient,
    row: dict[str, Any],
    *,
    force: bool = True,
) -> float | None:
    p1, p2 = str(row.get("player_a") or ""), str(row.get("player_b") or "")
    start = _as_utc(row.get("match_date"))
    start_ts = start.timestamp() if start else None
    market = str(row.get("market") or "").casefold()
    selection = str(row.get("selection") or "")

    if market == "moneyline":
        q = pinn.find_moneyline(p1, p2, start_timestamp=start_ts, force=force)
        pair = q.get("moneyline") if q else None
        if not pair or len(pair) != 2:
            return None
        if _same_name(selection, p1):
            return float(pair[0])
        if _same_name(selection, p2):
            return float(pair[1])
        return None

    if market == "total sets 3.5":
        q = pinn.find_total_sets_35(p1, p2, start_timestamp=start_ts, force=force)
        pair = q.get("sets35") if q else None
        if not pair or len(pair) != 2:
            return None
        return float(pair[0] if selection.casefold().startswith("over") else pair[1])
    return None


def refresh_open_bet_closes(pinn: PinnOddsClient, now: datetime | None = None) -> dict[str, int]:
    """Store the latest available pre-match Pinnacle quote on each open bet.

    The worker runs repeatedly before match start, so ``closing_odds`` becomes the
    last successfully observed pre-match Pinnacle price and is later frozen when the
    match is graded.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    updated = 0
    skipped = 0
    errors = 0
    for bet in supabase_store.list_bets():
        if bet.get("profit_loss") is not None:
            continue
        start = _as_utc(bet.get("match_date"))
        if start is None or start <= now:
            skipped += 1
            continue
        # No need to burn PinnOdds quota days in advance.
        if start - now > timedelta(hours=36):
            skipped += 1
            continue
        try:
            price = _price_for_selection(pinn, bet, force=False)
            if price and price > 1.0:
                supabase_store.update_bet_closing_odds(int(bet["id"]), price)
                updated += 1
            else:
                skipped += 1
        except Exception:
            errors += 1
    return {"updated": updated, "skipped": skipped, "errors": errors}


def settle_finished_tracking(client: MatchstatClient, now: datetime | None = None) -> dict[str, int]:
    """Grade every saved prediction and corresponding open bet that has finished."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    preds = supabase_store.list_predictions()
    bets = supabase_store.list_bets()

    candidates = [p for p in preds if p.get("actual_result") is None]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for pred in candidates:
        start = _as_utc(pred.get("match_date"))
        if start is None or start > now + timedelta(minutes=5):
            continue
        # Stop repeatedly querying ancient unresolved rows.
        if now - start > timedelta(days=7):
            continue
        key = (
            str(pred.get("match_id") or ""),
            str(pred.get("player_a") or ""),
            str(pred.get("player_b") or ""),
            str(pred.get("match_date") or ""),
        )
        grouped.setdefault(key, []).append(pred)

    events_checked = 0
    predictions_settled = 0
    bets_settled = 0
    abnormal_skipped = 0
    unresolved = 0

    for (_match_id, p1, p2, match_date), rows in grouped.items():
        payload = _fetch_event(client, p1, p2, match_date)
        if payload is None:
            unresolved += 1
            continue
        events_checked += 1
        outcome = outcome_from_event(payload, p1, p2)
        if outcome is None:
            continue
        if outcome.abnormal:
            # Retirement/walkover settlement rules vary by sportsbook and market.
            # Do not poison calibration or P&L with an assumption; leave for manual review.
            abnormal_skipped += 1
            continue
        if outcome.winner is None:
            unresolved += 1
            continue

        settled_at = datetime.now(timezone.utc).isoformat()
        for pred in rows:
            market = str(pred.get("market") or "").casefold()
            selection = str(pred.get("selection") or "")
            actual: float | None = None
            if market == "moneyline":
                actual = 1.0 if _same_name(selection, outcome.winner) else 0.0
            elif market == "total sets 3.5" and outcome.sets_played is not None:
                actual = 1.0 if outcome.sets_played >= 4 else 0.0
            if actual is None:
                continue
            supabase_store.update_prediction_outcome(int(pred["id"]), actual, settled_at=settled_at)
            predictions_settled += 1

        # Settle any actual bets for this match. The latest pre-match price has
        # already been stored in closing_odds by refresh_open_bet_closes().
        for bet in bets:
            if bet.get("profit_loss") is not None or str(bet.get("match_id") or "") != _match_id:
                continue
            market = str(bet.get("market") or "").casefold()
            selection = str(bet.get("selection") or "")
            result: str | None = None
            if market == "moneyline":
                result = "Win" if _same_name(selection, outcome.winner) else "Loss"
            elif market == "total sets 3.5" and outcome.sets_played is not None:
                went_over = outcome.sets_played >= 4
                picked_over = selection.casefold().startswith("over")
                result = "Win" if went_over == picked_over else "Loss"
            if result is None:
                continue
            close = bet.get("closing_odds")
            supabase_store.settle_bet(int(bet["id"]), result, float(close) if close not in (None, "") else None)
            bets_settled += 1

    return {
        "events_checked": events_checked,
        "predictions_settled": predictions_settled,
        "bets_settled": bets_settled,
        "abnormal_skipped": abnormal_skipped,
        "unresolved": unresolved,
    }

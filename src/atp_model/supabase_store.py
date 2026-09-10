from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from typing import Any

import requests


DEFAULT_TIMEOUT = 15


def _secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def config() -> tuple[str, str]:
    return _secret("SUPABASE_URL").rstrip("/"), _secret("SUPABASE_KEY")


def configured() -> bool:
    url, key = config()
    return bool(url and key)


def _headers(prefer: str | None = None) -> dict[str, str]:
    _, key = config()
    headers = {
        "apikey": key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # New sb_secret_* keys are API keys, not JWTs. Legacy service_role JWTs still
    # need the Authorization header, so support both without exposing either.
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        # Handles numpy scalar types.
        item = value.item()
        return _clean(item)
    except Exception:
        return str(value)


def _clean_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _clean(v) for k, v in payload.items()}


def _request(
    method: str,
    table: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | list[dict[str, Any]] | None = None,
    prefer: str | None = None,
) -> list[dict[str, Any]]:
    url, _ = config()
    if not configured():
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY are not configured")
    endpoint = f"{url}/rest/v1/{table}"
    if isinstance(payload, dict):
        body: Any = _clean_dict(payload)
    elif isinstance(payload, list):
        body = [_clean_dict(x) for x in payload]
    else:
        body = None
    response = requests.request(
        method.upper(),
        endpoint,
        headers=_headers(prefer),
        params=params,
        json=body,
        timeout=DEFAULT_TIMEOUT,
    )
    if not response.ok:
        text = response.text[:800]
        raise RuntimeError(f"Supabase {method.upper()} {table} failed ({response.status_code}): {text}")
    if not response.text.strip():
        return []
    data = response.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def healthcheck() -> tuple[bool, str]:
    if not configured():
        return False, "SUPABASE_URL / SUPABASE_KEY not configured"
    try:
        _request("GET", "bets", params={"select": "id", "limit": 1})
        return True, "connected"
    except Exception as exc:
        return False, str(exc)


def get_setting(key: str) -> Any:
    rows = _request(
        "GET",
        "model_settings",
        params={"select": "value", "key": f"eq.{key}", "limit": 1},
    )
    return rows[0].get("value") if rows else None


def set_setting(key: str, value: Any) -> None:
    existing = _request(
        "GET", "model_settings", params={"select": "key", "key": f"eq.{key}", "limit": 1}
    )
    payload = {"key": key, "value": _clean(value), "updated_at": datetime.now(timezone.utc).isoformat()}
    if existing:
        _request("PATCH", "model_settings", params={"key": f"eq.{key}"}, payload=payload)
    else:
        _request("POST", "model_settings", payload=payload)


def get_starting_bankroll() -> float | None:
    try:
        value = get_setting("starting_bankroll")
        if isinstance(value, dict):
            value = value.get("amount")
        if value is None:
            return None
        amount = float(value)
        return amount if amount >= 0 else None
    except Exception:
        return None


def set_starting_bankroll(amount: float) -> float:
    amount = float(amount)
    if amount < 0:
        raise ValueError("Starting bankroll must be zero or greater")
    set_setting("starting_bankroll", amount)
    return amount


def get_tracking_mode() -> str:
    """Return the persistent tracking mode. Percentage mode uses a normalized 100-point bankroll."""
    try:
        value = get_setting("tracking_mode")
        if isinstance(value, dict):
            value = value.get("mode")
        text = str(value or "percentage").strip().lower()
        return "currency" if text in {"currency", "cash", "bankroll", "cad"} else "percentage"
    except Exception:
        return "percentage"


def set_tracking_mode(mode: str) -> str:
    text = str(mode or "percentage").strip().lower()
    normalized = "currency" if text in {"currency", "cash", "bankroll", "cad"} else "percentage"
    set_setting("tracking_mode", normalized)
    return normalized


def effective_starting_bankroll() -> float | None:
    # Percentage-only tracking does not need the user's real bankroll. We normalize
    # the starting bankroll to 100.00 so stake amounts are internal bankroll points.
    if get_tracking_mode() == "percentage":
        return 100.0
    return get_starting_bankroll()


def list_bets() -> list[dict[str, Any]]:
    return _request("GET", "bets", params={"select": "*", "order": "created_at.asc"})


def list_predictions() -> list[dict[str, Any]]:
    return _request("GET", "predictions", params={"select": "*", "order": "created_at.asc"})


def list_odds_snapshots(match_id: str | None = None, market: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"select": "*", "order": "captured_at.asc"}
    if match_id:
        params["match_id"] = f"eq.{match_id}"
    if market:
        params["market"] = f"eq.{market}"
    return _request("GET", "odds_snapshots", params=params)


def _latest_odds_snapshot(match_id: str, market: str) -> dict[str, Any] | None:
    rows = _request(
        "GET",
        "odds_snapshots",
        params={
            "select": "*",
            "match_id": f"eq.{match_id}",
            "market": f"eq.{market}",
            "order": "captured_at.desc",
            "limit": 1,
        },
    )
    return rows[0] if rows else None


def record_odds_snapshot(
    *,
    match_id: str,
    match_date: str | None,
    market: str,
    player_a: str,
    player_b: str,
    selection_a: str,
    selection_b: str,
    odds_a: float | None,
    odds_b: float | None,
    source: str | None = None,
    captured_at: str | None = None,
) -> int | None:
    """Persist a Pinnacle price change. Unchanged consecutive quotes are deduplicated."""
    try:
        oa = float(odds_a) if odds_a not in (None, "") else None
        ob = float(odds_b) if odds_b not in (None, "") else None
    except Exception:
        return None
    if oa is None or ob is None or oa <= 1.0 or ob <= 1.0:
        return None
    last = _latest_odds_snapshot(str(match_id), str(market))
    if last:
        try:
            if abs(float(last.get("odds_a")) - oa) < 1e-9 and abs(float(last.get("odds_b")) - ob) < 1e-9:
                return int(last["id"])
        except Exception:
            pass
    payload = {
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
        "match_id": str(match_id),
        "match_date": match_date,
        "market": str(market),
        "player_a": player_a,
        "player_b": player_b,
        "selection_a": selection_a,
        "selection_b": selection_b,
        "odds_a": oa,
        "odds_b": ob,
        "source": source or "Pinnacle",
    }
    rows = _request("POST", "odds_snapshots", payload=payload, prefer="return=representation")
    return int(rows[0]["id"]) if rows else None


def current_bankroll() -> float | None:
    start = effective_starting_bankroll()
    if start is None:
        return None
    try:
        bets = list_bets()
    except Exception:
        return start
    profit = 0.0
    for row in bets:
        value = row.get("profit_loss")
        if value is not None:
            try:
                profit += float(value)
            except Exception:
                pass
    return float(start) + profit


def _prediction_lookup(match_id: str, market: str, model_version: str) -> list[dict[str, Any]]:
    return _request(
        "GET",
        "predictions",
        params={
            "select": "*",
            "match_id": f"eq.{match_id}",
            "market": f"eq.{market}",
            "model_version": f"eq.{model_version}",
            "limit": 1,
        },
    )


def upsert_prediction(payload: dict[str, Any]) -> int:
    """Store one immutable model forecast per match/market/version and refresh only its market price.

    The model probability/pick is frozen on first capture so later odds or data refreshes
    cannot rewrite history. Pinnacle latest/close fields continue updating separately.
    """
    match_id = str(payload.get("match_id") or "")
    market = str(payload.get("market") or "")
    model_version = str(payload.get("model_version") or "")
    if not match_id or not market or not model_version:
        raise ValueError("Prediction requires match_id, market and model_version")

    now = datetime.now(timezone.utc).isoformat()
    existing = _prediction_lookup(match_id, market, model_version)
    current_price = payload.get("latest_odds")
    try:
        current_price = float(current_price) if current_price not in (None, "") else None
    except Exception:
        current_price = None

    if existing:
        row = existing[0]
        pid = int(row["id"])
        if row.get("actual_result") is not None:
            return pid

        opening = row.get("opening_odds")
        if opening in (None, "") and current_price and current_price > 1.0:
            opening = current_price

        # Freeze the original forecast. Only backfill new v0.3.6 fields on legacy rows.
        update: dict[str, Any] = {
            "opening_odds": opening,
            "first_price_at": row.get("first_price_at") or (now if opening else None),
        }
        for key in ("predicted_selection", "predicted_probability"):
            if row.get(key) in (None, "") and payload.get(key) not in (None, ""):
                update[key] = payload.get(key)

        if current_price and current_price > 1.0:
            update["latest_odds"] = current_price
            update["latest_price_at"] = now
            if opening:
                try:
                    update["odds_change"] = current_price / float(opening) - 1.0
                except Exception:
                    pass
        _request("PATCH", "predictions", params={"id": f"eq.{pid}"}, payload=update)
        return pid

    insert = dict(payload)
    if current_price and current_price > 1.0:
        insert["opening_odds"] = current_price
        insert["latest_odds"] = current_price
        insert["first_price_at"] = now
        insert["latest_price_at"] = now
        insert["odds_change"] = 0.0
    rows = _request("POST", "predictions", payload=insert, prefer="return=representation")
    if not rows:
        rows = _prediction_lookup(match_id, market, model_version)
    if not rows:
        raise RuntimeError("Prediction was inserted but its id could not be recovered")
    return int(rows[0]["id"])


def prediction_payload_from_detail(detail: dict[str, Any], model_version: str, market: str = "Moneyline") -> dict[str, Any]:
    result = detail["result"]
    event = detail["event"]
    context = detail.get("context", {})
    quote = detail.get("quote", {})
    start = event.get("startTimestamp")
    match_date = None
    try:
        match_date = datetime.fromtimestamp(float(start), tz=timezone.utc).isoformat() if start else None
    except Exception:
        match_date = None
    tournament = result.get("tournament") or event.get("league") or context.get("tournament")
    level = result.get("tournament_level", context.get("level"))
    round_name = event.get("round") or context.get("round") or ""

    if market == "Total Sets 3.5":
        sets = detail.get("sets") or {}
        p_over = sets.get("probability_over35")
        p_under = sets.get("probability_under35")
        try:
            over_is_pick = float(p_over) >= float(p_under)
        except Exception:
            over_is_pick = True
        predicted_selection = "Over 3.5" if over_is_pick else "Under 3.5"
        predicted_probability = p_over if over_is_pick else p_under
        current_price = sets.get("odds_over35") if over_is_pick else sets.get("odds_under35")
        return {
            "match_id": str(event.get("id") or ""),
            "match_date": match_date,
            "model_version": model_version,
            "tournament": tournament,
            "tournament_level": str(level) if level is not None else None,
            "surface": result.get("surface"),
            "round": str(round_name),
            "player_a": result.get("player_a"),
            "player_b": result.get("player_b"),
            "market": market,
            "selection": "Over 3.5",
            "model_probability": p_over,
            "model_fair_odds": sets.get("fair_odds_over35"),
            "pinnacle_odds": sets.get("odds_over35"),
            "pinnacle_no_vig_probability": sets.get("market_probability_over35"),
            "edge": sets.get("edge_over35"),
            "expected_value": sets.get("ev_over35"),
            "court_speed": result.get("court_speed"),
            "predicted_selection": predicted_selection,
            "predicted_probability": predicted_probability,
            "latest_odds": current_price,
        }

    moneyline = quote.get("moneyline")
    odds_a, odds_b = moneyline if moneyline and len(moneyline) == 2 else (None, None)
    p_a = result.get("probability_a")
    p_b = result.get("probability_b")
    try:
        a_is_pick = float(p_a) >= float(p_b)
    except Exception:
        a_is_pick = True
    predicted_selection = result.get("player_a") if a_is_pick else result.get("player_b")
    predicted_probability = p_a if a_is_pick else p_b
    current_price = odds_a if a_is_pick else odds_b
    return {
        "match_id": str(event.get("id") or ""),
        "match_date": match_date,
        "model_version": model_version,
        "tournament": tournament,
        "tournament_level": str(level) if level is not None else None,
        "surface": result.get("surface"),
        "round": str(round_name),
        "player_a": result.get("player_a"),
        "player_b": result.get("player_b"),
        "market": "Moneyline",
        "selection": result.get("player_a"),
        "model_probability": p_a,
        "model_fair_odds": result.get("fair_odds_a"),
        "pinnacle_odds": odds_a,
        "pinnacle_no_vig_probability": result.get("market_probability_a") if moneyline else None,
        "edge": result.get("edge_a") if moneyline else None,
        "expected_value": result.get("ev_a") if moneyline else None,
        "court_speed": result.get("court_speed"),
        "predicted_selection": predicted_selection,
        "predicted_probability": predicted_probability,
        "latest_odds": current_price,
    }


def record_detail_predictions(detail: dict[str, Any], model_version: str) -> list[int]:
    """Persist every model forecast plus each observed Pinnacle price change."""
    result = detail.get("result") or {}
    event = detail.get("event") or {}
    quote = detail.get("quote") or {}
    sets = detail.get("sets") or {}
    start = event.get("startTimestamp")
    match_date = None
    try:
        match_date = datetime.fromtimestamp(float(start), tz=timezone.utc).isoformat() if start else None
    except Exception:
        match_date = None
    match_id = str(event.get("id") or "")
    player_a = str(result.get("player_a") or "")
    player_b = str(result.get("player_b") or "")
    source = str(quote.get("source") or "Pinnacle")

    ids = [upsert_prediction(prediction_payload_from_detail(detail, model_version, "Moneyline"))]
    ml = quote.get("moneyline")
    if ml and len(ml) == 2:
        record_odds_snapshot(
            match_id=match_id,
            match_date=match_date,
            market="Moneyline",
            player_a=player_a,
            player_b=player_b,
            selection_a=player_a,
            selection_b=player_b,
            odds_a=ml[0],
            odds_b=ml[1],
            source=source,
        )

    if sets.get("available") and float(result.get("best_of", 3) or 3) >= 5:
        ids.append(upsert_prediction(prediction_payload_from_detail(detail, model_version, "Total Sets 3.5")))
        oa, ob = sets.get("odds_over35"), sets.get("odds_under35")
        if oa not in (None, "") and ob not in (None, ""):
            record_odds_snapshot(
                match_id=match_id,
                match_date=match_date,
                market="Total Sets 3.5",
                player_a=player_a,
                player_b=player_b,
                selection_a="Over 3.5",
                selection_b="Under 3.5",
                odds_a=oa,
                odds_b=ob,
                source=str(quote.get("sets35_source") or source),
            )
    return ids



def update_prediction_outcome(prediction_id: int, actual_result: float, *, settled_at: str | None = None) -> None:
    actual = float(actual_result)
    if actual not in {0.0, 1.0}:
        raise ValueError("actual_result must be 0 or 1")
    _request(
        "PATCH",
        "predictions",
        params={"id": f"eq.{int(prediction_id)}"},
        payload={
            "actual_result": actual,
            "settled_at": settled_at or datetime.now(timezone.utc).isoformat(),
        },
    )



def _snapshot_price_for_prediction(prediction: dict[str, Any], snapshot: dict[str, Any]) -> float | None:
    predicted = str(prediction.get("predicted_selection") or "").strip()
    market = str(prediction.get("market") or "")
    if market == "Moneyline":
        if predicted and predicted == str(snapshot.get("selection_a") or ""):
            return float(snapshot.get("odds_a"))
        if predicted and predicted == str(snapshot.get("selection_b") or ""):
            return float(snapshot.get("odds_b"))
        p = float(prediction.get("model_probability") or 0.5)
        return float(snapshot.get("odds_a") if p >= 0.5 else snapshot.get("odds_b"))
    if market == "Total Sets 3.5":
        if predicted.casefold().startswith("under"):
            return float(snapshot.get("odds_b"))
        return float(snapshot.get("odds_a"))
    return None


def finalize_prediction_close(prediction_id: int, closing_odds: float, *, captured_at: str | None = None) -> None:
    rows = _request("GET", "predictions", params={"select": "*", "id": f"eq.{int(prediction_id)}", "limit": 1})
    if not rows:
        return
    pred = rows[0]
    close = float(closing_odds)
    if close <= 1.0:
        return
    opening = pred.get("opening_odds")
    try:
        opening_f = float(opening) if opening not in (None, "") else None
    except Exception:
        opening_f = None
    update: dict[str, Any] = {
        "latest_odds": close,
        "closing_odds": close,
        "latest_price_at": captured_at or datetime.now(timezone.utc).isoformat(),
    }
    if opening_f and opening_f > 1.0:
        update["odds_change"] = close / opening_f - 1.0
        update["clv"] = opening_f / close - 1.0
    _request("PATCH", "predictions", params={"id": f"eq.{int(prediction_id)}"}, payload=update)


def last_pre_start_odds_snapshot(match_id: str, market: str, match_date: str | None) -> dict[str, Any] | None:
    if not match_date:
        return None
    rows = _request(
        "GET",
        "odds_snapshots",
        params={
            "select": "*",
            "match_id": f"eq.{match_id}",
            "market": f"eq.{market}",
            "captured_at": f"lt.{match_date}",
            "order": "captured_at.desc",
            "limit": 1,
        },
    )
    return rows[0] if rows else None


def update_bet_closing_odds(bet_id: int, closing_odds: float) -> None:
    close = float(closing_odds)
    if close <= 1.0:
        raise ValueError("closing_odds must be greater than 1.00")
    # Pre-match worker may update this repeatedly. Once profit_loss is non-null the
    # bet is already settled and its closing price must remain frozen.
    _request(
        "PATCH",
        "bets",
        params={"id": f"eq.{int(bet_id)}", "profit_loss": "is.null"},
        payload={"closing_odds": close},
    )

def find_open_bet(match_id: str, market: str, selection: str) -> dict[str, Any] | None:
    rows = _request(
        "GET",
        "bets",
        params={
            "select": "*",
            "match_id": f"eq.{match_id}",
            "market": f"eq.{market}",
            "selection": f"eq.{selection}",
            "result": "is.null",
            "limit": 1,
        },
    )
    return rows[0] if rows else None


def place_bet(
    *,
    prediction_id: int | None,
    match_id: str,
    match_date: str | None,
    model_version: str,
    tournament: str | None,
    tournament_level: Any,
    surface: str | None,
    round_name: str | None,
    player_a: str,
    player_b: str,
    market: str,
    selection: str,
    model_probability: float,
    model_fair_odds: float,
    odds_taken: float,
    edge: float | None,
    expected_value: float | None,
    bankroll_before: float | None,
    kelly_fraction: float | None,
    stake_amount: float,
) -> int:
    if float(odds_taken) <= 1.0:
        raise ValueError("Odds taken must be greater than 1.00")
    if float(stake_amount) <= 0:
        raise ValueError("Stake must be greater than zero")
    bankroll = float(bankroll_before) if bankroll_before not in (None, 0) else None
    stake_pct = float(stake_amount) / bankroll if bankroll else None
    payload = {
        "prediction_id": prediction_id,
        "match_id": str(match_id),
        "match_date": match_date,
        "model_version": model_version,
        "tournament": tournament,
        "tournament_level": str(tournament_level) if tournament_level is not None else None,
        "surface": surface,
        "round": round_name or "",
        "player_a": player_a,
        "player_b": player_b,
        "market": market,
        "selection": selection,
        "model_probability": model_probability,
        "model_fair_odds": model_fair_odds,
        "odds_taken": odds_taken,
        "closing_odds": None,
        "edge": edge,
        "expected_value": expected_value,
        "bankroll_before": bankroll_before,
        "kelly_fraction": kelly_fraction,
        "stake_percent": stake_pct,
        "stake_amount": stake_amount,
        "result": None,
        "profit_loss": None,
        "clv": None,
        "bankroll_after": None,
        "settled_at": None,
    }
    rows = _request("POST", "bets", payload=payload, prefer="return=representation")
    if rows:
        return int(rows[0]["id"])
    candidates = _request(
        "GET", "bets", params={"select": "id", "match_id": f"eq.{match_id}", "order": "created_at.desc", "limit": 1}
    )
    if not candidates:
        raise RuntimeError("Bet was inserted but its id could not be recovered")
    return int(candidates[0]["id"])


def settle_bet(bet_id: int, result: str, closing_odds: float | None = None) -> dict[str, Any]:
    rows = _request("GET", "bets", params={"select": "*", "id": f"eq.{int(bet_id)}", "limit": 1})
    if not rows:
        raise ValueError(f"Bet #{bet_id} was not found")
    bet = rows[0]
    result_norm = str(result).strip().lower()
    if result_norm not in {"win", "loss", "void"}:
        raise ValueError("Result must be Win, Loss or Void")
    stake = float(bet.get("stake_amount") or 0.0)
    odds = float(bet.get("odds_taken") or 0.0)
    if result_norm == "win":
        profit = stake * (odds - 1.0)
        actual = 1.0
    elif result_norm == "loss":
        profit = -stake
        actual = 0.0
    else:
        profit = 0.0
        actual = None
    close = float(closing_odds) if closing_odds not in (None, 0) else None
    clv = odds / close - 1.0 if close and close > 1.0 and odds > 1.0 else None

    start = effective_starting_bankroll()
    all_bets = list_bets()
    realized_before = sum(
        float(x.get("profit_loss") or 0.0)
        for x in all_bets
        if int(x.get("id") or 0) != int(bet_id) and x.get("profit_loss") is not None
    )
    bankroll_after = (float(start) + realized_before + profit) if start is not None else None
    update = {
        "result": result_norm.title(),
        "closing_odds": close,
        "profit_loss": profit,
        "clv": clv,
        "bankroll_after": bankroll_after,
        "settled_at": datetime.now(timezone.utc).isoformat(),
    }
    _request("PATCH", "bets", params={"id": f"eq.{int(bet_id)}"}, payload=update)

    prediction_id = bet.get("prediction_id")
    if prediction_id and actual is not None:
        try:
            pred_rows = _request(
                "GET", "predictions", params={"select": "selection", "id": f"eq.{int(prediction_id)}", "limit": 1}
            )
            prediction_actual = actual
            if pred_rows:
                canonical_selection = str(pred_rows[0].get("selection") or "").strip().casefold()
                bet_selection = str(bet.get("selection") or "").strip().casefold()
                if canonical_selection and bet_selection and canonical_selection != bet_selection:
                    prediction_actual = 1.0 - actual
            _request(
                "PATCH",
                "predictions",
                params={"id": f"eq.{int(prediction_id)}"},
                payload={"actual_result": prediction_actual, "settled_at": update["settled_at"]},
            )
        except Exception:
            pass
    return {**bet, **update}


def delete_bet(bet_id: int) -> None:
    _request("DELETE", "bets", params={"id": f"eq.{int(bet_id)}"})

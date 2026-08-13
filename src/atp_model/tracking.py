from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import math
import os
import shutil
from typing import Any

from .config import ROOT


TRACKING_DIR = ROOT / "data" / "tracking"
TRACKING_FILE = TRACKING_DIR / "predictions.jsonl"
SETTINGS_FILE = TRACKING_DIR / "settings.json"
LEGACY_TRACKING_FILE = ROOT / "data" / "generated" / "predictions.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "win", "won"}:
        return True
    if text in {"0", "false", "no", "n", "loss", "lost"}:
        return False
    return None


def _no_vig_probability_a(odds_a: Any, odds_b: Any) -> float | None:
    oa = _float_or_none(odds_a)
    ob = _float_or_none(odds_b)
    if oa is None or ob is None or oa <= 1.0 or ob <= 1.0:
        return None
    ia, ib = 1.0 / oa, 1.0 / ob
    return ia / (ia + ib)


def _ensure_storage() -> None:
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve any predictions saved by pre-v4.1 versions.
    if not TRACKING_FILE.exists() and LEGACY_TRACKING_FILE.exists():
        try:
            shutil.copy2(LEGACY_TRACKING_FILE, TRACKING_FILE)
        except Exception:
            pass


def _read_raw_records() -> list[dict[str, Any]]:
    _ensure_storage()
    if not TRACKING_FILE.exists():
        return []
    records: list[dict[str, Any]] = []
    with TRACKING_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


def _atomic_write(records: list[dict[str, Any]]) -> None:
    _ensure_storage()
    tmp = TRACKING_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, TRACKING_FILE)


def _normalize_record(raw: dict[str, Any], fallback_id: int) -> dict[str, Any]:
    """Normalize both old and current tracking schemas."""
    created = raw.get("created_at") or raw.get("saved_at") or _utc_now()
    odds_a = _float_or_none(raw.get("odds_a"))
    odds_b = _float_or_none(raw.get("odds_b"))
    opening_no_vig = _float_or_none(
        raw.get("no_vig_probability_a", raw.get("market_probability_a"))
    )
    if opening_no_vig is None:
        opening_no_vig = _no_vig_probability_a(odds_a, odds_b)

    normalized = {
        "id": int(raw.get("id") or fallback_id),
        "created_at": str(created),
        "player_a": raw.get("player_a"),
        "player_b": raw.get("player_b"),
        "surface": raw.get("surface"),
        "tournament": raw.get("tournament") or "",
        "tournament_level": _float_or_none(raw.get("tournament_level")),
        "court_speed": _float_or_none(raw.get("court_speed")),
        "model_probability_a": _float_or_none(
            raw.get("model_probability_a", raw.get("probability_a"))
        ),
        "odds_a": odds_a,
        "odds_b": odds_b,
        "no_vig_probability_a": opening_no_vig,
        "edge": _float_or_none(raw.get("edge")),
        "expected_value": _float_or_none(raw.get("expected_value", raw.get("ev"))),
        "fair_odds_a": _float_or_none(raw.get("fair_odds_a")),
        "quarter_kelly": _float_or_none(raw.get("quarter_kelly")),
        "stake": _float_or_none(raw.get("stake")) or 0.0,
        "result_a": _bool_or_none(raw.get("result_a")),
        "settled_at": raw.get("settled_at"),
        "profit": _float_or_none(raw.get("profit")),
        "closing_odds_a": _float_or_none(raw.get("closing_odds_a")),
        "closing_odds_b": _float_or_none(raw.get("closing_odds_b")),
        "closing_no_vig_probability_a": _float_or_none(
            raw.get("closing_no_vig_probability_a")
        ),
        "probability_clv": _float_or_none(raw.get("probability_clv")),
        "price_clv": _float_or_none(raw.get("price_clv")),
        "notes": str(raw.get("notes") or ""),
    }
    return normalized


@dataclass(frozen=True)
class PredictionRecord:
    id: int
    created_at: str
    player_a: str | None
    player_b: str | None
    surface: str | None
    tournament: str
    tournament_level: float | None
    court_speed: float | None
    model_probability_a: float | None
    odds_a: float | None
    odds_b: float | None
    no_vig_probability_a: float | None
    edge: float | None
    expected_value: float | None
    fair_odds_a: float | None
    quarter_kelly: float | None
    stake: float
    result_a: bool | None
    settled_at: str | None
    profit: float | None
    closing_odds_a: float | None
    closing_odds_b: float | None
    closing_no_vig_probability_a: float | None
    probability_clv: float | None
    price_clv: float | None
    notes: str = ""


def get_predictions() -> list[PredictionRecord]:
    raw_records = _read_raw_records()
    normalized = [_normalize_record(r, i + 1) for i, r in enumerate(raw_records)]
    # IDs from legacy files may be missing or duplicated. Repair them deterministically.
    seen: set[int] = set()
    next_id = 1
    repaired: list[dict[str, Any]] = []
    changed = False
    for record in normalized:
        rid = int(record["id"])
        if rid in seen or rid <= 0:
            while next_id in seen:
                next_id += 1
            record["id"] = next_id
            rid = next_id
            changed = True
        seen.add(rid)
        next_id = max(next_id, rid + 1)
        repaired.append(record)
    if changed:
        _atomic_write(repaired)
    return [PredictionRecord(**r) for r in sorted(repaired, key=lambda x: x["id"])]


def save_prediction(result: dict[str, Any], odds_a: Any, odds_b: Any, stake: Any = 0.0, notes: str = "") -> int:
    records = [_normalize_record(r, i + 1) for i, r in enumerate(_read_raw_records())]
    next_id = max([int(r["id"]) for r in records], default=0) + 1
    oa = _float_or_none(odds_a)
    ob = _float_or_none(odds_b)
    market_p = _float_or_none(result.get("market_probability_a"))
    if market_p is None:
        market_p = _no_vig_probability_a(oa, ob)

    record = {
        "id": next_id,
        "created_at": _utc_now(),
        "player_a": result.get("player_a"),
        "player_b": result.get("player_b"),
        "surface": result.get("surface"),
        "tournament": result.get("tournament") or "",
        "tournament_level": _float_or_none(result.get("tournament_level")),
        "court_speed": _float_or_none(result.get("court_speed")),
        "model_probability_a": _float_or_none(result.get("probability_a")),
        "odds_a": oa,
        "odds_b": ob,
        "no_vig_probability_a": market_p,
        "edge": _float_or_none(result.get("edge")),
        "expected_value": _float_or_none(result.get("ev")),
        "fair_odds_a": _float_or_none(result.get("fair_odds_a")),
        "quarter_kelly": _float_or_none(result.get("quarter_kelly")),
        "stake": max(0.0, _float_or_none(stake) or 0.0),
        "result_a": None,
        "settled_at": None,
        "profit": None,
        "closing_odds_a": None,
        "closing_odds_b": None,
        "closing_no_vig_probability_a": None,
        "probability_clv": None,
        "price_clv": None,
        "notes": str(notes or ""),
    }
    records.append(record)
    _atomic_write(records)
    return next_id


def settle_prediction(
    prediction_id: int,
    player_a_won: bool,
    closing_odds_a: Any = None,
    closing_odds_b: Any = None,
) -> PredictionRecord:
    records = [_normalize_record(r, i + 1) for i, r in enumerate(_read_raw_records())]
    target = None
    for record in records:
        if int(record["id"]) == int(prediction_id):
            target = record
            break
    if target is None:
        raise ValueError(f"Prediction #{prediction_id} was not found.")

    target["result_a"] = bool(player_a_won)
    target["settled_at"] = _utc_now()
    stake = float(target.get("stake") or 0.0)
    opening_odds = _float_or_none(target.get("odds_a"))
    if player_a_won and opening_odds is not None:
        target["profit"] = stake * (opening_odds - 1.0)
    else:
        target["profit"] = -stake

    ca = _float_or_none(closing_odds_a)
    cb = _float_or_none(closing_odds_b)
    target["closing_odds_a"] = ca
    target["closing_odds_b"] = cb
    close_p = _no_vig_probability_a(ca, cb)
    target["closing_no_vig_probability_a"] = close_p

    open_p = _float_or_none(target.get("no_vig_probability_a"))
    target["probability_clv"] = (
        close_p - open_p if close_p is not None and open_p is not None else None
    )
    target["price_clv"] = (
        opening_odds / ca - 1.0
        if opening_odds is not None and ca is not None and opening_odds > 1.0 and ca > 1.0
        else None
    )

    _atomic_write(records)
    return PredictionRecord(**target)


def delete_prediction(prediction_id: int) -> None:
    records = [_normalize_record(r, i + 1) for i, r in enumerate(_read_raw_records())]
    kept = [r for r in records if int(r["id"]) != int(prediction_id)]
    if len(kept) == len(records):
        raise ValueError(f"Prediction #{prediction_id} was not found.")
    _atomic_write(kept)


def get_starting_bankroll() -> float | None:
    _ensure_storage()
    if not SETTINGS_FILE.exists():
        return None
    try:
        payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        value = _float_or_none(payload.get("starting_bankroll"))
        return value if value is not None and value >= 0 else None
    except Exception:
        return None


def set_starting_bankroll(amount: Any) -> float:
    _ensure_storage()
    value = _float_or_none(amount)
    if value is None or value < 0:
        raise ValueError("Starting bankroll must be zero or greater.")
    SETTINGS_FILE.write_text(
        json.dumps({"starting_bankroll": value, "updated_at": _utc_now()}, indent=2),
        encoding="utf-8",
    )
    return value


def predictions_csv() -> str:
    rows = get_predictions()
    fields = list(PredictionRecord.__dataclass_fields__.keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: getattr(row, field) for field in fields})
    return buffer.getvalue()


def restore_predictions_csv(csv_text: str, replace: bool = False) -> int:
    reader = csv.DictReader(io.StringIO(csv_text))
    incoming: list[dict[str, Any]] = []
    for i, row in enumerate(reader, start=1):
        if not row:
            continue
        incoming.append(_normalize_record(row, i))
    if not incoming:
        raise ValueError("The uploaded CSV did not contain any prediction rows.")

    if replace:
        records = incoming
    else:
        records = [_normalize_record(r, i + 1) for i, r in enumerate(_read_raw_records())]
        # Deduplicate by a stable tuple rather than trusting IDs from the backup.
        keys = {
            (r.get("created_at"), r.get("player_a"), r.get("player_b"), r.get("odds_a"))
            for r in records
        }
        next_id = max([int(r["id"]) for r in records], default=0) + 1
        for row in incoming:
            key = (row.get("created_at"), row.get("player_a"), row.get("player_b"), row.get("odds_a"))
            if key in keys:
                continue
            row["id"] = next_id
            next_id += 1
            keys.add(key)
            records.append(row)

    # Ensure unique sequential IDs after a full replacement.
    if replace:
        for i, row in enumerate(sorted(records, key=lambda x: str(x.get("created_at") or "")), start=1):
            row["id"] = i
    _atomic_write(records)
    return len(records)

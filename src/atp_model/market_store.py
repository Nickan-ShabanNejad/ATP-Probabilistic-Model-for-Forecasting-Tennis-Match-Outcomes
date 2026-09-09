from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from .config import ROOT

STORE_DIR = ROOT / "data/tracking"
SNAPSHOT_FILE = STORE_DIR / "market_snapshots.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_snapshot(
    event_id: str | int,
    start_timestamp: float | int,
    player_a: str,
    player_b: str,
    odds_a: float,
    odds_b: float,
    tournament: str = "",
    *,
    sets_over35: float | None = None,
    sets_under35: float | None = None,
) -> dict[str, Any]:
    """Append a timestamped Pinnacle snapshot for later CLV calculation."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "captured_at": _now(),
        "event_id": str(event_id),
        "start_timestamp": float(start_timestamp or 0),
        "player_a": player_a,
        "player_b": player_b,
        "odds_a": float(odds_a),
        "odds_b": float(odds_b),
        "tournament": tournament,
    }
    if sets_over35 is not None and sets_under35 is not None:
        record["sets_over35"] = float(sets_over35)
        record["sets_under35"] = float(sets_under35)
    with SNAPSHOT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def snapshots(event_id: str | int) -> list[dict[str, Any]]:
    if not SNAPSHOT_FILE.exists():
        return []
    out: list[dict[str, Any]] = []
    with SNAPSHOT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if str(row.get("event_id")) == str(event_id):
                    out.append(row)
            except Exception:
                continue
    return out


def last_pre_start_snapshot(event_id: str | int, start_timestamp: float | int | None = None) -> dict[str, Any] | None:
    rows = snapshots(event_id)
    if not rows:
        return None
    start = float(start_timestamp or rows[-1].get("start_timestamp") or 0)
    valid: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        try:
            ts = datetime.fromisoformat(str(row["captured_at"]).replace("Z", "+00:00")).timestamp()
            if not start or ts < start:
                valid.append((ts, row))
        except Exception:
            continue
    return max(valid, key=lambda x: x[0])[1] if valid else None

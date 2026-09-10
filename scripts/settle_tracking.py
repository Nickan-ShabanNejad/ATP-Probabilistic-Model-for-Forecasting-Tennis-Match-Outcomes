from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atp_model.matchstat import MatchstatClient
from atp_model.pinnodds import PinnOddsClient
from atp_model.settlement import refresh_open_bet_closes, settle_finished_tracking
from atp_model.supabase_store import configured, healthcheck


def main() -> int:
    if not configured():
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required")
    ok, status = healthcheck()
    if not ok:
        raise RuntimeError(f"Supabase healthcheck failed: {status}")

    matchstat_key = os.getenv("MATCHSTAT_API_KEY", "").strip()
    if not matchstat_key:
        raise RuntimeError("MATCHSTAT_API_KEY is required")
    client = MatchstatClient(
        api_key=matchstat_key,
        min_interval_seconds=float(os.getenv("MATCHSTAT_MIN_INTERVAL", "0.66")),
    )

    pinn_key = os.getenv("PINNODDS_API_KEY", "").strip()
    close_summary = {"updated": 0, "skipped": 0, "errors": 0}
    if pinn_key:
        pinn = PinnOddsClient(api_key=pinn_key, cache_seconds=300)
        close_summary = refresh_open_bet_closes(pinn)
    else:
        print("PINNODDS_API_KEY missing: automatic close capture skipped")

    settle_summary = settle_finished_tracking(client)
    print("close_capture", close_summary)
    print("settlement", settle_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

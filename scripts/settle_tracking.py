from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atp_model.matchstat import MatchstatClient
from atp_model.pinnodds import PinnOddsClient
from atp_model.model_service import load_bundle, load_state
from atp_model.slate import build_slate, eligible_events, tournament_context
from atp_model.settlement import finalize_prediction_closes, refresh_open_bet_closes, settle_finished_tracking
from atp_model.supabase_store import configured, healthcheck, record_detail_predictions

MODEL_VERSION = "v0.3.6"


def capture_upcoming(client: MatchstatClient, pinn: PinnOddsClient | None) -> dict[str, int]:
    """Generate/save the whole ATP 250+ slate even when Streamlit is closed."""
    events = client.upcoming_events("atp", max_events=200)
    context = tournament_context(ROOT / "data/generated/master_matches.csv.gz")
    eligible, _diag = eligible_events(
        events,
        context,
        include_qualifying=False,
        horizon_hours=36,
        today_only=False,
        timezone_name=os.getenv("APP_TIMEZONE", "America/Toronto"),
    )
    state = load_state()
    bundle = load_bundle()
    _board, detail = build_slate(
        client,
        eligible,
        state,
        bundle,
        pinnacle_client=pinn,
        bankroll=100.0,
        min_ev=0.02,
        min_edge=0.02,
        force_odds=False,
    )
    saved = 0
    errors = 0
    for match_detail in detail.values():
        try:
            record_detail_predictions(match_detail, MODEL_VERSION)
            saved += 1
        except Exception as exc:
            print("prediction_capture_error", exc)
            errors += 1
    return {"raw_events": len(events), "eligible": len(eligible), "matches_saved": saved, "errors": errors}


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
    pinn = PinnOddsClient(api_key=pinn_key, cache_seconds=300) if pinn_key else None

    capture_summary = capture_upcoming(client, pinn)

    close_summary = {"updated": 0, "skipped": 0, "errors": 0}
    if pinn is not None:
        close_summary = refresh_open_bet_closes(pinn)
    else:
        print("PINNODDS_API_KEY missing: automatic close capture skipped")

    prediction_close_summary = finalize_prediction_closes()
    settle_summary = settle_finished_tracking(client)

    print("prediction_capture", capture_summary)
    print("bet_close_capture", close_summary)
    print("prediction_close_capture", prediction_close_summary)
    print("settlement", settle_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

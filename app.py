from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from atp_model.matchstat import MatchstatClient
from atp_model.pinnodds import PinnOddsClient
from atp_model.model_service import load_bundle, load_state
from atp_model.slate import build_slate, eligible_events, tournament_context
from atp_model.tracking import current_bankroll as local_current_bankroll, get_starting_bankroll as local_get_starting_bankroll
from atp_model.supabase_store import (
    configured as supabase_configured,
    current_bankroll as supabase_current_bankroll,
    get_starting_bankroll as supabase_get_starting_bankroll,
    get_tracking_mode as supabase_get_tracking_mode,
    healthcheck as supabase_healthcheck,
    record_detail_predictions,
)

MODEL_VERSION = "v0.3.7"

st.set_page_config(page_title="ATP v0.3.7 Live Betting Board", page_icon="🎾", layout="wide")
st.title("🎾 ATP v0.3.7 — Tournament Match Feed")
st.caption(
    "Choose a date, open a tournament, then view its matches and Pinnacle prices. "
    "ATP 250 / 500 / Masters 1000 / ATP Finals / Grand Slam · model probabilities · EV · quarter-Kelly."
)


def _api_key() -> str:
    key = os.getenv("MATCHSTAT_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("MATCHSTAT_API_KEY", "")).strip()
    except Exception:
        return ""


def _pinnodds_key() -> str:
    key = os.getenv("PINNODDS_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("PINNODDS_API_KEY", "")).strip()
    except Exception:
        return ""


@st.cache_resource
def client_resource(key: str):
    return MatchstatClient(api_key=key, min_interval_seconds=float(os.getenv("MATCHSTAT_MIN_INTERVAL", "0.61")))


@st.cache_resource
def pinnacle_resource(key: str):
    return PinnOddsClient(
        api_key=key,
        base_url=os.getenv("PINNODDS_BASE_URL", "https://pinnodds.com"),
        cache_seconds=float(os.getenv("PINNODDS_CACHE_SECONDS", "1200")),
        stale_seconds=float(os.getenv("PINNODDS_STALE_SECONDS", "21600")),
        single_event_cache_seconds=float(os.getenv("PINNODDS_SINGLE_EVENT_CACHE_SECONDS", "300")),
    )


@st.cache_data
def state_resource():
    return load_state()


@st.cache_resource
def bundle_resource():
    return load_bundle()


@st.cache_data(ttl=60)
def upcoming_resource(key: str):
    # Events list changes much less often than prices. Prices are NOT cached here.
    return client_resource(key).upcoming_events("atp", max_events=200)


@st.cache_data(ttl=1800)
def context_resource():
    return tournament_context(ROOT / "data/generated/master_matches.csv.gz")


key = _api_key()
pinn_key = _pinnodds_key()
if not key:
    st.error("MATCHSTAT_API_KEY is required for the automated board.")
    st.code('MATCHSTAT_API_KEY = "your-RapidAPI-key"', language="toml")
    st.stop()

if not pinn_key:
    st.info(
        "Direct Pinnacle feed is not configured. Matchstat will still be tried as a fallback, "
        "but for reliable Pinnacle prices add PINNODDS_API_KEY to Streamlit Secrets."
    )

state = state_resource()
bundle = bundle_resource()
client = client_resource(key)
pinnacle_client = pinnacle_resource(pinn_key) if pinn_key else None
context = context_resource()
metrics = bundle.get("metrics", {})

if supabase_configured():
    tracking_mode = supabase_get_tracking_mode()
    bankroll = supabase_current_bankroll()
    if bankroll is None and tracking_mode == "currency":
        bankroll = supabase_get_starting_bankroll()
else:
    tracking_mode = "currency"
    bankroll = local_current_bankroll()
    if bankroll is None:
        bankroll = local_get_starting_bankroll()
bankroll = float(bankroll or (100.0 if tracking_mode == "percentage" else 0.0))

m1, m2, m3, m4 = st.columns(4)
if tracking_mode == "percentage":
    m1.metric("Bankroll index", f"{bankroll:,.2f}")
    m1.caption("Percentage mode · 100.00 = starting bankroll")
else:
    m1.metric("Current bankroll", f"CA${bankroll:,.2f}" if bankroll else "Not set")
m2.metric("Model", str(metrics.get("selected_model", "unknown")).replace("_", " ").title())
m3.metric("Latest match data", str(metrics.get("latest_data_date", "unknown")))
m4.metric("Model log loss", f"{float(metrics.get('log_loss', 0)):.3f}")

app_timezone = os.getenv("APP_TIMEZONE", "America/Toronto")
local_today = datetime.now(ZoneInfo(app_timezone)).date()

d1, d2, d3, d4 = st.columns([1.35, 1, 1, 2.25])
with d1:
    selected_date = st.date_input(
        "Match date",
        value=local_today,
        help="Choose the calendar day you want to browse. Times are shown in your app timezone.",
    )
with d2:
    min_ev = st.number_input("Minimum EV to call BET", min_value=0.0, max_value=0.25, value=0.02, step=0.005, format="%.3f")
with d3:
    min_edge = st.number_input("Minimum model edge", min_value=0.0, max_value=0.25, value=0.02, step=0.005, format="%.3f")
with d4:
    include_q = st.toggle("Include qualifying", value=False)
    st.caption(
        "The board refreshes every 30 seconds, but Pinnacle REST prices are cached for ~20 minutes "
        "to protect the API quota. Use Refresh slate now when you explicitly want a fresh Pinnacle pull."
    )

# If the user changes the calendar day, return to the tournament chooser instead
# of leaving a tournament from the previous date selected.
if st.session_state.get("feed_selected_date") != selected_date.isoformat():
    st.session_state["feed_selected_date"] = selected_date.isoformat()
    st.session_state.pop("selected_tournament", None)

if st.button("Refresh slate now", type="secondary"):
    upcoming_resource.clear()
    if pinnacle_client is not None:
        pinnacle_client.invalidate_prices()
    st.rerun()


@st.fragment(run_every="30s")
def live_board():
    try:
        events = upcoming_resource(key)
        eligible, diag = eligible_events(
            events,
            context,
            include_qualifying=include_q,
            horizon_hours=36,
            today_only=False,
            selected_date=selected_date,
            timezone_name=app_timezone,
        )
        board, detail = build_slate(
            client,
            eligible,
            state,
            bundle,
            pinnacle_client=pinnacle_client,
            bankroll=bankroll,
            min_ev=min_ev,
            min_edge=min_edge,
        )

        # Persist one current pre-match prediction per match/market/model version.
        # The row is updated only when its model/market snapshot changes, so the
        # 30-second live refresh does not flood Supabase with duplicates.
        tracking_sync = "local tracking"
        if supabase_configured():
            ok_db, db_status = supabase_healthcheck()
            tracking_sync = "Supabase connected" if ok_db else "Supabase configured; connection unavailable"
            signatures = st.session_state.setdefault("supabase_prediction_signatures", {})
            synced = 0
            sync_errors = []
            for event_id, match_detail in detail.items():
                if not ok_db:
                    break
                result = match_detail.get("result") or {}
                quote = match_detail.get("quote") or {}
                sets = match_detail.get("sets") or {}
                ml = quote.get("moneyline") or (None, None)
                sig = (
                    round(float(result.get("probability_a", 0) or 0), 6),
                    round(float(result.get("probability_b", 0) or 0), 6),
                    ml[0] if len(ml) > 0 else None,
                    ml[1] if len(ml) > 1 else None,
                    round(float(result.get("court_speed", 0) or 0), 4),
                    round(float(sets.get("probability_over35", 0) or 0), 6) if sets.get("available") else None,
                    sets.get("odds_over35") if sets.get("available") else None,
                    sets.get("odds_under35") if sets.get("available") else None,
                )
                if signatures.get(str(event_id)) == sig:
                    continue
                try:
                    record_detail_predictions(match_detail, MODEL_VERSION)
                    signatures[str(event_id)] = sig
                    synced += 1
                except Exception as exc:
                    sync_errors.append(f"{event_id}: {exc}")
            if sync_errors:
                tracking_sync = f"Supabase warning ({len(sync_errors)} sync error(s))"
    except Exception as exc:
        st.error(f"Could not refresh the live board: {exc}")
        return

    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    price_count = int(board["Pinnacle available"].fillna(False).sum()) if not board.empty and "Pinnacle available" in board.columns else 0
    pin_mode = "direct Pinnodds feed" if pinnacle_client is not None else "Matchstat fallback only"
    st.caption(
        f"Board refreshed {now} · {len(events)} raw ATP upcoming events received · "
        f"{len(eligible)} eligible ATP 250-or-higher events inspected · {len(board)} matches shown · "
        f"{price_count} currently have usable Pinnacle moneyline prices · "
        f"event source: {getattr(client, 'last_upcoming_source', 'cached/unknown')} · Pinnacle source: {pin_mode} · "
        f"Tracking: {tracking_sync}."
    )
    if board.empty:
        st.info("No ATP 250 / 500 / Masters / Finals / Grand Slam matches could currently be matched to the model.")
        with st.expander("Diagnostics"):
            st.write({
                "raw_events_received": len(events),
                "event_source": getattr(client, "last_upcoming_source", "cached/unknown"),
                **diag,
            })
        return

    recommended = board[board["Best market"].isin(["Moneyline", "Total sets 3.5"])]
    exposure = float(recommended["Best stake CA$"].sum()) if not recommended.empty else 0.0
    exposure_pct = float(recommended["Best Kelly %"].sum()) if not recommended.empty else 0.0
    r1, r2, r3 = st.columns(3)
    r1.metric("Matches on selected date", len(board))
    r2.metric("Recommended opportunities", len(recommended))
    if tracking_mode == "percentage":
        r3.metric("Recommended stake exposure", f"{exposure_pct:.2%} of bankroll")
    else:
        r3.metric("Recommended stake exposure", f"CA${exposure:,.2f}")

    level_labels = {2.0: "ATP 250", 3.0: "ATP 500", 4.0: "Masters 1000", 4.5: "ATP Finals", 5.0: "Grand Slam"}
    shown = board.copy()
    shown["Level label"] = shown["Level"].map(lambda x: level_labels.get(float(x), str(x)))
    shown["Start"] = pd.to_datetime(shown["Start UTC"], utc=True, errors="coerce").dt.tz_convert(
        app_timezone
    ).dt.strftime("%H:%M")

    tournaments = []
    for tournament_name, group in shown.groupby("Tournament", sort=False):
        first = group.iloc[0]
        tournaments.append({
            "name": str(tournament_name),
            "level": str(first["Level label"]),
            "surface": str(first["Surface"]),
            "matches": len(group),
            "first_start": str(group["Start"].min()),
        })
    tournaments.sort(key=lambda x: (x["first_start"], x["name"].casefold()))

    active_tournament = st.session_state.get("selected_tournament")
    available_names = {x["name"] for x in tournaments}
    if active_tournament not in available_names:
        active_tournament = None
        st.session_state.pop("selected_tournament", None)

    if active_tournament is None:
        st.subheader(f"Tournaments — {selected_date.strftime('%B %d, %Y')}")
        st.caption("Choose a tournament to open its match feed.")
        cols = st.columns(3)
        for i, tournament in enumerate(tournaments):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"### 🏆 {tournament['name']}")
                    st.write(f"{tournament['level']} · {tournament['surface']}")
                    st.caption(
                        f"{tournament['matches']} match{'es' if tournament['matches'] != 1 else ''} · "
                        f"first start {tournament['first_start']}"
                    )
                    if st.button(
                        "View matches →",
                        key=f"open_tournament_{i}_{tournament['name']}",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state["selected_tournament"] = tournament["name"]
                        st.rerun()
        with st.expander("Slate diagnostics"):
            st.json(diag)
        return

    if st.button("← All tournaments", type="secondary"):
        st.session_state.pop("selected_tournament", None)
        st.rerun()

    tournament_board = shown[shown["Tournament"] == active_tournament].copy()
    first = tournament_board.iloc[0]
    st.subheader(active_tournament)
    st.caption(
        f"{first['Level label']} · {first['Surface']} · {selected_date.strftime('%B %d, %Y')} · "
        f"{len(tournament_board)} match{'es' if len(tournament_board) != 1 else ''}"
    )

    h1, h2, h3, h4, h5 = st.columns([0.85, 2.9, 1.55, 1.35, 2.2])
    h1.markdown("**Start**")
    h2.markdown("**Match**")
    h3.markdown("**Most likely outcome**")
    h4.markdown("**Pinnacle ML**")
    h5.markdown("**What to bet / value bet**")

    for _, r in tournament_board.iterrows():
        eid = str(r["Event ID"])
        match_detail = detail.get(eid, {})
        result = match_detail.get("result") or {}
        quote = match_detail.get("quote") or {}
        sets = match_detail.get("sets") or {}

        try:
            pa = float(result.get("probability_a"))
            pb = float(result.get("probability_b"))
            if pa >= pb:
                likely_name, likely_prob = str(result.get("player_a") or ""), pa
            else:
                likely_name, likely_prob = str(result.get("player_b") or ""), pb
        except Exception:
            likely_name = str(r.get("ML pick") or "—")
            likely_prob = float(r.get("ML pick P")) if pd.notna(r.get("ML pick P")) else float("nan")

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([0.85, 2.9, 1.55, 1.35, 2.2])
            c1.write(str(r["Start"]))
            with c2:
                st.caption(f"BO{int(r['BO'])}")
                if st.button(str(r["Match"]), key=f"open_match_{eid}", use_container_width=True):
                    if eid in detail:
                        st.session_state["selected_event_id"] = eid
                        st.session_state["selected_match_detail"] = detail[eid]
                        st.switch_page("pages/1_Match_Detail.py")

            if pd.notna(likely_prob):
                c3.write(f"**{likely_name}**\n\n{likely_prob:.1%}")
            else:
                c3.write(likely_name or "—")

            q = quote.get("moneyline")
            if q and len(q) == 2:
                c4.write(f"{float(q[0]):.3f} / {float(q[1]):.3f}")
            else:
                c4.write("Unavailable")

            if str(r["Best market"]) != "No bet":
                market_label = "ML" if str(r["Best market"]) == "Moneyline" else "O/U 3.5 sets"
                c5.write(
                    f"**BET: {r['Best selection']}** · {market_label}\n\n"
                    f"EV {float(r['Best EV']):+.1%} · edge {float(r['Best Edge']):+.1%}\n\n"
                    f"Quarter-Kelly {float(r['Best Kelly %']):.2%} · "
                    + (f"stake {float(r['Best Kelly %']):.2%} bankroll" if tracking_mode == "percentage" else f"stake CA${float(r['Best stake CA$']):,.2f}")
                )
            else:
                candidates = []
                if q and len(q) == 2:
                    for name, ev_key, edge_key in (
                        (result.get("player_a"), "ev_a", "edge_a"),
                        (result.get("player_b"), "ev_b", "edge_b"),
                    ):
                        try:
                            ev, edge = float(result.get(ev_key)), float(result.get(edge_key))
                            if pd.notna(ev) and pd.notna(edge):
                                candidates.append((ev, edge, str(name), "ML"))
                        except Exception:
                            pass
                # Total Sets 3.5 is intentionally considered only when the slate
                # marked this as a BO5 Grand Slam match.
                is_grand_slam = float(r.get("Level", 0) or 0) == 5.0 and int(r.get("BO", 0) or 0) == 5
                if is_grand_slam and sets and sets.get("available") and sets.get("odds_over35") and sets.get("odds_under35"):
                    for name, ev_key, edge_key in (
                        ("Over 3.5", "ev_over35", "edge_over35"),
                        ("Under 3.5", "ev_under35", "edge_under35"),
                    ):
                        try:
                            ev, edge = float(sets.get(ev_key)), float(sets.get(edge_key))
                            if pd.notna(ev) and pd.notna(edge):
                                candidates.append((ev, edge, name, "O/U 3.5 sets"))
                        except Exception:
                            pass
                if candidates:
                    best_ev_raw, best_edge_raw, best_name_raw, best_market_raw = max(candidates, key=lambda x: x[0])
                    c5.write(
                        f"**NO BET at current price**\n\n"
                        f"Best candidate: {best_name_raw} · {best_market_raw}\n\n"
                        f"EV {best_ev_raw:+.1%} · edge {best_edge_raw:+.1%} "
                        f"(needs ≥{min_ev:.1%} EV and ≥{min_edge:.1%} edge)"
                    )
                else:
                    c5.write("**NO BET**\n\nNo usable Pinnacle price yet.")

    with st.expander("Slate diagnostics"):
        st.json(diag)
        st.write(
            "Total Sets 3.5 is disabled for ATP 250, ATP 500, Masters 1000 and ATP Finals. "
            "It is evaluated only for BO5 Grand Slam matches."
        )


live_board()

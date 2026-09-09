from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from atp_model.matchstat import MatchstatClient
from atp_model.pinnodds import PinnOddsClient
from atp_model.model_service import load_bundle, load_state
from atp_model.slate import build_slate, eligible_events, tournament_context
from atp_model.tracking import current_bankroll, get_starting_bankroll

st.set_page_config(page_title="ATP v0.3.2 Live Betting Board", page_icon="🎾", layout="wide")
st.title("🎾 ATP v0.3.2 — Live Value Board")
st.caption(
    "Automated ATP 250 / 500 / Masters 1000 / ATP Finals / Grand Slam slate · Pinnacle prices · "
    "model probabilities · EV · quarter-Kelly · bankroll stakes · click any match for the full breakdown."
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
        cache_seconds=float(os.getenv("PINNODDS_CACHE_SECONDS", "20")),
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

bankroll = current_bankroll()
if bankroll is None:
    bankroll = float(get_starting_bankroll() or 0.0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current bankroll", f"CA${bankroll:,.2f}" if bankroll else "Not set")
m2.metric("Model", str(metrics.get("selected_model", "unknown")).replace("_", " ").title())
m3.metric("Latest match data", str(metrics.get("latest_data_date", "unknown")))
m4.metric("Model log loss", f"{float(metrics.get('log_loss', 0)):.3f}")

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
with c1:
    min_ev = st.number_input("Minimum EV to call BET", min_value=0.0, max_value=0.25, value=0.02, step=0.005, format="%.3f")
with c2:
    min_edge = st.number_input("Minimum model edge", min_value=0.0, max_value=0.25, value=0.02, step=0.005, format="%.3f")
with c3:
    include_q = st.toggle("Include qualifying", value=False)
with c4:
    today_only = st.toggle("Today only", value=True)
with c5:
    st.caption(
        "Odds refresh while this page is open. Starts <1h: ~30s · 1–6h: ~60s · later: ~3 min. "
        "The event list refreshes every ~60 seconds with backup Matchstat feeds."
    )

if st.button("Refresh slate now", type="secondary"):
    upcoming_resource.clear()
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
            today_only=today_only,
            timezone_name=os.getenv("APP_TIMEZONE", "America/Toronto"),
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
        f"event source: {getattr(client, 'last_upcoming_source', 'cached/unknown')} · Pinnacle source: {pin_mode}."
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
    r1, r2, r3 = st.columns(3)
    r1.metric("Matches on board", len(board))
    r2.metric("Recommended opportunities", len(recommended))
    r3.metric("Recommended stake exposure", f"CA${exposure:,.2f}")

    level_labels = {2.0: "250", 3.0: "500", 4.0: "Masters", 4.5: "Finals", 5.0: "Grand Slam"}
    shown = board.copy()
    shown["Level label"] = shown["Level"].map(lambda x: level_labels.get(float(x), str(x)))
    shown["Start"] = pd.to_datetime(shown["Start UTC"], utc=True, errors="coerce").dt.tz_convert(
        os.getenv("APP_TIMEZONE", "America/Toronto")
    ).dt.strftime("%b %d %H:%M")

    # Table-like clickable rows. The match name itself is the navigation control,
    # so every event can be opened directly without a second picker underneath.
    h1, h2, h3, h4, h5, h6 = st.columns([1.0, 1.5, 3.2, 1.25, 1.45, 1.7])
    h1.markdown("**Start**")
    h2.markdown("**Tournament**")
    h3.markdown("**Match**")
    h4.markdown("**Model**")
    h5.markdown("**Pinnacle**")
    h6.markdown("**Recommendation**")

    for i, r in shown.iterrows():
        eid = str(r["Event ID"])
        with st.container(border=True):
            c1, c2, c3, c4, c5, c6 = st.columns([1.0, 1.5, 3.2, 1.25, 1.45, 1.7])
            c1.write(str(r["Start"]))
            c2.write(f"{r['Tournament']}\n\n{r['Level label']} · BO{int(r['BO'])}")
            with c3:
                if st.button(str(r["Match"]), key=f"open_match_{eid}", use_container_width=True):
                    if eid in detail:
                        st.session_state["selected_event_id"] = eid
                        st.session_state["selected_match_detail"] = detail[eid]
                        st.switch_page("pages/1_Match_Detail.py")
            c4.write(f"{r['ML pick']}\n\n{float(r['ML pick P']):.1%}" if pd.notna(r['ML pick P']) else str(r['ML pick']))
            if bool(r.get("Pinnacle available", False)):
                q = detail.get(eid, {}).get("quote", {}).get("moneyline")
                if q:
                    c5.write(f"{q[0]:.3f} / {q[1]:.3f}")
                else:
                    c5.write("Available")
            else:
                c5.write("Unavailable")
            if str(r["Best market"]) != "No bet":
                c6.write(
                    f"**{r['Best selection']}**\n\n"
                    f"EV {float(r['Best EV']):+.1%} · Kelly {float(r['Best Kelly %']):.2%}\n\n"
                    f"CA${float(r['Best stake CA$']):,.2f}"
                )
            else:
                c6.write("No bet")

    with st.expander("Slate diagnostics"):
        st.json(diag)
        st.write(
            "Grand Slam O/U 3.5 recommendations appear only when the separate sets model is trained. "
            "A Pinnacle totals price is required before it can become a bet recommendation."
        )


live_board()

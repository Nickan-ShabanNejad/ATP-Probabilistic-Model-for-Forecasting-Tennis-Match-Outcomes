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
from atp_model.model_service import load_bundle, load_state
from atp_model.slate import build_slate, eligible_events, tournament_context
from atp_model.tracking import current_bankroll, get_starting_bankroll

st.set_page_config(page_title="ATP v0.3 Live Betting Board", page_icon="🎾", layout="wide")
st.title("🎾 ATP v0.3 — Live Value Board")
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


@st.cache_resource
def client_resource(key: str):
    return MatchstatClient(api_key=key, min_interval_seconds=float(os.getenv("MATCHSTAT_MIN_INTERVAL", "0.61")))


@st.cache_data
def state_resource():
    return load_state()


@st.cache_resource
def bundle_resource():
    return load_bundle()


@st.cache_data(ttl=300)
def upcoming_resource(key: str):
    # Events list changes much less often than prices. Prices are NOT cached here.
    return client_resource(key).upcoming_events("atp", max_events=200)


@st.cache_data(ttl=1800)
def context_resource():
    return tournament_context(ROOT / "data/generated/master_matches.csv.gz")


key = _api_key()
if not key:
    st.error("MATCHSTAT_API_KEY is required for the automated board.")
    st.code('MATCHSTAT_API_KEY = "your-RapidAPI-key"', language="toml")
    st.stop()

state = state_resource()
bundle = bundle_resource()
client = client_resource(key)
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
        "The event list refreshes every 5 minutes."
    )


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
            bankroll=bankroll,
            min_ev=min_ev,
            min_edge=min_edge,
        )
    except Exception as exc:
        st.error(f"Could not refresh the live board: {exc}")
        return

    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    price_count = int(board["Pinnacle available"].fillna(False).sum()) if not board.empty and "Pinnacle available" in board.columns else 0
    st.caption(
        f"Board refreshed {now} · {len(eligible)} eligible ATP 250-or-higher events inspected · "
        f"{len(board)} matches shown · {price_count} currently have usable Pinnacle moneyline prices."
    )
    if board.empty:
        st.info("No ATP 250 / 500 / Masters / Finals / Grand Slam matches could currently be matched to the model.")
        with st.expander("Diagnostics"):
            st.json(diag)
        return

    recommended = board[board["Best market"].isin(["Moneyline", "Total sets 3.5"])]
    exposure = float(recommended["Best stake CA$"].sum()) if not recommended.empty else 0.0
    r1, r2, r3 = st.columns(3)
    r1.metric("Matches on board", len(board))
    r2.metric("Recommended opportunities", len(recommended))
    r3.metric("Recommended stake exposure", f"CA${exposure:,.2f}")

    level_labels = {2.0: "250", 3.0: "500", 4.0: "Masters", 4.5: "Finals", 5.0: "Grand Slam"}
    shown = board.copy()
    shown["Level"] = shown["Level"].map(lambda x: level_labels.get(float(x), str(x)))
    shown["Start"] = pd.to_datetime(shown["Start UTC"], utc=True, errors="coerce").dt.tz_convert(
        os.getenv("APP_TIMEZONE", "America/Toronto")
    ).dt.strftime("%b %d %H:%M")

    display_cols = [
        "Start", "Tournament", "Level", "Match", "BO", "Status",
        "Best selection", "Best market", "Best price", "Best Edge", "Best EV", "Best Kelly %", "Best stake CA$",
        "ML pick", "ML pick P", "O3.5 P", "O3.5 odds", "U3.5 odds", "Odds age s",
    ]
    selection = st.dataframe(
        shown[display_cols],
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Best price": st.column_config.NumberColumn(format="%.3f"),
            "Best Edge": st.column_config.NumberColumn(format="%+.1%%"),
            "Best EV": st.column_config.NumberColumn(format="%+.1%%"),
            "Best Kelly %": st.column_config.NumberColumn(format="%.2%%"),
            "Best stake CA$": st.column_config.NumberColumn(format="CA$%.2f"),
            "ML pick P": st.column_config.NumberColumn(format="%.1%%"),
            "O3.5 P": st.column_config.NumberColumn(format="%.1%%"),
            "O3.5 odds": st.column_config.NumberColumn(format="%.3f"),
            "U3.5 odds": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    selected = list(selection.selection.rows) if hasattr(selection, "selection") else []
    if selected:
        pos = int(selected[0])
        eid = str(board.iloc[pos]["Event ID"])
        if eid in detail:
            st.session_state["selected_event_id"] = eid
            st.session_state["selected_match_detail"] = detail[eid]
            st.switch_page("pages/1_Match_Detail.py")

    # Explicit navigation fallback. Streamlit row-selection behavior can vary when
    # a dataframe lives inside a timed fragment, so the app also exposes a reliable
    # match picker + button. This works even when Pinnacle odds are unavailable.
    event_ids = [str(x) for x in board["Event ID"].tolist() if str(x) in detail]
    if event_ids:
        label_by_id = {
            str(r["Event ID"]): f"{r['Match']} — {r['Tournament']}"
            for _, r in board.iterrows() if str(r["Event ID"]) in detail
        }
        d1, d2 = st.columns([4, 1])
        with d1:
            chosen_eid = st.selectbox(
                "Open detailed matchup",
                event_ids,
                format_func=lambda x: label_by_id.get(str(x), str(x)),
                key="detail_match_picker",
            )
        with d2:
            st.write("")
            st.write("")
            if st.button("Open details →", use_container_width=True, key="open_match_detail"):
                chosen_eid = str(chosen_eid)
                st.session_state["selected_event_id"] = chosen_eid
                st.session_state["selected_match_detail"] = detail[chosen_eid]
                st.switch_page("pages/1_Match_Detail.py")

    with st.expander("Slate diagnostics"):
        st.json(diag)
        st.write(
            "Grand Slam O/U 3.5 recommendations appear only when the separate sets model is trained. "
            "A Pinnacle totals price is required before it can become a bet recommendation."
        )


live_board()

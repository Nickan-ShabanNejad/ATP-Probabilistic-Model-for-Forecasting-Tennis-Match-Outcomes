from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
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
from atp_model.ui import APP_VERSION, SURFACE_EMOJI, badge, esc, hero, kpi, level_label, load_metrics, prob_bar, setup_page

MODEL_VERSION = APP_VERSION
setup_page("ATP Value Board")


# ---------------------------------------------------------------------------
# Secrets and cached resources
# ---------------------------------------------------------------------------
def _secret(name: str) -> str:
    key = os.getenv(name, "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get(name, "")).strip()
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
    return client_resource(key).upcoming_events("atp", max_events=200)


@st.cache_data(ttl=1800)
def context_resource():
    return tournament_context(ROOT / "data/generated/master_matches.csv.gz")


DEMO = os.getenv("ATP_DEMO", "").strip() == "1"
key = "demo" if DEMO else _secret("MATCHSTAT_API_KEY")
pinn_key = "demo" if DEMO else _secret("PINNODDS_API_KEY")
if not key:
    hero("ATP Value Board", "Add your data key to start.")
    st.error("MATCHSTAT_API_KEY is required for the live board. Add it to Streamlit Secrets:")
    st.code('MATCHSTAT_API_KEY = "your-RapidAPI-key"\nPINNODDS_API_KEY = "optional-direct-pinnacle-key"', language="toml")
    st.stop()

state = state_resource()
bundle = bundle_resource()
if DEMO:
    from atp_model.demo import DemoMatchstatClient, DemoPinnacleClient

    @st.cache_resource
    def demo_clients():
        c = DemoMatchstatClient(state)
        return c, DemoPinnacleClient(c._elo)

    client, pinnacle_client = demo_clients()
    upcoming_resource = st.cache_data(ttl=60)(lambda _k: client.upcoming_events("atp"))
else:
    client = client_resource(key)
    pinnacle_client = pinnacle_resource(pinn_key) if pinn_key else None
context = context_resource()
metrics = load_metrics() or bundle.get("metrics", {})
app_timezone = os.getenv("APP_TIMEZONE", "America/Toronto")
tz = ZoneInfo(app_timezone)

# Bankroll ------------------------------------------------------------------
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


def stake_text(fraction: float) -> str:
    if tracking_mode == "percentage" or not bankroll:
        return f"{fraction:.2%} of bankroll"
    return f"CA${bankroll * fraction:,.2f} ({fraction:.2%})"


# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode_label = st.radio(
        "Betting mode",
        ["Market-anchored (recommended)", "Model only (experimental)"],
        help=(
            "Market-anchored combines the model with Pinnacle's price using weights learned from 2018-2025 "
            "closing odds. It only calls a bet when a price is better than that fair value. "
            "Model only uses the raw model and flags many more bets; historically that lost money."
        ),
    )
    strategy = "market" if mode_label.startswith("Market") else "model"
    min_ev_pct = st.slider(
        "Minimum expected value", 0.0, 15.0, 2.0 if strategy == "market" else 5.0, 0.5, format="%.1f%%",
        help="Only call a bet when the expected return on the stake is at least this much.",
    )
    min_edge_pct = 0.0
    if strategy == "model":
        min_edge_pct = st.slider(
            "Minimum probability edge", 0.0, 15.0, 5.0, 0.5, format="%.1f%%",
            help="Model probability minus Pinnacle's no-vig probability.",
        )
    kelly_fraction = st.select_slider(
        "Kelly fraction", options=[0.1, 0.2, 0.25, 0.33, 0.5], value=0.25,
        format_func=lambda x: {0.1: "1/10", 0.2: "1/5", 0.25: "1/4", 0.33: "1/3", 0.5: "1/2"}[x],
        help="Share of the full Kelly stake to bet. Quarter-Kelly is a common, conservative choice.",
    )
    kelly_cap_pct = st.slider("Max stake per bet", 0.5, 5.0, 2.0, 0.5, format="%.1f%%", help="Hard cap as a share of bankroll.")
    include_q = st.toggle("Include qualifying", value=False)
    st.divider()
    if st.button("🔄 Refresh prices now", use_container_width=True):
        upcoming_resource.clear()
        if pinnacle_client is not None:
            pinnacle_client.invalidate_prices()
        st.rerun()
    st.caption(
        "The board refreshes every 30 s. Pinnacle prices are cached ~20 min to protect the API quota; "
        "use Refresh to force a new pull."
    )
    if not pinn_key:
        st.warning("PINNODDS_API_KEY not set: Pinnacle prices fall back to Matchstat and may be missing.")

min_ev, min_edge = min_ev_pct / 100.0, min_edge_pct / 100.0
kelly_cap = kelly_cap_pct / 100.0

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
bench = metrics.get("market_benchmark") or {}
latest_data = str(metrics.get("latest_data_date", "unknown"))
hero(
    "🎾 ATP Value Board",
    "Fair odds for every ATP 250 / 500 / Masters / Finals / Grand Slam match, compared with Pinnacle.",
    [
        (f"{APP_VERSION}", "blue"),
        (f"Data to {latest_data}", ""),
        ("Market-anchored" if strategy == "market" else "Model only · experimental", "ok" if strategy == "market" else "warn"),
    ] + ([("DEMO DATA · synthetic prices", "bad")] if DEMO else []),
)

if strategy == "model":
    bt = (metrics.get("backtests") or {}).get("model_only_ev2_edge2") or {}
    if bt.get("bets"):
        st.warning(
            f"**Model-only mode is experimental.** Replayed against Pinnacle closing odds for "
            f"{min(metrics.get('walk_forward_years') or [0])}–{max(metrics.get('walk_forward_years') or [0])}, "
            f"this rule flagged {bt['share_of_matches']:.0%} of matches and returned "
            f"**{bt['flat_roi']:+.1%}** per unit staked ({bt['bets']:,} bets). Use it to track ideas, not to stake real money."
        )

window_mode = st.segmented_control("Show", ["Next 48 hours", "Pick a date"], default="Next 48 hours", label_visibility="collapsed")
selected_date = None
if window_mode == "Pick a date":
    selected_date = st.date_input("Match date (your time zone)", value=datetime.now(tz).date())


# ---------------------------------------------------------------------------
# Live board
# ---------------------------------------------------------------------------
def _sync_supabase(detail: dict) -> str:
    if not supabase_configured():
        return "local tracking"
    ok_db, _ = supabase_healthcheck()
    if not ok_db:
        return "Supabase unavailable"
    signatures = st.session_state.setdefault("supabase_prediction_signatures", {})
    errors = 0
    for event_id, match_detail in detail.items():
        result = match_detail.get("result") or {}
        quote = match_detail.get("quote") or {}
        sets = match_detail.get("sets") or {}
        ml = quote.get("moneyline") or (None, None)
        sig = (
            round(float(result.get("probability_a", 0) or 0), 6),
            ml[0] if len(ml) > 0 else None,
            ml[1] if len(ml) > 1 else None,
            sets.get("odds_over35") if sets.get("available") else None,
            sets.get("odds_under35") if sets.get("available") else None,
        )
        if signatures.get(str(event_id)) == sig:
            continue
        try:
            record_detail_predictions(match_detail, MODEL_VERSION)
            signatures[str(event_id)] = sig
        except Exception:
            errors += 1
    return "Supabase connected" + (f" ({errors} sync errors)" if errors else "")


def _open_match(eid: str, detail: dict) -> None:
    if eid in detail:
        st.session_state["selected_event_id"] = eid
        st.session_state["selected_match_detail"] = detail[eid]
        st.switch_page("pages/1_Match_Detail.py")


def _empty_state(events: list, diag: dict) -> None:
    reasons = []
    if diag.get("already_started"):
        reasons.append(f"{diag['already_started']} already started or finished")
    if diag.get("wrong_date"):
        reasons.append(f"{diag['wrong_date']} are on a different day in your time zone")
    if diag.get("outside_window"):
        reasons.append(f"{diag['outside_window']} start more than 48 h from now")
    if diag.get("lower_tier"):
        reasons.append(f"{diag['lower_tier']} are Challenger/ITF")
    if diag.get("qualifying"):
        reasons.append(f"{diag['qualifying']} are qualifying (toggle it on in Settings)")
    if diag.get("unknown_context"):
        names = ", ".join(diag.get("unknown_tournaments", [])[:4])
        reasons.append(f"{diag['unknown_context']} are at tournaments the model doesn't recognise" + (f" ({names})" if names else ""))
    if diag.get("bad_status"):
        reasons.append(f"{diag['bad_status']} are live or finished")
    tip = ""
    if selected_date is not None and diag.get("wrong_date"):
        tip = ("<br><br>💡 Asian and Australian events often start overnight in North America, so "
               "tomorrow's matches can fall on a different calendar day. Switch to <b>Next 48 hours</b>.")
    body = "; ".join(reasons) if reasons else "the feed returned no ATP events"
    st.markdown(
        f'<div class="empty"><b>No tour-level matches to show.</b><br>'
        f'The feed returned {len(events)} ATP events: {esc(body)}.{tip}</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every="30s")
def live_board():
    try:
        events = upcoming_resource(key)
        eligible, diag = eligible_events(
            events, context, include_qualifying=include_q, horizon_hours=48,
            today_only=False, selected_date=selected_date, timezone_name=app_timezone,
        )
        board, detail = build_slate(
            client, eligible, state, bundle, pinnacle_client=pinnacle_client, bankroll=bankroll,
            min_ev=min_ev, min_edge=min_edge, strategy=strategy, kelly_fraction=kelly_fraction, kelly_cap=kelly_cap,
        )
        tracking_sync = _sync_supabase(detail)
    except Exception as exc:
        st.error(f"Could not refresh the live board: {exc}")
        return

    priced = int(board["Pinnacle available"].fillna(False).sum()) if not board.empty else 0
    bets = board[board["Best market"].isin(["Moneyline", "Total sets 3.5"])] if not board.empty else board
    exposure = float(bets["Best Kelly %"].sum()) if not bets.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    if tracking_mode == "percentage":
        kpi(k1, "Bankroll index", f"{bankroll:,.2f}", "100 = starting bankroll")
    elif bankroll:
        kpi(k1, "Bankroll", f"CA${bankroll:,.0f}", "stakes shown in CA$")
    else:
        kpi(k1, "Bankroll", "Not set", "set it on Tracking & CLV")
    kpi(k2, "Value bets now", str(len(bets)), f"{exposure:.1%} of bankroll at stake" if len(bets) else "nothing beats fair odds")
    kpi(k3, "Matches", str(len(board)), f"{priced} with a Pinnacle price")
    if bench.get("available"):
        kpi(k4, "Model vs market", f"{bench['model_log_loss']:.3f} / {bench['pinnacle_log_loss']:.3f}",
            "log loss, model / Pinnacle (lower wins)")
    else:
        kpi(k4, "Model log loss", f"{float(metrics.get('log_loss', 0)):.3f}", f"holdout {metrics.get('holdout', '')}")

    st.caption(
        f"Updated {datetime.now(tz).strftime('%H:%M:%S')} · {len(events)} feed events · "
        f"Pinnacle: {'direct feed' if pinnacle_client else 'Matchstat fallback'} · Tracking: {tracking_sync}"
    )

    if board.empty:
        _empty_state(events, diag)
        with st.expander("Technical diagnostics"):
            st.json({"raw_events": len(events), "event_source": getattr(client, "last_upcoming_source", "?"), **diag})
        return

    shown = board.copy()
    shown["start_dt"] = pd.to_datetime(shown["Start UTC"], utc=True, errors="coerce").dt.tz_convert(app_timezone)
    shown["Start"] = shown["start_dt"].dt.strftime("%a %H:%M")

    # ----- Recommended bets -------------------------------------------------
    st.markdown("### ✅ Recommended bets")
    if bets.empty:
        msg = (
            "No price currently beats our fair odds by your minimum EV. That is the normal state against "
            "Pinnacle, whose prices are very sharp. If your own sportsbook offers a better price, open a match "
            "and type it into <b>Check your price</b>."
            if strategy == "market" else
            "No match passes your EV and edge thresholds right now."
        )
        st.markdown(f'<div class="empty">{msg}</div>', unsafe_allow_html=True)
    else:
        cols = st.columns(min(3, len(bets)))
        for i, (_, r) in enumerate(shown[shown["Event ID"].isin(bets["Event ID"])].iterrows()):
            eid = str(r["Event ID"])
            res = (detail.get(eid) or {}).get("result") or {}
            is_ml = r["Best market"] == "Moneyline"
            fair = None
            if is_ml:
                fair = res.get("fair_odds_a") if r["Best selection"] == res.get("player_a") else res.get("fair_odds_b")
            with cols[i % len(cols)]:
                st.markdown(
                    f'<div class="bet-card"><div class="pick">{esc(r["Best selection"])} @ {float(r["Best price"]):.2f}</div>'
                    f'<div class="meta">{esc(r["Match"])}<br>{esc(r["Tournament"])} · {esc(r["Start"])}'
                    f'{" · Total sets 3.5" if not is_ml else ""}</div>'
                    f'<div class="nums"><div><span>EV</span><b>{float(r["Best EV"]):+.1%}</b></div>'
                    f'<div><span>Fair odds</span><b>{(f"{float(fair):.2f}" if fair else "—")}</b></div>'
                    f'<div><span>Stake</span><b>{esc(stake_text(float(r["Best Kelly %"])))}</b></div></div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Open match →", key=f"bet_{eid}", use_container_width=True):
                    _open_match(eid, detail)

    # ----- All matches by tournament ----------------------------------------
    st.markdown("### 📋 All matches")
    order = shown.groupby("Tournament", sort=False)["start_dt"].min().sort_values().index
    for t_i, tournament in enumerate(order):
        group = shown[shown["Tournament"] == tournament].sort_values("start_dt")
        first = group.iloc[0]
        n_bets = int(group["Best market"].isin(["Moneyline", "Total sets 3.5"]).sum())
        label = (f"{SURFACE_EMOJI.get(str(first['Surface']), '🎾')} {tournament} · {level_label(first['Level'])} · "
                 f"{first['Surface']} · {len(group)} match{'es' if len(group) != 1 else ''}"
                 + (f" · ✅ {n_bets} bet{'s' if n_bets != 1 else ''}" if n_bets else ""))
        with st.expander(label, expanded=t_i < 2):
            h = st.columns([0.9, 3.2, 1.3, 1.3, 2.2, 0.9])
            for c, text in zip(h, ["Start", "Match · fair win chance", "Pinnacle", "Fair odds", "Verdict", ""]):
                c.markdown(f'<span class="small-muted">{text}</span>', unsafe_allow_html=True)
            for _, r in group.iterrows():
                eid = str(r["Event ID"])
                md = detail.get(eid) or {}
                res = md.get("result") or {}
                q = (md.get("quote") or {}).get("moneyline")
                c = st.columns([0.9, 3.2, 1.3, 1.3, 2.2, 0.9], vertical_alignment="center")
                c[0].markdown(f"**{esc(r['Start'])}**", unsafe_allow_html=True)
                if not res:
                    c[1].markdown(f"{esc(r['Match'])}<br>{badge('No rating data for a player', 'warn')}", unsafe_allow_html=True)
                    c[2].write("—"); c[3].write("—"); c[4].write("—")
                    continue
                pa, pb = float(res["probability_a"]), float(res["probability_b"])
                c[1].markdown(
                    f'<div class="match-row"><div class="players">{esc(res["player_a"])} <span class="small-muted">{pa:.0%}</span>'
                    f' &nbsp;vs&nbsp; {esc(res["player_b"])} <span class="small-muted">{pb:.0%}</span></div>{prob_bar(pa)}</div>',
                    unsafe_allow_html=True,
                )
                c[2].markdown(f"{float(q[0]):.2f} / {float(q[1]):.2f}" if q and len(q) == 2 else '<span class="small-muted">no price</span>', unsafe_allow_html=True)
                c[3].markdown(f"{float(res['fair_odds_a']):.2f} / {float(res['fair_odds_b']):.2f}")
                if r["Best market"] != "No bet":
                    c[4].markdown(badge(f"BET {r['Best selection']} · EV {float(r['Best EV']):+.1%}", "ok"), unsafe_allow_html=True)
                elif q:
                    evs = [x for x in (res.get("ev_a"), res.get("ev_b")) if x is not None and np.isfinite(x)]
                    best = max(evs) if evs else float("nan")
                    c[4].markdown(badge(f"No bet · best EV {best:+.1%}", ""), unsafe_allow_html=True)
                else:
                    fav = res["player_a"] if pa >= pb else res["player_b"]
                    c[4].markdown(badge(f"Lean {fav} · waiting for price", "blue"), unsafe_allow_html=True)
                if c[5].button("Details", key=f"open_{eid}"):
                    _open_match(eid, detail)

    with st.expander("Technical diagnostics"):
        st.json({"raw_events": len(events), "event_source": getattr(client, "last_upcoming_source", "?"), **diag})
        st.caption("Total Sets 3.5 is evaluated only for best-of-five Grand Slam matches.")


live_board()

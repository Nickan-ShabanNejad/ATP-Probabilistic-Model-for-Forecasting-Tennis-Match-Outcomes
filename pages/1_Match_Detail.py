from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import requests

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atp_model.matchstat import MatchstatClient
from atp_model.model_service import court_speed_curve, load_bundle, load_state
from atp_model.tournament_features import court_speed_label
from atp_model.tracking import current_bankroll as local_current_bankroll, save_prediction as local_save_prediction
from atp_model.supabase_store import (
    configured as supabase_configured,
    current_bankroll as supabase_current_bankroll,
    get_tracking_mode as supabase_get_tracking_mode,
    find_open_bet,
    place_bet,
    prediction_payload_from_detail,
    upsert_prediction,
)
from atp_model.sets_service import predict_over35

MODEL_VERSION = "v0.3.4"

st.set_page_config(page_title="ATP Match Detail", page_icon="🎾", layout="wide")


# v0.3.2c: direct PinnOdds bridge for Grand Slam Total Sets 3.5.
# This deliberately does not depend on PinnOddsClient.find_total_sets_35 so a stale
# imported client class cannot disable the totals market on Streamlit Cloud.
def _secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def _tracking_mode() -> str:
    if supabase_configured():
        try:
            return supabase_get_tracking_mode()
        except Exception:
            pass
    return "currency"


def _tracking_bankroll() -> float:
    if supabase_configured():
        value = supabase_current_bankroll()
        if value is not None:
            return float(value)
    return float(local_current_bankroll() or 0.0)


def _norm_player(value: str) -> str:
    import unicodedata, re as _re
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = _re.sub(r"[^a-z0-9]+", " ", s).strip()
    return " ".join(s.split())


def _same_player(left: str, right: str) -> bool:
    a, b = _norm_player(left), _norm_player(right)
    if not a or not b:
        return False
    if a == b:
        return True
    ap, bp = a.split(), b.split()
    return len(ap) >= 2 and len(bp) >= 2 and ap[-1] == bp[-1] and ap[0][0] == bp[0][0]


def _event_timestamp(row: dict) -> float | None:
    from datetime import datetime, timezone
    raw = row.get("starts") or row.get("start_ts") or row.get("startTimestamp")
    if raw in (None, ""):
        return None
    try:
        if isinstance(raw, (int, float)):
            value = float(raw)
            return value / 1000.0 if value > 10_000_000_000 else value
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except Exception:
        return None


def _match_fixture(events: list[dict], player_a: str, player_b: str, match_start: float | None) -> dict | None:
    best = None
    for row in events or []:
        if not isinstance(row, dict):
            continue
        home = str(row.get("home") or row.get("participant1") or "").strip()
        away = str(row.get("away") or row.get("participant2") or "").strip()
        matches = ((_same_player(home, player_a) and _same_player(away, player_b)) or
                   (_same_player(home, player_b) and _same_player(away, player_a)))
        if not matches:
            continue
        ts = _event_timestamp(row)
        delta = abs(ts - match_start) if ts and match_start else 0.0
        if match_start and ts and delta > 36 * 3600:
            continue
        if best is None or delta < best[0]:
            best = (delta, row)
    return None if best is None else best[1]


def _valid_price(value):
    try:
        v = float(value)
        return v if v > 1.0 else None
    except Exception:
        return None


def _parse_total35(payload, *, require_set_context: bool = False):
    """Find an Over/Under 3.5 pair without mistaking unrelated props for sets."""
    found = {}

    def walk(node, path=""):
        if isinstance(node, list):
            for item in node:
                walk(item, path)
            return
        if not isinstance(node, dict):
            return

        label_bits = [
            node.get("special"), node.get("special_category"), node.get("name"),
            node.get("description"), node.get("market"), node.get("market_name"),
            node.get("type"), node.get("label"), path,
        ]
        context = " ".join(str(x or "") for x in label_bits).casefold()
        set_context = any(token in context for token in ("total sets", "number of sets", "sets total", "match sets", " set ", "sets"))

        # Standard PinnOdds periods.num_0.totals shape. On a BO5 tennis match a 3.5
        # full-match total is the set-count market; game totals are normally ~30-50.
        totals = node.get("totals")
        if isinstance(totals, dict):
            entry = totals.get("3.5") or totals.get(3.5)
            if isinstance(entry, dict) and (set_context or not require_set_context):
                over, under = _valid_price(entry.get("over")), _valid_price(entry.get("under"))
                if over and under:
                    return (over, under)
            for key, entry in totals.items():
                if not isinstance(entry, dict):
                    continue
                try:
                    points = float(entry.get("points", key))
                except Exception:
                    continue
                if abs(points - 3.5) < 1e-9 and (set_context or not require_set_context):
                    over, under = _valid_price(entry.get("over")), _valid_price(entry.get("under"))
                    if over and under:
                        return (over, under)

        # Special-market shape: prices: [{name: Over 3.5, price: ...}, ...]
        prices = node.get("prices")
        if isinstance(prices, list) and set_context:
            local = {}
            for item in prices:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("label") or "").casefold()
                price = _valid_price(item.get("price") or item.get("odds"))
                if not price:
                    continue
                if "over" in name and ("3.5" in name or str(node.get("points") or "") == "3.5"):
                    local["over"] = price
                elif "under" in name and ("3.5" in name or str(node.get("points") or "") == "3.5"):
                    local["under"] = price
            if "over" in local and "under" in local:
                return (local["over"], local["under"])

        # Alternate object shape: {points:3.5, over:..., under:...}.
        try:
            points = float(node.get("points")) if node.get("points") not in (None, "") else None
        except Exception:
            points = None
        if points is not None and abs(points - 3.5) < 1e-9 and set_context:
            over, under = _valid_price(node.get("over")), _valid_price(node.get("under"))
            if over and under:
                return (over, under)

        for key, value in node.items():
            if isinstance(value, (dict, list)):
                hit = walk(value, f"{path} {key}")
                if hit:
                    return hit
        return None

    return walk(payload)


@st.cache_data(ttl=60, show_spinner=False)
def _direct_pinnodds_total35(player_a: str, player_b: str, match_start: float | None, cache_bucket: int):
    key = _secret("PINNODDS_API_KEY")
    if not key:
        return None, None, "PINNODDS_API_KEY is not configured in Streamlit Secrets"
    headers = {"x-portal-apikey": key, "x-api-key": key, "accept": "application/json"}
    base = "https://pinnodds.com"
    try:
        r = requests.get(
            base + "/kit/v1/prematch/fixtures",
            params={"sport_id": 2, "include_specials": "nested"},
            headers=headers, timeout=15,
        )
        if r.status_code != 200:
            return None, None, f"PinnOdds fixtures returned HTTP {r.status_code}: {r.text[:180]}"
        data = r.json()
        events = data.get("events") or []
        parent = _match_fixture(events, player_a, player_b, match_start)
        if parent is None:
            return None, None, "PinnOdds tennis fixture could not be matched to these players"

        event_id = parent.get("event_id") or parent.get("id")
        # First inspect the parent payload itself, including nested specials.
        hit = _parse_total35(parent, require_set_context=False)
        if hit:
            return hit, "PinnOdds prematch fixture", None

        # Ask for every normal market on the exact Pinnacle event.
        if event_id is not None:
            for endpoint, params in (
                ("/kit/v1/prematch/lines", {"event_id": event_id, "market_type": "totals"}),
                ("/kit/v1/prematch/markets", {"event_id": event_id}),
            ):
                rr = requests.get(base + endpoint, params=params, headers=headers, timeout=15)
                if rr.status_code == 200:
                    hit = _parse_total35(rr.json(), require_set_context=False)
                    if hit:
                        return hit, f"PinnOdds {endpoint.rsplit('/',1)[-1]}", None

        # Some tennis set-count markets are specials. Pull flat specials so we can
        # identify child event IDs linked to the match, then fetch each set-related row.
        flat = requests.get(
            base + "/kit/v1/prematch/fixtures",
            params={"sport_id": 2, "include_specials": 1},
            headers=headers, timeout=15,
        )
        if flat.status_code == 200 and event_id is not None:
            children = []
            for row in flat.json().get("events") or []:
                if not isinstance(row, dict):
                    continue
                if str(row.get("parent_id") or "") != str(event_id):
                    continue
                label = " ".join(str(row.get(k) or "") for k in ("special", "special_category", "name", "description")).casefold()
                if "set" in label:
                    children.append(row)
            for child in children:
                hit = _parse_total35(child, require_set_context=True)
                if hit:
                    return hit, "PinnOdds nested set special", None
                child_id = child.get("event_id") or child.get("id")
                if child_id is None:
                    continue
                rr = requests.get(
                    base + "/kit/v1/prematch/markets",
                    params={"event_id": child_id}, headers=headers, timeout=15,
                )
                if rr.status_code == 200:
                    hit = _parse_total35(rr.json(), require_set_context=True)
                    if hit:
                        return hit, "PinnOdds set special", None

        return None, None, "Pinnacle does not currently expose a Total Sets 3.5 line for this matched event"
    except Exception as exc:
        return None, None, f"Direct PinnOdds totals request failed: {exc}"

detail = st.session_state.get("selected_match_detail")
if not detail:
    st.warning("Select a match from the Live Value Board first.")
    if st.button("← Back to live board"):
        st.switch_page("app.py")
    st.stop()

result = detail["result"]
event = detail["event"]
quote = detail["quote"]
sets = detail.get("sets")
row = detail["row"]

a, b = result["player_a"], result["player_b"]
ml_quote = quote.get("moneyline")
has_ml = bool(ml_quote and len(ml_quote) == 2)
oa, ob = ml_quote if has_ml else (np.nan, np.nan)
start_ts = float(event.get("startTimestamp") or 0)
eid = str(event.get("id"))

if st.button("← Back to live board"):
    st.switch_page("app.py")

st.title(f"🎾 {a} vs {b}")
st.caption(
    f"{result['tournament']} · {result['surface']} · BO{int(result['best_of'])} · "
    f"court speed {result['court_speed']:.2f} ({court_speed_label(result['court_speed'])})"
)

if not has_ml:
    books = quote.get("available_bookmakers") or []
    extra = f" Matchstat currently shows: {', '.join(books[:8])}." if books else ""
    if quote.get("direct_pinnacle_error") == "PINNODDS_API_KEY not configured":
        extra += " Add PINNODDS_API_KEY to Streamlit Secrets for the direct Pinnacle feed."
    st.warning(
        "No usable Pinnacle moneyline is available right now, so EV/Kelly/stake are intentionally disabled."
        + extra
    )

pick_side = str(result.get("recommended_side", "NO BET"))
actual_ml_bet = has_ml and pick_side in {"A", "B"} and str(row.get("ML pick", "No bet")) not in {"No bet", ""}
if actual_ml_bet:
    pick = result["recommended_pick"]
    pick_prob = result["probability_a"] if pick_side == "A" else result["probability_b"]
    st.success(
        f"**BET {pick} @ {result['recommended_odds']:.3f}** · model {pick_prob:.1%} · "
        f"edge {result['recommended_edge']:+.1%} · EV {result['recommended_ev']:+.1%} · "
        f"quarter-Kelly {result['recommended_quarter_kelly']:.2%} · "
        + (f"stake {result['recommended_quarter_kelly']:.2%} of bankroll" if _tracking_mode() == "percentage" else f"stake CA${float(row.get('ML stake CA$', 0)):,.2f}")
    )
elif has_ml:
    st.info("**NO MONEYLINE BET** at the current Pinnacle prices under your EV + edge thresholds.")
else:
    model_favorite = a if result["probability_a"] >= result["probability_b"] else b
    favorite_p = max(result["probability_a"], result["probability_b"])
    st.info(f"Model-only lean: **{model_favorite} {favorite_p:.1%}**. Waiting for Pinnacle before making a bet call.")

# v0.2-style headline metrics: compact, direct, and visible without tabs.
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Model P(A)", f"{result['probability_a']:.1%}")
c2.metric("Model P(B)", f"{result['probability_b']:.1%}")
c3.metric("Market no-vig P(A)", f"{result['market_probability_a']:.1%}" if has_ml else "—")
c4.metric("Edge A", f"{result['edge_a']:+.1%}" if has_ml else "—")
c5.metric("EV A", f"{result['ev_a']:+.1%}" if has_ml else "—")
c6.metric("Fair odds A", f"{result['fair_odds_a']:.2f}")

if has_ml:
    st.write(
        f"Pinnacle: **{a} {oa:.3f} · {b} {ob:.3f}** · source: **{quote.get('source') or 'Matchstat'}**"
    )
st.write(
    f"Court speed: **{result['court_speed']:.2f} ({court_speed_label(result['court_speed'])})** · "
    f"automatic format: **BO{int(result['best_of'])}** · "
    f"{'indoor' if result.get('indoor') else 'outdoor'}"
)

if actual_ml_bet:
    pick = result["recommended_pick"]
    pick_prob = result["probability_a"] if pick_side == "A" else result["probability_b"]
    pick_fair = result["fair_odds_a"] if pick_side == "A" else result["fair_odds_b"]
    pick_edge = result["edge_a"] if pick_side == "A" else result["edge_b"]
    pick_ev = result["ev_a"] if pick_side == "A" else result["ev_b"]
    pick_kelly = result["quarter_kelly_a"] if pick_side == "A" else result["quarter_kelly_b"]
    opposite = b if pick_side == "A" else a
    bankroll_now = _tracking_bankroll()

    if supabase_configured():
        existing_bet = None
        try:
            existing_bet = find_open_bet(eid, "Moneyline", pick)
        except Exception:
            existing_bet = None
        if existing_bet:
            st.success(
                f"Bet already recorded in Supabase: **{pick} @ {float(existing_bet.get('odds_taken') or 0):.3f}** · "
                + (f"stake {float(existing_bet.get('stake_percent') or 0):.2%} of bankroll." if _tracking_mode() == "percentage" else f"stake CA${float(existing_bet.get('stake_amount') or 0):,.2f}.")
            )
        else:
            with st.form(f"place_ml_{eid}", border=True):
                st.markdown("**Record this moneyline bet**")
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    placed_odds = st.number_input(
                        "Odds taken", min_value=1.01, value=float(result["recommended_odds"]), step=0.01, format="%.3f"
                    )
                with bc2:
                    if _tracking_mode() == "percentage":
                        placed_stake_pct = st.number_input(
                            "Stake (% of bankroll)", min_value=0.01, max_value=100.0,
                            value=max(0.01, float(pick_kelly) * 100.0), step=0.10, format="%.2f"
                        )
                        placed_stake = float(bankroll_now) * float(placed_stake_pct) / 100.0
                    else:
                        recommended_stake = float(row.get("ML stake CA$", 0) or 0)
                        placed_stake = st.number_input(
                            "Stake (CA$)", min_value=0.01, value=max(0.01, recommended_stake), step=1.0
                        )
                with bc3:
                    if _tracking_mode() == "percentage":
                        st.metric("Bankroll index", f"{bankroll_now:,.2f}")
                    else:
                        st.metric("Bankroll before", f"CA${bankroll_now:,.2f}" if bankroll_now else "Not set")
                submitted = st.form_submit_button(f"I placed {pick} — save bet", type="primary", use_container_width=True)
                if submitted:
                    try:
                        pred_payload = prediction_payload_from_detail(detail, MODEL_VERSION, "Moneyline")
                        prediction_id = upsert_prediction(pred_payload)
                        bet_id = place_bet(
                            prediction_id=prediction_id,
                            match_id=eid,
                            match_date=pred_payload.get("match_date"),
                            model_version=MODEL_VERSION,
                            tournament=result.get("tournament"),
                            tournament_level=result.get("tournament_level"),
                            surface=result.get("surface"),
                            round_name=pred_payload.get("round"),
                            player_a=a,
                            player_b=b,
                            market="Moneyline",
                            selection=pick,
                            model_probability=float(pick_prob),
                            model_fair_odds=float(pick_fair),
                            odds_taken=float(placed_odds),
                            edge=float(pick_edge),
                            expected_value=float(pick_ev),
                            bankroll_before=bankroll_now or None,
                            kelly_fraction=float(pick_kelly),
                            stake_amount=float(placed_stake),
                        )
                        st.success(f"Bet #{bet_id} saved permanently to Supabase Tracking.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not save bet to Supabase: {exc}")
    else:
        # Local fallback for development if Supabase is not configured.
        if pick_side == "A":
            tracked = {**result, "event_id": eid, "start_timestamp": start_ts, "market_type": "Moneyline", "selection": a}
            track_oa, track_ob = oa, ob
        else:
            tracked = {
                **result, "event_id": eid, "start_timestamp": start_ts, "market_type": "Moneyline", "selection": b,
                "player_a": b, "player_b": a, "probability_a": result["probability_b"],
                "probability_b": result["probability_a"], "market_probability_a": result["market_probability_b"],
                "edge": result["edge_b"], "ev": result["ev_b"], "fair_odds_a": result["fair_odds_b"],
                "quarter_kelly": result["quarter_kelly_b"],
            }
            track_oa, track_ob = ob, oa
        if st.button(f"Save {result['recommended_pick']} bet to local Tracking", type="primary"):
            pid = local_save_prediction(
                tracked, track_oa, track_ob, float(row.get("ML stake CA$", 0)),
                notes=f"Event {eid}; automated {MODEL_VERSION} board",
            )
            st.success(f"Saved local tracking row #{pid}.")

# Keep the new explainability work, but present it like the v0.2 expanders instead of a tabbed dashboard.
with st.expander("Why does the model lean this way?", expanded=True):
    impacts = result.get("factor_impacts", {})
    explain_side = pick_side if pick_side in {"A", "B"} else ("A" if result["probability_a"] >= result["probability_b"] else "B")
    explained = a if explain_side == "A" else b
    sign = -1.0 if explain_side == "B" else 1.0
    reason_rows = []
    for name, impact in impacts.items():
        adj = sign * float(impact)
        reason_rows.append(
            {
                "Factor": name,
                f"Matchup impact on {explained}": adj,
                "Direction": "Helps" if adj > 0 else ("Hurts" if adj < 0 else "Neutral"),
            }
        )
    if reason_rows:
        reason_df = pd.DataFrame(reason_rows).sort_values(
            f"Matchup impact on {explained}", key=lambda x: x.abs(), ascending=False
        )
        st.dataframe(
            reason_df.style.format({f"Matchup impact on {explained}": "{:+.2%}"}),
            hide_index=True,
            use_container_width=True,
        )
    cp = result.get("context_profile", {})
    w1, w2, w3 = st.columns(3)
    w1.metric("Exact-event edge A−B", f"{float(cp.get('event_perf_edge', 0)):+.3f}")
    w2.metric("Tournament-level edge A−B", f"{float(cp.get('level_perf_edge', 0)):+.3f}")
    w3.metric("Court-speed fit edge A−B", f"{float(cp.get('speed_fit_edge', 0)):+.3f}")
    st.caption(
        "These are counterfactual sensitivity checks. The model is nonlinear, so the factor impacts should not be added together as if they were fixed probability bonuses."
    )

hr = result.get("h2h_record", {})
with st.expander("Head-to-head model inputs", expanded=True):
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Career H2H", f"{int(hr.get('a_wins', 0))}–{int(hr.get('b_wins', 0))}")
    h2.metric(
        f"H2H on {result['surface']}",
        f"{int(hr.get('surface_a_wins', 0))}–{int(hr.get('surface_b_wins', 0))}",
    )
    h3.metric("Shrunk overall edge", f"{float(result.get('h2h_overall_edge', 0)):+.3f}")
    h4.metric("Shrunk surface edge", f"{float(result.get('h2h_surface_edge', 0)):+.3f}")
    st.caption("Small H2H samples are shrunk toward neutral rather than treated as a raw 100%/0% signal.")


def metric_table(row: pd.Series) -> pd.DataFrame:
    metrics = [
        ("ATP rank", row.get("rank")),
        ("Overall Elo", row.get("overall_elo")),
        ("Surface Elo", row.get("surface_elo")),
        ("Serve rating (EWMA)", row.get("serve")),
        ("Return rating (EWMA)", row.get("return_rating")),
        ("Last 3 win rate", row.get("win3")),
        ("Last 5 win rate", row.get("win5")),
        ("Last 10 win rate", row.get("win10")),
        ("Surface last 10", row.get("surface_win10")),
        ("Service points won — last match", row.get("spw1")),
        ("Service points won — last 5", row.get("spw5")),
        ("Return points won — last match", row.get("rpw1")),
        ("Return points won — last 5", row.get("rpw5")),
        ("1st serve in — last 5", row.get("first_in5")),
        ("1st serve points won — last 5", row.get("first_won5")),
        ("2nd serve points won — last 5", row.get("second_won5")),
        ("Ace rate — last 5", row.get("ace_rate5")),
        ("Double-fault rate — last 5", row.get("df_rate5")),
        ("Point share — last 5", row.get("point_share5")),
        ("Break points saved — last 5", row.get("bp_save5")),
        ("Break points converted — last 5", row.get("bp_convert5")),
        ("Recent form EWMA", row.get("form_ewma")),
        ("Surface form EWMA", row.get("surface_form_ewma")),
        ("Average opponent Elo — last 10", row.get("opp_elo10")),
        ("Recent performance vs expectation", row.get("recent_perf10")),
        ("Matches in 7 days", row.get("matches7")),
        ("Matches in 14 days", row.get("matches14")),
        ("Rest days", row.get("rest_days")),
        ("Recent Elo change", row.get("elo_change10")),
        ("Winner rate — advanced coverage", row.get("winner_rate")),
        ("UE rate — advanced coverage", row.get("ue_rate")),
        ("Advanced-stat coverage", row.get("advanced_coverage")),
        ("Average 1st-serve speed", row.get("avg_first_serve_speed")),
        ("Last match", row.get("last_match")),
    ]
    percentage_names = ("rate", "won", "share", "saved", "converted", "form", "coverage")
    formatted = []
    for name, value in metrics:
        if pd.isna(value):
            shown = ""
        elif isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            if "rank" == name.lower():
                shown = f"{int(value)}"
            elif any(token in name.lower() for token in percentage_names):
                shown = f"{value:.1%}"
            elif "elo" in name.lower():
                shown = f"{value:.1f}"
            else:
                shown = f"{value:.3f}"
        else:
            shown = str(value)
        formatted.append((name, shown))
    return pd.DataFrame(formatted, columns=["Metric", "Value"])

st.subheader("Player comparison")
left, right = st.columns(2)
with left:
    st.markdown(f"### {a}")
    st.dataframe(metric_table(result["row_a"]), hide_index=True, use_container_width=True)
with right:
    st.markdown(f"### {b}")
    st.dataframe(metric_table(result["row_b"]), hide_index=True, use_container_width=True)

with st.expander("Event and tournament-level history"):
    cp = result.get("context_profile", {})
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"#### {a}")
        st.metric("Exact-event excess performance", f"{float(cp.get('event_player_a', 0)):+.2%}")
        st.metric("Exact-event prior matches", int(cp.get("event_matches_a", 0)))
        st.metric("Tournament-level excess performance", f"{float(cp.get('level_player_a', 0)):+.2%}")
        st.metric("Tournament-level prior matches", int(cp.get("level_matches_a", 0)))
        st.metric("Court-speed sensitivity slope", f"{float(cp.get('speed_slope_a', 0)):+.3f}")
    with p2:
        st.markdown(f"#### {b}")
        st.metric("Exact-event excess performance", f"{float(cp.get('event_player_b', 0)):+.2%}")
        st.metric("Exact-event prior matches", int(cp.get("event_matches_b", 0)))
        st.metric("Tournament-level excess performance", f"{float(cp.get('level_player_b', 0)):+.2%}")
        st.metric("Tournament-level prior matches", int(cp.get("level_matches_b", 0)))
        st.metric("Court-speed sensitivity slope", f"{float(cp.get('speed_slope_b', 0)):+.3f}")
    st.caption("Excess performance is result minus pre-match expectation, with shrinkage for small samples.")

with st.expander("Court-speed analysis", expanded=True):
    st.write(
        f"Current estimate: **{result['court_speed']:.2f} ({court_speed_label(result['court_speed'])})** · "
        f"prior {float(result.get('court_speed_prior') or result['court_speed']):.2f} · "
        f"live weight {float(result.get('court_speed_live_weight', 0)):.1%}."
    )
    curve = court_speed_curve(
        load_state(),
        load_bundle(),
        player_a=a,
        player_b=b,
        surface=result["surface"],
        rank_a=detail["rank_a"],
        rank_b=detail["rank_b"],
        odds_a=oa,
        odds_b=ob,
        tournament_level=result["tournament_level"],
        best_of=result["best_of"],
        tournament=result["tournament"],
        indoor=result["indoor"],
    )
    st.line_chart(curve.set_index("Court speed")[["P(A)", "P(B)"]])
    neutral = float(curve.loc[np.isclose(curve["Court speed"], 1.0), "P(A)"].iloc[0])
    st.metric(
        f"Court-speed impact on {a} vs neutral 1.00",
        f"{result['probability_a'] - neutral:+.2%}",
    )

with st.expander("Grand Slam O/U 3.5 sets", expanded=True):
    if not sets or not sets.get("available"):
        st.info("The O/U 3.5 sets model is shown only for BO5 Grand Slam matches with the required model artifact.")
    else:
        # v0.3.2c: get the set-count price directly from PinnOdds on this page.
        # This avoids any stale PinnOddsClient class held by Streamlit's module cache.
        direct_pair, direct_source, direct_error = _direct_pinnodds_total35(
            a, b, start_ts or None, int(time.time() // 60)
        )
        if direct_pair:
            try:
                sets = predict_over35(
                    load_state(), a, b, result["surface"], result["tournament"], result["court_speed"],
                    odds_over=float(direct_pair[0]), odds_under=float(direct_pair[1]),
                )
            except Exception as exc:
                direct_error = f"Price found, but totals-model repricing failed: {exc}"
            else:
                quote["sets35"] = direct_pair
                quote["sets35_source"] = direct_source
                quote["sets35_error"] = None

        has_sets_price = bool(sets.get("odds_over35") and sets.get("odds_under35"))
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Model P(Over 3.5)", f"{sets['probability_over35']:.1%}")
        s2.metric("Model fair odds — Over", f"{sets['fair_odds_over35']:.2f}")
        s3.metric("Model P(Under 3.5)", f"{sets['probability_under35']:.1%}")
        s4.metric("Model fair odds — Under", f"{sets['fair_odds_under35']:.2f}")

        if has_sets_price:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pinnacle Over 3.5", f"{sets['odds_over35']:.3f}")
            m2.metric("Pinnacle Under 3.5", f"{sets['odds_under35']:.3f}")
            m3.metric("No-vig P(Over)", f"{sets.get('market_probability_over35', 0):.1%}")
            m4.metric("No-vig P(Under)", f"{sets.get('market_probability_under35', 0):.1%}")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Over edge", f"{sets.get('edge_over35', 0):+.1%}")
            v2.metric("Over EV", f"{sets.get('ev_over35', 0):+.1%}")
            v3.metric("Under edge", f"{sets.get('edge_under35', 0):+.1%}")
            v4.metric("Under EV", f"{sets.get('ev_under35', 0):+.1%}")
            k1, k2 = st.columns(2)
            k1.metric("Quarter-Kelly — Over", f"{sets.get('quarter_kelly_over35', 0):.2%}")
            k2.metric("Quarter-Kelly — Under", f"{sets.get('quarter_kelly_under35', 0):.2%}")
            if sets.get("recommended_market") != "No bet":
                stake = _tracking_bankroll() * float(sets.get("recommended_quarter_kelly", 0))
                st.success(
                    f"**{sets['recommended_market']}** · EV {sets.get('recommended_ev', 0):+.1%} · "
                    f"edge {sets.get('recommended_edge', 0):+.1%} · quarter-Kelly "
                    f"{sets.get('recommended_quarter_kelly', 0):.2%} · "
                    + (f"{sets.get('recommended_quarter_kelly', 0):.2%} bankroll" if _tracking_mode() == "percentage" else f"CA${stake:,.2f}")
                )
            else:
                st.info("**NO O/U 3.5 BET** at the current Pinnacle prices.")

            if sets.get("recommended_market") != "No bet" and supabase_configured():
                selection = str(sets["recommended_market"])
                is_over_bet = selection.casefold().startswith("over")
                selected_prob = float(sets["probability_over35"] if is_over_bet else sets["probability_under35"])
                selected_fair = float(sets["fair_odds_over35"] if is_over_bet else sets["fair_odds_under35"])
                selected_odds = float(sets["odds_over35"] if is_over_bet else sets["odds_under35"])
                selected_edge = float(sets["edge_over35"] if is_over_bet else sets["edge_under35"])
                selected_ev = float(sets["ev_over35"] if is_over_bet else sets["ev_under35"])
                selected_kelly = float(sets["quarter_kelly_over35"] if is_over_bet else sets["quarter_kelly_under35"])
                bankroll_now = _tracking_bankroll()
                existing_sets_bet = None
                try:
                    existing_sets_bet = find_open_bet(eid, "Total Sets 3.5", selection)
                except Exception:
                    existing_sets_bet = None
                if existing_sets_bet:
                    st.success(
                        f"Totals bet already recorded: **{selection} @ {float(existing_sets_bet.get('odds_taken') or 0):.3f}** · "
                        + (f"stake {float(existing_sets_bet.get('stake_percent') or 0):.2%} of bankroll." if _tracking_mode() == "percentage" else f"stake CA${float(existing_sets_bet.get('stake_amount') or 0):,.2f}.")
                    )
                else:
                    with st.form(f"place_sets_{eid}", border=True):
                        st.markdown("**Record this Total Sets bet**")
                        tc1, tc2, tc3 = st.columns(3)
                        with tc1:
                            sets_odds_taken = st.number_input(
                                "Totals odds taken", min_value=1.01, value=selected_odds, step=0.01, format="%.3f"
                            )
                        with tc2:
                            if _tracking_mode() == "percentage":
                                sets_stake_pct = st.number_input(
                                    "Totals stake (% of bankroll)", min_value=0.01, max_value=100.0,
                                    value=max(0.01, selected_kelly * 100.0), step=0.10, format="%.2f"
                                )
                                sets_stake = float(bankroll_now) * float(sets_stake_pct) / 100.0
                            else:
                                suggested = bankroll_now * selected_kelly if bankroll_now else 0.0
                                sets_stake = st.number_input(
                                    "Totals stake (CA$)", min_value=0.01, value=max(0.01, suggested), step=1.0
                                )
                        with tc3:
                            if _tracking_mode() == "percentage":
                                st.metric("Bankroll index", f"{bankroll_now:,.2f}")
                            else:
                                st.metric("Bankroll before", f"CA${bankroll_now:,.2f}" if bankroll_now else "Not set")
                        totals_submit = st.form_submit_button(
                            f"I placed {selection} — save bet", type="primary", use_container_width=True
                        )
                        if totals_submit:
                            try:
                                pred_payload = prediction_payload_from_detail(detail, MODEL_VERSION, "Total Sets 3.5")
                                # Use the direct price/model values rendered on this page, not a potentially stale slate copy.
                                pred_payload.update({
                                    "pinnacle_odds": float(sets["odds_over35"]),
                                    "pinnacle_no_vig_probability": sets.get("market_probability_over35"),
                                    "edge": sets.get("edge_over35"),
                                    "expected_value": sets.get("ev_over35"),
                                })
                                prediction_id = upsert_prediction(pred_payload)
                                bet_id = place_bet(
                                    prediction_id=prediction_id, match_id=eid, match_date=pred_payload.get("match_date"),
                                    model_version=MODEL_VERSION, tournament=result.get("tournament"),
                                    tournament_level=result.get("tournament_level"), surface=result.get("surface"),
                                    round_name=pred_payload.get("round"), player_a=a, player_b=b,
                                    market="Total Sets 3.5", selection=selection, model_probability=selected_prob,
                                    model_fair_odds=selected_fair, odds_taken=float(sets_odds_taken), edge=selected_edge,
                                    expected_value=selected_ev, bankroll_before=bankroll_now or None,
                                    kelly_fraction=selected_kelly, stake_amount=float(sets_stake),
                                )
                                st.success(f"Bet #{bet_id} saved permanently to Supabase Tracking.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not save totals bet to Supabase: {exc}")

            st.caption(f"Total Sets price source: {direct_source or quote.get('sets35_source') or 'PinnOdds'} · direct-v0.3.2c")
        else:
            st.warning(
                "The O/U probability model is working, but Pinnacle does not currently have a usable **Total Sets 3.5** "
                "pair in the PinnOdds response, so EV/Kelly cannot be calculated yet."
            )
            st.caption(f"Direct PinnOdds diagnostic: {direct_error or 'No 3.5 set-count pair found'} · direct-v0.3.2c")

        p = sets["profiles"]
        prof = pd.DataFrame([
            {"Player": a, "GS O3.5 rate": p["player_a_gs"]["over35_rate"], "GS matches": p["player_a_gs"]["matches"], "Event O3.5 rate": p["player_a_event"]["over35_rate"], "Event matches": p["player_a_event"]["matches"], "Five-set rate": p["player_a_gs"]["five_set_rate"]},
            {"Player": b, "GS O3.5 rate": p["player_b_gs"]["over35_rate"], "GS matches": p["player_b_gs"]["matches"], "Event O3.5 rate": p["player_b_event"]["over35_rate"], "Event matches": p["player_b_event"]["matches"], "Five-set rate": p["player_b_gs"]["five_set_rate"]},
        ])
        st.dataframe(
            prof.style.format({"GS O3.5 rate": "{:.1%}", "Event O3.5 rate": "{:.1%}", "Five-set rate": "{:.1%}"}),
            hide_index=True, use_container_width=True,
        )
        t1, t2, t3, t4 = st.columns(4)
        avg_gs = (p["player_a_gs"]["over35_rate"] + p["player_b_gs"]["over35_rate"]) / 2
        avg_event = (p["player_a_event"]["over35_rate"] + p["player_b_event"]["over35_rate"]) / 2
        avg_five = (p["player_a_gs"]["five_set_rate"] + p["player_b_gs"]["five_set_rate"]) / 2
        t1.metric("Pair GS O3.5 tendency", f"{avg_gs:.1%}")
        t2.metric("Pair event O3.5 tendency", f"{avg_event:.1%}")
        t3.metric("Pair five-set tendency", f"{avg_five:.1%}")
        t4.metric("Elo closeness", f"{float(p.get('elo_closeness', 0)):.1%}")
        st.caption(
            "These historical rates are inputs to the separate match-length model. The final O/U probability also "
            "uses Elo closeness, serve/return profile, court speed, surface and sample-size information."
        )

with st.expander("Market / CLV diagnostics"):
    if has_ml:
        st.write(f"Latest Pinnacle moneyline: **{a} {oa:.3f} · {b} {ob:.3f}**")
        st.caption(f"Quote source: {quote.get('source') or 'Matchstat odds feed'}")
    else:
        st.warning("No Pinnacle quote is currently stored for this event.")
    key = os.getenv("MATCHSTAT_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("MATCHSTAT_API_KEY", "")).strip()
        except Exception:
            key = ""
    if key:
        try:
            movements = MatchstatClient(api_key=key, min_interval_seconds=.61).last_ten_odds_movements(eid)
            st.write("Matchstat odds-movement payload")
            st.json(movements, expanded=False)
        except Exception as exc:
            st.caption(f"Matchstat odds-movement endpoint unavailable: {exc}")
    st.caption("The live board saves Pinnacle snapshots; the final pre-start snapshot is the intended CLV close reference.")

st.warning("Model outputs are estimates, not certainty. Injuries, withdrawals, travel and late news still require review.")

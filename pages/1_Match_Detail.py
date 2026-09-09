from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atp_model.matchstat import MatchstatClient
from atp_model.model_service import court_speed_curve, load_bundle, load_state
from atp_model.tournament_features import court_speed_label
from atp_model.tracking import current_bankroll, save_prediction

st.set_page_config(page_title="ATP Match Detail", page_icon="🎾", layout="wide")

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
        f"stake CA${float(row.get('ML stake CA$', 0)):,.2f}"
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
    if pick_side == "A":
        tracked = {**result, "event_id": eid, "start_timestamp": start_ts, "market_type": "Moneyline", "selection": a}
        track_oa, track_ob = oa, ob
    else:
        tracked = {
            **result,
            "event_id": eid,
            "start_timestamp": start_ts,
            "market_type": "Moneyline",
            "selection": b,
            "player_a": b,
            "player_b": a,
            "probability_a": result["probability_b"],
            "probability_b": result["probability_a"],
            "market_probability_a": result["market_probability_b"],
            "edge": result["edge_b"],
            "ev": result["ev_b"],
            "fair_odds_a": result["fair_odds_b"],
            "quarter_kelly": result["quarter_kelly_b"],
        }
        track_oa, track_ob = ob, oa
    if st.button(f"Save {result['recommended_pick']} bet to Tracking", type="primary"):
        pid = save_prediction(
            tracked,
            track_oa,
            track_ob,
            float(row.get("ML stake CA$", 0)),
            notes=f"Event {eid}; automated v0.3.2 board",
        )
        st.success(f"Saved tracking row #{pid}.")

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
        has_sets_price = bool(sets.get("odds_over35") and sets.get("odds_under35"))

        # Always show the model's probabilities AND fair prices, even if the market
        # quote has not opened yet.  This makes the totals model useful on its own.
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
                stake = (current_bankroll() or 0) * float(sets.get("recommended_quarter_kelly", 0))
                st.success(
                    f"**{sets['recommended_market']}** · EV {sets.get('recommended_ev', 0):+.1%} · "
                    f"edge {sets.get('recommended_edge', 0):+.1%} · quarter-Kelly "
                    f"{sets.get('recommended_quarter_kelly', 0):.2%} · CA${stake:,.2f}"
                )
            else:
                st.info("**NO O/U 3.5 BET** at the current Pinnacle prices.")

            if quote.get("sets35_source"):
                st.caption(f"Total Sets price source: {quote.get('sets35_source')}")
        else:
            st.warning(
                "The O/U model is working, but a Pinnacle **Total Sets 3.5** price has not been found yet, "
                "so EV, edge, Kelly and stake cannot be calculated. v0.3.2 checks both Pinnodds standard "
                "tennis totals and Pinnacle special-market rows rather than relying on Matchstat for this market."
            )
            if quote.get("sets35_error"):
                st.caption(f"Pinnacle Total Sets diagnostic: {quote.get('sets35_error')}")

        p = sets["profiles"]
        prof = pd.DataFrame(
            [
                {"Player": a, "GS O3.5 rate": p["player_a_gs"]["over35_rate"], "GS matches": p["player_a_gs"]["matches"], "Event O3.5 rate": p["player_a_event"]["over35_rate"], "Event matches": p["player_a_event"]["matches"], "Five-set rate": p["player_a_gs"]["five_set_rate"]},
                {"Player": b, "GS O3.5 rate": p["player_b_gs"]["over35_rate"], "GS matches": p["player_b_gs"]["matches"], "Event O3.5 rate": p["player_b_event"]["over35_rate"], "Event matches": p["player_b_event"]["matches"], "Five-set rate": p["player_b_gs"]["five_set_rate"]},
            ]
        )
        st.dataframe(
            prof.style.format({"GS O3.5 rate": "{:.1%}", "Event O3.5 rate": "{:.1%}", "Five-set rate": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )

        # Explain what the totals model is seeing without pretending these are
        # independent additive effects.
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

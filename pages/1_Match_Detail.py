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
from atp_model.tracking import current_bankroll, save_prediction

st.set_page_config(page_title="ATP Match Detail", page_icon="🔎", layout="wide")

detail = st.session_state.get("selected_match_detail")
if not detail:
    st.warning("Select a match from the Live Value Board first.")
    if st.button("← Back to live board"):
        st.switch_page("app.py")
    st.stop()

result = detail["result"]
event = detail["event"]
ctx = detail["context"]
quote = detail["quote"]
sets = detail.get("sets")
row = detail["row"]

a, b = result["player_a"], result["player_b"]
ml_quote = quote.get("moneyline")
has_ml = bool(ml_quote and len(ml_quote) == 2)
oa, ob = ml_quote if has_ml else (np.nan, np.nan)
start_ts = float(event.get("startTimestamp") or 0)
eid = str(event.get("id"))

st.title(f"🔎 {a} vs {b}")
st.caption(
    f"{result['tournament']} · {result['surface']} · BO{int(result['best_of'])} · "
    f"court speed {result['court_speed']:.2f}"
)
if st.button("← Back to live board"):
    st.switch_page("app.py")

if not has_ml:
    books = quote.get("available_bookmakers") or []
    extra = f" Available books seen by the API: {', '.join(books[:8])}." if books else ""
    st.warning(
        "Pinnacle moneyline is not currently exposed by Matchstat for this event. "
        "The detailed model view still works; EV, Kelly and stake stay disabled until a Pinnacle quote appears."
        + extra
    )

best_selection = str(row.get("Best selection", "No bet"))
best_market = str(row.get("Best market", "No bet"))
if best_market != "No bet":
    st.success(
        f"Best current recommendation: **{best_selection}** ({best_market}) @ {float(row.get('Best price')):.3f} · "
        f"edge {float(row.get('Best Edge', 0)):+.1%} · EV {float(row.get('Best EV', 0)):+.1%} · "
        f"quarter-Kelly {float(row.get('Best Kelly %', 0)):.2%} · CA${float(row.get('Best stake CA$', 0)):,.2f}"
    )
else:
    st.info("No market currently clears the board EV + edge thresholds.")

pick_side = str(result.get("recommended_side", "NO BET"))
actual_ml_bet = has_ml and pick_side in {"A", "B"} and str(row.get("ML pick", "No bet")) not in {"No bet", ""}
if actual_ml_bet:
    pick = result["recommended_pick"]
    pick_prob = result["probability_a"] if pick_side == "A" else result["probability_b"]
    st.success(
        f"Model moneyline recommendation: **{pick} @ {result['recommended_odds']:.3f}** · "
        f"P={pick_prob:.1%} · edge={result['recommended_edge']:+.1%} · EV={result['recommended_ev']:+.1%}"
    )
elif has_ml:
    st.info("Moneyline: NO BET at the current Pinnacle price under the board thresholds.")
else:
    model_favorite = a if result["probability_a"] >= result["probability_b"] else b
    favorite_p = max(result["probability_a"], result["probability_b"])
    st.info(f"Model-only view: **{model_favorite} {favorite_p:.1%}**. Waiting for a usable Pinnacle price before making a bet call.")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric(a, f"{result['probability_a']:.1%}")
m2.metric(b, f"{result['probability_b']:.1%}")
m3.metric("Pinnacle A", f"{oa:.3f}" if has_ml else "Unavailable")
m4.metric("Pinnacle B", f"{ob:.3f}" if has_ml else "Unavailable")
m5.metric("Court speed", f"{result['court_speed']:.2f}")
m6.metric("Best-bet stake", f"CA${float(row.get('Best stake CA$', 0)):,.2f}")

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
            notes=f"Event {eid}; automated v0.3 board",
        )
        st.success(f"Saved tracking row #{pid}.")

overview, reasons, stats, eventtab, h2h, speedtab, market, sets_tab = st.tabs(
    ["Overview", "Why?", "Player stats", "Event history", "H2H", "Court speed", "Market", "O/U 3.5 sets"]
)

with overview:
    rows = [
        {"Metric": "Model probability", a: result["probability_a"], b: result["probability_b"]},
        {"Metric": "Model fair odds", a: result["fair_odds_a"], b: result["fair_odds_b"]},
    ]
    if has_ml:
        rows.extend(
            [
                {"Metric": "Pinnacle no-vig probability", a: result["market_probability_a"], b: result["market_probability_b"]},
                {"Metric": "Model edge", a: result["edge_a"], b: result["edge_b"]},
                {"Metric": "Expected value", a: result["ev_a"], b: result["ev_b"]},
                {"Metric": "Quarter Kelly", a: result["quarter_kelly_a"], b: result["quarter_kelly_b"]},
            ]
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.write(
        f"**Automatic format:** BO{int(result['best_of'])}. Grand Slams default to BO5; "
        "ATP 250/500/Masters/Finals default to BO3 unless the API explicitly provides a valid format."
    )

with reasons:
    st.subheader("Counterfactual model sensitivity")
    impacts = result.get("factor_impacts", {})
    explain_side = pick_side if pick_side in {"A", "B"} else ("A" if result["probability_a"] >= result["probability_b"] else "B")
    sign = -1.0 if explain_side == "B" else 1.0
    reason_rows = []
    for name, impact in impacts.items():
        adj = sign * float(impact)
        reason_rows.append(
            {
                "Factor": name,
                "Impact on explained side": adj,
                "Direction": "Helps" if adj > 0 else ("Hurts" if adj < 0 else "Neutral"),
            }
        )
    if reason_rows:
        rdf = pd.DataFrame(reason_rows)
        st.dataframe(
            rdf.style.format({"Impact on explained side": "{:+.2%}"}),
            hide_index=True,
            use_container_width=True,
        )
    explained = a if explain_side == "A" else b
    st.caption(
        f"Explaining {explained}. These are counterfactual sensitivity checks; gradient-boosting impacts are nonlinear and should not be summed."
    )
    cp = result.get("context_profile", {})
    st.write(f"**Exact-event excess-performance edge (A−B):** {cp.get('event_perf_edge', 0):+.3f}")
    st.write(f"**Tournament-level excess-performance edge (A−B):** {cp.get('level_perf_edge', 0):+.3f}")
    st.write(f"**Historical court-speed slope edge (A−B):** {cp.get('speed_fit_edge', 0):+.3f}")

with stats:
    ra, rb = result["row_a"], result["row_b"]
    metrics = [
        ("ATP rank", "rank"), ("Overall Elo", "overall_elo"), ("Surface Elo", "surface_elo"),
        ("Serve rating", "serve"), ("Return rating", "return_rating"), ("Last 10 win rate", "win10"),
        ("Surface last 10", "surface_win10"), ("Serve points won L5", "spw5"), ("Return points won L5", "rpw5"),
        ("Ace rate L5", "ace_rate5"), ("2nd serve won L5", "second_won5"), ("Point share L5", "point_share5"),
        ("BP saved L5", "bp_save5"), ("BP converted L5", "bp_convert5"),
        ("Recent performance vs expectation", "recent_perf10"), ("Rest days", "rest_days"),
    ]
    stat_rows = []
    for label, key in metrics:
        av = float(ra.get(key, 0))
        bv = float(rb.get(key, 0))
        if key == "rank":
            advantage = a if av < bv else b if bv < av else "Even"
        else:
            advantage = a if av > bv else b if bv > av else "Even"
        stat_rows.append({"Metric": label, a: av, b: bv, "Advantage": advantage})
    st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)
    st.caption("Raw player-state inputs. Percent-like metrics are stored on the 0–1 scale in the model state.")

with eventtab:
    cp = result.get("context_profile", {})
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(a)
        st.metric("Exact-event excess performance", f"{cp.get('event_player_a', 0):+.2%}")
        st.metric("Exact-event prior matches", int(cp.get("event_matches_a", 0)))
        st.metric("Level excess performance", f"{cp.get('level_player_a', 0):+.2%}")
        st.metric("Level prior matches", int(cp.get("level_matches_a", 0)))
        st.metric("Court-speed sensitivity slope", f"{cp.get('speed_slope_a', 0):+.3f}")
    with c2:
        st.subheader(b)
        st.metric("Exact-event excess performance", f"{cp.get('event_player_b', 0):+.2%}")
        st.metric("Exact-event prior matches", int(cp.get("event_matches_b", 0)))
        st.metric("Level excess performance", f"{cp.get('level_player_b', 0):+.2%}")
        st.metric("Level prior matches", int(cp.get("level_matches_b", 0)))
        st.metric("Court-speed sensitivity slope", f"{cp.get('speed_slope_b', 0):+.3f}")
    st.caption("Excess performance is historical result minus pre-match Elo expectation, shrunk toward zero for small samples.")

with h2h:
    hr = result.get("h2h_record", {})
    h1, h2, h3 = st.columns(3)
    h1.metric("Career H2H", f"{int(hr.get('a_wins', 0))}–{int(hr.get('b_wins', 0))}")
    h2.metric(f"H2H on {result['surface']}", f"{int(hr.get('surface_a_wins', 0))}–{int(hr.get('surface_b_wins', 0))}")
    h3.metric("Prior meetings", int(hr.get("matches", 0)))
    st.write(
        f"Shrunk overall H2H edge: **{result.get('h2h_overall_edge', 0):+.3f}** · "
        f"surface H2H edge: **{result.get('h2h_surface_edge', 0):+.3f}**"
    )

with speedtab:
    st.write(
        f"Current estimate: **{result['court_speed']:.2f}** · prior {result.get('court_speed_prior')} · "
        f"live weight {result.get('court_speed_live_weight', 0):.1%}."
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
        f"Court-speed effect on {a}",
        f"{result['probability_a'] - neutral:+.2%}",
        help="Current prediction minus the same matchup at neutral court speed 1.00.",
    )

with market:
    if has_ml:
        st.write(f"Latest stored Pinnacle ML: **{a} {oa:.3f} · {b} {ob:.3f}**")
        st.caption(f"Quote source: {quote.get('source') or 'Matchstat odds feed'}")
    else:
        st.warning("No usable Pinnacle moneyline is currently available for this event.")
        books = quote.get("available_bookmakers") or []
        if books:
            st.write("Bookmakers visible in the current Matchstat response: " + ", ".join(books))
        if quote.get("error"):
            st.code(str(quote["error"]))
    key = os.getenv("MATCHSTAT_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("MATCHSTAT_API_KEY", "")).strip()
        except Exception:
            key = ""
    if key:
        try:
            movements = MatchstatClient(api_key=key, min_interval_seconds=.61).last_ten_odds_movements(eid)
            st.subheader("Recent odds movements from Matchstat")
            st.json(movements, expanded=False)
        except Exception as exc:
            st.caption(f"Odds movement endpoint unavailable: {exc}")
    st.caption(
        "The live board saves Pinnacle moneyline snapshots while it is running. The last snapshot strictly before match start is the intended CLV close reference."
    )

with sets_tab:
    if not sets or not sets.get("available"):
        st.info("O/U 3.5 sets is available only for BO5 Grand Slam matches after the separate sets model has been trained.")
    else:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Model P(Over 3.5)", f"{sets['probability_over35']:.1%}")
        s2.metric("Model P(Under 3.5)", f"{sets['probability_under35']:.1%}")
        s3.metric("Pinnacle Over", f"{sets.get('odds_over35'):.3f}" if sets.get("odds_over35") else "Not exposed")
        s4.metric("Pinnacle Under", f"{sets.get('odds_under35'):.3f}" if sets.get("odds_under35") else "Not exposed")
        p = sets["profiles"]
        prof = pd.DataFrame(
            [
                {
                    "Player": a,
                    "GS O3.5 rate": p["player_a_gs"]["over35_rate"],
                    "GS matches": p["player_a_gs"]["matches"],
                    "Event O3.5 rate": p["player_a_event"]["over35_rate"],
                    "Event matches": p["player_a_event"]["matches"],
                    "Five-set rate": p["player_a_gs"]["five_set_rate"],
                },
                {
                    "Player": b,
                    "GS O3.5 rate": p["player_b_gs"]["over35_rate"],
                    "GS matches": p["player_b_gs"]["matches"],
                    "Event O3.5 rate": p["player_b_event"]["over35_rate"],
                    "Event matches": p["player_b_event"]["matches"],
                    "Five-set rate": p["player_b_gs"]["five_set_rate"],
                },
            ]
        )
        st.dataframe(
            prof.style.format({"GS O3.5 rate": "{:.1%}", "Event O3.5 rate": "{:.1%}", "Five-set rate": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )
        if sets.get("recommended_market") != "No bet":
            stake = (current_bankroll() or 0) * float(sets.get("recommended_quarter_kelly", 0))
            st.success(
                f"Totals model: **{sets['recommended_market']}** · EV {sets.get('recommended_ev', 0):+.1%} · "
                f"edge {sets.get('recommended_edge', 0):+.1%} · quarter-Kelly {sets.get('recommended_quarter_kelly', 0):.2%} · CA${stake:,.2f}"
            )
            if sets.get("odds_over35") and sets.get("odds_under35"):
                is_over = sets["recommended_market"].startswith("Over")
                tr = {
                    "event_id": eid,
                    "start_timestamp": start_ts,
                    "market_type": "Total Sets 3.5",
                    "selection": sets["recommended_market"],
                    "player_a": sets["recommended_market"],
                    "player_b": "Under 3.5 sets" if is_over else "Over 3.5 sets",
                    "surface": result["surface"],
                    "tournament": result["tournament"],
                    "tournament_level": 5.0,
                    "court_speed": result["court_speed"],
                    "probability_a": sets["probability_over35"] if is_over else sets["probability_under35"],
                    "market_probability_a": sets.get("market_probability_over35") if is_over else sets.get("market_probability_under35"),
                    "edge": sets.get("edge_over35") if is_over else sets.get("edge_under35"),
                    "ev": sets.get("ev_over35") if is_over else sets.get("ev_under35"),
                    "fair_odds_a": sets["fair_odds_over35"] if is_over else sets["fair_odds_under35"],
                    "quarter_kelly": sets.get("quarter_kelly_over35") if is_over else sets.get("quarter_kelly_under35"),
                }
                so, su = sets["odds_over35"], sets["odds_under35"]
                first, second = (so, su) if is_over else (su, so)
                if st.button(f"Save {sets['recommended_market']} to Tracking"):
                    pid = save_prediction(tr, first, second, stake, notes=f"Grand Slam totals model; event {eid}")
                    st.success(f"Saved tracking row #{pid}.")

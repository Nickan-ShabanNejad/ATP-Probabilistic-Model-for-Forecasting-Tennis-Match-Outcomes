
import os, sys, json
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/"src"))

import pandas as pd
import streamlit as st
from atp_model.model_service import (
    load_state, load_bundle, predict_match, available_tournaments, tournament_speed
)
from atp_model.tracking import save_prediction

st.set_page_config(page_title="ATP Pro Value Model",page_icon="🎾",layout="wide")
st.title("🎾 ATP Professional Probability & Value Model")

@st.cache_data
def state(): return load_state()
@st.cache_resource
def bundle(): return load_bundle()

df=state(); model=bundle()
metrics=model.get("metrics",{})
freshness_path=ROOT/"data/generated/freshness.json"
freshness={}
if freshness_path.exists():
    try:
        freshness=json.loads(freshness_path.read_text(encoding="utf-8"))
    except Exception:
        freshness={}
ranking_date=freshness.get("reference",{}).get("rankings",{}).get("ranking_date","unknown")
st.caption(
    f"Model: {metrics.get('selected_model', model.get('model_name', 'unknown'))} · "
    f"Latest match: {metrics.get('latest_data_date','unknown')} · "
    f"rankings: {ranking_date} · holdout {metrics.get('holdout','unknown')} · "
    f"accuracy: {metrics.get('accuracy',0):.1%} · "
    f"log loss: {metrics.get('log_loss',0):.3f} · "
    f"Brier: {metrics.get('brier',0):.3f}"
)

match_age=freshness.get("matches",{}).get("age_days")
if freshness.get("matches",{}).get("stale"):
    st.error(
        f"⚠️ Match data is {match_age} days old. Treat predictions as stale until the source updates."
    )
ranking_method=freshness.get("reference",{}).get("rankings",{}).get("method")
if ranking_method=="latest observed ranking in match data":
    st.warning(
        "Current ranking defaults are derived from each player's latest recorded match, "
        "not a guaranteed live ATP ranking. You can override the ranking inputs manually."
    )
if s.get("accuracy",0)>0.9:
    st.error("Model accuracy is suspiciously high. Review the Data Health page before relying on predictions.")

names=sorted(df.player.dropna().unique())
with st.sidebar:
    st.header("Match")
    pa=st.selectbox("Player A",names,index=names.index("Lorenzo Musetti") if "Lorenzo Musetti" in names else 0)
    pb=st.selectbox("Player B",names,index=names.index("Matteo Arnaldi") if "Matteo Arnaldi" in names else 1)
    surface=st.selectbox("Surface",["Hard","Clay","Grass"])
    level_label=st.selectbox(
        "Tournament level",
        ["Challenger","ATP 250","ATP 500","Masters 1000","ATP Finals","Grand Slam"],
        index=1,
    )
    level_map={
        "Challenger":1.0,
        "ATP 250":2.0,
        "ATP 500":3.0,
        "Masters 1000":4.0,
        "ATP Finals":4.5,
        "Grand Slam":5.0,
    }
    # Call without arguments for compatibility with both old and new model_service versions.
    tournaments=available_tournaments()
    tournament=st.selectbox(
        "Tournament / court conditions",
        ["Unknown / surface average"] + tournaments,
    )
    tournament_value="" if tournament.startswith("Unknown") else tournament
    estimated_speed, speed_fallback=tournament_speed(tournament_value,surface)
    override_speed=st.checkbox("Override estimated court speed")
    if override_speed:
        court_speed=st.slider(
            "Court speed (0.75 slow – 1.25 fast)",
            min_value=0.75,max_value=1.25,value=float(round(estimated_speed,2)),step=0.01,
        )
    else:
        court_speed=None
        label="surface fallback" if speed_fallback else "tournament history"
        st.caption(f"Estimated court speed: {estimated_speed:.2f} ({label})")
    indoor=st.checkbox("Indoor")
    best_of=st.selectbox("Format",[3,5])
    arow=df[(df.player==pa)&(df.surface==surface)].iloc[0]
    brow=df[(df.player==pb)&(df.surface==surface)].iloc[0]
    rank_a=st.number_input("ATP rank A",1,2500,int(arow["rank"]))
    rank_b=st.number_input("ATP rank B",1,2500,int(brow["rank"]))
    odds_a=st.number_input("Decimal odds A",1.01,value=1.80,step=.01,format="%.3f")
    odds_b=st.number_input("Decimal odds B",1.01,value=2.05,step=.01,format="%.3f")
    stake=st.number_input("Optional tracked stake (CA$)",0.0,value=0.0,step=5.0)

if pa==pb:
    st.error("Choose two different players.")
    st.stop()

r=predict_match(
    df,model,pa,pb,surface,rank_a,rank_b,odds_a,odds_b,
    level_map[level_label],best_of,
    tournament=tournament_value,
    court_speed_override=court_speed,
    indoor=indoor,
)
c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric("Model P(A)", f"{r['probability_a']:.1%}")
c2.metric("Model P(B)", f"{r['probability_b']:.1%}")
c3.metric("Market no-vig P(A)", f"{r['market_probability_a']:.1%}")
c4.metric("Edge", f"{r['edge']:+.1%}")
c5.metric("Expected value", f"{r['ev']:+.1%}")
c6.metric("Fair odds A", f"{r['fair_odds_a']:.2f}")

if r["ev"]>=.05: st.success("Strong positive-EV signal — still subject to model and data risk.")
elif r["ev"]>=.02: st.info("Small positive-EV signal.")
elif r["ev"]>0: st.warning("Marginal signal; likely vulnerable to estimation noise.")
else: st.error("No estimated value on Player A at this price.")

st.write(f"Quarter-Kelly reference: **{r['quarter_kelly']:.2%} of bankroll** (uncapped).")

with st.expander("Player-order symmetry audit"):
    st.success(
        f"Displayed probabilities sum to {(r['probability_a'] + r['probability_b']):.2%}. "
        "The model now averages both player orientations, so reversing A and B gives the exact complement."
    )
    st.caption(
        f"Original A→B raw prediction: {r.get('forward_raw_probability_a', r['raw_probability_a']):.2%} · "
        f"Reverse prediction converted to P(A): {r.get('reverse_raw_probability_a', r['raw_probability_a']):.2%} · "
        f"pre-fix order gap: {r.get('symmetry_gap_before_fix', 0.0):+.2%}."
    )

h2h = r.get("h2h_record", {})
h2h_matches = int(h2h.get("matches", 0))
h2h_surface_matches = int(h2h.get("surface_matches", 0))
h2h_impact = float(r.get("h2h_impact", 0.0))

with st.expander("Head-to-head impact", expanded=True):
    if h2h_matches == 0:
        st.info("No recorded head-to-head matches were found for these players.")
    else:
        h1, h2, h3 = st.columns(3)
        h1.(
            "Overall H2H",
            f"{int(h2h.get('player_a_wins', 0))}–{int(h2h.get('player_b_wins', 0))}",
            help=f"{pa} wins – {pb} wins",
        )
        h2.(
            f"H2H on {surface}",
            f"{int(h2h.get('surface_player_a_wins', 0))}–{int(h2h.get('surface_player_b_wins', 0))}",
            help=f"{h2h_surface_matches} recorded matches on {surface}",
        )
        h3.(
            "Probability impact",
            f"{h2h_impact:+.2%}",
            help="The change applied to Player A's model probability.",
        )
        st.caption(
            f"Base model probability: {r.get('base_probability_a', r['probability_a']):.2%} · "
            f"After H2H: {r['probability_a']:.2%}. "
            "The H2H adjustment is sample-size weighted and limited to ±6 percentage points."
        )

if st.button("Save prediction to tracking database"):
    pid=save_prediction(r,odds_a,odds_b,stake)
    st.success(f"Saved prediction #{pid}.")

def metric_table(row):
    metrics = [
        ("Overall Elo", row.overall_elo),
        ("Surface Elo", row.surface_elo),
        ("Serve rating", row.serve),
        ("Return rating", row.return_rating),
        ("Last 5 win rate", row.win5),
        ("Last 10 win rate", row.win10),
        ("Surface last 10", row.surface_win10),
        ("Average opponent Elo (last 10)", row.opp_elo10),
        ("Recent performance vs expectation", row.recent_perf10),
        ("Matches in 7 days", row.matches7),
        ("Matches in 14 days", row.matches14),
        ("Rest days", row.rest_days),
        ("Recent Elo change", row.elo_change10),
        ("Charted serve points won", getattr(row, "chart_serve", None)),
        ("Charted return points won", getattr(row, "chart_return", None)),
        ("Charted winner rate", getattr(row, "chart_winner_rate", None)),
        ("Charted unforced-error rate", getattr(row, "chart_ue_rate", None)),
        ("Charted net success", getattr(row, "chart_net_win", None)),
        ("Charted matches used", getattr(row, "charted_matches", 0)),
        ("Last match", row.last_match),
    ]

    formatted = []

    for name, value in metrics:
        if pd.isna(value):
            value = ""
        elif isinstance(value, (int, float)):
            if "rate" in name.lower() or "win" in name.lower():
                value = f"{value:.3f}"
            elif "elo" in name.lower():
                value = f"{value:.1f}"
            else:
                value = f"{value}"
        else:
            value = str(value)

        formatted.append((name, value))

    return pd.DataFrame(formatted, columns=["Metric", "Value"])

l, rcol = st.columns(2)

with l:
    st.subheader(pa)
    st.dataframe(
        metric_table(r["row_a"]),
        hide_index=True,
        width="stretch",
    )

with rcol:
    st.subheader(pb)
    st.dataframe(
        metric_table(r["row_b"]),
        hide_index=True,
        width="stretch",
    )

st.caption(
    f"Prediction context: {level_label} · "
    f"court speed {r['court_speed']:.2f} · "
    f"{'indoor' if r['indoor'] else 'outdoor'}"
)
st.warning("Do not treat model output as certainty. Injuries, withdrawals, travel and late news require manual review.")


with st.expander("Data-source status"):
    st.json(freshness if freshness else {"status":"Run the GitHub workflow to generate freshness data."})



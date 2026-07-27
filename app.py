
import os, sys, json
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/"src"))

import pandas as pd
import streamlit as st
from atp_model.model_service import load_state, load_bundle, predict_match, available_tournaments
from atp_model.tracking import save_prediction

st.set_page_config(page_title="ATP Pro Value Model",page_icon="🎾",layout="wide")
st.title("🎾 ATP Professional Probability & Value Model")

@st.cache_data
def state(): return load_state()
@st.cache_resource
def bundle(): return load_bundle()

df = state()
model = bundle()


metrics = model.get("metrics", {})
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
if metrics.get("accuracy",0)>0.9:
    st.error("Model accuracy is suspiciously high. Review the Data Health page before relying on predictions.")

names=sorted(df.player.dropna().unique())
with st.sidebar:
    st.header("Match")
    pa=st.selectbox("Player A",names,index=names.index("Lorenzo Musetti") if "Lorenzo Musetti" in names else 0)
    pb=st.selectbox("Player B",names,index=names.index("Matteo Arnaldi") if "Matteo Arnaldi" in names else 1)
    surface=st.selectbox("Surface",["Hard","Clay","Grass"])
    tournaments=available_tournaments()
    tournament=st.selectbox("Tournament", tournaments if tournaments else ["Unknown tournament"])
    level_label=st.selectbox("Tournament level",["ATP 250 / standard","ATP 500","Masters 1000","Grand Slam","Challenger"])
    level_map={"Challenger":1.0,"ATP 250 / standard":2.0,"ATP 500":2.0,"Masters 1000":3.0,"Grand Slam":4.0}
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

r=predict_match(df,model,pa,pb,surface,rank_a,rank_b,odds_a,odds_b,level_map[level_label],best_of,tournament=tournament)
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Calibrated model P(A)",f"{r['probability_a']:.1%}")
c2.metric("Market no-vig P(A)",f"{r['market_probability_a']:.1%}")
c3.metric("Edge",f"{r['edge']:+.1%}")
c4.metric("Expected value",f"{r['ev']:+.1%}")
c5.metric("Fair odds A",f"{r['fair_odds_a']:.2f}")

if r["ev"]>=.05: st.success("Strong positive-EV signal — still subject to model and data risk.")
elif r["ev"]>=.02: st.info("Small positive-EV signal.")
elif r["ev"]>0: st.warning("Marginal signal; likely vulnerable to estimation noise.")
else: st.error("No estimated value on Player A at this price.")

st.write(f"Quarter-Kelly reference: **{r['quarter_kelly']:.2%} of bankroll** (capped at 5%).")
if r.get("probability_guardrail_applied"):
    st.warning(
        "A probability safety guardrail was applied because the calibrated output "
        "was outside the supported 5%–95% range."
    )
with st.expander("Probability diagnostics"):
    st.write({
        "raw_model_probability": round(r.get("raw_probability_a", r["probability_a"]), 4),
        "calibrated_probability": round(r.get("calibrated_probability_a", r["probability_a"]), 4),
        "displayed_probability": round(r["probability_a"], 4),
        "guardrail_applied": bool(r.get("probability_guardrail_applied", False)),
    })
h=r["h2h_record"]
st.caption(
    f"Court speed: {r['court_speed']:.2f}" + (" (surface fallback)" if r["court_speed_fallback"] else "") +
    f" · H2H: {pa} {h['a_wins']}-{h['b_wins']} {pb}" +
    f" · On {surface}: {h['surface_a_wins']}-{h['surface_b_wins']}"
)
if st.button("Save prediction to tracking database"):
    pid=save_prediction(r,odds_a,odds_b,stake)
    st.success(f"Saved prediction #{pid}.")

def metric_table(row):
    return pd.DataFrame({
      "Metric":["Overall Elo","Surface Elo","Serve rating","Return rating","Last 5 win rate",
      "Last 10 win rate","Surface last 10","Average opponent Elo (last 10)",
      "Recent performance vs expectation","Matches in 7 days","Matches in 14 days",
      "Rest days","Recent Elo change","Charted serve points won","Charted return points won",
      "Charted winner rate","Charted unforced-error rate","Charted net success",
      "Charted matches used","Last match"],
      "Value":[row.overall_elo,row.surface_elo,row.serve,row.return_rating,row.win5,row.win10,
      row.surface_win10,row.opp_elo10,row.recent_perf10,row.matches7,row.matches14,
      row.rest_days,row.elo_change10,
      getattr(row,"chart_serve",None),getattr(row,"chart_return",None),
      getattr(row,"chart_winner_rate",None),getattr(row,"chart_ue_rate",None),
      getattr(row,"chart_net_win",None),getattr(row,"charted_matches",0),row.last_match]
    })

l,rcol=st.columns(2)
with l:
    st.subheader(pa); st.dataframe(metric_table(r["row_a"]),hide_index=True,use_container_width=True)
with rcol:
    st.subheader(pb); st.dataframe(metric_table(r["row_b"]),hide_index=True,use_container_width=True)

st.warning("Do not treat model output as certainty. Injuries, withdrawals, travel and late news require manual review.")


with st.expander("Data-source status"):
    st.json(freshness if freshness else {"status":"Run the GitHub workflow to generate freshness data."})

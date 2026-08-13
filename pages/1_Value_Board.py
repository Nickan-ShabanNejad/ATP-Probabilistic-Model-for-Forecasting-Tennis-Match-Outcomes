
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import pandas as pd
import streamlit as st
from atp_model.odds import fetch_current_odds
from atp_model.model_service import load_state,load_bundle,predict_match

st.title("Today's automated value board")
events=fetch_current_odds()
if not events:
    st.info("No odds feed configured or no supported Pinnacle events returned. Add ODDS_API_KEY in Streamlit secrets to enable this page.")
    st.stop()
df=load_state(); bundle=load_bundle(); names=set(df.player.unique())
rows=[]
for e in events:
    a,b=e["home_team"],e["away_team"]
    if a not in names or b not in names: continue
    prices=e["prices"]
    if a not in prices or b not in prices: continue
    for surface in ["Hard"]:
        ar=df[(df.player==a)&(df.surface==surface)]
        br=df[(df.player==b)&(df.surface==surface)]
        if ar.empty or br.empty: continue
        result=predict_match(df,bundle,a,b,surface,int(ar.iloc[0]["rank"]),int(br.iloc[0]["rank"]),prices[a],prices[b])
        rows.append({"Match":f"{a} vs {b}","P(A)":result["probability_a"],"Market P(A)":result["market_probability_a"],
                     "Edge":result["edge"],"EV":result["ev"],"Odds A":prices[a]})
if rows:
    out=pd.DataFrame(rows).sort_values("EV",ascending=False)
    st.dataframe(out,hide_index=True,use_container_width=True,
                 column_config={"P(A)":st.column_config.NumberColumn(format="%.1%%"),
                 "Market P(A)":st.column_config.NumberColumn(format="%.1%%"),
                 "Edge":st.column_config.NumberColumn(format="%+.1%%"),
                 "EV":st.column_config.NumberColumn(format="%+.1%%")})
else:
    st.warning("Odds were received, but player names could not be matched to the ATP database.")

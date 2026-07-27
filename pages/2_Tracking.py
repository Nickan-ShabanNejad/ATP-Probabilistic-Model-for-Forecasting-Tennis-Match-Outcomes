
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import pandas as pd
import streamlit as st
from atp_model.tracking import get_predictions,settle_prediction

st.title("Prediction, ROI and closing-line tracking")
rows=get_predictions()
if not rows:
    st.info("No saved predictions yet.")
    st.stop()
df=pd.DataFrame([{
 "ID":x.id,"Created":x.created_at,"Match":f"{x.player_a} vs {x.player_b}",
 "Surface":x.surface,"Model P(A)":x.model_probability_a,"Odds A":x.odds_a,
 "Edge":x.edge,"EV":x.expected_value,"Closing odds":x.closing_odds_a,
 "Result A":x.result_a,"Stake":x.stake,"Profit":x.profit
} for x in rows])
st.dataframe(df,hide_index=True,use_container_width=True)
settled=df[df["Profit"].notna()]
if not settled.empty:
    total_stake=settled["Stake"].sum()
    profit=settled["Profit"].sum()
    c1,c2,c3=st.columns(3)
    c1.metric("Tracked profit",f"CA${profit:,.2f}")
    c2.metric("Tracked stake",f"CA${total_stake:,.2f}")
    c3.metric("ROI",f"{profit/total_stake:.1%}" if total_stake else "—")
st.subheader("Settle a prediction")
pid=st.number_input("Prediction ID",min_value=1,step=1)
result=st.selectbox("Did Player A win?",["Yes","No"])
closing=st.number_input("Closing odds A (optional)",min_value=0.0,value=0.0,step=.01)
if st.button("Settle"):
    settle_prediction(int(pid),result=="Yes",closing or None)
    st.success("Updated.")

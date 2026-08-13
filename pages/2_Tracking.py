import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from atp_model.tracking import (
    delete_prediction,
    get_predictions,
    get_starting_bankroll,
    predictions_csv,
    restore_predictions_csv,
    set_starting_bankroll,
    settle_prediction,
)

st.set_page_config(page_title="ATP Bet Tracking", page_icon="📈", layout="wide")
st.title("📈 Betting performance and closing-line value")
st.caption(
    "Save predictions from the main model page, then settle the bets here. "
    "The dashboard separates model quality from realized betting results."
)

with st.expander("Tracking storage and backup", expanded=False):
    st.warning(
        "Tracking is stored on the app's local filesystem. Hosted Streamlit instances can reset local files "
        "during a restart/redeploy, so download a CSV backup regularly. Your model data pipeline does not "
        "commit this private betting log to GitHub."
    )
    st.download_button(
        "Download tracking CSV backup",
        data=predictions_csv(),
        file_name="atp_bet_tracking_backup.csv",
        mime="text/csv",
        use_container_width=True,
    )
    upload = st.file_uploader("Restore/merge a previous tracking CSV", type=["csv"])
    replace = st.checkbox("Replace the current log instead of merging", value=False)
    if upload is not None and st.button("Restore tracking backup"):
        try:
            count = restore_predictions_csv(upload.getvalue().decode("utf-8"), replace=replace)
            st.success(f"Tracking backup restored. {count} rows are now stored.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not restore backup: {exc}")

rows = get_predictions()

st.subheader("Bankroll settings")
starting_bankroll = get_starting_bankroll()
bankroll_default = float(starting_bankroll or 0.0)
b1, b2 = st.columns([1, 3])
with b1:
    bankroll_input = st.number_input(
        "Starting bankroll (CA$)", min_value=0.0, value=bankroll_default, step=50.0
    )
with b2:
    st.caption(
        "Optional. Set this once to show current bankroll, peak bankroll, and maximum drawdown. "
        "It does not affect model probabilities or Kelly calculations."
    )
if st.button("Save starting bankroll"):
    set_starting_bankroll(bankroll_input)
    st.success("Starting bankroll saved.")
    st.rerun()

if not rows:
    st.info("No saved predictions yet. Go to the main model page and click ‘Save prediction / bet to tracking’. ")
    st.stop()

records = []
for x in rows:
    records.append(
        {
            "ID": x.id,
            "Created": pd.to_datetime(x.created_at, errors="coerce", utc=True),
            "Match": f"{x.player_a} vs {x.player_b}",
            "Player A": x.player_a,
            "Player B": x.player_b,
            "Surface": x.surface,
            "Tournament": x.tournament,
            "Model P(A)": x.model_probability_a,
            "Opening odds A": x.odds_a,
            "Opening odds B": x.odds_b,
            "Opening no-vig P(A)": x.no_vig_probability_a,
            "Opening edge": x.edge,
            "EV": x.expected_value,
            "Quarter Kelly": x.quarter_kelly,
            "Stake": x.stake,
            "Result A": x.result_a,
            "Profit": x.profit,
            "Closing odds A": x.closing_odds_a,
            "Closing odds B": x.closing_odds_b,
            "Closing no-vig P(A)": x.closing_no_vig_probability_a,
            "Probability CLV": x.probability_clv,
            "Price CLV": x.price_clv,
        }
    )

df = pd.DataFrame(records).sort_values("Created")
df["Placed bet"] = df["Stake"].fillna(0) > 0
settled = df[df["Profit"].notna() & df["Placed bet"]].copy()
clv_rows = settled[settled["Price CLV"].notna()].copy()

wins = int((settled["Result A"] == True).sum()) if not settled.empty else 0
losses = int((settled["Result A"] == False).sum()) if not settled.empty else 0
total_stake = float(settled["Stake"].sum()) if not settled.empty else 0.0
profit = float(settled["Profit"].sum()) if not settled.empty else 0.0
roi = profit / total_stake if total_stake else np.nan
avg_edge = float(settled["Opening edge"].mean()) if not settled.empty else np.nan
avg_clv = float(clv_rows["Price CLV"].mean()) if not clv_rows.empty else np.nan
beat_close = float((clv_rows["Price CLV"] > 0).mean()) if not clv_rows.empty else np.nan

start_bank = get_starting_bankroll()
current_bankroll = None
peak_bankroll = None
max_drawdown = None
if start_bank is not None:
    curve = settled.sort_values("Created").copy()
    if curve.empty:
        current_bankroll = peak_bankroll = float(start_bank)
        max_drawdown = 0.0
    else:
        curve["Bankroll"] = float(start_bank) + curve["Profit"].cumsum()
        curve["Peak"] = curve["Bankroll"].cummax().clip(lower=1e-9)
        curve["Drawdown"] = curve["Bankroll"] / curve["Peak"] - 1.0
        current_bankroll = float(curve["Bankroll"].iloc[-1])
        peak_bankroll = float(curve["Peak"].max())
        max_drawdown = float(curve["Drawdown"].min())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Settled bets", f"{len(settled)}", f"{wins}–{losses}" if len(settled) else None)
m2.metric("Profit", f"CA${profit:,.2f}")
m3.metric("ROI", f"{roi:+.1%}" if np.isfinite(roi) else "—")
m4.metric("Average model edge", f"{avg_edge:+.1%}" if np.isfinite(avg_edge) else "—")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Average price CLV", f"{avg_clv:+.2%}" if np.isfinite(avg_clv) else "—")
m6.metric("Bets beating close", f"{beat_close:.1%}" if np.isfinite(beat_close) else "—")
m7.metric("Current bankroll", f"CA${current_bankroll:,.2f}" if current_bankroll is not None else "Set bankroll")
m8.metric("Max drawdown", f"{max_drawdown:.1%}" if max_drawdown is not None else "—")

st.subheader("Prediction and bet log")
st.dataframe(
    df.drop(columns=["Player A", "Player B", "Placed bet"]).style.format(
        {
            "Model P(A)": "{:.1%}",
            "Opening no-vig P(A)": "{:.1%}",
            "Opening edge": "{:+.1%}",
            "EV": "{:+.1%}",
            "Quarter Kelly": "{:.2%}",
            "Closing no-vig P(A)": "{:.1%}",
            "Probability CLV": "{:+.1%}",
            "Price CLV": "{:+.1%}",
            "Stake": "CA${:,.2f}",
            "Profit": "CA${:,.2f}",
        },
        na_rep="—",
    ),
    hide_index=True,
    use_container_width=True,
)

if not settled.empty:
    st.subheader("Profit / bankroll progression")
    curve = settled.sort_values("Created").copy()
    curve["Cumulative profit"] = curve["Profit"].cumsum()
    if start_bank is not None:
        curve["Bankroll"] = float(start_bank) + curve["Cumulative profit"]
        st.line_chart(curve.set_index("Created")[["Bankroll"]])
    else:
        st.line_chart(curve.set_index("Created")[["Cumulative profit"]])

if not clv_rows.empty:
    st.subheader("Closing-line value")
    chart = clv_rows.sort_values("Created").copy()
    chart["Cumulative average price CLV"] = chart["Price CLV"].expanding().mean()
    st.line_chart(chart.set_index("Created")[["Price CLV", "Cumulative average price CLV"]])
    st.caption(
        "Price CLV = odds you saved ÷ closing odds − 1. Positive means your price beat the close. "
        "Probability CLV = closing no-vig probability − opening no-vig probability."
    )

if not settled.empty:
    st.subheader("Performance breakdowns")
    left, right = st.columns(2)
    with left:
        surface = settled.groupby("Surface", dropna=False).agg(
            Bets=("ID", "count"), Stake=("Stake", "sum"), Profit=("Profit", "sum")
        ).reset_index()
        surface["ROI"] = np.where(surface["Stake"] > 0, surface["Profit"] / surface["Stake"], np.nan)
        st.write("**By surface**")
        st.dataframe(surface, hide_index=True, use_container_width=True)
    with right:
        buckets = settled.copy()
        buckets["Edge bucket"] = pd.cut(
            buckets["Opening edge"],
            bins=[-np.inf, 0.02, 0.04, 0.06, 0.10, np.inf],
            labels=["<2%", "2–4%", "4–6%", "6–10%", "10%+"],
        )
        edge = buckets.groupby("Edge bucket", observed=False).agg(
            Bets=("ID", "count"), Stake=("Stake", "sum"), Profit=("Profit", "sum")
        ).reset_index()
        edge["ROI"] = np.where(edge["Stake"] > 0, edge["Profit"] / edge["Stake"], np.nan)
        st.write("**By model edge**")
        st.dataframe(edge, hide_index=True, use_container_width=True)

st.divider()
st.subheader("Settle a saved bet")
open_rows = [x for x in rows if x.result_a is None]
if open_rows:
    labels = {
        x.id: f"#{x.id} — {x.player_a} vs {x.player_b} ({x.tournament or x.surface})"
        for x in open_rows
    }
    pid = st.selectbox("Prediction", options=list(labels), format_func=lambda x: labels[x])
    selected = next(x for x in open_rows if x.id == pid)
    result = st.selectbox(f"Did {selected.player_a} win?", ["Yes", "No"])
    c1, c2 = st.columns(2)
    with c1:
        closing_a = st.number_input("Closing decimal odds A", min_value=0.0, value=0.0, step=0.01)
    with c2:
        closing_b = st.number_input("Closing decimal odds B", min_value=0.0, value=0.0, step=0.01)
    st.caption("Closing odds are optional. Enter both sides if you want no-vig probability CLV calculated.")
    if st.button("Settle and calculate CLV", type="primary"):
        try:
            settle_prediction(int(pid), result == "Yes", closing_a or None, closing_b or None)
            st.success("Prediction settled and tracking metrics updated.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not settle prediction: {exc}")
else:
    st.success("There are no unsettled saved predictions.")

with st.expander("Delete a tracking row"):
    delete_id = st.number_input("Prediction ID to delete", min_value=1, step=1, key="delete_prediction_id")
    st.warning("Deletion cannot be undone unless you have a CSV backup.")
    if st.button("Delete prediction"):
        try:
            delete_prediction(int(delete_id))
            st.success(f"Prediction #{int(delete_id)} deleted.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

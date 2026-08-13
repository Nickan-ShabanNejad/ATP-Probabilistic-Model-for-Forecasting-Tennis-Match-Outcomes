
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
from atp_model.tracking import get_predictions, settle_prediction

st.set_page_config(page_title="ATP Bet Tracking", page_icon="📈", layout="wide")
st.title("📈 ROI and Closing-Line Value")

rows = get_predictions()
if not rows:
    st.info("No saved predictions yet.")
    st.stop()

df = pd.DataFrame(
    [
        {
            "ID": x.id,
            "Created": x.created_at,
            "Match": f"{x.player_a} vs {x.player_b}",
            "Surface": x.surface,
            "Model P(A)": x.model_probability_a,
            "Opening odds A": x.odds_a,
            "Opening no-vig P(A)": x.no_vig_probability_a,
            "Opening edge": x.edge,
            "EV": x.expected_value,
            "Closing odds A": x.closing_odds_a,
            "Closing odds B": getattr(x, "closing_odds_b", None),
            "Closing no-vig P(A)": getattr(
                x, "closing_no_vig_probability_a", None
            ),
            "Probability CLV": getattr(x, "probability_clv", None),
            "Price CLV": getattr(x, "price_clv", None),
            "Result A": x.result_a,
            "Stake": x.stake,
            "Profit": x.profit,
        }
        for x in rows
    ]
)

st.dataframe(
    df.style.format(
        {
            "Model P(A)": "{:.1%}",
            "Opening no-vig P(A)": "{:.1%}",
            "Opening edge": "{:+.1%}",
            "EV": "{:+.1%}",
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

settled = df[df["Profit"].notna()]
clv_rows = df[df["Price CLV"].notna()]

c1, c2, c3, c4, c5 = st.columns(5)
total_stake = settled["Stake"].sum()
profit = settled["Profit"].sum()
c1.metric("Tracked profit", f"CA${profit:,.2f}")
c2.metric("Tracked stake", f"CA${total_stake:,.2f}")
c3.metric("ROI", f"{profit / total_stake:.1%}" if total_stake else "—")
c4.metric(
    "Average price CLV",
    f"{clv_rows['Price CLV'].mean():+.2%}" if not clv_rows.empty else "—",
)
c5.metric(
    "Bets beating close",
    f"{(clv_rows['Price CLV'] > 0).mean():.1%}" if not clv_rows.empty else "—",
)

if not clv_rows.empty:
    st.subheader("CLV history")
    chart = clv_rows.sort_values("Created").copy()
    chart["Cumulative average price CLV"] = chart["Price CLV"].expanding().mean()
    st.line_chart(
        chart.set_index("Created")[["Price CLV", "Cumulative average price CLV"]]
    )

st.caption(
    "Price CLV = opening decimal odds ÷ closing decimal odds − 1. "
    "Positive values mean your saved price beat the closing price."
)

st.subheader("Settle a prediction")
pid = st.number_input("Prediction ID", min_value=1, step=1)
result = st.selectbox("Did Player A win?", ["Yes", "No"])
closing_a = st.number_input(
    "Closing decimal odds A", min_value=0.0, value=0.0, step=0.01
)
closing_b = st.number_input(
    "Closing decimal odds B", min_value=0.0, value=0.0, step=0.01
)

if st.button("Settle and calculate CLV"):
    settle_prediction(
        int(pid),
        result == "Yes",
        closing_a or None,
        closing_b or None,
    )
    st.success("Prediction settled and CLV calculated.")
    st.rerun()

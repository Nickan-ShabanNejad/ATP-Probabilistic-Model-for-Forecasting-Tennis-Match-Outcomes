
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="ATP Data Health", page_icon="🩺", layout="wide")
st.title("🩺 Data and Model Health")

freshness_path = ROOT / "data/generated/freshness.json"
metrics_path = ROOT / "data/generated/metrics.json"
source_history_path = ROOT / "data/generated/source_history.csv"
model_history_path = ROOT / "data/generated/model_history.csv"

if not freshness_path.exists():
    st.error("No freshness report exists yet. Run the GitHub workflow.")
    st.stop()

freshness = json.loads(freshness_path.read_text(encoding="utf-8"))
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

matches = freshness.get("matches", {})
rankings = freshness.get("reference", {}).get("rankings", {})
charting = freshness.get("match_charting_project", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest match date", matches.get("latest_tourney_date", "unknown"))
c2.metric("Match-data age", f"{matches.get('age_days', '?')} days")
c3.metric("Master matches", f"{matches.get('master_rows', 0):,}")
c4.metric("Current-year matches", f"{matches.get('current_year_rows', 0):,}")

if matches.get("stale"):
    st.error(
        f"Match data is older than {matches.get('freshness_threshold_days', 7)} days. "
        "Predictions should be treated as stale."
    )
else:
    st.success("Match-data freshness is within the configured threshold.")

st.subheader("Source contribution")
contribution = pd.DataFrame(
    [
        {"Source": source, "Selected rows": rows}
        for source, rows in matches.get("selected_rows_by_source", {}).items()
    ]
)
if not contribution.empty:
    st.dataframe(contribution, hide_index=True, use_container_width=True)
else:
    st.warning("No source contribution data is available.")

st.subheader("Rankings")
st.write(f"Method: **{rankings.get('method', 'unavailable')}**")
st.write(f"Ranking date: **{rankings.get('ranking_date', 'unknown')}**")
if rankings.get("warning"):
    st.warning(rankings["warning"])

st.subheader("Match Charting Project")
if charting.get("available"):
    st.success("All configured Match Charting files downloaded and parsed.")
else:
    st.warning("At least one Match Charting file is unavailable or invalid.")
st.json(charting)

st.subheader("Current model evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
m2.metric("Log loss", f"{metrics.get('log_loss', 0):.4f}")
m3.metric("Brier score", f"{metrics.get('brier', 0):.4f}")

if metrics.get("accuracy", 0) > 0.9:
    st.error(
        "Accuracy above 90% is suspicious for professional tennis and should trigger "
        "a leakage or evaluation audit."
    )

if source_history_path.exists():
    st.subheader("Source freshness history")
    source_history = pd.read_csv(source_history_path)
    st.dataframe(source_history.tail(30), hide_index=True, use_container_width=True)

if model_history_path.exists():
    st.subheader("Model-performance history")
    model_history = pd.read_csv(model_history_path)
    st.dataframe(model_history.tail(30), hide_index=True, use_container_width=True)
    if len(model_history) > 1:
        st.line_chart(model_history.set_index("trained_at_utc")[["accuracy", "log_loss", "brier"]])

with st.expander("Complete freshness report"):
    st.json(freshness)

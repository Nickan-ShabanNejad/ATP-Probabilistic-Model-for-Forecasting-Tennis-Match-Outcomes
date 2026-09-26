
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atp_model.ui import setup_page
setup_page("ATP Data Health")
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

latest_raw = matches.get("latest_tourney_date") or metrics.get("latest_data_date")
latest_dt = pd.to_datetime(str(latest_raw), format="%Y%m%d", errors="coerce")
if pd.isna(latest_dt):
    latest_dt = pd.to_datetime(latest_raw, errors="coerce")
dynamic_age = (
    max(0, int((pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - latest_dt.normalize()).days))
    if pd.notna(latest_dt) else matches.get("age_days")
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest match date", latest_raw or "unknown")
c2.metric("Match-data age", f"{dynamic_age if dynamic_age is not None else '?'} days")
c3.metric("Master matches", f"{matches.get('master_rows', 0):,}")
c4.metric("Current-year matches", f"{matches.get('current_year_rows', 0):,}")

if dynamic_age is not None and dynamic_age > matches.get("freshness_threshold_days", 7):
    st.error(
        f"Match data is older than {matches.get('freshness_threshold_days', 7)} days. "
        "Predictions should be treated as stale."
    )
else:
    st.success("Match-data freshness is within the configured threshold.")

# ---------------------------------------------------------------------------
# v0.4: the honest scoreboard - model vs the betting market
# ---------------------------------------------------------------------------
bench = metrics.get("market_benchmark") or {}
blend = metrics.get("market_blend") or {}
backtests = metrics.get("backtests") or {}
st.subheader("Model vs the betting market")
if not bench.get("available"):
    st.info("Retrain with `python scripts/train_model.py` to generate the market benchmark.")
else:
    st.caption(
        "Every number here is walk-forward: each season is predicted by a model trained only on earlier seasons, "
        f"then compared with {bench.get('source', 'closing odds')} for the same matches "
        f"({bench.get('matches_with_odds', 0):,} matches, {min(bench.get('years', [0]))}–{max(bench.get('years', [0]))})."
    )
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Model log loss", f"{bench['model_log_loss']:.4f}", help="Lower is better. 0.693 = coin flip.")
    b2.metric("Pinnacle log loss", f"{bench['pinnacle_log_loss']:.4f}")
    b3.metric("Model accuracy", f"{bench['model_accuracy']:.1%}", f"{bench['model_accuracy'] - bench['pinnacle_accuracy']:+.1%} vs Pinnacle")
    b4.metric("Model weight in blend", f"{blend.get('model_coef', 0):.2f}",
              help="How much the model's opinion adds on top of Pinnacle's price (0 = nothing measurable).")

    per_year = pd.DataFrame(bench.get("per_year", []))
    if not per_year.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=per_year["year"], y=per_year["model_log_loss"], name="Model", mode="lines+markers",
                                 line=dict(color="#3987e5", width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=per_year["year"], y=per_year["pinnacle_log_loss"], name="Closing market", mode="lines+markers",
                                 line=dict(color="#d95926", width=2), marker=dict(size=8)))
        fig.update_layout(
            title="Log loss by season (lower is better)", height=320, margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
            xaxis=dict(dtick=1, gridcolor="rgba(139,148,158,.15)"), yaxis=dict(gridcolor="rgba(139,148,158,.15)", tickformat=".3f"),
        )
        st.plotly_chart(fig, use_container_width=True)
        if "market_source" in per_year.columns and (per_year["market_source"] != "Pinnacle close").any():
            st.caption("Seasons without Pinnacle odds in the archive are compared with the market-average closing price.")

    rows = []
    labels = {
        "model_only_ev2_edge2": "Model only · EV ≥ 2% & edge ≥ 2% (old default)",
        "model_only_ev5_edge5": "Model only · EV ≥ 5% & edge ≥ 5%",
        "market_blend_ev2": "Market-anchored · EV ≥ 2% at Pinnacle (new default)",
    }
    for rule, s in backtests.items():
        if not s.get("bets"):
            rows.append({"Rule": labels.get(rule, rule), "Bets": 0})
            continue
        rows.append({
            "Rule": labels.get(rule, rule), "Bets": s["bets"], "Share of matches": s["share_of_matches"],
            "Win rate": s["win_rate"], "Avg odds": s["avg_odds"], "Claimed EV": s["avg_claimed_ev"],
            "Actual ROI (flat)": s["flat_roi"], "ROI ± 1 s.e.": s.get("flat_roi_se"),
        })
    if rows:
        st.markdown("**Betting-rule backtest at Pinnacle closing prices**")
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True,
            column_config={
                "Share of matches": st.column_config.NumberColumn(format="percent"),
                "Win rate": st.column_config.NumberColumn(format="percent"),
                "Avg odds": st.column_config.NumberColumn(format="%.2f"),
                "Claimed EV": st.column_config.NumberColumn(format="percent"),
                "Actual ROI (flat)": st.column_config.NumberColumn(format="percent"),
                "ROI ± 1 s.e.": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(
            "Closing prices are the hardest test: earlier prices and softer sportsbooks can be beaten more often. "
            "The live test that matters is closing-line value (CLV) on the Tracking page."
        )

dq = metrics.get("data_quality") or {}
if dq:
    with st.expander("Data-quality cleaning applied before training"):
        st.write({
            "Rows in": dq.get("input_rows"),
            "ITF / Challenger rows removed": dq.get("lower_tier_rows_removed"),
            "Qualifying rows removed": dq.get("qualifying_rows_removed"),
            "Duplicate matches removed (same match from two providers)": dq.get("cross_provider_duplicates_removed"),
            "Tournament levels corrected": dq.get("levels_corrected"),
            "Rows used": dq.get("output_rows"),
        })

matchstat = matches.get("matchstat", {})
st.subheader("Matchstat production feed")
if matchstat.get("ok"):
    st.success(
        f"Matchstat refresh succeeded: {matchstat.get('usable_match_rows', 0):,} usable current-season rows "
        f"from {matchstat.get('ranking_players_requested', 0):,} ranked players."
    )
    if matchstat.get("player_failure_count", 0):
        st.warning(f"{matchstat.get('player_failure_count')} player-history calls failed; see freshness report for details.")
else:
    st.error(f"Matchstat refresh unavailable: {matchstat.get('error') or matchstat.get('reason') or 'unknown reason'}")

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
rank_date = pd.to_datetime(str(rankings.get("ranking_date", "")), format="%Y%m%d", errors="coerce")
if pd.notna(rank_date):
    rank_age = max(0, int((pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - rank_date.normalize()).days))
    st.write(f"Ranking age: **{rank_age} days**")
    if rank_age > 7 and not rankings.get("warning"):
        st.warning(f"The ranking snapshot is {rank_age} days old even though match data may be current.")
if rankings.get("warning"):
    st.warning(rankings["warning"])

st.subheader("Match Charting Project")
if charting.get("available"):
    st.success("All configured Match Charting files downloaded and parsed.")
else:
    st.warning("At least one Match Charting file is unavailable or invalid.")
st.json(charting)

st.subheader("Model selection")
st.write(
    f"Selected model: **{metrics.get('selected_model', 'unknown')}**"
)
candidate_metrics = metrics.get("candidate_metrics", {})
if candidate_metrics:
    comparison = pd.DataFrame.from_dict(candidate_metrics, orient="index")
    comparison.index.name = "Model"
    st.dataframe(comparison.reset_index(), hide_index=True, use_container_width=True)

st.subheader("Current model evaluation")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
m2.metric("Log loss", f"{metrics.get('log_loss', 0):.4f}")
m3.metric("Brier score", f"{metrics.get('brier', 0):.4f}")
m4.metric("ECE (10 bins)", f"{metrics.get('ece_10bin', 0):.4f}")

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

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atp_model.matchstat import MatchstatClient
from atp_model.model_service import load_bundle, load_state, predict_match
from atp_model.odds import decimal
from atp_model.tournament_features import canonical_tournament, encode_tournament_level

st.set_page_config(page_title="ATP Value Board", page_icon="💰", layout="wide")
st.title("💰 Today's automated value board")
st.caption("Upcoming ATP matches priced with Pinnacle through Matchstat. Only pre-match events are considered.")


def _matchstat_key() -> str:
    key = os.getenv("MATCHSTAT_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("MATCHSTAT_API_KEY", "")).strip()
    except Exception:
        return ""


def _tournament_context() -> dict[str, dict]:
    path = ROOT / "data/generated/master_matches.csv.gz"
    if not path.exists():
        return {}
    try:
        matches = pd.read_csv(
            path,
            usecols=lambda c: c in {
                "tourney_name", "surface", "tourney_level", "indoor", "tourney_date"
            },
            low_memory=False,
        )
    except Exception:
        return {}
    required = {"tourney_name", "surface", "tourney_date"}
    if matches.empty or not required.issubset(matches.columns):
        return {}
    matches = matches.dropna(subset=["tourney_name", "surface"]).copy()
    matches["tourney_date"] = pd.to_numeric(matches["tourney_date"], errors="coerce")
    matches["key"] = matches["tourney_name"].map(canonical_tournament)
    latest = matches.sort_values("tourney_date").drop_duplicates("key", keep="last")
    out = {}
    for _, row in latest.iterrows():
        out[str(row["key"])] = {
            "surface": str(row["surface"]).title(),
            "level": encode_tournament_level(row.get("tourney_level", "A")),
            "indoor": str(row.get("indoor", "")).strip().upper() in {"I", "1", "TRUE"},
            "tournament": str(row["tourney_name"]),
        }
    return out


api_key = _matchstat_key()
if not api_key:
    st.info(
        "The Value Board needs your Matchstat key at app runtime. Your GitHub Actions secret does not "
        "automatically pass into Streamlit Cloud. Add MATCHSTAT_API_KEY under Manage app → Settings → Secrets."
    )
    st.code('MATCHSTAT_API_KEY = "your-rotated-RapidAPI-key"', language="toml")
    st.stop()

try:
    client = MatchstatClient(api_key=api_key)
    events = client.upcoming_events("atp", max_events=80)
except Exception as exc:
    st.error(f"Matchstat upcoming-events request failed: {exc}")
    st.stop()

if not events:
    st.info("Matchstat returned no upcoming ATP events.")
    st.stop()

state = load_state()
bundle = load_bundle()
player_names = set(state["player"].dropna().astype(str).unique())
context = _tournament_context()
now_ts = datetime.now(timezone.utc).timestamp()
max_ts = now_ts + 48 * 3600

rows = []
skipped_no_context = 0
skipped_no_pinnacle = 0
errors = []

for event in events:
    try:
        if not isinstance(event, dict):
            continue
        status = str(event.get("status") or "").casefold()
        if status not in {"not started", "upcoming", "scheduled", ""}:
            continue
        start_ts = float(event.get("startTimestamp") or 0)
        if start_ts and (start_ts < now_ts or start_ts > max_ts):
            continue

        a = str(event.get("participant1") or "").strip()
        b = str(event.get("participant2") or "").strip()
        event_id = event.get("id")
        league = str(event.get("league") or "").strip()
        if not a or not b or event_id is None or a not in player_names or b not in player_names:
            continue

        ctx = context.get(canonical_tournament(league))
        if not ctx:
            skipped_no_context += 1
            continue
        surface = ctx["surface"]
        if surface not in {"Hard", "Clay", "Grass"}:
            skipped_no_context += 1
            continue

        odds_payload = client.compared_odds(event_id, market_id=1)
        offers = odds_payload.get("results", []) if isinstance(odds_payload, dict) else []
        pinnacle = next(
            (x for x in offers if str(x.get("bookmaker") or "").casefold() == "pinnacle"),
            None,
        )
        if not pinnacle:
            skipped_no_pinnacle += 1
            continue
        odds_a = decimal(pinnacle.get("od1"))
        odds_b = decimal(pinnacle.get("od2"))
        if odds_a is None or odds_b is None:
            skipped_no_pinnacle += 1
            continue

        arow = state[(state.player == a) & (state.surface == surface)]
        brow = state[(state.player == b) & (state.surface == surface)]
        if arow.empty or brow.empty:
            continue
        rank_a = int(pd.to_numeric(arow.iloc[0].get("rank"), errors="coerce") or 999)
        rank_b = int(pd.to_numeric(brow.iloc[0].get("rank"), errors="coerce") or 999)
        prediction_year = datetime.fromtimestamp(start_ts, tz=timezone.utc).year if start_ts else datetime.now(timezone.utc).year

        result = predict_match(
            state,
            bundle,
            a,
            b,
            surface,
            rank_a,
            rank_b,
            odds_a,
            odds_b,
            tournament_level=ctx["level"],
            tournament=league or ctx["tournament"],
            prediction_year=prediction_year,
            indoor=ctx["indoor"],
        )
        rows.append(
            {
                "Start (UTC)": datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if start_ts else "",
                "Match": f"{a} vs {b}",
                "Tournament": league,
                "Surface": surface,
                "P(A)": result["probability_a"],
                "Market P(A)": result["market_probability_a"],
                "Edge": result["edge"],
                "EV": result["ev"],
                "Pinnacle A": odds_a,
                "Pinnacle B": odds_b,
                "Quarter Kelly": result["quarter_kelly"],
            }
        )
    except Exception as exc:
        errors.append(f"{event.get('name', event.get('id', 'event'))}: {exc}")

if rows:
    out = pd.DataFrame(rows).sort_values(["EV", "Edge"], ascending=False)
    st.dataframe(
        out,
        hide_index=True,
        use_container_width=True,
        column_config={
            "P(A)": st.column_config.NumberColumn(format="%.1%%"),
            "Market P(A)": st.column_config.NumberColumn(format="%.1%%"),
            "Edge": st.column_config.NumberColumn(format="%+.1%%"),
            "EV": st.column_config.NumberColumn(format="%+.1%%"),
            "Quarter Kelly": st.column_config.NumberColumn(format="%.2%%"),
            "Pinnacle A": st.column_config.NumberColumn(format="%.3f"),
            "Pinnacle B": st.column_config.NumberColumn(format="%.3f"),
        },
    )
else:
    st.warning(
        "No upcoming events could be fully matched to the model with a known tournament surface and Pinnacle price."
    )

with st.expander("Value-board diagnostics"):
    st.write(f"Upcoming events inspected: **{len(events)}**")
    st.write(f"Skipped because tournament/surface context was unavailable: **{skipped_no_context}**")
    st.write(f"Skipped because Pinnacle moneyline was unavailable: **{skipped_no_pinnacle}**")
    if errors:
        st.write("Per-event errors:")
        st.code("\n".join(errors[:20]))

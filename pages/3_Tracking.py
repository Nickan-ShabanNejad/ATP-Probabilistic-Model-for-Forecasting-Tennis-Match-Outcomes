from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from atp_model.market_store import last_pre_start_snapshot
from atp_model.supabase_store import (
    configured as supabase_configured,
    current_bankroll,
    delete_bet,
    get_starting_bankroll,
    get_tracking_mode,
    set_tracking_mode,
    healthcheck,
    list_bets,
    list_predictions,
    set_starting_bankroll,
    settle_bet,
)

st.set_page_config(page_title="ATP Model Tracking", page_icon="📈", layout="wide")
st.title("📈 Model performance & bet tracking")
st.caption(
    "Persistent Supabase tracking for placed bets, ROI/yield, bankroll, CLV, and model calibration. "
    "The live board also keeps one current pre-match prediction per match/market/model version."
)

if not supabase_configured():
    st.error("Supabase is not configured for this app.")
    st.code(
        'SUPABASE_URL = "https://your-project.supabase.co"\nSUPABASE_KEY = "your-secret-key"',
        language="toml",
    )
    st.stop()

ok, status = healthcheck()
if not ok:
    st.error(f"Could not connect to Supabase: {status}")
    if "NameResolutionError" in status or "Failed to resolve" in status or "Name or service not known" in status:
        st.warning(
            "This is a Supabase URL/DNS problem, not a table or API-key error. Go to the Supabase project home, "
            "make sure the project status is Active/Healthy (not 'Coming up…'), click Copy beside the Project URL, "
            "replace SUPABASE_URL in Streamlit Secrets, save, then reboot the app."
        )
    else:
        st.caption("If you just created the tables, also run supabase_tracking_upgrade.sql from this repo in Supabase SQL Editor.")
    st.stop()

st.success("Supabase connected — tracking survives Streamlit reboots and redeploys.")

# ----------------------------- Tracking mode / bankroll -----------------------------
st.subheader("Staking basis")
mode = get_tracking_mode()
choice = st.radio(
    "How do you want to track stakes?",
    ["Percentage only", "Currency bankroll (CA$)"],
    index=0 if mode == "percentage" else 1,
    horizontal=True,
)
selected_mode = "percentage" if choice == "Percentage only" else "currency"
if selected_mode != mode:
    if st.button("Save staking mode", type="secondary"):
        set_tracking_mode(selected_mode)
        st.success("Staking mode saved.")
        st.rerun()

if mode == "percentage":
    starting = 100.0
    st.info(
        "Percentage-only mode is active. You do not need to enter your real bankroll. "
        "The tracker uses a normalized bankroll index starting at 100.00; a 2% stake is recorded as 2% of the current index."
    )
else:
    st.subheader("Bankroll")
    starting = get_starting_bankroll()
    left, right = st.columns([1, 3])
    with left:
        bankroll_input = st.number_input(
            "Starting bankroll (CA$)",
            min_value=0.0,
            value=float(starting or 0.0),
            step=50.0,
        )
    with right:
        st.caption(
            "This value is stored in Supabase. Current bankroll = starting bankroll + realized P&L from settled bets."
        )
    if st.button("Save starting bankroll", type="secondary"):
        try:
            set_starting_bankroll(float(bankroll_input))
            st.success("Starting bankroll saved permanently.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save bankroll: {exc}")

try:
    bets = list_bets()
    predictions = list_predictions()
except Exception as exc:
    st.error(f"Could not load tracking data: {exc}")
    st.stop()

bets_df = pd.DataFrame(bets)
pred_df = pd.DataFrame(predictions)

# ----------------------------- Top metrics -----------------------------
if bets_df.empty:
    settled = pd.DataFrame()
else:
    bets_df["created_at"] = pd.to_datetime(bets_df.get("created_at"), utc=True, errors="coerce")
    bets_df["match_date"] = pd.to_datetime(bets_df.get("match_date"), utc=True, errors="coerce")
    bets_df["settled_at"] = pd.to_datetime(bets_df.get("settled_at"), utc=True, errors="coerce")
    for col in [
        "model_probability", "model_fair_odds", "odds_taken", "closing_odds", "edge", "expected_value",
        "bankroll_before", "kelly_fraction", "stake_percent", "stake_amount", "profit_loss", "clv", "bankroll_after",
    ]:
        if col in bets_df:
            bets_df[col] = pd.to_numeric(bets_df[col], errors="coerce")
    settled = bets_df[bets_df["profit_loss"].notna()].copy()

wins = int((settled.get("result", pd.Series(dtype=str)).astype(str).str.casefold() == "win").sum()) if not settled.empty else 0
losses = int((settled.get("result", pd.Series(dtype=str)).astype(str).str.casefold() == "loss").sum()) if not settled.empty else 0
voids = int((settled.get("result", pd.Series(dtype=str)).astype(str).str.casefold() == "void").sum()) if not settled.empty else 0
stake_total = float(settled.get("stake_amount", pd.Series(dtype=float)).fillna(0).sum()) if not settled.empty else 0.0
profit = float(settled.get("profit_loss", pd.Series(dtype=float)).fillna(0).sum()) if not settled.empty else 0.0
yield_pct = profit / stake_total if stake_total else np.nan
avg_clv = float(settled["clv"].dropna().mean()) if not settled.empty and "clv" in settled and settled["clv"].notna().any() else np.nan
beat_close = float((settled["clv"].dropna() > 0).mean()) if not settled.empty and "clv" in settled and settled["clv"].notna().any() else np.nan
bank = current_bankroll()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Settled bets", str(len(settled)), f"{wins}W–{losses}L–{voids}V" if len(settled) else None)
m2.metric("Bankroll P&L" if mode == "percentage" else "Profit", f"{profit:+.2f} pts" if mode == "percentage" else f"CA${profit:,.2f}")
m3.metric("Yield / ROI on stake", f"{yield_pct:+.1%}" if np.isfinite(yield_pct) else "—")
m4.metric("Bankroll index" if mode == "percentage" else "Current bankroll", f"{bank:,.2f}" if mode == "percentage" and bank is not None else (f"CA${bank:,.2f}" if bank is not None else "Set bankroll"))

m5, m6, m7, m8 = st.columns(4)
m5.metric("Average CLV", f"{avg_clv:+.2%}" if np.isfinite(avg_clv) else "—")
m6.metric("Bets beating close", f"{beat_close:.1%}" if np.isfinite(beat_close) else "—")
m7.metric("Open bets", str(int(bets_df["profit_loss"].isna().sum())) if not bets_df.empty else "0")
m8.metric("Tracked predictions", str(len(pred_df)))

# ----------------------------- Calibration -----------------------------
st.subheader("Model calibration")
if not pred_df.empty and "actual_result" in pred_df:
    pred_df["model_probability"] = pd.to_numeric(pred_df.get("model_probability"), errors="coerce")
    pred_df["actual_result"] = pd.to_numeric(pred_df.get("actual_result"), errors="coerce")
    evaluated = pred_df[pred_df["model_probability"].notna() & pred_df["actual_result"].notna()].copy()
else:
    evaluated = pd.DataFrame()

if evaluated.empty:
    st.info(
        "No settled prediction outcomes yet. The background settlement workflow grades saved predictions after Matchstat marks a match finished. "
        "That includes NO BET matches, so Brier score and log loss evaluate the model rather than only the bets you chose to place."
    )
else:
    p = evaluated["model_probability"].clip(1e-6, 1 - 1e-6)
    y = evaluated["actual_result"]
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Evaluated predictions", len(evaluated))
    c2.metric("Brier score", f"{brier:.4f}")
    c3.metric("Log loss", f"{logloss:.4f}")

    calib = evaluated.copy()
    calib["Probability bucket"] = pd.cut(
        calib["model_probability"],
        bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        include_lowest=True,
    )
    calib_table = calib.groupby("Probability bucket", observed=False).agg(
        Predictions=("id", "count"),
        Average_model_probability=("model_probability", "mean"),
        Actual_win_rate=("actual_result", "mean"),
    ).reset_index()
    st.dataframe(
        calib_table.style.format({"Average_model_probability": "{:.1%}", "Actual_win_rate": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )

# ----------------------------- Bet log -----------------------------
st.subheader("Bet log")
if bets_df.empty:
    st.info("No bets recorded yet. Open a recommended matchup and press ‘I placed … — save bet’. ")
else:
    display = bets_df.copy()
    display["Match"] = display["player_a"].fillna("") + " vs " + display["player_b"].fillna("")
    display = display.rename(columns={
        "created_at": "Placed",
        "match_date": "Match date",
        "model_version": "Version",
        "tournament": "Tournament",
        "tournament_level": "Level",
        "surface": "Surface",
        "market": "Market",
        "selection": "Selection",
        "model_probability": "Model P",
        "model_fair_odds": "Fair odds",
        "odds_taken": "Odds taken",
        "closing_odds": "Close",
        "edge": "Edge",
        "expected_value": "EV",
        "kelly_fraction": "Kelly",
        "stake_percent": "Stake %",
        "stake_amount": "Stake points" if mode == "percentage" else "Stake",
        "result": "Result",
        "profit_loss": "P&L points" if mode == "percentage" else "P&L",
        "clv": "CLV",
    })
    cols = [
        "id", "Placed", "Match date", "Tournament", "Level", "Match", "Market", "Selection",
        "Model P", "Fair odds", "Odds taken", "Close", "Edge", "EV", "Kelly", "Stake %",
        ("Stake points" if mode == "percentage" else "Stake"), "Result",
        ("P&L points" if mode == "percentage" else "P&L"), "CLV", "Version",
    ]
    cols = [c for c in cols if c in display.columns]
    st.dataframe(
        display[cols].sort_values("Placed", ascending=False).style.format({
            "Model P": "{:.1%}", "Edge": "{:+.1%}", "EV": "{:+.1%}", "Kelly": "{:.2%}",
            "Stake %": "{:.2%}",
            **({"Stake points": "{:.2f}", "P&L points": "{:+.2f}"} if mode == "percentage" else {"Stake": "CA${:,.2f}", "P&L": "CA${:,.2f}"}),
            "CLV": "{:+.2%}",
            "Fair odds": "{:.3f}", "Odds taken": "{:.3f}", "Close": "{:.3f}",
        }, na_rep="—"),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download bets CSV",
        data=bets_df.to_csv(index=False).encode("utf-8"),
        file_name="atp_bets_supabase.csv",
        mime="text/csv",
    )

# ----------------------------- Curves -----------------------------
if not settled.empty:
    st.subheader("Bankroll progression")
    curve = settled.sort_values("settled_at").copy()
    curve["Cumulative P&L"] = curve["profit_loss"].fillna(0).cumsum()
    if starting is not None:
        curve["Bankroll index" if mode == "percentage" else "Bankroll"] = float(starting) + curve["Cumulative P&L"]
        curve_col = "Bankroll index" if mode == "percentage" else "Bankroll"
        curve["Peak"] = curve[curve_col].cummax().clip(lower=1e-9)
        curve["Drawdown"] = curve[curve_col] / curve["Peak"] - 1
        st.line_chart(curve.set_index("settled_at")[[curve_col]])
        st.metric("Maximum drawdown", f"{float(curve['Drawdown'].min()):.1%}")
    else:
        st.line_chart(curve.set_index("settled_at")[["Cumulative P&L"]])

    clv_curve = curve[curve["clv"].notna()].copy() if "clv" in curve else pd.DataFrame()
    if not clv_curve.empty:
        st.subheader("Closing-line value")
        clv_curve["Running average CLV"] = clv_curve["clv"].expanding().mean()
        st.line_chart(clv_curve.set_index("settled_at")[["clv", "Running average CLV"]])
        st.caption("Price CLV = odds taken ÷ closing odds − 1. Positive means you beat the Pinnacle close.")

# ----------------------------- Breakdowns -----------------------------
if not settled.empty:
    st.subheader("Performance breakdowns")
    b1, b2 = st.columns(2)
    with b1:
        by_market = settled.groupby("market", dropna=False).agg(
            Bets=("id", "count"), Stake=("stake_amount", "sum"), Profit=("profit_loss", "sum")
        ).reset_index()
        by_market["ROI"] = np.where(by_market["Stake"] > 0, by_market["Profit"] / by_market["Stake"], np.nan)
        st.write("**By market**")
        st.dataframe(by_market.style.format({"Stake": "{:.2f} pts" if mode == "percentage" else "CA${:,.2f}", "Profit": "{:+.2f} pts" if mode == "percentage" else "CA${:,.2f}", "ROI": "{:+.1%}"}), hide_index=True, use_container_width=True)
    with b2:
        by_surface = settled.groupby("surface", dropna=False).agg(
            Bets=("id", "count"), Stake=("stake_amount", "sum"), Profit=("profit_loss", "sum")
        ).reset_index()
        by_surface["ROI"] = np.where(by_surface["Stake"] > 0, by_surface["Profit"] / by_surface["Stake"], np.nan)
        st.write("**By surface**")
        st.dataframe(by_surface.style.format({"Stake": "{:.2f} pts" if mode == "percentage" else "CA${:,.2f}", "Profit": "{:+.2f} pts" if mode == "percentage" else "CA${:,.2f}", "ROI": "{:+.1%}"}), hide_index=True, use_container_width=True)

    b3, b4 = st.columns(2)
    with b3:
        by_level = settled.groupby("tournament_level", dropna=False).agg(
            Bets=("id", "count"), Stake=("stake_amount", "sum"), Profit=("profit_loss", "sum")
        ).reset_index()
        by_level["ROI"] = np.where(by_level["Stake"] > 0, by_level["Profit"] / by_level["Stake"], np.nan)
        st.write("**By tournament level**")
        st.dataframe(by_level.style.format({"Stake": "{:.2f} pts" if mode == "percentage" else "CA${:,.2f}", "Profit": "{:+.2f} pts" if mode == "percentage" else "CA${:,.2f}", "ROI": "{:+.1%}"}), hide_index=True, use_container_width=True)
    with b4:
        edge_buckets = settled.copy()
        edge_buckets["Edge bucket"] = pd.cut(
            edge_buckets["edge"],
            bins=[-np.inf, .02, .04, .06, .10, np.inf],
            labels=["<2%", "2–4%", "4–6%", "6–10%", "10%+"],
        )
        by_edge = edge_buckets.groupby("Edge bucket", observed=False).agg(
            Bets=("id", "count"), Stake=("stake_amount", "sum"), Profit=("profit_loss", "sum")
        ).reset_index()
        by_edge["ROI"] = np.where(by_edge["Stake"] > 0, by_edge["Profit"] / by_edge["Stake"], np.nan)
        st.write("**By model edge**")
        st.dataframe(by_edge.style.format({"Stake": "{:.2f} pts" if mode == "percentage" else "CA${:,.2f}", "Profit": "{:+.2f} pts" if mode == "percentage" else "CA${:,.2f}", "ROI": "{:+.1%}"}), hide_index=True, use_container_width=True)

# ----------------------------- Settlement -----------------------------
st.divider()
st.subheader("Open bets / manual fallback")
open_bets = [] if bets_df.empty else bets_df[bets_df["profit_loss"].isna()].to_dict("records")
if not open_bets:
    st.success("No unsettled bets. Normal completed matches are settled automatically by GitHub Actions.")
else:
    labels = {
        int(x["id"]): f"#{int(x['id'])} — {x.get('selection')} · {x.get('market')} · {x.get('player_a')} vs {x.get('player_b')}"
        for x in open_bets
    }
    bet_id = st.selectbox("Bet", list(labels), format_func=lambda x: labels[x])
    selected = next(x for x in open_bets if int(x["id"]) == int(bet_id))
    result_choice = st.selectbox("Result", ["Win", "Loss", "Void"])

    auto_close = 0.0
    try:
        match_ts = pd.to_datetime(selected.get("match_date"), utc=True, errors="coerce")
        start_ts = match_ts.timestamp() if pd.notna(match_ts) else None
        snap = last_pre_start_snapshot(str(selected.get("match_id") or ""), start_ts)
        if snap:
            if str(selected.get("market")) == "Moneyline":
                if str(snap.get("player_a")) == str(selected.get("selection")):
                    auto_close = float(snap.get("odds_a") or 0)
                elif str(snap.get("player_b")) == str(selected.get("selection")):
                    auto_close = float(snap.get("odds_b") or 0)
            elif str(selected.get("market")) == "Total Sets 3.5":
                if str(selected.get("selection", "")).casefold().startswith("over"):
                    auto_close = float(snap.get("sets_over35") or 0)
                else:
                    auto_close = float(snap.get("sets_under35") or 0)
    except Exception:
        auto_close = 0.0

    closing = st.number_input(
        "Pinnacle closing odds for your selection",
        min_value=0.0,
        value=float(auto_close),
        step=0.01,
        format="%.3f",
    )
    if auto_close:
        st.caption(f"Last local pre-start Pinnacle snapshot found: {auto_close:.3f}.")
    else:
        st.caption("The background worker normally stores the latest observed pre-match Pinnacle quote in Supabase. If no close was captured, enter it manually or leave 0 to settle without CLV.")

    if st.button("Settle bet", type="primary"):
        try:
            settle_bet(int(bet_id), result_choice, closing or None)
            st.success("Bet settled. P&L, bankroll and CLV were updated in Supabase.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not settle bet: {exc}")

with st.expander("Delete a bet entered by mistake"):
    if bets_df.empty:
        st.caption("No bets to delete.")
    else:
        delete_id = st.number_input("Bet ID", min_value=1, step=1)
        confirm = st.checkbox("I understand this permanently deletes the bet row")
        if st.button("Delete bet", disabled=not confirm):
            try:
                delete_bet(int(delete_id))
                st.success(f"Bet #{int(delete_id)} deleted.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete bet: {exc}")

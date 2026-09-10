from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

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
    list_odds_snapshots,
    set_starting_bankroll,
    settle_bet,
)

st.set_page_config(page_title="ATP Model Tracking", page_icon="📈", layout="wide")
st.title("📈 Model tracking")
st.caption(
    "Two separate records: every model prediction is graded for accuracy/calibration, while "
    "‘My placed bets’ tracks only wagers you actually placed, including ROI, stake and CLV."
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
    st.caption("Check SUPABASE_URL / SUPABASE_KEY in Streamlit Secrets, then reboot the app.")
    st.stop()

# ----------------------------- Tracking mode -----------------------------
mode = get_tracking_mode()
with st.expander("Staking settings", expanded=False):
    choice = st.radio(
        "How do you want to track actual bets?",
        ["Percentage only", "Currency bankroll (CA$)"],
        index=0 if mode == "percentage" else 1,
        horizontal=True,
    )
    selected_mode = "percentage" if choice == "Percentage only" else "currency"
    if selected_mode != mode and st.button("Save staking mode", type="secondary"):
        set_tracking_mode(selected_mode)
        st.success("Staking mode saved.")
        st.rerun()

    if mode == "percentage":
        starting = 100.0
        st.caption("Percentage mode uses a normalized bankroll index starting at 100.00.")
    else:
        starting = get_starting_bankroll()
        bankroll_input = st.number_input(
            "Starting bankroll (CA$)",
            min_value=0.0,
            value=float(starting or 0.0),
            step=50.0,
        )
        if st.button("Save starting bankroll", type="secondary"):
            set_starting_bankroll(float(bankroll_input))
            st.success("Starting bankroll saved.")
            st.rerun()

try:
    bets_df = pd.DataFrame(list_bets())
    pred_df = pd.DataFrame(list_predictions())
    try:
        odds_df = pd.DataFrame(list_odds_snapshots())
    except Exception as exc:
        odds_df = pd.DataFrame()
        st.warning(
            "Odds-history table is not available yet. Run `supabase_tracking_v036.sql` once in Supabase SQL Editor. "
            f"Details: {exc}"
        )
except Exception as exc:
    st.error(f"Could not load tracking data: {exc}")
    st.stop()

# ----------------------------- Normalization helpers -----------------------------
def _numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _prepare_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in ("created_at", "match_date", "settled_at", "first_price_at", "latest_price_at"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    _numeric(out, [
        "model_probability", "predicted_probability", "actual_result",
        "pinnacle_odds", "opening_odds", "latest_odds", "closing_odds",
        "odds_change", "clv", "edge", "expected_value", "court_speed",
    ])

    p = out.get("model_probability", pd.Series(np.nan, index=out.index))
    out["Model confidence"] = out.get("predicted_probability", pd.Series(np.nan, index=out.index))
    missing_conf = out["Model confidence"].isna()
    out.loc[missing_conf, "Model confidence"] = np.maximum(p[missing_conf], 1 - p[missing_conf])

    if "predicted_selection" not in out:
        out["predicted_selection"] = None
    for idx, row in out[out["predicted_selection"].isna() | (out["predicted_selection"].astype(str).str.len() == 0)].iterrows():
        prob = row.get("model_probability")
        market = str(row.get("market") or "")
        if pd.isna(prob):
            pick = ""
        elif market == "Total Sets 3.5":
            pick = "Over 3.5" if float(prob) >= 0.5 else "Under 3.5"
        else:
            pick = row.get("player_a") if float(prob) >= 0.5 else row.get("player_b")
        out.at[idx, "predicted_selection"] = pick

    out["Correct"] = np.nan
    if "actual_result" in out:
        mask = out["actual_result"].notna() & out["model_probability"].notna()
        # actual_result is canonical Player-A/Over outcome. Flip when the model predicted B/Under.
        out.loc[mask, "Correct"] = np.where(
            out.loc[mask, "model_probability"] >= 0.5,
            out.loc[mask, "actual_result"],
            1.0 - out.loc[mask, "actual_result"],
        )
    out["Result"] = np.where(
        out["Correct"].isna(),
        "Pending",
        np.where(out["Correct"] >= 0.5, "Correct", "Wrong"),
    )
    return out


def _prepare_bets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in ("created_at", "match_date", "settled_at"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    _numeric(out, [
        "model_probability", "model_fair_odds", "odds_taken", "closing_odds", "edge",
        "expected_value", "bankroll_before", "kelly_fraction", "stake_percent",
        "stake_amount", "profit_loss", "clv", "bankroll_after",
    ])
    if "odds_taken" in out and "closing_odds" in out:
        out["Odds move"] = np.where(
            out["odds_taken"].notna() & out["closing_odds"].notna() & (out["odds_taken"] > 1),
            out["closing_odds"] / out["odds_taken"] - 1.0,
            np.nan,
        )
    return out


pred_df = _prepare_predictions(pred_df)
bets_df = _prepare_bets(bets_df)
if not odds_df.empty:
    odds_df["captured_at"] = pd.to_datetime(odds_df.get("captured_at"), utc=True, errors="coerce")
    odds_df["match_date"] = pd.to_datetime(odds_df.get("match_date"), utc=True, errors="coerce")
    _numeric(odds_df, ["odds_a", "odds_b"])

all_predictions_tab, placed_bets_tab = st.tabs(["🎯 All model predictions", "💵 My placed bets"])

# ======================================================================================
# ALL MODEL PREDICTIONS
# ======================================================================================
with all_predictions_tab:
    st.caption(
        "Every ATP 250+ forecast is stored whether or not you bet it. The model pick is graded after the match, "
        "so this tab measures the model itself rather than your betting decisions."
    )

    evaluated = pred_df[pred_df.get("Correct", pd.Series(dtype=float)).notna()].copy() if not pred_df.empty else pd.DataFrame()
    accuracy = float(evaluated["Correct"].mean()) if not evaluated.empty else np.nan

    if not evaluated.empty:
        canonical_p = evaluated["model_probability"].clip(1e-6, 1 - 1e-6)
        canonical_y = evaluated["actual_result"]
        brier = float(np.mean((canonical_p - canonical_y) ** 2))
        logloss = float(-np.mean(canonical_y * np.log(canonical_p) + (1 - canonical_y) * np.log(1 - canonical_p)))
    else:
        brier = np.nan
        logloss = np.nan

    pred_clv = pred_df["clv"].dropna() if "clv" in pred_df else pd.Series(dtype=float)
    avg_model_clv = float(pred_clv.mean()) if len(pred_clv) else np.nan
    beat_close = float((pred_clv > 0).mean()) if len(pred_clv) else np.nan

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Predictions tracked", len(pred_df))
    p2.metric("Predictions graded", len(evaluated))
    p3.metric("Model accuracy", f"{accuracy:.1%}" if np.isfinite(accuracy) else "—")
    p4.metric("Average confidence", f"{evaluated['Model confidence'].mean():.1%}" if not evaluated.empty else "—")

    p5, p6, p7, p8 = st.columns(4)
    p5.metric("Brier score", f"{brier:.4f}" if np.isfinite(brier) else "—")
    p6.metric("Log loss", f"{logloss:.4f}" if np.isfinite(logloss) else "—")
    p7.metric("Model-side avg CLV", f"{avg_model_clv:+.2%}" if np.isfinite(avg_model_clv) else "—")
    p8.metric("Model picks beating close", f"{beat_close:.1%}" if np.isfinite(beat_close) else "—")

    if evaluated.empty:
        st.info(
            "No graded predictions yet. The background workflow saves the ATP 250+ slate and grades matches after completion."
        )
    else:
        st.subheader("Accuracy by model confidence")
        confidence_bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.001]
        confidence_labels = ["50–55%", "55–60%", "60–65%", "65–70%", "70–75%", "75–80%", "80–90%", "90%+"]
        bucketed = evaluated.copy()
        bucketed["Confidence bucket"] = pd.cut(
            bucketed["Model confidence"],
            bins=confidence_bins,
            labels=confidence_labels,
            include_lowest=True,
            right=False,
        )
        acc_table = bucketed.groupby("Confidence bucket", observed=False).agg(
            Predictions=("id", "count"),
            Average_confidence=("Model confidence", "mean"),
            Accuracy=("Correct", "mean"),
        ).reset_index()
        acc_table["Calibration gap"] = acc_table["Accuracy"] - acc_table["Average_confidence"]
        st.dataframe(
            acc_table.style.format({
                "Average_confidence": "{:.1%}",
                "Accuracy": "{:.1%}",
                "Calibration gap": "{:+.1%}",
            }, na_rep="—"),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Accuracy is the percentage of model picks that won. Calibration gap compares actual accuracy with the model's average confidence."
        )

    st.subheader("Prediction log")
    if pred_df.empty:
        st.info("No predictions have been saved yet.")
    else:
        display = pred_df.copy()
        display["Match"] = display["player_a"].fillna("") + " vs " + display["player_b"].fillna("")
        display = display.rename(columns={
            "match_date": "Start",
            "tournament": "Tournament",
            "market": "Market",
            "predicted_selection": "Model pick",
            "Model confidence": "Confidence",
            "opening_odds": "First Pinnacle",
            "latest_odds": "Latest Pinnacle",
            "closing_odds": "Pinnacle close",
            "odds_change": "Odds move",
            "clv": "CLV",
            "model_version": "Version",
        })
        cols = [
            "Start", "Tournament", "Match", "Market", "Model pick", "Confidence", "Result",
            "First Pinnacle", "Latest Pinnacle", "Pinnacle close", "Odds move", "CLV", "Version",
        ]
        cols = [c for c in cols if c in display.columns]
        st.dataframe(
            display[cols].sort_values("Start", ascending=False).style.format({
                "Confidence": "{:.1%}",
                "First Pinnacle": "{:.3f}",
                "Latest Pinnacle": "{:.3f}",
                "Pinnacle close": "{:.3f}",
                "Odds move": "{:+.2%}",
                "CLV": "{:+.2%}",
            }, na_rep="—"),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download all predictions CSV",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="atp_all_model_predictions.csv",
            mime="text/csv",
        )

    if not pred_df.empty and not odds_df.empty:
        with st.expander("Pinnacle odds movement history", expanded=False):
            options = pred_df.sort_values("match_date", ascending=False).copy()
            options["label"] = (
                options["player_a"].fillna("") + " vs " + options["player_b"].fillna("")
                + " · " + options["market"].fillna("")
            )
            selected_id = st.selectbox(
                "Prediction",
                options["id"].astype(int).tolist(),
                format_func=lambda pid: options.loc[options["id"].astype(int) == pid, "label"].iloc[0],
            )
            pred = options.loc[options["id"].astype(int) == int(selected_id)].iloc[0]
            movement = odds_df[
                (odds_df["match_id"].astype(str) == str(pred.get("match_id")))
                & (odds_df["market"].astype(str) == str(pred.get("market")))
            ].copy()
            if movement.empty:
                st.caption("No persistent Pinnacle price changes have been captured for this prediction yet.")
            else:
                pick = str(pred.get("predicted_selection") or "")
                def selected_price(row):
                    if pick == str(row.get("selection_a") or ""):
                        return row.get("odds_a")
                    if pick == str(row.get("selection_b") or ""):
                        return row.get("odds_b")
                    if str(pred.get("market")) == "Total Sets 3.5" and pick.casefold().startswith("under"):
                        return row.get("odds_b")
                    return row.get("odds_a")
                movement["Model-pick Pinnacle odds"] = movement.apply(selected_price, axis=1)
                movement = movement.dropna(subset=["captured_at", "Model-pick Pinnacle odds"]).sort_values("captured_at")
                if not movement.empty:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("First tracked", f"{movement['Model-pick Pinnacle odds'].iloc[0]:.3f}")
                    c2.metric("Latest tracked", f"{movement['Model-pick Pinnacle odds'].iloc[-1]:.3f}")
                    c3.metric(
                        "Change",
                        f"{movement['Model-pick Pinnacle odds'].iloc[-1] / movement['Model-pick Pinnacle odds'].iloc[0] - 1:+.2%}",
                    )
                    st.line_chart(movement.set_index("captured_at")[["Model-pick Pinnacle odds"]])

    if not evaluated.empty:
        st.subheader("Accuracy breakdowns")
        b1, b2 = st.columns(2)
        with b1:
            by_market = evaluated.groupby("market", dropna=False).agg(
                Predictions=("id", "count"), Accuracy=("Correct", "mean"), Avg_confidence=("Model confidence", "mean")
            ).reset_index()
            st.write("**By market**")
            st.dataframe(by_market.style.format({"Accuracy": "{:.1%}", "Avg_confidence": "{:.1%}"}), hide_index=True, use_container_width=True)
        with b2:
            by_surface = evaluated.groupby("surface", dropna=False).agg(
                Predictions=("id", "count"), Accuracy=("Correct", "mean"), Avg_confidence=("Model confidence", "mean")
            ).reset_index()
            st.write("**By surface**")
            st.dataframe(by_surface.style.format({"Accuracy": "{:.1%}", "Avg_confidence": "{:.1%}"}), hide_index=True, use_container_width=True)

# ======================================================================================
# MY PLACED BETS
# ======================================================================================
with placed_bets_tab:
    st.caption("Only wagers you explicitly recorded with ‘I placed this bet’ appear here.")

    settled = bets_df[bets_df["profit_loss"].notna()].copy() if not bets_df.empty and "profit_loss" in bets_df else pd.DataFrame()
    wins = int((settled.get("result", pd.Series(dtype=str)).astype(str).str.casefold() == "win").sum()) if not settled.empty else 0
    losses = int((settled.get("result", pd.Series(dtype=str)).astype(str).str.casefold() == "loss").sum()) if not settled.empty else 0
    voids = int((settled.get("result", pd.Series(dtype=str)).astype(str).str.casefold() == "void").sum()) if not settled.empty else 0
    stake_total = float(settled.get("stake_amount", pd.Series(dtype=float)).fillna(0).sum()) if not settled.empty else 0.0
    profit = float(settled.get("profit_loss", pd.Series(dtype=float)).fillna(0).sum()) if not settled.empty else 0.0
    yield_pct = profit / stake_total if stake_total else np.nan
    avg_clv = float(settled["clv"].dropna().mean()) if not settled.empty and "clv" in settled and settled["clv"].notna().any() else np.nan
    beat_close_bets = float((settled["clv"].dropna() > 0).mean()) if not settled.empty and "clv" in settled and settled["clv"].notna().any() else np.nan
    bank = current_bankroll()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Settled bets", str(len(settled)), f"{wins}W–{losses}L–{voids}V" if len(settled) else None)
    m2.metric("P&L" if mode == "percentage" else "Profit", f"{profit:+.2f} pts" if mode == "percentage" else f"CA${profit:,.2f}")
    m3.metric("ROI on stake", f"{yield_pct:+.1%}" if np.isfinite(yield_pct) else "—")
    m4.metric("Bankroll index" if mode == "percentage" else "Current bankroll", f"{bank:,.2f}" if mode == "percentage" and bank is not None else (f"CA${bank:,.2f}" if bank is not None else "—"))

    m5, m6, m7 = st.columns(3)
    m5.metric("Average CLV", f"{avg_clv:+.2%}" if np.isfinite(avg_clv) else "—")
    m6.metric("Bets beating close", f"{beat_close_bets:.1%}" if np.isfinite(beat_close_bets) else "—")
    m7.metric("Open bets", str(int(bets_df["profit_loss"].isna().sum())) if not bets_df.empty and "profit_loss" in bets_df else "0")

    st.subheader("Bet log")
    if bets_df.empty:
        st.info("No bets recorded yet. Open a matchup and press ‘I placed … — save bet’.")
    else:
        display = bets_df.copy()
        display["Match"] = display["player_a"].fillna("") + " vs " + display["player_b"].fillna("")
        display = display.rename(columns={
            "created_at": "Placed",
            "match_date": "Match date",
            "model_version": "Version",
            "tournament": "Tournament",
            "tournament_level": "Level",
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
            "id", "Placed", "Match date", "Tournament", "Match", "Market", "Selection",
            "Model P", "Odds taken", "Close", "Odds move", "Edge", "EV", "Stake %",
            ("Stake points" if mode == "percentage" else "Stake"), "Result",
            ("P&L points" if mode == "percentage" else "P&L"), "CLV", "Version",
        ]
        cols = [c for c in cols if c in display.columns]
        st.dataframe(
            display[cols].sort_values("Placed", ascending=False).style.format({
                "Model P": "{:.1%}", "Edge": "{:+.1%}", "EV": "{:+.1%}",
                "Stake %": "{:.2%}",
                **({"Stake points": "{:.2f}", "P&L points": "{:+.2f}"} if mode == "percentage" else {"Stake": "CA${:,.2f}", "P&L": "CA${:,.2f}"}),
                "CLV": "{:+.2%}", "Odds move": "{:+.2%}", "Odds taken": "{:.3f}", "Close": "{:.3f}",
            }, na_rep="—"),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download my bets CSV",
            data=bets_df.to_csv(index=False).encode("utf-8"),
            file_name="atp_my_bets.csv",
            mime="text/csv",
        )

    if not settled.empty:
        st.subheader("Bankroll progression")
        curve = settled.sort_values("settled_at").copy()
        curve["Cumulative P&L"] = curve["profit_loss"].fillna(0).cumsum()
        base = 100.0 if mode == "percentage" else float(starting or 0.0)
        curve["Bankroll index" if mode == "percentage" else "Bankroll"] = base + curve["Cumulative P&L"]
        curve_col = "Bankroll index" if mode == "percentage" else "Bankroll"
        st.line_chart(curve.set_index("settled_at")[[curve_col]])

        if settled["clv"].notna().any():
            st.subheader("Bet CLV")
            clv_curve = settled[settled["clv"].notna()].sort_values("settled_at").copy()
            clv_curve["Running average CLV"] = clv_curve["clv"].expanding().mean()
            st.line_chart(clv_curve.set_index("settled_at")[["clv", "Running average CLV"]])
            st.caption("Bet CLV = odds you took ÷ Pinnacle closing odds − 1.")

    if not settled.empty:
        st.subheader("Performance breakdowns")
        b1, b2 = st.columns(2)
        with b1:
            by_market = settled.groupby("market", dropna=False).agg(
                Bets=("id", "count"), Stake=("stake_amount", "sum"), Profit=("profit_loss", "sum")
            ).reset_index()
            by_market["ROI"] = np.where(by_market["Stake"] > 0, by_market["Profit"] / by_market["Stake"], np.nan)
            st.write("**By market**")
            st.dataframe(by_market.style.format({"ROI": "{:+.1%}"}), hide_index=True, use_container_width=True)
        with b2:
            by_surface = settled.groupby("surface", dropna=False).agg(
                Bets=("id", "count"), Stake=("stake_amount", "sum"), Profit=("profit_loss", "sum")
            ).reset_index()
            by_surface["ROI"] = np.where(by_surface["Stake"] > 0, by_surface["Profit"] / by_surface["Stake"], np.nan)
            st.write("**By surface**")
            st.dataframe(by_surface.style.format({"ROI": "{:+.1%}"}), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Open bets / manual fallback")
    open_bets = [] if bets_df.empty else bets_df[bets_df["profit_loss"].isna()].to_dict("records")
    if not open_bets:
        st.success("No unsettled bets. Normal completed matches are settled automatically.")
    else:
        labels = {
            int(x["id"]): f"#{int(x['id'])} — {x.get('selection')} · {x.get('market')} · {x.get('player_a')} vs {x.get('player_b')}"
            for x in open_bets
        }
        bet_id = st.selectbox("Bet", list(labels), format_func=lambda x: labels[x])
        selected = next(x for x in open_bets if int(x["id"]) == int(bet_id))
        result_choice = st.selectbox("Result", ["Win", "Loss", "Void"])
        auto_close = float(selected.get("closing_odds") or 0.0)
        closing = st.number_input(
            "Pinnacle closing odds for your selection",
            min_value=0.0,
            value=auto_close,
            step=0.01,
            format="%.3f",
        )
        st.caption("Normally filled automatically from the last persistent pre-start Pinnacle snapshot.")
        if st.button("Settle bet", type="primary"):
            try:
                settle_bet(int(bet_id), result_choice, closing or None)
                st.success("Bet settled.")
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
                delete_bet(int(delete_id))
                st.success(f"Bet #{int(delete_id)} deleted.")
                st.rerun()

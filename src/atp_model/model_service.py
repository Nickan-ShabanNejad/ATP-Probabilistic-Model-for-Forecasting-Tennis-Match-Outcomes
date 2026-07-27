
from pathlib import Path
import math
import joblib
import pandas as pd
from .config import ROOT

STATE_GENERATED = ROOT / "data/generated/player_state.csv.gz"
STATE_BOOTSTRAP = ROOT / "data/bootstrap/player_state.csv.gz"
MODEL_GENERATED = ROOT / "model/model.joblib"
MODEL_BOOTSTRAP = ROOT / "model/bootstrap_model.joblib"


def state_path():
    return STATE_GENERATED if STATE_GENERATED.exists() else STATE_BOOTSTRAP


def model_path():
    return MODEL_GENERATED if MODEL_GENERATED.exists() else MODEL_BOOTSTRAP


def load_state():
    return pd.read_csv(state_path())


def load_bundle():
    return joblib.load(model_path())


def get_player_row(df, player, surface):
    rows = df[(df["player"] == player) & (df["surface"] == surface)]
    if rows.empty:
        raise ValueError(f"No rating state for {player} on {surface}.")
    return rows.iloc[0]


def predict_match(
    df, bundle, player_a, player_b, surface, rank_a, rank_b, odds_a, odds_b,
    tournament_level=2.0, best_of=3.0
):
    a = get_player_row(df, player_a, surface)
    b = get_player_row(df, player_b, surface)

    def diff(key, default=0.0):
        av = float(a[key]) if key in a and pd.notna(a[key]) else default
        bv = float(b[key]) if key in b and pd.notna(b[key]) else default
        return av - bv

    feature_values = {
        "overall_elo_diff": diff("overall_elo"),
        "surface_elo_diff": diff("surface_elo"),
        "serve_diff": diff("serve"),
        "return_diff": diff("return_rating"),
        "log_rank_advantage": math.log(max(rank_b, 1)) - math.log(max(rank_a, 1)),
        "win5_diff": diff("win5"),
        "win10_diff": diff("win10"),
        "surface_win10_diff": diff("surface_win10"),
        "opp_elo10_diff": diff("opp_elo10"),
        "recent_perf10_diff": diff("recent_perf10"),
        "matches7_diff": diff("matches7"),
        "matches14_diff": diff("matches14"),
        "rest_days_diff": diff("rest_days"),
        "elo_change10_diff": diff("elo_change10"),
        "age_diff": diff("age"),
        "chart_serve_diff": diff("chart_serve"),
        "chart_return_diff": diff("chart_return"),
        "chart_winner_rate_diff": diff("chart_winner_rate"),
        "chart_ue_rate_diff": diff("chart_ue_rate"),
        "chart_net_win_diff": diff("chart_net_win"),
        "charted_matches_diff": diff("charted_matches"),
        "chart_available_diff": diff("chart_available"),
        "tournament_level": float(tournament_level),
        "best_of": float(best_of),
    }

    feature_order = bundle.get("features", list(feature_values))
    x = [[feature_values.get(name, 0.0) for name in feature_order]]
    probability = float(bundle["pipeline"].predict_proba(x)[0, 1])

    raw_a, raw_b = 1 / odds_a, 1 / odds_b
    no_vig = raw_a / (raw_a + raw_b)
    edge = probability - no_vig
    ev = probability * odds_a - 1
    fair = 1 / probability
    full_kelly = max(0.0, ev / (odds_a - 1))

    return {
        "player_a": player_a,
        "player_b": player_b,
        "surface": surface,
        "probability_a": probability,
        "probability_b": 1 - probability,
        "market_probability_a": no_vig,
        "edge": edge,
        "ev": ev,
        "fair_odds_a": fair,
        "quarter_kelly": full_kelly * 0.25,
        "row_a": a,
        "row_b": b,
    }

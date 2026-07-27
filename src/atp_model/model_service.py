from pathlib import Path
import math
import joblib
import pandas as pd
from .config import ROOT
from .tournament_features import load_surface_speeds, canonical_tournament

STATE_GENERATED = ROOT / "data/generated/player_state.csv.gz"
STATE_BOOTSTRAP = ROOT / "data/bootstrap/player_state.csv.gz"
MODEL_GENERATED = ROOT / "model/model.joblib"
MODEL_BOOTSTRAP = ROOT / "model/bootstrap_model.joblib"
H2H_GENERATED = ROOT / "data/generated/head_to_head.csv.gz"
SPEED_PATH = ROOT / "data/tournament_surface_speed.csv"


def state_path():
    return STATE_GENERATED if STATE_GENERATED.exists() else STATE_BOOTSTRAP


def model_path():
    return MODEL_GENERATED if MODEL_GENERATED.exists() else MODEL_BOOTSTRAP


def load_state():
    return pd.read_csv(state_path())


def load_bundle():
    return joblib.load(model_path())


def load_h2h():
    if H2H_GENERATED.exists():
        return pd.read_csv(H2H_GENERATED, dtype={"player_1_id": str, "player_2_id": str})
    return pd.DataFrame()


def load_tournament_speeds():
    return load_surface_speeds(SPEED_PATH) if SPEED_PATH.exists() else pd.DataFrame()


def available_tournaments():
    speeds = load_tournament_speeds()
    return sorted(speeds["tournament"].dropna().astype(str).unique()) if not speeds.empty else []


def tournament_speed(tournament, surface, prediction_year=None):
    year = int(prediction_year or pd.Timestamp.today().year)
    speeds = load_tournament_speeds()
    if speeds.empty:
        return 1.0, 1.0
    key = canonical_tournament(tournament)
    candidates = speeds[(speeds["tournament_key"] == key) & (speeds["season"] < year)]
    if not candidates.empty:
        return float(candidates.sort_values("season").iloc[-1]["surface_speed"]), 0.0
    fallback = speeds[(speeds["surface"].astype(str).str.title() == str(surface).title()) & (speeds["season"] < year)]
    if not fallback.empty:
        latest = int(fallback["season"].max())
        return float(fallback[fallback["season"] == latest]["surface_speed"].median()), 1.0
    return 1.0, 1.0


def get_player_row(df, player, surface):
    rows = df[(df["player"] == player) & (df["surface"] == surface)]
    if rows.empty:
        raise ValueError(f"No rating state for {player} on {surface}.")
    return rows.iloc[0]


def head_to_head_features(player_a, player_b, surface):
    h2h = load_h2h()
    if h2h.empty:
        return 0.0, 0.0, 0, {"a_wins": 0, "b_wins": 0, "surface_a_wins": 0, "surface_b_wins": 0}
    pair = h2h[
        ((h2h["player_1"] == player_a) & (h2h["player_2"] == player_b)) |
        ((h2h["player_1"] == player_b) & (h2h["player_2"] == player_a))
    ]
    if pair.empty:
        return 0.0, 0.0, 0, {"a_wins": 0, "b_wins": 0, "surface_a_wins": 0, "surface_b_wins": 0}
    all_row = pair[pair["surface"] == "All"].iloc[0]
    surf_rows = pair[pair["surface"] == surface]
    surf_row = surf_rows.iloc[0] if not surf_rows.empty else all_row
    direct = all_row["player_1"] == player_a
    a_wins = int(all_row["player_1_wins"] if direct else all_row["player_2_wins"])
    b_wins = int(all_row["player_2_wins"] if direct else all_row["player_1_wins"])
    sa = int(surf_row["surface_player_1_wins"] if direct else surf_row["surface_player_2_wins"])
    sb = int(surf_row["surface_player_2_wins"] if direct else surf_row["surface_player_1_wins"])
    total = a_wins + b_wins
    h2h_edge = ((a_wins + 2) / (total + 4) - 0.5) * 2
    stotal = sa + sb
    surface_edge = ((sa + 2) / (stotal + 4) - 0.5) * 2
    return h2h_edge, surface_edge, total, {
        "a_wins": a_wins, "b_wins": b_wins,
        "surface_a_wins": sa, "surface_b_wins": sb,
    }


def predict_match(
    df, bundle, player_a, player_b, surface, rank_a, rank_b, odds_a, odds_b,
    tournament_level=2.0, best_of=3.0, tournament="", prediction_year=None
):
    a = get_player_row(df, player_a, surface)
    b = get_player_row(df, player_b, surface)

    def diff(key, default=0.0):
        av = float(a[key]) if key in a and pd.notna(a[key]) else default
        bv = float(b[key]) if key in b and pd.notna(b[key]) else default
        return av - bv

    h2h_edge, h2h_surface_edge, h2h_matches, h2h_record = head_to_head_features(
        player_a, player_b, surface
    )
    speed, speed_missing = tournament_speed(tournament, surface, prediction_year)

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
        "h2h_edge": h2h_edge,
        "h2h_surface_edge": h2h_surface_edge,
        "tournament_level": float(tournament_level),
        "best_of": float(best_of),
        "log_h2h_matches": math.log1p(h2h_matches),
        "court_speed": speed,
        "court_speed_missing": speed_missing,
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
        "player_a": player_a, "player_b": player_b, "surface": surface,
        "tournament": tournament, "court_speed": speed,
        "court_speed_fallback": bool(speed_missing), "h2h_record": h2h_record,
        "probability_a": probability, "probability_b": 1 - probability,
        "market_probability_a": no_vig, "edge": edge, "ev": ev,
        "fair_odds_a": fair, "quarter_kelly": full_kelly * 0.25,
        "row_a": a, "row_b": b,
    }

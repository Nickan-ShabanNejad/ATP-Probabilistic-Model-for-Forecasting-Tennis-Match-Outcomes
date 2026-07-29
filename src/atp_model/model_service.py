from pathlib import Path
import math

import joblib
import pandas as pd

from .config import ROOT
from .tournament_features import load_surface_speeds, lookup_surface_speed

STATE_GENERATED = ROOT / "data/generated/player_state.csv.gz"
STATE_BOOTSTRAP = ROOT / "data/bootstrap/player_state.csv.gz"
MODEL_GENERATED = ROOT / "model/model.joblib"
MODEL_BOOTSTRAP = ROOT / "model/bootstrap_model.joblib"
SPEED_PATH = ROOT / "data/tournament_surface_speed.csv"
H2H_PATH = ROOT / "data/generated/h2h.csv"

# Prevent numerical/model outliers from producing impossible-looking 0% or 100%
# probabilities. The unclipped value is still returned for diagnostics.
MIN_PROBABILITY = 0.05
MAX_PROBABILITY = 0.95


def state_path():
    return STATE_GENERATED if STATE_GENERATED.exists() else STATE_BOOTSTRAP


def model_path():
    return MODEL_GENERATED if MODEL_GENERATED.exists() else MODEL_BOOTSTRAP


def load_state():
    return pd.read_csv(state_path())


def load_bundle():
    return joblib.load(model_path())


def load_h2h():
    """Load optional descriptive H2H data.

    H2H is deliberately not used as a predictive feature unless it is built
    into the trained model. This avoids treating tiny samples as real edge.
    """
    if not H2H_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(H2H_PATH)
    except Exception:
        return pd.DataFrame()


def _h2h_record(player_a, player_b):
    """Return a safe, descriptive H2H summary without changing probability."""
    h2h = load_h2h()
    empty = {
        "player_a_wins": 0,
        "player_b_wins": 0,
        "matches": 0,
        "used_as_predictive_edge": False,
    }
    if h2h.empty:
        return empty

    columns = set(h2h.columns)

    # Supported format 1: one row per match with winner/loser names.
    if {"winner", "loser"}.issubset(columns):
        relevant = h2h[
            ((h2h["winner"] == player_a) & (h2h["loser"] == player_b))
            | ((h2h["winner"] == player_b) & (h2h["loser"] == player_a))
        ]
        a_wins = int(((relevant["winner"] == player_a)).sum())
        b_wins = int(((relevant["winner"] == player_b)).sum())
        return {
            "player_a_wins": a_wins,
            "player_b_wins": b_wins,
            "matches": a_wins + b_wins,
            "used_as_predictive_edge": False,
        }

    # Supported format 2: one aggregated row per player pair.
    if {"player_a", "player_b"}.issubset(columns):
        direct = h2h[(h2h["player_a"] == player_a) & (h2h["player_b"] == player_b)]
        reverse = h2h[(h2h["player_a"] == player_b) & (h2h["player_b"] == player_a)]

        def value(row, candidates, default=0):
            for col in candidates:
                if col in row.index and pd.notna(row[col]):
                    return int(row[col])
            return default

        if not direct.empty:
            row = direct.iloc[0]
            a_wins = value(row, ["player_a_wins", "a_wins", "wins_a"])
            b_wins = value(row, ["player_b_wins", "b_wins", "wins_b"])
            matches = value(row, ["matches", "n_matches", "total"], a_wins + b_wins)
            return {
                "player_a_wins": a_wins,
                "player_b_wins": b_wins,
                "matches": matches,
                "used_as_predictive_edge": False,
            }

        if not reverse.empty:
            row = reverse.iloc[0]
            a_wins = value(row, ["player_b_wins", "b_wins", "wins_b"])
            b_wins = value(row, ["player_a_wins", "a_wins", "wins_a"])
            matches = value(row, ["matches", "n_matches", "total"], a_wins + b_wins)
            return {
                "player_a_wins": a_wins,
                "player_b_wins": b_wins,
                "matches": matches,
                "used_as_predictive_edge": False,
            }

    return empty



def head_to_head_features(player_a, player_b, surface, min_matches=10):
    """Return conservative H2H features and a descriptive record.

    H2H samples below ``min_matches`` are reported for display but return
    zero predictive edge. This prevents a tiny or duplicated sample from
    materially changing the model probability.
    """
    h2h = load_h2h()
    record = {
        "player_a": player_a,
        "player_b": player_b,
        "player_a_wins": 0,
        "player_b_wins": 0,
        "surface_player_a_wins": 0,
        "surface_player_b_wins": 0,
        "matches": 0,
        "surface_matches": 0,
        "used_as_predictive_edge": False,
    }
    if h2h.empty:
        return 0.0, 0.0, 0, record

    cols = set(h2h.columns)

    # Aggregated schema used by this project/tests.
    if {"player_1", "player_2"}.issubset(cols):
        direct = h2h[(h2h["player_1"] == player_a) & (h2h["player_2"] == player_b)]
        reverse = h2h[(h2h["player_1"] == player_b) & (h2h["player_2"] == player_a)]

        def total(frame, column):
            if frame.empty or column not in frame.columns:
                return 0
            return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())

        a_wins = total(direct, "player_1_wins") + total(reverse, "player_2_wins")
        b_wins = total(direct, "player_2_wins") + total(reverse, "player_1_wins")

        direct_surface = direct[direct.get("surface", pd.Series(index=direct.index, dtype=object)).astype(str).str.casefold() == str(surface).casefold()]
        reverse_surface = reverse[reverse.get("surface", pd.Series(index=reverse.index, dtype=object)).astype(str).str.casefold() == str(surface).casefold()]
        sa_wins = total(direct_surface, "surface_player_1_wins") + total(reverse_surface, "surface_player_2_wins")
        sb_wins = total(direct_surface, "surface_player_2_wins") + total(reverse_surface, "surface_player_1_wins")

    # One-row-per-match schema.
    elif {"winner", "loser"}.issubset(cols):
        relevant = h2h[
            ((h2h["winner"] == player_a) & (h2h["loser"] == player_b))
            | ((h2h["winner"] == player_b) & (h2h["loser"] == player_a))
        ]
        a_wins = int((relevant["winner"] == player_a).sum())
        b_wins = int((relevant["winner"] == player_b).sum())
        if "surface" in relevant.columns:
            on_surface = relevant[relevant["surface"].astype(str).str.casefold() == str(surface).casefold()]
        else:
            on_surface = relevant.iloc[0:0]
        sa_wins = int((on_surface["winner"] == player_a).sum())
        sb_wins = int((on_surface["winner"] == player_b).sum())
    else:
        return 0.0, 0.0, 0, record

    matches = a_wins + b_wins
    surface_matches = sa_wins + sb_wins
    record.update({
        "player_a_wins": a_wins,
        "player_b_wins": b_wins,
        "surface_player_a_wins": sa_wins,
        "surface_player_b_wins": sb_wins,
        "matches": matches,
        "surface_matches": surface_matches,
    })

    # Descriptive only until the sample is large enough.
    if matches < int(min_matches):
        return 0.0, 0.0, 0, record

    overall_edge = (a_wins - b_wins) / matches
    surface_edge = ((sa_wins - sb_wins) / surface_matches) if surface_matches >= int(min_matches) else 0.0
    record["used_as_predictive_edge"] = True
    return float(overall_edge), float(surface_edge), int(matches), record

def load_tournament_speeds():
    return load_surface_speeds(SPEED_PATH)


def available_tournaments(surface=None):
    speeds = load_tournament_speeds()
    if speeds.empty:
        return []
    if surface:
        speeds = speeds[
            speeds["surface"].astype(str).str.title() == str(surface).title()
        ]
    return sorted(speeds["tournament"].dropna().astype(str).unique())


def tournament_speed(tournament, surface, prediction_year=None):
    year = int(prediction_year or pd.Timestamp.today().year)
    return lookup_surface_speed(
        load_tournament_speeds(), tournament, surface, year
    )


def get_player_row(df, player, surface):
    rows = df[(df["player"] == player) & (df["surface"] == surface)]
    if rows.empty:
        raise ValueError(f"No rating state for {player} on {surface}.")
    return rows.iloc[0]


def predict_match(
    df,
    bundle,
    player_a,
    player_b,
    surface,
    rank_a,
    rank_b,
    odds_a,
    odds_b,
    tournament_level=2.0,
    best_of=3.0,
    tournament="",
    prediction_year=None,
    court_speed_override=None,
    indoor=False,
):
    a = get_player_row(df, player_a, surface)
    b = get_player_row(df, player_b, surface)

    def diff(key, default=0.0):
        av = float(a[key]) if key in a and pd.notna(a[key]) else default
        bv = float(b[key]) if key in b and pd.notna(b[key]) else default
        return av - bv

    surface_elo_diff = diff("surface_elo")
    serve_diff = diff("serve")
    return_diff = diff("return_rating")
    rank_advantage = math.log(max(float(rank_b), 1.0)) - math.log(
        max(float(rank_a), 1.0)
    )
    level = float(tournament_level)
    level_centered = level - 3.0

    estimated_speed, speed_missing = tournament_speed(
        tournament, surface, prediction_year
    )
    if court_speed_override is None:
        speed = float(estimated_speed)
    else:
        speed = float(court_speed_override)
        speed_missing = 0.0

    speed_centered = speed - 1.0
    indoor_value = 1.0 if indoor else 0.0

    feature_values = {
        "overall_elo_diff": diff("overall_elo"),
        "surface_elo_diff": surface_elo_diff,
        "serve_diff": serve_diff,
        "return_diff": return_diff,
        "log_rank_advantage": rank_advantage,
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
        "level_surface_elo_interaction": level_centered * surface_elo_diff / 400.0,
        "level_rank_interaction": level_centered * rank_advantage,
        "level_serve_interaction": level_centered * serve_diff * 10.0,
        "speed_surface_elo_interaction": speed_centered * surface_elo_diff / 100.0,
        "speed_serve_interaction": speed_centered * serve_diff * 10.0,
        "speed_return_interaction": speed_centered * return_diff * 10.0,
        "indoor_serve_interaction": indoor_value * serve_diff * 10.0,
        "indoor_return_interaction": indoor_value * return_diff * 10.0,
        "tournament_level": level,
        "best_of": float(best_of),
        "court_speed": speed,
        "court_speed_missing": float(speed_missing),
        "indoor": indoor_value,
    }

    feature_order = bundle.get("features", list(feature_values))
    x = pd.DataFrame(
        [[feature_values.get(name, 0.0) for name in feature_order]],
        columns=feature_order,
    )
    raw_probability = float(bundle["pipeline"].predict_proba(x)[0, 1])
    probability = min(MAX_PROBABILITY, max(MIN_PROBABILITY, raw_probability))

    odds_a = float(odds_a)
    odds_b = float(odds_b)
    if odds_a <= 1.0 or odds_b <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0.")

    raw_a, raw_b = 1.0 / odds_a, 1.0 / odds_b
    no_vig = raw_a / (raw_a + raw_b)
    edge = probability - no_vig
    ev = probability * odds_a - 1.0
    fair = 1.0 / probability
    full_kelly = max(0.0, ev / (odds_a - 1.0))
    _, _, _, h2h_record = head_to_head_features(player_a, player_b, surface)

    return {
        "player_a": player_a,
        "player_b": player_b,
        "surface": surface,
        "tournament": tournament,
        "tournament_level": level,
        "court_speed": speed,
        "court_speed_fallback": bool(speed_missing),
        "indoor": bool(indoor),
        "raw_probability_a": raw_probability,
        "calibrated_probability_a": probability,
        "probability_a": probability,
        "probability_b": 1.0 - probability,
        "market_probability_a": no_vig,
        "edge": edge,
        "ev": ev,
        "fair_odds_a": fair,
        "quarter_kelly": min(0.05, full_kelly * 0.25),
        "h2h_record": h2h_record,
        "row_a": a,
        "row_b": b,
    }


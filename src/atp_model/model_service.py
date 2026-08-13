from __future__ import annotations

from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd

from .config import ROOT
from .features import FEATURES
from .tournament_features import (
    load_surface_speeds,
    prediction_surface_speed,
)

STATE_GENERATED = ROOT / "data/generated/player_state.csv.gz"
STATE_BOOTSTRAP = ROOT / "data/bootstrap/player_state.csv.gz"
MODEL_GENERATED = ROOT / "model/model.joblib"
MODEL_BOOTSTRAP = ROOT / "model/bootstrap_model.joblib"
H2H_PATH = ROOT / "data/generated/head_to_head.csv.gz"
EXTERNAL_SPEED_PATH = ROOT / "data/tournament_surface_speed_external.csv"
LEGACY_SPEED_PATH = ROOT / "data/tournament_surface_speed.csv"
EMPIRICAL_SPEED_PATH = ROOT / "data/generated/tournament_surface_speed_empirical.csv"

MIN_PROBABILITY = 0.05
MAX_PROBABILITY = 0.95
H2H_PRIOR_MATCHES = 6.0
H2H_METRIC_PRIOR_MATCHES = 6.0


def state_path():
    return STATE_GENERATED if STATE_GENERATED.exists() else STATE_BOOTSTRAP


def model_path():
    return MODEL_GENERATED if MODEL_GENERATED.exists() else MODEL_BOOTSTRAP


def load_state():
    return pd.read_csv(state_path())


def load_bundle():
    return joblib.load(model_path())


def load_h2h():
    if not H2H_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(H2H_PATH, compression="gzip")
    except Exception:
        return pd.DataFrame()


def _numeric(row, key, default=0.0):
    try:
        value = row.get(key, default)
        value = float(value)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _h2h_model_state(player_a, player_b, surface):
    h2h = load_h2h()
    empty = {
        "h2h_overall_edge": 0.0,
        "h2h_surface_edge": 0.0,
        "h2h_serve_diff": 0.0,
        "h2h_second_serve_diff": 0.0,
        "h2h_bp_convert_diff": 0.0,
        "h2h_matches_log": 0.0,
        "h2h_surface_matches_log": 0.0,
        "matches": 0,
        "surface_matches": 0,
        "a_wins": 0,
        "b_wins": 0,
        "surface_a_wins": 0,
        "surface_b_wins": 0,
        "player_a_wins": 0,
        "player_b_wins": 0,
        "surface_player_a_wins": 0,
        "surface_player_b_wins": 0,
        "used_as_predictive_edge": False,
    }
    if h2h.empty or not {"player_1", "player_2"}.issubset(h2h.columns):
        return empty

    direct = h2h[(h2h["player_1"] == player_a) & (h2h["player_2"] == player_b)]
    reverse = h2h[(h2h["player_1"] == player_b) & (h2h["player_2"] == player_a)]
    orientation = 1.0
    rows = direct
    if rows.empty:
        rows = reverse
        orientation = -1.0
    if rows.empty:
        return empty

    all_rows = rows[rows["surface"].astype(str).str.casefold() == "all"]
    overall_row = (all_rows if not all_rows.empty else rows).iloc[0]
    surf_rows = rows[rows["surface"].astype(str).str.casefold() == str(surface).casefold()]
    surface_row = surf_rows.iloc[0] if not surf_rows.empty else None

    if orientation > 0:
        a_wins = int(_numeric(overall_row, "player_1_wins", 0))
        b_wins = int(_numeric(overall_row, "player_2_wins", 0))
    else:
        a_wins = int(_numeric(overall_row, "player_2_wins", 0))
        b_wins = int(_numeric(overall_row, "player_1_wins", 0))
    matches = a_wins + b_wins

    if surface_row is not None:
        if orientation > 0:
            sa = int(_numeric(surface_row, "player_1_wins", 0))
            sb = int(_numeric(surface_row, "player_2_wins", 0))
        else:
            sa = int(_numeric(surface_row, "player_2_wins", 0))
            sb = int(_numeric(surface_row, "player_1_wins", 0))
    else:
        sa = sb = 0
    surface_matches = sa + sb

    # H2H contributes from the first prior meeting, but is continuously
    # shrunk toward neutral (50/50) instead of using a hard sample cutoff.
    # With a six-match neutral prior, a 3-0 record has an effective H2H
    # win probability of 66.7%, rather than being treated as either 50% or 100%.
    predictive_overall = matches > 0
    predictive_surface = surface_matches > 0
    overall_edge = (a_wins - b_wins) / (matches + H2H_PRIOR_MATCHES) if predictive_overall else 0.0
    surface_edge = (sa - sb) / (surface_matches + H2H_PRIOR_MATCHES) if predictive_surface else 0.0

    def metric_diff(row, p1_col, p2_col, evidence):
        if row is None or evidence <= 0:
            return 0.0
        p1 = _numeric(row, p1_col, np.nan)
        p2 = _numeric(row, p2_col, np.nan)
        if not np.isfinite(p1) or not np.isfinite(p2):
            return 0.0
        raw_diff = (p1 - p2) * orientation
        return float(raw_diff * evidence / (evidence + H2H_METRIC_PRIOR_MATCHES))

    record = {
        "h2h_overall_edge": float(overall_edge),
        "h2h_surface_edge": float(surface_edge),
        "h2h_serve_diff": metric_diff(overall_row, "player_1_serve", "player_2_serve", matches),
        "h2h_second_serve_diff": metric_diff(overall_row, "player_1_second_serve", "player_2_second_serve", matches),
        "h2h_bp_convert_diff": metric_diff(overall_row, "player_1_bp_convert", "player_2_bp_convert", matches),
        "h2h_matches_log": float(math.log1p(matches)),
        "h2h_surface_matches_log": float(math.log1p(surface_matches)),
        "matches": matches,
        "surface_matches": surface_matches,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "surface_a_wins": sa,
        "surface_b_wins": sb,
        "player_a_wins": a_wins,
        "player_b_wins": b_wins,
        "surface_player_a_wins": sa,
        "surface_player_b_wins": sb,
        "used_as_predictive_edge": predictive_overall,
    }
    return record


def head_to_head_features(player_a, player_b, surface, min_matches=1):
    """Backward-compatible H2H accessor.

    The returned edges are now model inputs. No probability is manually added or
    subtracted after prediction.
    """
    record = _h2h_model_state(player_a, player_b, surface)
    if record["matches"] < int(min_matches):
        return 0.0, 0.0, int(record["matches"]), record
    return (
        float(record["h2h_overall_edge"]),
        float(record["h2h_surface_edge"]),
        int(record["matches"]),
        record,
    )


def load_tournament_speeds():
    prior = load_surface_speeds(EXTERNAL_SPEED_PATH)
    if prior.empty:
        prior = load_surface_speeds(LEGACY_SPEED_PATH)
    return prior


def load_live_tournament_speeds():
    return load_surface_speeds(EMPIRICAL_SPEED_PATH)


def available_tournaments(surface=None):
    frames = [x for x in (load_tournament_speeds(), load_live_tournament_speeds()) if not x.empty]
    if not frames:
        return []
    speeds = pd.concat(frames, ignore_index=True)
    if surface:
        speeds = speeds[speeds["surface"].astype(str).str.title() == str(surface).title()]
    return sorted(speeds["tournament"].dropna().astype(str).unique())


def tournament_speed_details(tournament, surface, prediction_year=None):
    year = int(prediction_year or pd.Timestamp.today().year)
    speed, meta = prediction_surface_speed(
        load_tournament_speeds(), load_live_tournament_speeds(), tournament, surface, year
    )
    return float(speed), meta


def tournament_speed(tournament, surface, prediction_year=None):
    speed, meta = tournament_speed_details(tournament, surface, prediction_year)
    missing = 1.0 if meta.get("prior_missing") and not meta.get("live_matches") else 0.0
    return speed, missing


def get_player_row(df, player, surface):
    rows = df[(df["player"] == player) & (df["surface"] == surface)]
    if rows.empty:
        raise ValueError(f"No rating state for {player} on {surface}.")
    return rows.iloc[0]


def _feature_frame(row_a, row_b, rank_a, rank_b, level, best_of, speed, speed_meta, indoor, h2h, feature_order):
    def diff(key, default=0.0):
        av = _numeric(row_a, key, default)
        bv = _numeric(row_b, key, default)
        return av - bv

    surface_elo_diff = diff("surface_elo")
    serve_diff = diff("serve", 0.635)
    return_diff = diff("return_rating", 0.365)
    rank_advantage = math.log(max(float(rank_b), 1.0)) - math.log(max(float(rank_a), 1.0))
    level_centered = float(level) - 3.0
    speed_centered = float(speed) - 1.0
    bestof_centered = float(best_of) - 3.0
    indoor_value = 1.0 if indoor else 0.0

    values = {
        "overall_elo_diff": diff("overall_elo"),
        "surface_elo_diff": surface_elo_diff,
        "serve_diff": serve_diff,
        "return_diff": return_diff,
        "log_rank_advantage": rank_advantage,
        "win3_diff": diff("win3", 0.5), "win5_diff": diff("win5", 0.5), "win10_diff": diff("win10", 0.5),
        "surface_win10_diff": diff("surface_win10", 0.5),
        "spw1_diff": diff("spw1", 0.635), "spw3_diff": diff("spw3", 0.635), "spw5_diff": diff("spw5", 0.635), "spw10_diff": diff("spw10", 0.635),
        "rpw1_diff": diff("rpw1", 0.365), "rpw3_diff": diff("rpw3", 0.365), "rpw5_diff": diff("rpw5", 0.365), "rpw10_diff": diff("rpw10", 0.365),
        "first_in5_diff": diff("first_in5", 0.62), "first_won5_diff": diff("first_won5", 0.70),
        "second_won5_diff": diff("second_won5", 0.50), "ace_rate5_diff": diff("ace_rate5", 0.08), "df_rate5_diff": diff("df_rate5", 0.035),
        "point_share5_diff": diff("point_share5", 0.50), "point_share10_diff": diff("point_share10", 0.50),
        "bp_save5_diff": diff("bp_save5", 0.62), "bp_convert5_diff": diff("bp_convert5", 0.38),
        "form_ewma_diff": diff("form_ewma", 0.5), "surface_form_ewma_diff": diff("surface_form_ewma", 0.5),
        "opp_elo10_diff": diff("opp_elo10", 1500.0), "recent_perf10_diff": diff("recent_perf10", 0.0),
        "matches7_diff": diff("matches7"), "matches14_diff": diff("matches14"), "rest_days_diff": diff("rest_days", 30),
        "elo_change10_diff": diff("elo_change10"), "age_diff": diff("age"),
        "winner_rate_diff": diff("winner_rate", 0.15), "ue_rate_diff": diff("ue_rate", 0.15),
        "aggression_quality_diff": diff("aggression_quality"), "advanced_coverage_diff": diff("advanced_coverage"),
        "net_win_diff": diff("net_win", 0.65), "avg_first_serve_speed_diff": diff("avg_first_serve_speed"),
        "h2h_overall_edge": float(h2h["h2h_overall_edge"]),
        "h2h_surface_edge": float(h2h["h2h_surface_edge"]),
        "h2h_serve_diff": float(h2h["h2h_serve_diff"]),
        "h2h_second_serve_diff": float(h2h["h2h_second_serve_diff"]),
        "h2h_bp_convert_diff": float(h2h["h2h_bp_convert_diff"]),
        "level_surface_elo_interaction": level_centered * surface_elo_diff / 400.0,
        "level_rank_interaction": level_centered * rank_advantage,
        "level_serve_interaction": level_centered * serve_diff * 10.0,
        "level_form_interaction": level_centered * diff("form_ewma", 0.5),
        "speed_surface_elo_interaction": speed_centered * surface_elo_diff / 100.0,
        "speed_serve_interaction": speed_centered * serve_diff * 10.0,
        "speed_return_interaction": speed_centered * return_diff * 10.0,
        "speed_ace_interaction": speed_centered * diff("ace_rate5", 0.08) * 10.0,
        "speed_second_serve_interaction": speed_centered * diff("second_won5", 0.50) * 10.0,
        "speed_point_share_interaction": speed_centered * diff("point_share5", 0.50) * 10.0,
        "indoor_serve_interaction": indoor_value * serve_diff * 10.0,
        "indoor_return_interaction": indoor_value * return_diff * 10.0,
        "bestof_surface_elo_interaction": bestof_centered * surface_elo_diff / 400.0,
        "tournament_level": float(level),
        "best_of": float(best_of),
        "court_speed": float(speed),
        "court_speed_prior": float(speed_meta.get("prior_speed", speed)),
        "court_speed_live_weight": float(speed_meta.get("live_weight", 0.0)),
        "court_speed_missing": 1.0 if speed_meta.get("prior_missing") else 0.0,
        "indoor": indoor_value,
        "h2h_matches_log": float(h2h["h2h_matches_log"]),
        "h2h_surface_matches_log": float(h2h["h2h_surface_matches_log"]),
        # Legacy aliases allow the bootstrap model to remain loadable if the
        # generated v4 artifacts are absent.
        "chart_winner_rate_diff": diff("winner_rate", 0.15),
        "chart_ue_rate_diff": diff("ue_rate", 0.15),
        "chart_available_diff": diff("advanced_coverage"),
    }
    order = feature_order or FEATURES
    return pd.DataFrame([[values.get(name, 0.0) for name in order]], columns=order)


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
    """Predict a match with order invariance and no post-hoc H2H bump."""
    a = get_player_row(df, player_a, surface)
    b = get_player_row(df, player_b, surface)
    level = float(tournament_level)

    estimated_speed, speed_meta = tournament_speed_details(tournament, surface, prediction_year)
    if court_speed_override is None:
        speed = float(estimated_speed)
    else:
        speed = float(court_speed_override)
        speed_meta = {
            "prior_speed": speed,
            "prior_missing": False,
            "live_speed": None,
            "live_matches": 0,
            "live_weight": 0.0,
        }

    feature_order = bundle.get("features") or FEATURES
    h2h_forward = _h2h_model_state(player_a, player_b, surface)
    h2h_reverse = _h2h_model_state(player_b, player_a, surface)

    x_forward = _feature_frame(
        a, b, rank_a, rank_b, level, best_of, speed, speed_meta, indoor, h2h_forward, feature_order
    )
    x_reverse = _feature_frame(
        b, a, rank_b, rank_a, level, best_of, speed, speed_meta, indoor, h2h_reverse, feature_order
    )

    # The training pipeline was fitted on numpy arrays. Passing arrays here
    # avoids scikit-learn's harmless feature-name warning in smoke tests.
    forward_probability = float(bundle["pipeline"].predict_proba(x_forward.to_numpy(dtype=float))[0, 1])
    reverse_probability_for_b = float(bundle["pipeline"].predict_proba(x_reverse.to_numpy(dtype=float))[0, 1])
    reverse_probability_for_a = 1.0 - reverse_probability_for_b
    symmetric_raw_probability = 0.5 * (forward_probability + reverse_probability_for_a)
    symmetry_gap_before_fix = forward_probability - reverse_probability_for_a
    probability = min(MAX_PROBABILITY, max(MIN_PROBABILITY, symmetric_raw_probability))

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

    return {
        "player_a": player_a,
        "player_b": player_b,
        "surface": surface,
        "tournament": tournament,
        "tournament_level": level,
        "court_speed": speed,
        "court_speed_prior": speed_meta.get("prior_speed"),
        "court_speed_live": speed_meta.get("live_speed"),
        "court_speed_live_matches": speed_meta.get("live_matches", 0),
        "court_speed_live_weight": speed_meta.get("live_weight", 0.0),
        "court_speed_fallback": bool(speed_meta.get("prior_missing")),
        "indoor": bool(indoor),
        "raw_probability_a": symmetric_raw_probability,
        "forward_raw_probability_a": forward_probability,
        "reverse_raw_probability_a": reverse_probability_for_a,
        "symmetry_gap_before_fix": symmetry_gap_before_fix,
        "calibrated_probability_a": probability,
        "probability_a": probability,
        "probability_b": 1.0 - probability,
        "market_probability_a": no_vig,
        "edge": edge,
        "ev": ev,
        "fair_odds_a": fair,
        "quarter_kelly": full_kelly * 0.25,
        "base_probability_a": probability,
        "h2h_impact": 0.0,
        "h2h_in_model": True,
        "h2h_overall_edge": h2h_forward["h2h_overall_edge"],
        "h2h_surface_edge": h2h_forward["h2h_surface_edge"],
        "h2h_record": h2h_forward,
        "row_a": a,
        "row_b": b,
    }

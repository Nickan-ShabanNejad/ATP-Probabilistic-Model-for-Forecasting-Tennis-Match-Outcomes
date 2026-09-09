import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atp_model.matchstat import past_matches_to_master
from atp_model.model_service import load_state, load_bundle, predict_match
from atp_model.odds import last_pre_match_quote, no_vig_two_way, safe_opening_quote
from atp_model.tournament_features import shrink_live_speed


def _two_players(df):
    hard = df[df.surface == "Hard"]
    names = list(hard.player.dropna().unique())
    assert len(names) >= 2
    return names[0], names[1]


def test_prediction_smoke():
    df = load_state()
    bundle = load_bundle()
    a, b = _two_players(df)
    r = predict_match(df, bundle, a, b, "Hard", 100, 120, 1.8, 2.05, tournament="Australian Open")
    assert 0 < r["probability_a"] < 1
    assert r["fair_odds_a"] > 1
    assert "h2h_record" in r
    assert r["h2h_in_model"] is True
    assert r["h2h_impact"] == 0.0
    assert r["court_speed"] > 0


def test_probability_safety_and_order_invariance():
    df = load_state()
    bundle = load_bundle()
    a, b = _two_players(df)
    ab = predict_match(df, bundle, a, b, "Hard", 1, 2000, 1.10, 8.00, tournament="Australian Open")
    ba = predict_match(df, bundle, b, a, "Hard", 2000, 1, 8.00, 1.10, tournament="Australian Open")
    assert 0.05 <= ab["probability_a"] <= 0.95
    assert 0.0 <= ab["quarter_kelly"] <= 0.25
    assert "raw_probability_a" in ab
    assert "calibrated_probability_a" in ab
    assert abs(ab["probability_a"] + ba["probability_a"] - 1.0) < 1e-10


def test_small_h2h_samples_are_continuously_shrunk_not_ignored():
    from atp_model import model_service

    mock = pd.DataFrame([
        {
            "player_1": "Player A", "player_2": "Player B", "surface": "All",
            "player_1_wins": 3, "player_2_wins": 0,
            "player_1_serve": 0.67, "player_2_serve": 0.61,
            "player_1_second_serve": 0.55, "player_2_second_serve": 0.48,
            "player_1_bp_convert": 0.42, "player_2_bp_convert": 0.31,
        },
        {
            "player_1": "Player A", "player_2": "Player B", "surface": "Hard",
            "player_1_wins": 3, "player_2_wins": 0,
            "player_1_serve": 0.67, "player_2_serve": 0.61,
            "player_1_second_serve": 0.55, "player_2_second_serve": 0.48,
            "player_1_bp_convert": 0.42, "player_2_bp_convert": 0.31,
        },
    ])
    original = model_service.load_h2h
    try:
        model_service.load_h2h = lambda: mock
        overall, surface, n, record = model_service.head_to_head_features("Player A", "Player B", "Hard")
        reverse = model_service._h2h_model_state("Player B", "Player A", "Hard")
        assert record["a_wins"] == 3 and record["b_wins"] == 0
        assert n == 3
        # Six-match neutral prior: 3-0 => edge 3/(3+6)=1/3, equivalent to 66.7% H2H strength.
        assert np.isclose(overall, 1.0 / 3.0)
        assert np.isclose(surface, 1.0 / 3.0)
        assert 0.0 < record["h2h_serve_diff"] < 0.06
        assert 0.0 < record["h2h_second_serve_diff"] < 0.07
        assert 0.0 < record["h2h_bp_convert_diff"] < 0.11
        assert record["used_as_predictive_edge"] is True
        assert np.isclose(reverse["h2h_overall_edge"], -overall)
        assert np.isclose(reverse["h2h_surface_edge"], -surface)
    finally:
        model_service.load_h2h = original

def test_large_h2h_sample_is_shrunk_and_order_symmetric():
    from atp_model import model_service

    mock = pd.DataFrame([
        {
            "player_1": "Player A", "player_2": "Player B", "surface": "All",
            "player_1_wins": 9, "player_2_wins": 3,
            "player_1_serve": 0.67, "player_2_serve": 0.61,
            "player_1_second_serve": 0.55, "player_2_second_serve": 0.48,
            "player_1_bp_convert": 0.42, "player_2_bp_convert": 0.31,
        },
        {
            "player_1": "Player A", "player_2": "Player B", "surface": "Hard",
            "player_1_wins": 6, "player_2_wins": 2,
            "player_1_serve": 0.68, "player_2_serve": 0.60,
            "player_1_second_serve": 0.56, "player_2_second_serve": 0.47,
            "player_1_bp_convert": 0.43, "player_2_bp_convert": 0.30,
        },
    ])
    original = model_service.load_h2h
    try:
        model_service.load_h2h = lambda: mock
        overall, surface, n, record = model_service.head_to_head_features("Player A", "Player B", "Hard")
        reverse = model_service._h2h_model_state("Player B", "Player A", "Hard")
        assert n == 12
        assert overall > 0 and surface > 0
        assert 0 < overall < 1
        assert 0 < surface < 1
        assert record["used_as_predictive_edge"] is True
        assert np.isclose(reverse["h2h_overall_edge"], -overall)
        assert np.isclose(reverse["h2h_surface_edge"], -surface)
        assert np.isclose(reverse["h2h_serve_diff"], -record["h2h_serve_diff"])
    finally:
        model_service.load_h2h = original

def test_matchstat_parser_preserves_advanced_missingness_and_stats():
    records = [{
        "id": 123,
        "date": "2026-08-12T01:00:00.000Z",
        "player1Id": 1,
        "player2Id": 2,
        "result": "6-3 6-1",
        "tournamentId": 99,
        "player1": {
            "id": 1, "name": "Winner", "currentRank": 10,
            "stats": {
                "firstServe": 28, "firstServeOf": 44, "aces": 1, "doubleFaults": 1,
                "winningOnFirstServe": 25, "winningOnSecondServe": 8,
                "breakPointsConverted": 4, "breakPointsConvertedOf": 6,
                "totalPointsWon": 60, "winners": None, "unforcedErrors": None,
            },
        },
        "player2": {
            "id": 2, "name": "Loser", "currentRank": 17,
            "stats": {
                "firstServe": 33, "firstServeOf": 53, "aces": 1, "doubleFaults": 3,
                "winningOnFirstServe": 21, "winningOnSecondServe": 5,
                "breakPointsConverted": 0, "breakPointsConvertedOf": 0,
                "totalPointsWon": 37, "winners": 20, "unforcedErrors": 25,
            },
        },
        "tournament": {
            "id": 99, "name": "National Bank Open - Montreal", "rankId": 3,
            "tier": "ATP Masters 1000", "court": {"name": "Hard"},
        },
    }]
    df = past_matches_to_master(records, {"winner": "old-w", "loser": "old-l"})
    assert len(df) == 1
    row = df.iloc[0]
    assert row.winner_id == "old-w"
    assert row.loser_id == "old-l"
    assert row.w_svpt == 44
    assert pd.isna(row.winner_rank) and pd.isna(row.loser_rank)
    assert row.w_bpSaved == 0
    assert row.w_bpFaced == 0
    assert pd.isna(row.w_winners) and pd.isna(row.w_unforced_errors)
    assert row.l_winners == 20 and row.l_unforced_errors == 25
    assert row.tourney_level == "M"


def test_odds_selector_rejects_inplay_quotes():
    payload = {
        "result": {
            "Pinnacle": {
                "Full Time Result": [
                    {"od1": "1.95", "od2": "1.90", "sourceAddTime": "900"},
                    {"od1": "2.25", "od2": "1.725", "sourceAddTime": "1100"},
                ]
            }
        }
    }
    quote = last_pre_match_quote(payload, 1000)
    assert quote["sourceAddTime"] == 900
    assert quote["od1"] == 1.95
    probs = no_vig_two_way(quote["od1"], quote["od2"])
    assert probs is not None and abs(sum(probs) - 1.0) < 1e-12

    summary = {
        "result": {"Pinnacle": {"Full Time Result": {
            "start": {"od1": "1.971", "od2": "1.917", "sourceAddTime": 800},
            "end": {"od1": "2.25", "od2": "1.725", "sourceAddTime": 1100},
        }}}
    }
    opening = safe_opening_quote(summary)
    assert opening["od1"] == 1.971 and opening["od2"] == 1.917


def test_live_court_speed_is_shrunk_toward_prior():
    combined, weight = shrink_live_speed(0.90, 1.40, matches_used=2, k=10)
    assert 0.90 < combined < 1.40
    assert np.isclose(weight, 2 / 12)


def test_cross_provider_dedup_coalesces_archive_rank_into_matchstat_row():
    import importlib.util
    spec = importlib.util.spec_from_file_location("update_data", ROOT / "scripts" / "update_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = {
        "tourney_date": 20260812,
        "tourney_name": "Montreal",
        "round": "R16",
        "winner_id": "w",
        "loser_id": "l",
        "winner_name": "Winner Name",
        "loser_name": "Loser Name",
        "surface": "Hard",
        "match_num": 1,
    }
    archival = pd.DataFrame([{**base, "data_source": "TennisMyLife", "winner_rank": 12, "w_winners": np.nan, "w_ace": 5}])
    matchstat = pd.DataFrame([{**base, "data_source": "Matchstat", "winner_rank": np.nan, "w_winners": 31, "w_ace": 6}])
    merged, _ = module.deduplicate_matches([archival, matchstat])
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row.data_source == "Matchstat"
    assert row.w_winners == 31
    assert row.w_ace == 6
    assert row.winner_rank == 12


def test_tracking_round_trip_and_clv(tmp_path):
    from atp_model import tracking

    old_file = tracking.TRACKING_FILE
    old_dir = tracking.TRACKING_DIR
    old_settings = tracking.SETTINGS_FILE
    old_legacy = tracking.LEGACY_TRACKING_FILE
    try:
        tracking.TRACKING_DIR = tmp_path / "tracking"
        tracking.TRACKING_FILE = tracking.TRACKING_DIR / "predictions.jsonl"
        tracking.SETTINGS_FILE = tracking.TRACKING_DIR / "settings.json"
        tracking.LEGACY_TRACKING_FILE = tmp_path / "missing_legacy.jsonl"

        result = {
            "player_a": "Player A", "player_b": "Player B", "surface": "Hard",
            "tournament": "Montreal", "tournament_level": 4.0, "court_speed": 1.08,
            "probability_a": 0.60, "market_probability_a": 0.55, "edge": 0.05,
            "ev": 0.08, "fair_odds_a": 1 / 0.60, "quarter_kelly": 0.025,
        }
        pid = tracking.save_prediction(result, 1.80, 2.10, 25.0)
        assert pid == 1
        rows = tracking.get_predictions()
        assert len(rows) == 1 and rows[0].stake == 25.0

        settled = tracking.settle_prediction(pid, True, 1.65, 2.25)
        assert np.isclose(settled.profit, 20.0)
        assert settled.price_clv > 0
        assert settled.probability_clv is not None

        tracking.set_starting_bankroll(1000)
        assert tracking.get_starting_bankroll() == 1000
        csv_text = tracking.predictions_csv()
        assert "Player A" in csv_text and "price_clv" in csv_text
    finally:
        tracking.TRACKING_FILE = old_file
        tracking.TRACKING_DIR = old_dir
        tracking.SETTINGS_FILE = old_settings
        tracking.LEGACY_TRACKING_FILE = old_legacy

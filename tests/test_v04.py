"""Regression tests for the v0.4 fixes."""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _row(**kw):
    base = {
        "tourney_name": "X", "surface": "Hard", "tourney_level": "A", "tourney_date": 20260105,
        "winner_id": "a", "loser_id": "b", "winner_name": "Jaume Munar", "loser_name": "Taylor Fritz",
        "score": "6-4 6-4", "round": "R32", "data_source": "TennisMyLife",
    }
    base.update(kw)
    return base


def test_cross_provider_duplicate_with_different_dates_is_removed():
    from atp_model.data_quality import clean_master
    df = pd.DataFrame([
        _row(tourney_name="Auckland", tourney_level="250", tourney_date=20260112, w_svpt=None),
        _row(tourney_name="ASB Classic - Auckland", tourney_date=20260115, data_source="Matchstat",
             winner_name="Jaume Antoni Munar Clar", w_svpt=60, round="First"),
    ])
    out, report = clean_master(df)
    assert len(out) == 1
    assert report["cross_provider_duplicates_removed"] == 1
    assert float(out.iloc[0]["w_svpt"]) == 60  # the richer row is kept


def test_genuine_rematch_from_same_provider_is_kept():
    from atp_model.data_quality import clean_master
    df = pd.DataFrame([
        _row(tourney_name="Hamburg Open - Hamburg", round="Q3", data_source="Matchstat"),
        _row(tourney_name="Hamburg Open - Hamburg", round="First", tourney_date=20260107, data_source="Matchstat",
             winner_name="Taylor Fritz", loser_name="Jaume Munar"),
    ])
    out, _ = clean_master(df)
    # Q3 row is removed as qualifying; main-draw row remains
    assert len(out) == 1


def test_itf_challenger_and_qualifying_rows_are_removed():
    from atp_model.data_quality import clean_master
    df = pd.DataFrame([
        _row(tourney_name="M25 Marrakech 1", tourney_level="C", data_source="Matchstat"),
        _row(tourney_name="Pune Challenger", tourney_level="C", data_source="Matchstat", winner_name="A B", loser_name="C D"),
        _row(tourney_name="Brisbane International - Brisbane", round="Q1", data_source="Matchstat", winner_name="E F", loser_name="G H"),
        _row(tourney_name="Brisbane International - Brisbane", round="First", data_source="Matchstat", winner_name="I J", loser_name="K L"),
    ])
    out, report = clean_master(df)
    assert len(out) == 1
    assert report["lower_tier_rows_removed"] == 2
    assert report["qualifying_rows_removed"] == 1


def test_matchstat_levels_are_corrected_from_history():
    from atp_model.data_quality import clean_master
    df = pd.DataFrame([
        _row(tourney_name="Barcelona", tourney_level="500", tourney_date=20250414, winner_name="P Q", loser_name="R S"),
        _row(tourney_name="Barcelona Open Banc Sabadell - Barcelona", tourney_level="A", tourney_date=20260411,
             data_source="Matchstat", winner_name="T U", loser_name="V W"),
    ])
    out, _ = clean_master(df)
    assert set(out["tourney_level"].astype(str)) == {"500"}


def test_houston_is_not_the_us_open():
    from atp_model.tournament_features import canonical_tournament
    assert canonical_tournament("US Men's Clay Court Championship - Houston") != "us new york"
    assert canonical_tournament("US Open") == "us new york"
    assert canonical_tournament("U.S. Open - New York") == "us new york"
    assert canonical_tournament("Next Gen ATP Finals") != canonical_tournament("Nitto ATP Finals")


def test_live_context_resolves_sponsor_names_and_paris_masters():
    from atp_model.slate import tournament_context, _resolve_context
    ctx = tournament_context(ROOT / "data/generated/master_matches.csv.gz")
    assert _resolve_context("China Open", ctx)["level"] == 3.0
    assert _resolve_context("Chengdu Open", ctx)["level"] == 2.0
    paris = _resolve_context("Rolex Paris Masters", ctx)
    assert paris["level"] == 4.0 and paris["surface"] == "Hard"


def test_no_fake_longshot_value_from_probability_clamp():
    from atp_model.model_service import MIN_PROBABILITY, _market_metrics
    assert MIN_PROBABILITY <= 0.01
    # A 2% player at 30.0 must not look like +50% EV any more.
    m = _market_metrics(0.02, 30.0, 1.02)
    assert m["ev"] < 0


def test_stake_is_capped():
    from atp_model.model_service import _market_metrics
    m = _market_metrics(0.80, 2.0, 1.9, kelly_fraction=0.25, kelly_cap=0.02)
    assert m["quarter_kelly"] == 0.02
    assert m["full_kelly"] > 0.5


def test_market_blend_with_zero_model_weight_ignores_model():
    from atp_model.historical_odds import blend_probability
    blend = {"available": True, "model_coef": 0.0, "market_coef": 1.0}
    assert abs(blend_probability(0.9, 0.6, blend) - 0.6) < 1e-9
    fav = blend_probability(0.5, 0.8, {"model_coef": 0.0, "market_coef": 1.06})
    assert fav > 0.8  # favourite-longshot correction


def test_prediction_is_fast_enough_for_the_live_board():
    from atp_model.model_service import load_bundle, load_state, predict_match
    state, bundle = load_state(), load_bundle()
    hard = state[state.surface == "Hard"].sort_values("rank")
    a, b = hard.player.iloc[0], hard.player.iloc[1]
    predict_match(state, bundle, a, b, "Hard", 1, 2, 1.8, 2.1, tournament_level=3.0, tournament="Beijing")
    t = time.time()
    for _ in range(3):
        r = predict_match(state, bundle, a, b, "Hard", 1, 2, 1.8, 2.1, tournament_level=3.0, tournament="Beijing")
    assert (time.time() - t) / 3 < 2.0
    assert abs(r["probability_a"] + r["probability_b"] - 1) < 1e-9
    assert "model_probability_a" in r


def test_eligible_events_explains_exclusions():
    from atp_model.slate import eligible_events
    now = time.time()
    context = {"beijing": {"surface": "Hard", "level": 3.0, "indoor": False, "tournament": "Beijing", "best_of": 3.0}}
    events = [
        {"id": 1, "participant1": "A", "participant2": "B", "league": "Beijing", "status": "Not started", "startTimestamp": now - 3600},
        {"id": 2, "participant1": "C", "participant2": "D", "league": "Beijing", "status": "Not started", "startTimestamp": now + 90 * 3600},
        {"id": 3, "participant1": "E", "participant2": "F", "league": "Mystery Cup", "status": "Not started", "startTimestamp": now + 3600},
        {"id": 4, "participant1": "G", "participant2": "H", "league": "Beijing", "status": "Not started", "startTimestamp": now + 3600},
    ]
    kept, diag = eligible_events(events, context, today_only=False, horizon_hours=48)
    assert [e["id"] for e in kept] == [4]
    assert diag["already_started"] == 1 and diag["outside_window"] == 1
    assert diag["unknown_tournaments"] == ["Mystery Cup"]


def test_tennis_data_name_keys_match_full_names():
    from atp_model.historical_odds import full_name_keys, td_key
    assert td_key("Auger-Aliassime F.") in full_name_keys("Felix Auger Aliassime")
    assert td_key("Bautista Agut R.") in full_name_keys("Roberto Bautista Agut")
    assert td_key("O Connell C.") in full_name_keys("Christopher O'Connell")


def test_backtest_counts_one_bet_per_match():
    from atp_model.historical_odds import backtest
    s, bets = backtest([0.7, 0.4], [2.0, 2.0], [2.0, 2.0], min_ev=0.02)
    assert s["bets"] == 2
    assert list(bets.won) == [1, 0]

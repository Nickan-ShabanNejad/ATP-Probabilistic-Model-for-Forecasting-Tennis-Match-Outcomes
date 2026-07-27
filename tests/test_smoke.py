
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from atp_model.model_service import load_state,load_bundle,predict_match

def test_prediction_smoke():
    df=load_state(); bundle=load_bundle()
    names=list(df.player.unique())
    a,b=names[0],names[1]
    r=predict_match(df,bundle,a,b,"Hard",100,120,1.8,2.05,tournament="Australian Open")
    assert 0 < r["probability_a"] < 1
    assert r["fair_odds_a"] > 1

    assert "h2h_record" in r
    assert r["court_speed"] > 0


def test_probability_safety_outputs():
    df = load_state(); bundle = load_bundle()
    names = list(df.player.unique())
    r = predict_match(df, bundle, names[0], names[1], "Hard", 1, 2000, 1.10, 8.00, tournament="Australian Open")
    assert 0.05 <= r["probability_a"] <= 0.95
    assert 0.0 <= r["quarter_kelly"] <= 0.05
    assert "raw_probability_a" in r
    assert "calibrated_probability_a" in r


def test_small_h2h_samples_are_not_used_as_predictive_edge():
    from atp_model import model_service
    original = model_service.load_h2h
    try:
        model_service.load_h2h = lambda: __import__("pandas").DataFrame([{
            "player_1": "Player A", "player_2": "Player B", "surface": "All",
            "player_1_wins": 3, "player_2_wins": 0,
            "surface_player_1_wins": 3, "surface_player_2_wins": 0,
        }, {
            "player_1": "Player A", "player_2": "Player B", "surface": "Hard",
            "player_1_wins": 3, "player_2_wins": 0,
            "surface_player_1_wins": 3, "surface_player_2_wins": 0,
        }])
        overall, surface, effective_matches, record = model_service.head_to_head_features(
            "Player A", "Player B", "Hard"
        )
        assert record["a_wins"] == 3
        assert overall == 0.0
        assert surface == 0.0
        assert effective_matches == 0
    finally:
        model_service.load_h2h = original


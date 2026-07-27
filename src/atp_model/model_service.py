
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

def predict_match(df, bundle, player_a, player_b, surface, rank_a, rank_b, odds_a, odds_b):
    a = get_player_row(df, player_a, surface)
    b = get_player_row(df, player_b, surface)
    def d(k): return float(a[k]) - float(b[k])
    x = [[
        d("overall_elo"), d("surface_elo"), d("serve"), d("return_rating"),
        math.log(max(rank_b,1))-math.log(max(rank_a,1)),
        d("win5"), d("win10"), d("surface_win10"), d("opp_elo10"),
        d("recent_perf10"), d("matches7"), d("matches14"),
        d("rest_days"), d("elo_change10"), d("age")
    ]]
    p = float(bundle["pipeline"].predict_proba(x)[0,1])
    raw_a, raw_b = 1/odds_a, 1/odds_b
    no_vig = raw_a/(raw_a+raw_b)
    edge = p-no_vig
    ev = p*odds_a-1
    fair = 1/p
    full_kelly = max(0.0, ev/(odds_a-1))
    return {
        "player_a":player_a,"player_b":player_b,"surface":surface,
        "probability_a":p,"probability_b":1-p,"market_probability_a":no_vig,
        "edge":edge,"ev":ev,"fair_odds_a":fair,
        "quarter_kelly":full_kelly*.25,"row_a":a,"row_b":b
    }

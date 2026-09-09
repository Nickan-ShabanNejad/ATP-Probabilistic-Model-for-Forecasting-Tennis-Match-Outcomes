from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import math

import joblib
import numpy as np
import pandas as pd

from .config import ROOT
from .model_service import get_player_row
from .odds import no_vig_two_way
from .tournament_features import canonical_tournament

SETS_MODEL_PATH = ROOT / "model/sets_model.joblib"
SETS_PROFILE_PATH = ROOT / "data/generated/player_sets_profiles.csv.gz"


@lru_cache(maxsize=1)
def load_sets_bundle():
    if not SETS_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(SETS_MODEL_PATH)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_sets_profiles() -> pd.DataFrame:
    if not SETS_PROFILE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(SETS_PROFILE_PATH, compression="gzip")
    except Exception:
        return pd.DataFrame()


def _profile(player: str, scope: str, tournament: str = "") -> dict:
    frame = load_sets_profiles()
    neutral = {"over35_rate": .5, "five_set_rate": .2, "straight_rate": .45, "matches": 0}
    if frame.empty:
        return neutral
    rows = frame[(frame.player.astype(str) == str(player)) & (frame.scope.astype(str) == scope)]
    if scope == "Event":
        key = canonical_tournament(tournament)
        rows = rows[rows.tournament.fillna("").astype(str) == key]
    if rows.empty:
        return neutral
    row = rows.iloc[0]
    out = {}
    for key, default in neutral.items():
        try:
            value = float(row.get(key, default))
            out[key] = value if np.isfinite(value) else default
        except Exception:
            out[key] = default
    out["matches"] = int(out["matches"])
    return out


def _feature_vector(df, player_a, player_b, surface, tournament, court_speed, tournament_level=5.0):
    a, b = get_player_row(df, player_a, surface), get_player_row(df, player_b, surface)
    ea, eb = float(a.surface_elo), float(b.surface_elo)
    expected_a = 1 / (1 + 10 ** ((eb - ea) / 400))
    ga, gb = _profile(player_a, "Grand Slam"), _profile(player_b, "Grand Slam")
    va, vb = _profile(player_a, "Event", tournament), _profile(player_b, "Event", tournament)
    surface_code = {"Hard": 0.0, "Clay": 1.0, "Grass": 2.0}.get(str(surface).title(), 0.0)
    x = [
        1.0 - abs(2.0 * expected_a - 1.0),
        abs(ea - eb) / 400.0,
        (float(a.serve) + float(b.serve)) / 2.0,
        abs(float(a.serve) - float(b.serve)),
        (float(a.return_rating) + float(b.return_rating)) / 2.0,
        (float(a.ace_rate5) + float(b.ace_rate5)) / 2.0,
        abs(float(a.point_share5) - float(b.point_share5)),
        float(court_speed),
        (float(court_speed)-1.0) * ((float(a.serve)+float(b.serve))/2.0),
        (ga["over35_rate"]+gb["over35_rate"])/2.0,
        abs(ga["over35_rate"]-gb["over35_rate"]),
        (va["over35_rate"]+vb["over35_rate"])/2.0,
        (ga["five_set_rate"]+gb["five_set_rate"])/2.0,
        (ga["straight_rate"]+gb["straight_rate"])/2.0,
        math.log1p(min(ga["matches"],gb["matches"])),
        math.log1p(min(va["matches"],vb["matches"])),
        surface_code,
        float(tournament_level),
    ]
    return np.array([x], dtype=float), {"player_a_gs":ga,"player_b_gs":gb,"player_a_event":va,"player_b_event":vb,"elo_closeness":x[0]}


def predict_over35(df, player_a, player_b, surface, tournament, court_speed, odds_over=None, odds_under=None):
    bundle = load_sets_bundle()
    if bundle is None:
        return {"available": False, "reason": "Grand Slam sets model has not been trained yet."}
    x, profiles = _feature_vector(df,player_a,player_b,surface,tournament,court_speed,5.0)
    p_over = float(bundle["pipeline"].predict_proba(x)[0,1])
    p_over = min(.95,max(.05,p_over)); p_under=1-p_over
    result = {
        "available":True,"probability_over35":p_over,"probability_under35":p_under,
        "fair_odds_over35":1/p_over,"fair_odds_under35":1/p_under,"profiles":profiles,
        "recommended_market":"No bet","recommended_quarter_kelly":0.0,
    }
    if odds_over and odds_under:
        market = no_vig_two_way(odds_over,odds_under)
        if market:
            mo,mu=market
            evo=p_over*float(odds_over)-1; evu=p_under*float(odds_under)-1
            ko=max(0,evo/(float(odds_over)-1))*.25; ku=max(0,evu/(float(odds_under)-1))*.25
            result.update({
                "odds_over35":float(odds_over),"odds_under35":float(odds_under),
                "market_probability_over35":mo,"market_probability_under35":mu,
                "edge_over35":p_over-mo,"edge_under35":p_under-mu,"ev_over35":evo,"ev_under35":evu,
                "quarter_kelly_over35":ko,"quarter_kelly_under35":ku,
            })
            if evo>evu and evo>0:
                result["recommended_market"]="Over 3.5 sets"; result["recommended_quarter_kelly"]=ko; result["recommended_ev"]=evo; result["recommended_edge"]=p_over-mo
            elif evu>0:
                result["recommended_market"]="Under 3.5 sets"; result["recommended_quarter_kelly"]=ku; result["recommended_ev"]=evu; result["recommended_edge"]=p_under-mu
    return result

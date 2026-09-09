from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import math

import joblib
import numpy as np
import pandas as pd

from .config import ROOT
from .features import FEATURES
from .tournament_features import canonical_tournament, load_surface_speeds, prediction_surface_speed

STATE_GENERATED = ROOT / "data/generated/player_state.csv.gz"
STATE_BOOTSTRAP = ROOT / "data/bootstrap/player_state.csv.gz"
MODEL_GENERATED = ROOT / "model/model.joblib"
MODEL_BOOTSTRAP = ROOT / "model/bootstrap_model.joblib"
H2H_PATH = ROOT / "data/generated/head_to_head.csv.gz"
EVENT_PROFILE_PATH = ROOT / "data/generated/player_event_profiles.csv.gz"
LEVEL_PROFILE_PATH = ROOT / "data/generated/player_level_profiles.csv.gz"
SPEED_PROFILE_PATH = ROOT / "data/generated/player_speed_profiles.csv.gz"
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


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, compression="gzip" if path.suffix == ".gz" else None)
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_h2h():
    return _safe_csv(H2H_PATH)


@lru_cache(maxsize=1)
def load_event_profiles():
    return _safe_csv(EVENT_PROFILE_PATH)


@lru_cache(maxsize=1)
def load_level_profiles():
    return _safe_csv(LEVEL_PROFILE_PATH)


@lru_cache(maxsize=1)
def load_speed_profiles():
    return _safe_csv(SPEED_PROFILE_PATH)


def _numeric(row, key, default=0.0):
    try:
        value = float(row.get(key, default))
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _h2h_model_state(player_a, player_b, surface):
    h2h = load_h2h()
    empty = {
        "h2h_overall_edge": 0.0, "h2h_surface_edge": 0.0, "h2h_serve_diff": 0.0,
        "h2h_second_serve_diff": 0.0, "h2h_bp_convert_diff": 0.0,
        "h2h_matches_log": 0.0, "h2h_surface_matches_log": 0.0,
        "matches": 0, "surface_matches": 0, "a_wins": 0, "b_wins": 0,
        "surface_a_wins": 0, "surface_b_wins": 0, "player_a_wins": 0, "player_b_wins": 0,
        "surface_player_a_wins": 0, "surface_player_b_wins": 0, "used_as_predictive_edge": False,
    }
    if h2h.empty or not {"player_1", "player_2"}.issubset(h2h.columns):
        return empty
    direct = h2h[(h2h.player_1 == player_a) & (h2h.player_2 == player_b)]
    reverse = h2h[(h2h.player_1 == player_b) & (h2h.player_2 == player_a)]
    orientation = 1.0
    rows = direct
    if rows.empty:
        rows, orientation = reverse, -1.0
    if rows.empty:
        return empty
    all_rows = rows[rows.surface.astype(str).str.casefold() == "all"]
    overall_row = (all_rows if not all_rows.empty else rows).iloc[0]
    surf_rows = rows[rows.surface.astype(str).str.casefold() == str(surface).casefold()]
    surface_row = surf_rows.iloc[0] if not surf_rows.empty else None
    if orientation > 0:
        a_wins, b_wins = int(_numeric(overall_row, "player_1_wins", 0)), int(_numeric(overall_row, "player_2_wins", 0))
    else:
        a_wins, b_wins = int(_numeric(overall_row, "player_2_wins", 0)), int(_numeric(overall_row, "player_1_wins", 0))
    matches = a_wins + b_wins
    if surface_row is not None:
        if orientation > 0:
            sa, sb = int(_numeric(surface_row, "player_1_wins", 0)), int(_numeric(surface_row, "player_2_wins", 0))
        else:
            sa, sb = int(_numeric(surface_row, "player_2_wins", 0)), int(_numeric(surface_row, "player_1_wins", 0))
    else:
        sa = sb = 0
    sn = sa + sb

    def metric_diff(row, p1, p2, evidence):
        if row is None or evidence <= 0:
            return 0.0
        x, y = _numeric(row, p1, np.nan), _numeric(row, p2, np.nan)
        if not np.isfinite(x) or not np.isfinite(y):
            return 0.0
        return float((x - y) * orientation * evidence / (evidence + H2H_METRIC_PRIOR_MATCHES))

    return {
        "h2h_overall_edge": float((a_wins - b_wins) / (matches + H2H_PRIOR_MATCHES)) if matches else 0.0,
        "h2h_surface_edge": float((sa - sb) / (sn + H2H_PRIOR_MATCHES)) if sn else 0.0,
        "h2h_serve_diff": metric_diff(overall_row, "player_1_serve", "player_2_serve", matches),
        "h2h_second_serve_diff": metric_diff(overall_row, "player_1_second_serve", "player_2_second_serve", matches),
        "h2h_bp_convert_diff": metric_diff(overall_row, "player_1_bp_convert", "player_2_bp_convert", matches),
        "h2h_matches_log": float(math.log1p(matches)), "h2h_surface_matches_log": float(math.log1p(sn)),
        "matches": matches, "surface_matches": sn, "a_wins": a_wins, "b_wins": b_wins,
        "surface_a_wins": sa, "surface_b_wins": sb, "player_a_wins": a_wins, "player_b_wins": b_wins,
        "surface_player_a_wins": sa, "surface_player_b_wins": sb, "used_as_predictive_edge": matches > 0,
    }


def head_to_head_features(player_a, player_b, surface, min_matches=1):
    record = _h2h_model_state(player_a, player_b, surface)
    if record["matches"] < int(min_matches):
        return 0.0, 0.0, int(record["matches"]), record
    return float(record["h2h_overall_edge"]), float(record["h2h_surface_edge"]), int(record["matches"]), record


def _profile_lookup(frame: pd.DataFrame, player: str, filters: dict, value_col: str, matches_col="matches") -> tuple[float, int]:
    if frame.empty or "player" not in frame.columns:
        return 0.0, 0
    rows = frame[frame.player.astype(str) == str(player)]
    for key, value in filters.items():
        if key not in rows.columns:
            return 0.0, 0
        if isinstance(value, float):
            rows = rows[np.isclose(pd.to_numeric(rows[key], errors="coerce"), value, equal_nan=False)]
        else:
            rows = rows[rows[key].astype(str) == str(value)]
    if rows.empty:
        return 0.0, 0
    row = rows.iloc[0]
    return _numeric(row, value_col, 0.0), int(_numeric(row, matches_col, 0))


def context_profile_state(player_a, player_b, tournament, level, speed):
    event_key = canonical_tournament(str(tournament or ""))
    event = load_event_profiles(); levels = load_level_profiles(); speeds = load_speed_profiles()
    ea, ena = _profile_lookup(event, player_a, {"tournament": event_key}, "event_perf")
    eb, enb = _profile_lookup(event, player_b, {"tournament": event_key}, "event_perf")
    la, lna = _profile_lookup(levels, player_a, {"level": float(level)}, "level_perf")
    lb, lnb = _profile_lookup(levels, player_b, {"level": float(level)}, "level_perf")
    sa, sna = _profile_lookup(speeds, player_a, {}, "speed_slope")
    sb, snb = _profile_lookup(speeds, player_b, {}, "speed_slope")
    slope_diff = sa - sb
    return {
        "event_perf_edge": ea - eb, "level_perf_edge": la - lb, "speed_fit_edge": slope_diff,
        "speed_fit_current_interaction": (float(speed) - 1.0) * slope_diff,
        "event_matches_log": float(math.log1p(min(ena, enb))),
        "level_matches_log": float(math.log1p(min(lna, lnb))),
        "speed_fit_matches_log": float(math.log1p(min(sna, snb))),
        "event_player_a": ea, "event_player_b": eb, "event_matches_a": ena, "event_matches_b": enb,
        "level_player_a": la, "level_player_b": lb, "level_matches_a": lna, "level_matches_b": lnb,
        "speed_slope_a": sa, "speed_slope_b": sb, "speed_matches_a": sna, "speed_matches_b": snb,
    }


@lru_cache(maxsize=1)
def load_tournament_speeds():
    prior = load_surface_speeds(EXTERNAL_SPEED_PATH)
    if prior.empty:
        prior = load_surface_speeds(LEGACY_SPEED_PATH)
    return prior


@lru_cache(maxsize=1)
def load_live_tournament_speeds():
    return load_surface_speeds(EMPIRICAL_SPEED_PATH)


def available_tournaments(surface=None):
    frames = [x for x in (load_tournament_speeds(), load_live_tournament_speeds()) if not x.empty]
    if not frames:
        return []
    speeds = pd.concat(frames, ignore_index=True)
    if surface:
        speeds = speeds[speeds.surface.astype(str).str.title() == str(surface).title()]
    return sorted(speeds.tournament.dropna().astype(str).unique())


def tournament_speed_details(tournament, surface, prediction_year=None):
    year = int(prediction_year or pd.Timestamp.today().year)
    speed, meta = prediction_surface_speed(load_tournament_speeds(), load_live_tournament_speeds(), tournament, surface, year)
    return float(speed), meta


def tournament_speed(tournament, surface, prediction_year=None):
    speed, meta = tournament_speed_details(tournament, surface, prediction_year)
    missing = 1.0 if meta.get("prior_missing") and not meta.get("live_matches") else 0.0
    return speed, missing


def get_player_row(df, player, surface):
    rows = df[(df.player == player) & (df.surface == surface)]
    if rows.empty:
        raise ValueError(f"No rating state for {player} on {surface}.")
    return rows.iloc[0]


def automatic_best_of(tournament_level: float, api_best_of=None) -> float:
    try:
        parsed = float(api_best_of)
        if parsed in {3.0, 5.0}:
            return parsed
    except Exception:
        pass
    return 5.0 if float(tournament_level) >= 5.0 else 3.0


def _feature_frame(row_a, row_b, rank_a, rank_b, level, best_of, speed, speed_meta, indoor, h2h, context, feature_order):
    def diff(key, default=0.0):
        return _numeric(row_a, key, default) - _numeric(row_b, key, default)
    surface_elo_diff, serve_diff, return_diff = diff("surface_elo"), diff("serve", .635), diff("return_rating", .365)
    rank_advantage = math.log(max(float(rank_b), 1.0)) - math.log(max(float(rank_a), 1.0))
    level_centered, speed_centered, bestof_centered = float(level) - 3.0, float(speed) - 1.0, float(best_of) - 3.0
    indoor_value = 1.0 if indoor else 0.0
    values = {
        "overall_elo_diff": diff("overall_elo"), "surface_elo_diff": surface_elo_diff, "serve_diff": serve_diff,
        "return_diff": return_diff, "log_rank_advantage": rank_advantage,
        "win3_diff": diff("win3",.5), "win5_diff": diff("win5",.5), "win10_diff": diff("win10",.5), "surface_win10_diff": diff("surface_win10",.5),
        "spw1_diff": diff("spw1",.635), "spw3_diff": diff("spw3",.635), "spw5_diff": diff("spw5",.635), "spw10_diff": diff("spw10",.635),
        "rpw1_diff": diff("rpw1",.365), "rpw3_diff": diff("rpw3",.365), "rpw5_diff": diff("rpw5",.365), "rpw10_diff": diff("rpw10",.365),
        "first_in5_diff": diff("first_in5",.62), "first_won5_diff": diff("first_won5",.70), "second_won5_diff": diff("second_won5",.50),
        "ace_rate5_diff": diff("ace_rate5",.08), "df_rate5_diff": diff("df_rate5",.035), "point_share5_diff": diff("point_share5",.5), "point_share10_diff": diff("point_share10",.5),
        "bp_save5_diff": diff("bp_save5",.62), "bp_convert5_diff": diff("bp_convert5",.38),
        "form_ewma_diff": diff("form_ewma",.5), "surface_form_ewma_diff": diff("surface_form_ewma",.5), "opp_elo10_diff": diff("opp_elo10",1500), "recent_perf10_diff": diff("recent_perf10",0),
        "matches7_diff": diff("matches7"), "matches14_diff": diff("matches14"), "rest_days_diff": diff("rest_days",30), "elo_change10_diff": diff("elo_change10"), "age_diff": diff("age"),
        "winner_rate_diff": diff("winner_rate",.15), "ue_rate_diff": diff("ue_rate",.15), "aggression_quality_diff": diff("aggression_quality"), "advanced_coverage_diff": diff("advanced_coverage"),
        "net_win_diff": diff("net_win",.65), "avg_first_serve_speed_diff": diff("avg_first_serve_speed"),
        "h2h_overall_edge": float(h2h["h2h_overall_edge"]), "h2h_surface_edge": float(h2h["h2h_surface_edge"]),
        "h2h_serve_diff": float(h2h["h2h_serve_diff"]), "h2h_second_serve_diff": float(h2h["h2h_second_serve_diff"]), "h2h_bp_convert_diff": float(h2h["h2h_bp_convert_diff"]),
        "event_perf_edge": float(context.get("event_perf_edge",0)), "level_perf_edge": float(context.get("level_perf_edge",0)), "speed_fit_edge": float(context.get("speed_fit_edge",0)),
        "level_surface_elo_interaction": level_centered*surface_elo_diff/400.0, "level_rank_interaction": level_centered*rank_advantage,
        "level_serve_interaction": level_centered*serve_diff*10.0, "level_form_interaction": level_centered*diff("form_ewma",.5),
        "speed_surface_elo_interaction": speed_centered*surface_elo_diff/100.0, "speed_serve_interaction": speed_centered*serve_diff*10.0,
        "speed_return_interaction": speed_centered*return_diff*10.0, "speed_ace_interaction": speed_centered*diff("ace_rate5",.08)*10.0,
        "speed_second_serve_interaction": speed_centered*diff("second_won5",.5)*10.0, "speed_point_share_interaction": speed_centered*diff("point_share5",.5)*10.0,
        "speed_fit_current_interaction": float(context.get("speed_fit_current_interaction",0)),
        "indoor_serve_interaction": indoor_value*serve_diff*10.0, "indoor_return_interaction": indoor_value*return_diff*10.0,
        "bestof_surface_elo_interaction": bestof_centered*surface_elo_diff/400.0,
        "tournament_level": float(level), "best_of": float(best_of), "court_speed": float(speed), "court_speed_prior": float(speed_meta.get("prior_speed", speed)),
        "court_speed_live_weight": float(speed_meta.get("live_weight",0)), "court_speed_missing": 1.0 if speed_meta.get("prior_missing") else 0.0, "indoor": indoor_value,
        "h2h_matches_log": float(h2h["h2h_matches_log"]), "h2h_surface_matches_log": float(h2h["h2h_surface_matches_log"]),
        "event_matches_log": float(context.get("event_matches_log",0)), "level_matches_log": float(context.get("level_matches_log",0)), "speed_fit_matches_log": float(context.get("speed_fit_matches_log",0)),
        # legacy bootstrap aliases
        "chart_winner_rate_diff": diff("winner_rate",.15), "chart_ue_rate_diff": diff("ue_rate",.15), "chart_available_diff": diff("advanced_coverage"),
    }
    order = feature_order or FEATURES
    return pd.DataFrame([[values.get(name,0.0) for name in order]], columns=order)


def _symmetric_probability(bundle, x_forward, x_reverse):
    fp = float(bundle["pipeline"].predict_proba(x_forward.to_numpy(dtype=float))[0,1])
    rb = float(bundle["pipeline"].predict_proba(x_reverse.to_numpy(dtype=float))[0,1])
    ra = 1.0-rb
    raw = 0.5*(fp+ra)
    return min(MAX_PROBABILITY,max(MIN_PROBABILITY,raw)), fp, ra


def _market_metrics(probability, odds_self, odds_other):
    oa, ob = float(odds_self), float(odds_other)
    if oa <= 1 or ob <= 1:
        raise ValueError("Decimal odds must be greater than 1.0.")
    ia, ib = 1/oa, 1/ob
    market = ia/(ia+ib)
    ev = probability*oa-1
    full_kelly = max(0.0, ev/(oa-1.0))
    return {"market":market,"edge":probability-market,"ev":ev,"fair":1/probability,"quarter_kelly":full_kelly*.25}


def _group_impacts(bundle, x_f, x_r, base_probability):
    groups = {
        "Court speed": [x for x in x_f.columns if "speed" in x or "court_speed" in x],
        "Exact event history": [x for x in x_f.columns if x.startswith("event_")],
        "Tournament-level history": [x for x in x_f.columns if x.startswith("level_perf")],
        "H2H": [x for x in x_f.columns if x.startswith("h2h_")],
        "Recent form": [x for x in x_f.columns if any(k in x for k in ("win3","win5","win10","form_ewma","recent_perf","elo_change"))],
        "Serve / return profile": [x for x in x_f.columns if any(k in x for k in ("serve_diff","return_diff","spw","rpw","ace_rate","second_won","point_share"))],
        "Surface Elo": [x for x in x_f.columns if "surface_elo" in x],
    }
    impacts = {}
    for label, cols in groups.items():
        if not cols:
            continue
        af, ar = x_f.copy(), x_r.copy()
        for col in cols:
            # Zeroing differential/interaction features is a neutral counterfactual.
            # Raw context values are left alone except court speed, which is reset to neutral.
            if col == "court_speed":
                af[col] = ar[col] = 1.0
            elif col == "court_speed_prior":
                af[col] = ar[col] = 1.0
            elif col == "court_speed_live_weight":
                af[col] = ar[col] = 0.0
            else:
                af[col] = ar[col] = 0.0
        p,_,_ = _symmetric_probability(bundle, af, ar)
        impacts[label] = float(base_probability-p)
    return dict(sorted(impacts.items(), key=lambda kv: abs(kv[1]), reverse=True))


def predict_match(df,bundle,player_a,player_b,surface,rank_a,rank_b,odds_a,odds_b,tournament_level=2.0,best_of=None,tournament="",prediction_year=None,court_speed_override=None,indoor=False,api_best_of=None):
    a,b = get_player_row(df,player_a,surface), get_player_row(df,player_b,surface)
    level=float(tournament_level); best_of=automatic_best_of(level, api_best_of if api_best_of is not None else best_of)
    estimated_speed,speed_meta=tournament_speed_details(tournament,surface,prediction_year)
    if court_speed_override is None:
        speed=float(estimated_speed)
    else:
        speed=float(court_speed_override); speed_meta={"prior_speed":speed,"prior_missing":False,"live_speed":None,"live_matches":0,"live_weight":0.0}
    feature_order=bundle.get("features") or FEATURES
    hf,hr=_h2h_model_state(player_a,player_b,surface),_h2h_model_state(player_b,player_a,surface)
    cf=context_profile_state(player_a,player_b,tournament,level,speed)
    cr=context_profile_state(player_b,player_a,tournament,level,speed)
    xf=_feature_frame(a,b,rank_a,rank_b,level,best_of,speed,speed_meta,indoor,hf,cf,feature_order)
    xr=_feature_frame(b,a,rank_b,rank_a,level,best_of,speed,speed_meta,indoor,hr,cr,feature_order)
    probability,fp,ra=_symmetric_probability(bundle,xf,xr)
    ma=_market_metrics(probability,odds_a,odds_b); pb=1-probability; mb=_market_metrics(pb,odds_b,odds_a)
    if ma["ev"] > mb["ev"] and ma["ev"] > 0:
        side, pick, pick_ev, pick_edge, pick_kelly, pick_odds = "A", player_a, ma["ev"], ma["edge"], ma["quarter_kelly"], float(odds_a)
    elif mb["ev"] > 0:
        side, pick, pick_ev, pick_edge, pick_kelly, pick_odds = "B", player_b, mb["ev"], mb["edge"], mb["quarter_kelly"], float(odds_b)
    else:
        side, pick, pick_ev, pick_edge, pick_kelly, pick_odds = "NO BET", "No bet", max(ma["ev"],mb["ev"]), max(ma["edge"],mb["edge"]), 0.0, np.nan
    impacts=_group_impacts(bundle,xf,xr,probability)
    return {
        "player_a":player_a,"player_b":player_b,"surface":surface,"tournament":tournament,"tournament_level":level,"best_of":best_of,
        "court_speed":speed,"court_speed_prior":speed_meta.get("prior_speed"),"court_speed_live":speed_meta.get("live_speed"),"court_speed_live_matches":speed_meta.get("live_matches",0),"court_speed_live_weight":speed_meta.get("live_weight",0.0),"court_speed_fallback":bool(speed_meta.get("prior_missing")),"indoor":bool(indoor),
        "raw_probability_a":probability,"forward_raw_probability_a":fp,"reverse_raw_probability_a":ra,"symmetry_gap_before_fix":fp-ra,"calibrated_probability_a":probability,
        "probability_a":probability,"probability_b":pb,"market_probability_a":ma["market"],"market_probability_b":mb["market"],
        "edge":ma["edge"],"edge_a":ma["edge"],"edge_b":mb["edge"],"ev":ma["ev"],"ev_a":ma["ev"],"ev_b":mb["ev"],
        "fair_odds_a":ma["fair"],"fair_odds_b":mb["fair"],"quarter_kelly":ma["quarter_kelly"],"quarter_kelly_a":ma["quarter_kelly"],"quarter_kelly_b":mb["quarter_kelly"],
        "recommended_side":side,"recommended_pick":pick,"recommended_ev":pick_ev,"recommended_edge":pick_edge,"recommended_quarter_kelly":pick_kelly,"recommended_odds":pick_odds,
        "base_probability_a":probability,"h2h_impact":0.0,"h2h_in_model":True,"h2h_overall_edge":hf["h2h_overall_edge"],"h2h_surface_edge":hf["h2h_surface_edge"],"h2h_record":hf,
        "context_profile":cf,"factor_impacts":impacts,"row_a":a,"row_b":b,"feature_frame":xf,
    }


def court_speed_curve(df,bundle,**kwargs):
    points=[]
    for speed in (0.70,0.85,1.00,1.15,1.30,1.45,1.55):
        r=predict_match(df,bundle,court_speed_override=speed,**kwargs)
        points.append({"Court speed":speed,"P(A)":r["probability_a"],"P(B)":r["probability_b"]})
    return pd.DataFrame(points)

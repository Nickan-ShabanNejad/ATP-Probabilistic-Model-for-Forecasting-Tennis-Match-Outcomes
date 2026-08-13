from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import random
import re
import sys
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "data" / "generated"
RAW = ROOT / "data" / "raw"
MODEL = ROOT / "model"
MODEL.mkdir(parents=True, exist_ok=True)
GENERATED.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from atp_model.features import DIFF_FEATURES, CONTEXT_FEATURES, FEATURES
from atp_model.tournament_features import (
    build_empirical_surface_speed_table,
    canonical_tournament,
    encode_tournament_level,
    load_surface_speeds,
    lookup_surface_speed,
    shrink_live_speed,
)

INITIAL_ELO = 1500.0
K = 28.0
ALPHA = 0.15
SURFACES = {"Hard", "Clay", "Grass"}
RETIREMENT_MARKERS = ("RET", "W/O", "WO", "DEF", "ABN", "ABD", "WALKOVER")


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def safe_ratio(num, den):
    num, den = number(num), number(den)
    if num is None or den is None or den <= 0:
        return None
    return num / den


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return " ".join(text.split())


def find_column(columns, aliases):
    lower = {str(c).lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    for c in columns:
        compact = re.sub(r"[^a-z0-9]", "", str(c).lower())
        for alias in aliases:
            if compact == re.sub(r"[^a-z0-9]", "", alias.lower()):
                return c
    return None


def load_charting_events():
    """Optional historical advanced-stat source.

    Matchstat is the production source for new data. Match Charting Project is
    retained only as a leakage-safe historical fallback for winners/UE where
    those fields are otherwise sparse.
    """
    overview_path = RAW / "charting-m-stats-Overview.csv"
    matches_path = RAW / "charting-m-matches.csv"
    if not overview_path.exists():
        return defaultdict(list), {"available": False, "reason": "overview file missing"}
    try:
        overview = pd.read_csv(overview_path, low_memory=False)
    except Exception as exc:
        return defaultdict(list), {"available": False, "reason": str(exc)}
    columns = list(overview.columns)
    match_col = find_column(columns, ["match_id", "matchid"])
    player_col = find_column(columns, ["player", "player_name"])
    if match_col is None or player_col is None:
        return defaultdict(list), {"available": False, "reason": "match/player columns not found"}

    date_by_match = {}
    if matches_path.exists():
        try:
            metadata = pd.read_csv(matches_path, low_memory=False)
            mmatch = find_column(metadata.columns, ["match_id", "matchid"])
            mdate = find_column(metadata.columns, ["date", "match_date"])
            if mmatch and mdate:
                for _, row in metadata[[mmatch, mdate]].dropna().iterrows():
                    parsed = pd.to_datetime(row[mdate], errors="coerce")
                    if pd.notna(parsed):
                        date_by_match[str(row[mmatch])] = parsed.to_pydatetime()
        except Exception:
            pass

    winners_col = find_column(columns, ["winners", "winner"])
    ue_col = find_column(columns, ["unforced", "unforced_errors", "ues", "ue"])
    serve_col = find_column(columns, ["serve_pts", "servepoints", "svpts"])
    return_won_col = find_column(columns, ["return_pts_won", "returnpointswon"])
    return_col = find_column(columns, ["return_pts", "returnpoints"])

    events = defaultdict(list)
    for _, row in overview.iterrows():
        match_id = str(row[match_col])
        player = normalize_name(row[player_col])
        if not player:
            continue
        date = date_by_match.get(match_id)
        if date is None:
            found = re.search(r"(19|20)\d{6}", match_id)
            if found:
                date = datetime.strptime(found.group(0), "%Y%m%d")
        if date is None:
            continue
        winners = number(row[winners_col]) if winners_col else None
        ue = number(row[ue_col]) if ue_col else None
        sp = number(row[serve_col]) if serve_col else None
        rp = number(row[return_col]) if return_col else None
        rpw = number(row[return_won_col]) if return_won_col else None
        points = max((sp or 0) + (rp or 0), 1)
        events[player].append(
            {
                "date": date,
                "winner_rate": winners / points if winners is not None else None,
                "ue_rate": ue / points if ue is not None else None,
                "return_point_win": rpw / rp if rpw is not None and rp and rp > 0 else None,
            }
        )
    for player in events:
        events[player].sort(key=lambda x: x["date"])
    return events, {
        "available": bool(events),
        "players": len(events),
        "rows": sum(len(v) for v in events.values()),
    }


def charting_profile(events, player_name, date):
    recent = [x for x in events.get(normalize_name(player_name), []) if x["date"] < date][-20:]
    if not recent:
        return {"chart_winner_rate": np.nan, "chart_ue_rate": np.nan, "chart_available": 0.0}

    def avg(key):
        vals = [x[key] for x in recent if x.get(key) is not None and np.isfinite(x[key])]
        return float(np.mean(vals)) if vals else np.nan

    return {
        "chart_winner_rate": avg("winner_rate"),
        "chart_ue_rate": avg("ue_rate"),
        "chart_available": 1.0,
    }


def ewm(old, new, alpha=ALPHA):
    if new is None or not np.isfinite(new):
        return old
    return float(new) if old is None else float(alpha * new + (1 - alpha) * old)


def weighted_recent(records, key, default, max_n=10, decay=0.78):
    vals = [m.get(key) for m in records[-max_n:]]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return default
    weights = np.array([decay ** (len(vals) - 1 - i) for i in range(len(vals))], dtype=float)
    return float(np.average(vals, weights=weights))


def avg_metric(records, key, default=np.nan):
    vals = [m.get(key) for m in records if m.get(key) is not None and np.isfinite(m.get(key))]
    return float(np.mean(vals)) if vals else default


def load_current_rankings():
    current_rankings = {}
    current_rankings_by_name = {}
    path = GENERATED / "current_rankings.csv"
    if not path.exists():
        return current_rankings, current_rankings_by_name
    rankings = pd.read_csv(path, dtype={"player_id": str})
    rankings["ranking"] = pd.to_numeric(rankings.get("ranking"), errors="coerce")
    if "player_id" in rankings:
        rankings["player_id"] = rankings["player_id"].str.replace(r"\.0$", "", regex=True)
        valid = rankings.dropna(subset=["player_id", "ranking"])
        current_rankings = dict(zip(valid["player_id"], valid["ranking"]))
    if "player" in rankings:
        valid = rankings.dropna(subset=["player", "ranking"])
        current_rankings_by_name = dict(zip(valid["player"].map(normalize_name), valid["ranking"]))
    return current_rankings, current_rankings_by_name


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
master_path = GENERATED / "master_matches.csv.gz"
if not master_path.exists():
    raise RuntimeError("Run scripts/update_data.py first; master match data is missing.")

matches = pd.read_csv(master_path, low_memory=False)
matches["tourney_date"] = pd.to_numeric(matches["tourney_date"], errors="coerce")
matches = matches.dropna(subset=["tourney_date", "winner_id", "loser_id", "surface"])
matches["date"] = pd.to_datetime(matches["tourney_date"].astype(int).astype(str), format="%Y%m%d")
matches["match_num"] = pd.to_numeric(matches.get("match_num", 0), errors="coerce").fillna(0)
matches = matches.sort_values(["date", "tourney_name", "match_num"], na_position="last").reset_index(drop=True)

empirical_path = GENERATED / "tournament_surface_speed_empirical.csv"
empirical_speeds = build_empirical_surface_speed_table(matches, empirical_path)
external_path = ROOT / "data" / "tournament_surface_speed_external.csv"
legacy_path = ROOT / "data" / "tournament_surface_speed.csv"
prior_speeds = load_surface_speeds(external_path)
prior_speed_source = "Tennis Abstract external surface-speed table"
if prior_speeds.empty:
    prior_speeds = load_surface_speeds(legacy_path)
    prior_speed_source = "legacy/empirical surface-speed table"
if prior_speeds.empty:
    prior_speeds = empirical_speeds
    prior_speed_source = "empirical tournament-season proxy"

chart_events, chart_meta = load_charting_events()
current_rankings, current_rankings_by_name = load_current_rankings()

# Speed lookup is identical for every match in the same tournament/surface/year.
# Cache it so training stays fast even when the external speed table is large.
_prior_speed_cache = {}
def cached_prior_speed(tournament, surface, season):
    key = (canonical_tournament(str(tournament or "")), str(surface).title(), int(season))
    if key not in _prior_speed_cache:
        _prior_speed_cache[key] = lookup_surface_speed(
            prior_speeds, str(tournament or ""), str(surface), int(season), prior_only=True
        )
    return _prior_speed_cache[key]

# ---------------------------------------------------------------------------
# Chronological state
# ---------------------------------------------------------------------------
elo = defaultdict(lambda: INITIAL_ELO)
surface_elo = defaultdict(lambda: INITIAL_ELO)
serve_ewma = defaultdict(lambda: {"serve": None, "return": None})
surface_ewma = defaultdict(lambda: {"serve": None, "return": None})
history = defaultdict(lambda: deque(maxlen=80))
elo_trail = defaultdict(lambda: deque(maxlen=10))
h2h_history = defaultdict(lambda: deque(maxlen=40))
last_seen = {}
names = {}
historical_ranks = {}
ages = {}
rows = []
rng = random.Random(123)

# Online condition baselines and current-event samples. These are updated only
# AFTER each match, so the feature for a match never sees itself or the future.
surface_condition = defaultdict(lambda: {"n": 0, "ace_sum": 0.0, "ace_sq": 0.0, "serve_sum": 0.0, "serve_sq": 0.0})
event_condition = defaultdict(lambda: {"n": 0, "ace_sum": 0.0, "serve_sum": 0.0})


def condition_live_speed(surface, event_key):
    event = event_condition[event_key]
    base = surface_condition[surface]
    if event["n"] < 3 or base["n"] < 100:
        return None, int(event["n"])
    n = float(base["n"])
    ace_mean = base["ace_sum"] / n
    serve_mean = base["serve_sum"] / n
    ace_var = max(base["ace_sq"] / n - ace_mean ** 2, 0.00002)
    serve_var = max(base["serve_sq"] / n - serve_mean ** 2, 0.00005)
    event_ace = event["ace_sum"] / event["n"]
    event_serve = event["serve_sum"] / event["n"]
    z_ace = float(np.clip((event_ace - ace_mean) / math.sqrt(ace_var), -3, 3))
    z_serve = float(np.clip((event_serve - serve_mean) / math.sqrt(serve_var), -3, 3))
    speed = float(np.clip(math.exp(0.18 * (0.68 * z_ace + 0.32 * z_serve)), 0.55, 1.55))
    return speed, int(event["n"])


def player_state(pid, name, surface, date):
    h = [m for m in history[pid] if m["date"] < date]
    last1, last3, last5, last10 = h[-1:], h[-3:], h[-5:], h[-10:]
    surface10 = [m for m in h if m["surface"] == surface][-10:]
    overall = serve_ewma[pid]
    surf = surface_ewma[(pid, surface)]
    chart = charting_profile(chart_events, name, date)

    winner_rate = avg_metric(last10, "winner_rate")
    ue_rate = avg_metric(last10, "ue_rate")
    advanced_count = sum(
        (m.get("winner_rate") is not None and np.isfinite(m.get("winner_rate")))
        or (m.get("ue_rate") is not None and np.isfinite(m.get("ue_rate")))
        for m in last10
    )
    if not np.isfinite(winner_rate) and np.isfinite(chart["chart_winner_rate"]):
        winner_rate = chart["chart_winner_rate"]
    if not np.isfinite(ue_rate) and np.isfinite(chart["chart_ue_rate"]):
        ue_rate = chart["chart_ue_rate"]
    aggression_quality = (
        winner_rate - ue_rate if np.isfinite(winner_rate) and np.isfinite(ue_rate) else np.nan
    )

    def window(key, records, default):
        value = avg_metric(records, key, default)
        return default if value is None or not np.isfinite(value) else float(value)

    state = {
        "overall_elo": float(elo[pid]),
        "surface_elo": float(surface_elo[(pid, surface)]),
        "serve": float(surf["serve"] if surf["serve"] is not None else (overall["serve"] if overall["serve"] is not None else 0.635)),
        "return_rating": float(surf["return"] if surf["return"] is not None else (overall["return"] if overall["return"] is not None else 0.365)),
        "win3": window("win", last3, 0.5),
        "win5": window("win", last5, 0.5),
        "win10": window("win", last10, 0.5),
        "surface_win10": window("win", surface10, 0.5),
        "spw1": window("service_point_win", last1, 0.635),
        "spw3": window("service_point_win", last3, 0.635),
        "spw5": window("service_point_win", last5, 0.635),
        "spw10": window("service_point_win", last10, 0.635),
        "rpw1": window("return_point_win", last1, 0.365),
        "rpw3": window("return_point_win", last3, 0.365),
        "rpw5": window("return_point_win", last5, 0.365),
        "rpw10": window("return_point_win", last10, 0.365),
        "first_in5": window("first_in", last5, 0.62),
        "first_won5": window("first_won", last5, 0.70),
        "second_won5": window("second_won", last5, 0.50),
        "ace_rate5": window("ace_rate", last5, 0.08),
        "df_rate5": window("df_rate", last5, 0.035),
        "point_share5": window("point_share", last5, 0.50),
        "point_share10": window("point_share", last10, 0.50),
        "bp_save5": window("bp_save", last5, 0.62),
        "bp_convert5": window("bp_convert", last5, 0.38),
        "form_ewma": weighted_recent(h, "win", 0.5),
        "surface_form_ewma": weighted_recent(surface10, "win", 0.5),
        "opp_elo10": window("opp_elo", last10, 1500.0),
        "recent_perf10": window("perf", last10, 0.0),
        "matches7": float(sum((date - m["date"]).days <= 7 for m in h)),
        "matches14": float(sum((date - m["date"]).days <= 14 for m in h)),
        "rest_days": float(min(max((date - last_seen[pid]).days if pid in last_seen else 30, 0), 60)),
        "elo_change10": float(elo[pid] - elo_trail[pid][0] if elo_trail[pid] else 0.0),
        "winner_rate": float(winner_rate) if np.isfinite(winner_rate) else 0.15,
        "ue_rate": float(ue_rate) if np.isfinite(ue_rate) else 0.15,
        "aggression_quality": float(aggression_quality) if np.isfinite(aggression_quality) else 0.0,
        "advanced_coverage": float(advanced_count / max(len(last10), 1)),
        "net_win": window("net_win", last10, 0.65),
        "avg_first_serve_speed": window("avg_first_serve_speed", last10, 0.0),
        "chart_available": chart["chart_available"],
    }
    return state


def h2h_state(pid_a, pid_b, surface):
    pair = tuple(sorted((str(pid_a), str(pid_b))))
    records = list(h2h_history[pair])
    n = len(records)
    a_wins = sum(m["winner"] == str(pid_a) for m in records)
    b_wins = sum(m["winner"] == str(pid_b) for m in records)
    surface_records = [m for m in records if m["surface"] == surface]
    sn = len(surface_records)
    sa = sum(m["winner"] == str(pid_a) for m in surface_records)
    sb = sum(m["winner"] == str(pid_b) for m in surface_records)

    raw = ((a_wins - b_wins) / n) if n else 0.0
    sraw = ((sa - sb) / sn) if sn else 0.0
    overall_edge = raw * n / (n + 4.0) if n else 0.0
    surface_edge = sraw * sn / (sn + 3.0) if sn else 0.0

    def matchup_diff(key, source=records):
        av = [m["stats"].get(str(pid_a), {}).get(key) for m in source]
        bv = [m["stats"].get(str(pid_b), {}).get(key) for m in source]
        av = [x for x in av if x is not None and np.isfinite(x)]
        bv = [x for x in bv if x is not None and np.isfinite(x)]
        if not av or not bv:
            return 0.0
        evidence = min(len(av), len(bv))
        return float((np.mean(av) - np.mean(bv)) * evidence / (evidence + 3.0))

    return {
        "h2h_overall_edge": float(overall_edge),
        "h2h_surface_edge": float(surface_edge),
        "h2h_serve_diff": matchup_diff("service_point_win"),
        "h2h_second_serve_diff": matchup_diff("second_won"),
        "h2h_bp_convert_diff": matchup_diff("bp_convert"),
        "h2h_matches_log": float(math.log1p(n)),
        "h2h_surface_matches_log": float(math.log1p(sn)),
        "matches": int(n),
        "surface_matches": int(sn),
        "a_wins": int(a_wins),
        "b_wins": int(b_wins),
        "surface_a_wins": int(sa),
        "surface_b_wins": int(sb),
    }


def side_match_metrics(row, side, opponent_side):
    svpt = number(row.get(f"{side}_svpt"))
    first_in = number(row.get(f"{side}_1stIn"))
    first_won_n = number(row.get(f"{side}_1stWon"))
    second_won_n = number(row.get(f"{side}_2ndWon"))
    ace = number(row.get(f"{side}_ace"))
    df = number(row.get(f"{side}_df"))
    bp_saved = number(row.get(f"{side}_bpSaved"))
    bp_faced = number(row.get(f"{side}_bpFaced"))
    opp_svpt = number(row.get(f"{opponent_side}_svpt"))
    opp_first_won = number(row.get(f"{opponent_side}_1stWon"))
    opp_second_won = number(row.get(f"{opponent_side}_2ndWon"))
    opp_bp_saved = number(row.get(f"{opponent_side}_bpSaved"))
    opp_bp_faced = number(row.get(f"{opponent_side}_bpFaced"))

    service_points_won = (
        first_won_n + second_won_n
        if first_won_n is not None and second_won_n is not None
        else None
    )
    opp_service_won = (
        opp_first_won + opp_second_won
        if opp_first_won is not None and opp_second_won is not None
        else None
    )
    spw = safe_ratio(service_points_won, svpt)
    rpw = 1.0 - safe_ratio(opp_service_won, opp_svpt) if safe_ratio(opp_service_won, opp_svpt) is not None else None
    second_total = svpt - first_in if svpt is not None and first_in is not None else None
    total_points = svpt + opp_svpt if svpt is not None and opp_svpt is not None else None
    derived_points_won = (
        service_points_won + (opp_svpt - opp_service_won)
        if service_points_won is not None and opp_svpt is not None and opp_service_won is not None
        else None
    )
    explicit_points = number(row.get(f"{side}_total_points_won"))
    points_won = explicit_points if explicit_points is not None else derived_points_won

    winners = number(row.get(f"{side}_winners"))
    ue = number(row.get(f"{side}_unforced_errors"))
    net_won = number(row.get(f"{side}_net_won"))
    net_total = number(row.get(f"{side}_net_total"))
    break_won = (
        opp_bp_faced - opp_bp_saved
        if opp_bp_faced is not None and opp_bp_saved is not None
        else None
    )
    return {
        "service_point_win": spw,
        "return_point_win": rpw,
        "first_in": safe_ratio(first_in, svpt),
        "first_won": safe_ratio(first_won_n, first_in),
        "second_won": safe_ratio(second_won_n, second_total),
        "ace_rate": safe_ratio(ace, svpt),
        "df_rate": safe_ratio(df, svpt),
        "bp_save": safe_ratio(bp_saved, bp_faced),
        "bp_convert": safe_ratio(break_won, opp_bp_faced),
        "point_share": safe_ratio(points_won, total_points),
        "winner_rate": safe_ratio(winners, total_points),
        "ue_rate": safe_ratio(ue, total_points),
        "net_win": safe_ratio(net_won, net_total),
        "avg_first_serve_speed": number(row.get(f"{side}_avg_first_serve_speed")),
    }


def feature_values(a, b, rank_a, rank_b, age_a, age_b, level, best_of, speed, speed_meta, indoor, h2h):
    def diff(key):
        return float(a[key]) - float(b[key])

    surface_elo_diff = diff("surface_elo")
    serve_diff = diff("serve")
    return_diff = diff("return_rating")
    rank_advantage = math.log(max(float(rank_b), 1.0)) - math.log(max(float(rank_a), 1.0))
    level_centered = float(level) - 3.0
    speed_centered = float(speed) - 1.0
    bestof_centered = float(best_of) - 3.0

    values = {
        "overall_elo_diff": diff("overall_elo"),
        "surface_elo_diff": surface_elo_diff,
        "serve_diff": serve_diff,
        "return_diff": return_diff,
        "log_rank_advantage": rank_advantage,
        "win3_diff": diff("win3"), "win5_diff": diff("win5"), "win10_diff": diff("win10"),
        "surface_win10_diff": diff("surface_win10"),
        "spw1_diff": diff("spw1"), "spw3_diff": diff("spw3"), "spw5_diff": diff("spw5"), "spw10_diff": diff("spw10"),
        "rpw1_diff": diff("rpw1"), "rpw3_diff": diff("rpw3"), "rpw5_diff": diff("rpw5"), "rpw10_diff": diff("rpw10"),
        "first_in5_diff": diff("first_in5"), "first_won5_diff": diff("first_won5"), "second_won5_diff": diff("second_won5"),
        "ace_rate5_diff": diff("ace_rate5"), "df_rate5_diff": diff("df_rate5"),
        "point_share5_diff": diff("point_share5"), "point_share10_diff": diff("point_share10"),
        "bp_save5_diff": diff("bp_save5"), "bp_convert5_diff": diff("bp_convert5"),
        "form_ewma_diff": diff("form_ewma"), "surface_form_ewma_diff": diff("surface_form_ewma"),
        "opp_elo10_diff": diff("opp_elo10"), "recent_perf10_diff": diff("recent_perf10"),
        "matches7_diff": diff("matches7"), "matches14_diff": diff("matches14"),
        "rest_days_diff": diff("rest_days"), "elo_change10_diff": diff("elo_change10"),
        "age_diff": (float(age_a) - float(age_b)) if age_a is not None and age_b is not None else 0.0,
        "winner_rate_diff": diff("winner_rate"), "ue_rate_diff": diff("ue_rate"),
        "aggression_quality_diff": diff("aggression_quality"), "advanced_coverage_diff": diff("advanced_coverage"),
        "net_win_diff": diff("net_win"), "avg_first_serve_speed_diff": diff("avg_first_serve_speed"),
        "h2h_overall_edge": h2h["h2h_overall_edge"], "h2h_surface_edge": h2h["h2h_surface_edge"],
        "h2h_serve_diff": h2h["h2h_serve_diff"], "h2h_second_serve_diff": h2h["h2h_second_serve_diff"],
        "h2h_bp_convert_diff": h2h["h2h_bp_convert_diff"],
        "level_surface_elo_interaction": level_centered * surface_elo_diff / 400.0,
        "level_rank_interaction": level_centered * rank_advantage,
        "level_serve_interaction": level_centered * serve_diff * 10.0,
        "level_form_interaction": level_centered * diff("form_ewma"),
        "speed_surface_elo_interaction": speed_centered * surface_elo_diff / 100.0,
        "speed_serve_interaction": speed_centered * serve_diff * 10.0,
        "speed_return_interaction": speed_centered * return_diff * 10.0,
        "speed_ace_interaction": speed_centered * diff("ace_rate5") * 10.0,
        "speed_second_serve_interaction": speed_centered * diff("second_won5") * 10.0,
        "speed_point_share_interaction": speed_centered * diff("point_share5") * 10.0,
        "indoor_serve_interaction": float(indoor) * serve_diff * 10.0,
        "indoor_return_interaction": float(indoor) * return_diff * 10.0,
        "bestof_surface_elo_interaction": bestof_centered * surface_elo_diff / 400.0,
        "tournament_level": float(level), "best_of": float(best_of), "court_speed": float(speed),
        "court_speed_prior": float(speed_meta["prior"]), "court_speed_live_weight": float(speed_meta["live_weight"]),
        "court_speed_missing": float(speed_meta["prior_missing"]), "indoor": float(indoor),
        "h2h_matches_log": h2h["h2h_matches_log"], "h2h_surface_matches_log": h2h["h2h_surface_matches_log"],
    }
    return [float(values[name]) for name in FEATURES]


# ---------------------------------------------------------------------------
# Chronological feature construction
# ---------------------------------------------------------------------------
for _, r in matches.iterrows():
    date = r["date"]
    surface = str(r.get("surface", "")).title()
    if surface not in SURFACES:
        continue
    winner = str(r["winner_id"]).replace(".0", "")
    loser = str(r["loser_id"]).replace(".0", "")
    winner_name = str(r.get("winner_name", "")).strip()
    loser_name = str(r.get("loser_name", "")).strip()
    if not winner_name or not loser_name:
        continue
    names[winner], names[loser] = winner_name, loser_name

    score = str(r.get("score", "")).upper()
    completed = not any(marker in score for marker in RETIREMENT_MARKERS)
    if not completed:
        # Do not let retirements/walkovers distort ratings, rolling form, or the
        # training target. They remain in raw data for auditability.
        continue

    winner_state = player_state(winner, winner_name, surface, date)
    loser_state = player_state(loser, loser_name, surface, date)
    wrank = number(r.get("winner_rank")) or historical_ranks.get(winner, 500.0) or 500.0
    lrank = number(r.get("loser_rank")) or historical_ranks.get(loser, 500.0) or 500.0
    wage = number(r.get("winner_age"))
    lage = number(r.get("loser_age"))
    level = encode_tournament_level(r.get("tourney_level", ""))
    best_of = number(r.get("best_of")) or (5.0 if level >= 5.0 else 3.0)
    indoor_raw = str(r.get("indoor", "")).strip().lower()
    indoor = 1.0 if indoor_raw in {"1", "1.0", "true", "yes", "y", "indoor"} else 0.0

    prior, prior_missing = cached_prior_speed(
        r.get("tourney_name", ""), surface, date.year
    )
    event_key = (date.year, canonical_tournament(str(r.get("tourney_name", ""))), surface)
    live_speed, live_n = condition_live_speed(surface, event_key)
    court_speed, live_weight = shrink_live_speed(prior, live_speed, live_n, k=10.0)
    speed_meta = {
        "prior": prior,
        "prior_missing": prior_missing,
        "live": live_speed,
        "live_matches": live_n,
        "live_weight": live_weight,
    }

    h2h = h2h_state(winner, loser, surface)
    features = feature_values(
        winner_state, loser_state, wrank, lrank, wage, lage, level, best_of,
        court_speed, speed_meta, indoor, h2h,
    )
    flip = rng.random() < 0.5
    diff = features[: len(DIFF_FEATURES)]
    context = features[len(DIFF_FEATURES) :]
    rows.append(
        {
            "date": date,
            "year": date.year,
            "y": 0 if flip else 1,
            "x": ([-x for x in diff] + context) if flip else features,
        }
    )

    # Metrics from the match are added only after the pre-match row is captured.
    wmetrics = side_match_metrics(r, "w", "l")
    lmetrics = side_match_metrics(r, "l", "w")

    pre_winner, pre_loser = float(elo[winner]), float(elo[loser])
    expected = 1 / (1 + 10 ** ((pre_loser - pre_winner) / 400))
    delta = K * (1 - expected)
    elo[winner] += delta
    elo[loser] -= delta

    pre_ws, pre_ls = float(surface_elo[(winner, surface)]), float(surface_elo[(loser, surface)])
    expected_surface = 1 / (1 + 10 ** ((pre_ls - pre_ws) / 400))
    surface_delta = K * (1 - expected_surface)
    surface_elo[(winner, surface)] += surface_delta
    surface_elo[(loser, surface)] -= surface_delta

    for pid, metrics in ((winner, wmetrics), (loser, lmetrics)):
        serve_ewma[pid]["serve"] = ewm(serve_ewma[pid]["serve"], metrics["service_point_win"])
        serve_ewma[pid]["return"] = ewm(serve_ewma[pid]["return"], metrics["return_point_win"])
        surface_ewma[(pid, surface)]["serve"] = ewm(surface_ewma[(pid, surface)]["serve"], metrics["service_point_win"])
        surface_ewma[(pid, surface)]["return"] = ewm(surface_ewma[(pid, surface)]["return"], metrics["return_point_win"])

    w_perf = 1.0 - expected
    l_perf = -w_perf
    history[winner].append({"date": date, "surface": surface, "win": 1.0, "opp_elo": pre_loser, "perf": w_perf, **wmetrics})
    history[loser].append({"date": date, "surface": surface, "win": 0.0, "opp_elo": pre_winner, "perf": l_perf, **lmetrics})
    elo_trail[winner].append(pre_winner)
    elo_trail[loser].append(pre_loser)
    last_seen[winner] = date
    last_seen[loser] = date
    historical_ranks[winner] = wrank
    historical_ranks[loser] = lrank
    ages[winner] = wage
    ages[loser] = lage

    pair = tuple(sorted((winner, loser)))
    h2h_history[pair].append({
        "winner": winner,
        "surface": surface,
        "stats": {winner: wmetrics, loser: lmetrics},
    })

    # Update court-condition state after the match.
    total_svpt = number(r.get("w_svpt"))
    lsv = number(r.get("l_svpt"))
    wace = number(r.get("w_ace"))
    lace = number(r.get("l_ace"))
    if total_svpt is not None and lsv is not None and wace is not None and lace is not None:
        total = total_svpt + lsv
        w_spw_count = (number(r.get("w_1stWon")) or 0) + (number(r.get("w_2ndWon")) or 0)
        l_spw_count = (number(r.get("l_1stWon")) or 0) + (number(r.get("l_2ndWon")) or 0)
        if total > 0:
            ace_rate = (wace + lace) / total
            service_rate = (w_spw_count + l_spw_count) / total
            base = surface_condition[surface]
            base["n"] += 1
            base["ace_sum"] += ace_rate
            base["ace_sq"] += ace_rate ** 2
            base["serve_sum"] += service_rate
            base["serve_sq"] += service_rate ** 2
            event = event_condition[event_key]
            event["n"] += 1
            event["ace_sum"] += ace_rate
            event["serve_sum"] += service_rate


if len(rows) < 1000:
    raise RuntimeError(f"Only {len(rows)} completed matches were available.")

# ---------------------------------------------------------------------------
# Model selection on chronological holdout
# ---------------------------------------------------------------------------
latest_year = max(row["year"] for row in rows)
holdout_year = latest_year - 1 if latest_year >= 2025 else latest_year
train = [row for row in rows if row["year"] < holdout_year]
test = [row for row in rows if row["year"] == holdout_year]
if len(train) < 500 or len(test) < 50:
    split = int(len(rows) * 0.85)
    train, test = rows[:split], rows[split:]
    holdout_label = "latest chronological 15%"
else:
    holdout_label = str(holdout_year)

x_train = np.array([r["x"] for r in train], dtype=float)
y_train = np.array([r["y"] for r in train], dtype=int)
x_test = np.array([r["x"] for r in test], dtype=float)
y_test = np.array([r["y"] for r in test], dtype=int)

candidates = [
    (
        "logistic_regression",
        Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.55, max_iter=5000)),
        ]),
    ),
    (
        "hist_gradient_boosting",
        Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=45,
                l2_regularization=3.0,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=25,
                random_state=123,
            )),
        ]),
    ),
]


def evaluate_candidate(name, estimator):
    estimator.fit(x_train, y_train)
    probability = estimator.predict_proba(x_test)[:, 1]
    return {
        "name": name,
        "pipeline": estimator,
        "probability": probability,
        "accuracy": float(accuracy_score(y_test, probability >= 0.5)),
        "log_loss": float(log_loss(y_test, probability)),
        "brier": float(brier_score_loss(y_test, probability)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }


evaluations = [evaluate_candidate(name, estimator) for name, estimator in candidates]
winner_eval = min(evaluations, key=lambda item: (item["log_loss"], item["brier"]))
selected_name = winner_eval["name"]

# Calibration diagnostic: expected calibration error on the untouched holdout.
def calibration_error(y_true, prob, bins=10):
    frame = pd.DataFrame({"y": y_true, "p": prob})
    frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    total = len(frame)
    err = 0.0
    for _, group in frame.groupby("bin", observed=True):
        if len(group):
            err += len(group) / total * abs(group["p"].mean() - group["y"].mean())
    return float(err)

candidate_metrics = {
    item["name"]: {
        "accuracy": item["accuracy"], "log_loss": item["log_loss"],
        "brier": item["brier"], "roc_auc": item["roc_auc"],
        "ece_10bin": calibration_error(y_test, item["probability"]),
    }
    for item in evaluations
}

metrics = {
    "training_matches": len(train),
    "test_matches": len(test),
    "holdout": holdout_label,
    "selected_model": selected_name,
    "selection_metric": "lowest chronological holdout log loss, then Brier score",
    "candidate_metrics": candidate_metrics,
    "accuracy": winner_eval["accuracy"],
    "log_loss": winner_eval["log_loss"],
    "brier": winner_eval["brier"],
    "roc_auc": winner_eval["roc_auc"],
    "ece_10bin": calibration_error(y_test, winner_eval["probability"]),
    "latest_data_date": matches["date"].max().strftime("%Y-%m-%d"),
    "features": FEATURES,
    "feature_groups": {
        "recent_form": [x for x in FEATURES if any(k in x for k in ("spw", "rpw", "win", "form", "point_share", "ace_rate", "df_rate"))],
        "h2h": [x for x in FEATURES if x.startswith("h2h_")],
        "court_speed": [x for x in FEATURES if "speed" in x or "court_speed" in x],
    },
    "charting": chart_meta,
    "court_speed_rows": int(len(empirical_speeds)),
    "court_speed_source": prior_speed_source,
    "pipeline_version": "4.0-matchstat-recent-form-h2h-speed",
}

# Refit selected family on all completed rows after honest model selection.
selected_template = dict(candidates)[selected_name]
selected_template.fit(
    np.array([r["x"] for r in rows], dtype=float),
    np.array([r["y"] for r in rows], dtype=int),
)
joblib.dump(
    {
        "pipeline": selected_template,
        "features": FEATURES,
        "diff_features": DIFF_FEATURES,
        "context_features": CONTEXT_FEATURES,
        "metrics": metrics,
        "model_name": selected_name,
    },
    MODEL / "model.joblib",
)

# ---------------------------------------------------------------------------
# Current player state
# ---------------------------------------------------------------------------
latest = matches["date"].max()
state_rows = []
for pid, player_name in names.items():
    if pid not in last_seen or (latest - last_seen[pid]).days > 730:
        continue
    for surface in ["Hard", "Clay", "Grass"]:
        state = player_state(pid, player_name, surface, latest + timedelta(days=1))
        rank = current_rankings.get(
            pid,
            current_rankings_by_name.get(normalize_name(player_name), historical_ranks.get(pid, 500)),
        )
        state_rows.append({
            "player_id": pid,
            "player": player_name,
            "surface": surface,
            **state,
            "rank": int(number(rank) or 500),
            "age": float(ages.get(pid) or 0),
            "last_match": last_seen[pid].strftime("%Y-%m-%d"),
        })

state_df = pd.DataFrame(state_rows).sort_values(["player", "surface"])
state_df.to_csv(GENERATED / "player_state.csv.gz", index=False, compression="gzip")

# ---------------------------------------------------------------------------
# H2H snapshot used at prediction time. It is descriptive input to the trained
# model, never an arbitrary post-model probability bump.
# ---------------------------------------------------------------------------
h2h_rows = []
for pair, records_deque in h2h_history.items():
    records = list(records_deque)
    if not records:
        continue
    p1, p2 = pair
    for surface in ["All", "Hard", "Clay", "Grass"]:
        subset = records if surface == "All" else [m for m in records if m["surface"] == surface]
        if not subset:
            continue
        p1_wins = sum(m["winner"] == p1 for m in subset)
        p2_wins = sum(m["winner"] == p2 for m in subset)

        def mean_for(pid, key):
            vals = [m["stats"].get(pid, {}).get(key) for m in subset]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            return float(np.mean(vals)) if vals else np.nan

        h2h_rows.append({
            "player_1": names.get(p1, p1), "player_2": names.get(p2, p2), "surface": surface,
            "player_1_wins": int(p1_wins), "player_2_wins": int(p2_wins), "matches": int(len(subset)),
            "player_1_serve": mean_for(p1, "service_point_win"), "player_2_serve": mean_for(p2, "service_point_win"),
            "player_1_second_serve": mean_for(p1, "second_won"), "player_2_second_serve": mean_for(p2, "second_won"),
            "player_1_bp_convert": mean_for(p1, "bp_convert"), "player_2_bp_convert": mean_for(p2, "bp_convert"),
        })

pd.DataFrame(h2h_rows).to_csv(GENERATED / "head_to_head.csv.gz", index=False, compression="gzip")

# Empirical table includes current completed matches and is used as the live
# tournament estimate at prediction time, shrunk toward the historical prior.
empirical_speeds.to_csv(empirical_path, index=False)

(GENERATED / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

history_path = GENERATED / "model_history.csv"
history_row = pd.DataFrame([{
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "latest_data_date": metrics["latest_data_date"],
    "holdout": metrics["holdout"],
    "training_matches": metrics["training_matches"],
    "test_matches": metrics["test_matches"],
    "accuracy": metrics["accuracy"],
    "log_loss": metrics["log_loss"],
    "brier": metrics["brier"],
    "roc_auc": metrics["roc_auc"],
    "ece_10bin": metrics["ece_10bin"],
    "selected_model": metrics["selected_model"],
    "pipeline_version": metrics["pipeline_version"],
}])
if history_path.exists():
    try:
        previous = pd.read_csv(history_path)
        history_row = pd.concat([previous, history_row], ignore_index=True).tail(365)
    except Exception:
        pass
history_row.to_csv(history_path, index=False)

print(json.dumps(metrics, indent=2))

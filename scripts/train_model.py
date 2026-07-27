
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
import math
import random
import json
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "data" / "generated"
RAW = ROOT / "data" / "raw"
MODEL = ROOT / "model"
MODEL.mkdir(parents=True, exist_ok=True)

INITIAL_ELO = 1500.0
K = 28.0
ALPHA = 0.12
SURFACES = {"Hard", "Clay", "Grass"}
TOUR_LEVEL = {"G": 4.0, "M": 3.0, "A": 2.0, "D": 1.5, "F": 1.5, "C": 1.0}


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


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
    overview_path = RAW / "charting-m-stats-Overview.csv"
    matches_path = RAW / "charting-m-matches.csv"
    if not overview_path.exists():
        return defaultdict(list), {"available": False, "reason": "overview file missing"}

    overview = pd.read_csv(overview_path, low_memory=False)
    columns = list(overview.columns)
    match_col = find_column(columns, ["match_id", "matchid"])
    player_col = find_column(columns, ["player", "player_name"])
    if match_col is None or player_col is None:
        return defaultdict(list), {
            "available": False,
            "reason": "could not identify match_id/player columns",
            "columns": columns,
        }

    # Optional metadata supplies authoritative date when available.
    date_by_match = {}
    if matches_path.exists():
        metadata = pd.read_csv(matches_path, low_memory=False)
        mmatch = find_column(metadata.columns, ["match_id", "matchid"])
        mdate = find_column(metadata.columns, ["date", "match_date"])
        if mmatch and mdate:
            for _, row in metadata[[mmatch, mdate]].dropna().iterrows():
                raw = str(row[mdate])
                parsed = pd.to_datetime(raw, errors="coerce")
                if pd.notna(parsed):
                    date_by_match[str(row[mmatch])] = parsed.to_pydatetime()

    aliases = {
        "serve_points_won": ["serve_pts_won", "servepointswon", "svptswon", "serve_won"],
        "serve_points": ["serve_pts", "servepoints", "svpts", "serve_total"],
        "return_points_won": ["return_pts_won", "returnpointswon", "return_won"],
        "return_points": ["return_pts", "returnpoints", "return_total"],
        "winners": ["winners", "winner"],
        "unforced_errors": ["unforced", "unforced_errors", "ues", "ue"],
        "net_points_won": ["net_pts_won", "netpointswon", "net_won"],
        "net_points": ["net_pts", "netpoints", "net_total"],
    }
    mapped = {k: find_column(columns, v) for k, v in aliases.items()}

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

        def val(key):
            col = mapped.get(key)
            return number(row[col]) if col else None

        spw, sp = val("serve_points_won"), val("serve_points")
        rpw, rp = val("return_points_won"), val("return_points")
        winners, ue = val("winners"), val("unforced_errors")
        npw, np_total = val("net_points_won"), val("net_points")

        event = {
            "date": date,
            "serve_point_win": spw / sp if spw is not None and sp and sp > 0 else None,
            "return_point_win": rpw / rp if rpw is not None and rp and rp > 0 else None,
            "winner_rate": winners / max((sp or 0) + (rp or 0), 1) if winners is not None else None,
            "ue_rate": ue / max((sp or 0) + (rp or 0), 1) if ue is not None else None,
            "net_win": npw / np_total if npw is not None and np_total and np_total > 0 else None,
        }
        events[player].append(event)

    for player in events:
        events[player].sort(key=lambda x: x["date"])
    return events, {
        "available": True,
        "players": len(events),
        "rows": sum(len(v) for v in events.values()),
        "mapped_columns": mapped,
    }


def charting_profile(events, player_name, date):
    player_events = [x for x in events.get(normalize_name(player_name), []) if x["date"] < date]
    recent = player_events[-30:]
    defaults = {
        "chart_serve": 0.64,
        "chart_return": 0.36,
        "chart_winner_rate": 0.15,
        "chart_ue_rate": 0.15,
        "chart_net_win": 0.65,
        "charted_matches": 0.0,
        "chart_available": 0.0,
    }
    if not recent:
        return defaults

    def avg(key, default):
        vals = [x[key] for x in recent if x.get(key) is not None and np.isfinite(x[key])]
        return float(np.mean(vals)) if vals else default

    return {
        "chart_serve": avg("serve_point_win", defaults["chart_serve"]),
        "chart_return": avg("return_point_win", defaults["chart_return"]),
        "chart_winner_rate": avg("winner_rate", defaults["chart_winner_rate"]),
        "chart_ue_rate": avg("ue_rate", defaults["chart_ue_rate"]),
        "chart_net_win": avg("net_win", defaults["chart_net_win"]),
        "charted_matches": float(len(recent)),
        "chart_available": 1.0,
    }


master_path = GENERATED / "master_matches.csv.gz"
if not master_path.exists():
    raise RuntimeError("Run scripts/update_data.py first; master match data is missing.")

matches = pd.read_csv(master_path, low_memory=False)
matches["tourney_date"] = pd.to_numeric(matches["tourney_date"], errors="coerce")
matches = matches.dropna(subset=["tourney_date", "winner_id", "loser_id", "surface"])
matches["date"] = pd.to_datetime(matches["tourney_date"].astype(int).astype(str), format="%Y%m%d")
matches["match_num"] = pd.to_numeric(matches.get("match_num", 0), errors="coerce").fillna(0)
matches = matches.sort_values(["date", "match_num"])

chart_events, chart_meta = load_charting_events()

current_rankings = {}
rankings_path = GENERATED / "current_rankings.csv"
if rankings_path.exists():
    rankings = pd.read_csv(rankings_path, dtype={"player_id": str})
    rankings["player_id"] = rankings["player_id"].str.replace(r"\.0$", "", regex=True)
    current_rankings = dict(zip(rankings["player_id"], rankings["ranking"]))

elo = defaultdict(lambda: INITIAL_ELO)
surface_elo = defaultdict(lambda: INITIAL_ELO)
serve_stats = defaultdict(lambda: {"serve": None, "return": None})
surface_stats = defaultdict(lambda: {"serve": None, "return": None})
history = defaultdict(lambda: deque(maxlen=60))
elo_trail = defaultdict(lambda: deque(maxlen=10))
last_seen = {}
names = {}
historical_ranks = {}
ages = {}
rows = []
rng = random.Random(123)


def ewm(old, new):
    return new if old is None else ALPHA * new + (1 - ALPHA) * old


def player_state(pid, name, surface, date):
    overall_stats = serve_stats[pid]
    surf_stats = surface_stats[(pid, surface)]
    h = [m for m in history[pid] if m["date"] < date]
    last5, last10 = h[-5:], h[-10:]
    surface10 = [m for m in h if m["surface"] == surface][-10:]
    chart = charting_profile(chart_events, name, date)
    overall = elo[pid]
    return {
        "overall_elo": overall,
        "surface_elo": surface_elo[(pid, surface)],
        "serve": surf_stats["serve"] if surf_stats["serve"] is not None else (
            overall_stats["serve"] if overall_stats["serve"] is not None else 0.635
        ),
        "return_rating": surf_stats["return"] if surf_stats["return"] is not None else (
            overall_stats["return"] if overall_stats["return"] is not None else 0.365
        ),
        "win5": sum(m["win"] for m in last5) / len(last5) if last5 else 0.5,
        "win10": sum(m["win"] for m in last10) / len(last10) if last10 else 0.5,
        "surface_win10": (
            sum(m["win"] for m in surface10) / len(surface10) if surface10 else 0.5
        ),
        "opp_elo10": float(np.mean([m["opp"] for m in last10])) if last10 else 1500.0,
        "recent_perf10": float(np.mean([m["perf"] for m in last10])) if last10 else 0.0,
        "matches7": sum((date - m["date"]).days <= 7 for m in h),
        "matches14": sum((date - m["date"]).days <= 14 for m in h),
        "rest_days": min(max((date - last_seen[pid]).days if pid in last_seen else 30, 0), 60),
        "elo_change10": overall - elo_trail[pid][0] if elo_trail[pid] else 0.0,
        **chart,
    }


def feature_difference(a, b, rank_a, rank_b, age_a, age_b, level, best_of):
    return [
        a["overall_elo"] - b["overall_elo"],
        a["surface_elo"] - b["surface_elo"],
        a["serve"] - b["serve"],
        a["return_rating"] - b["return_rating"],
        math.log(max(rank_b, 1)) - math.log(max(rank_a, 1)),
        a["win5"] - b["win5"],
        a["win10"] - b["win10"],
        a["surface_win10"] - b["surface_win10"],
        a["opp_elo10"] - b["opp_elo10"],
        a["recent_perf10"] - b["recent_perf10"],
        a["matches7"] - b["matches7"],
        a["matches14"] - b["matches14"],
        a["rest_days"] - b["rest_days"],
        a["elo_change10"] - b["elo_change10"],
        (age_a - age_b) if age_a is not None and age_b is not None else 0.0,
        a["chart_serve"] - b["chart_serve"],
        a["chart_return"] - b["chart_return"],
        a["chart_winner_rate"] - b["chart_winner_rate"],
        a["chart_ue_rate"] - b["chart_ue_rate"],
        a["chart_net_win"] - b["chart_net_win"],
        a["charted_matches"] - b["charted_matches"],
        a["chart_available"] - b["chart_available"],
        level,
        best_of,
    ]


for _, r in matches.iterrows():
    date = r["date"]
    surface = str(r.get("surface", "")).title()
    if surface not in SURFACES:
        continue
    winner = str(r["winner_id"]).replace(".0", "")
    loser = str(r["loser_id"]).replace(".0", "")
    winner_name = str(r.get("winner_name", ""))
    loser_name = str(r.get("loser_name", ""))
    if not winner_name or not loser_name:
        continue

    names[winner], names[loser] = winner_name, loser_name
    winner_state = player_state(winner, winner_name, surface, date)
    loser_state = player_state(loser, loser_name, surface, date)

    wrank = number(r.get("winner_rank")) or 500.0
    lrank = number(r.get("loser_rank")) or 500.0
    wage = number(r.get("winner_age"))
    lage = number(r.get("loser_age"))
    level = TOUR_LEVEL.get(str(r.get("tourney_level", "")).upper(), 1.0)
    best_of = number(r.get("best_of")) or 3.0

    score = str(r.get("score", "")).upper()
    completed = not any(marker in score for marker in ["RET", "W/O", "DEF", "ABN"])
    features = feature_difference(
        winner_state, loser_state, wrank, lrank, wage, lage, level, best_of
    )
    flip = rng.random() < 0.5
    if completed:
        rows.append(
            {
                "date": date,
                "year": date.year,
                "y": 0 if flip else 1,
                "x": [-x for x in features] if flip else features,
            }
        )

    expected = 1 / (1 + 10 ** ((elo[loser] - elo[winner]) / 400))
    pre_winner, pre_loser = elo[winner], elo[loser]
    delta = K * (1 - expected)
    elo[winner] += delta
    elo[loser] -= delta

    expected_surface = 1 / (
        1 + 10 ** ((surface_elo[(loser, surface)] - surface_elo[(winner, surface)]) / 400)
    )
    surface_delta = K * (1 - expected_surface)
    surface_elo[(winner, surface)] += surface_delta
    surface_elo[(loser, surface)] -= surface_delta

    values = [
        number(r.get(key))
        for key in ["w_svpt", "l_svpt", "w_1stWon", "w_2ndWon", "l_1stWon", "l_2ndWon"]
    ]
    if all(v is not None for v in values) and values[0] > 0 and values[1] > 0:
        wsv, lsv, w1, w2, l1, l2 = values
        winner_sp = (w1 + w2) / wsv
        loser_sp = (l1 + l2) / lsv
        for pid, sp, rp in [
            (winner, winner_sp, 1 - loser_sp),
            (loser, loser_sp, 1 - winner_sp),
        ]:
            serve_stats[pid]["serve"] = ewm(serve_stats[pid]["serve"], sp)
            serve_stats[pid]["return"] = ewm(serve_stats[pid]["return"], rp)
            surface_stats[(pid, surface)]["serve"] = ewm(
                surface_stats[(pid, surface)]["serve"], sp
            )
            surface_stats[(pid, surface)]["return"] = ewm(
                surface_stats[(pid, surface)]["return"], rp
            )

    history[winner].append(
        {"date": date, "surface": surface, "win": 1, "opp": pre_loser, "perf": 1 - expected}
    )
    history[loser].append(
        {"date": date, "surface": surface, "win": 0, "opp": pre_winner, "perf": -(1 - expected)}
    )
    elo_trail[winner].append(pre_winner)
    elo_trail[loser].append(pre_loser)
    last_seen[winner] = date
    last_seen[loser] = date
    historical_ranks[winner] = wrank
    historical_ranks[loser] = lrank
    ages[winner] = wage
    ages[loser] = lage

FEATURES = [
    "overall_elo_diff", "surface_elo_diff", "serve_diff", "return_diff",
    "log_rank_advantage", "win5_diff", "win10_diff", "surface_win10_diff",
    "opp_elo10_diff", "recent_perf10_diff", "matches7_diff", "matches14_diff",
    "rest_days_diff", "elo_change10_diff", "age_diff",
    "chart_serve_diff", "chart_return_diff", "chart_winner_rate_diff",
    "chart_ue_rate_diff", "chart_net_win_diff", "charted_matches_diff",
    "chart_available_diff", "tournament_level", "best_of",
]

if len(rows) < 1000:
    raise RuntimeError(f"Only {len(rows)} completed matches were available.")

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

base = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.7, max_iter=5000)),
    ]
)
base.fit(np.array([r["x"] for r in train]), np.array([r["y"] for r in train]))
pred = base.predict_proba(np.array([r["x"] for r in test]))[:, 1]

metrics = {
    "training_matches": len(train),
    "test_matches": len(test),
    "holdout": holdout_label,
    "accuracy": float(accuracy_score([r["y"] for r in test], pred >= 0.5)),
    "log_loss": float(log_loss([r["y"] for r in test], pred)),
    "brier": float(brier_score_loss([r["y"] for r in test], pred)),
    "latest_data_date": matches["date"].max().strftime("%Y-%m-%d"),
    "features": FEATURES,
    "charting": chart_meta,
}

base.fit(np.array([r["x"] for r in rows]), np.array([r["y"] for r in rows]))
joblib.dump({"pipeline": base, "features": FEATURES, "metrics": metrics}, MODEL / "model.joblib")

latest = matches["date"].max()
state_rows = []
for pid, player_name in names.items():
    if pid not in last_seen or (latest - last_seen[pid]).days > 730:
        continue
    for surface in ["Hard", "Clay", "Grass"]:
        state = player_state(pid, player_name, surface, latest + timedelta(days=1))
        rank = current_rankings.get(pid, historical_ranks.get(pid, 500))
        state_rows.append(
            {
                "player_id": pid,
                "player": player_name,
                "surface": surface,
                **state,
                "rank": int(number(rank) or 500),
                "age": float(ages.get(pid) or 0),
                "last_match": last_seen[pid].strftime("%Y-%m-%d"),
            }
        )

state_df = pd.DataFrame(state_rows).sort_values(["player", "surface"])
state_df.to_csv(GENERATED / "player_state.csv.gz", index=False, compression="gzip")
(GENERATED / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
print(json.dumps(metrics, indent=2))

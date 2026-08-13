from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_tournament(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\b(atp|wta|open|masters|championships?)\b", " ", text, flags=re.I)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    aliases = {
        "roland garros": "french paris",
        "french paris": "french paris",
        "wimbledon london": "wimbledon",
        "australian melbourne": "australian melbourne",
        "us new york": "us new york",
    }
    compact = " ".join(text.split())
    return aliases.get(compact, compact)


def encode_tournament_level(value) -> float:
    """Encode tour level on a genuinely distinct 1-5 scale."""
    raw = str(value or "").strip().upper()
    compact = re.sub(r"[^A-Z0-9]", "", raw)
    mapping = {
        "C": 1.0,
        "CH": 1.0,
        "CHALLENGER": 1.0,
        "125": 1.25,
        "D": 1.5,
        "DAVISCUP": 1.5,
        "A": 2.0,
        "250": 2.0,
        "ATP250": 2.0,
        "500": 3.0,
        "ATP500": 3.0,
        "M": 4.0,
        "1000": 4.0,
        "MASTERS": 4.0,
        "MASTERS1000": 4.0,
        "F": 4.5,
        "FINALS": 4.5,
        "ATPFINALS": 4.5,
        "G": 5.0,
        "GS": 5.0,
        "GRANDSLAM": 5.0,
    }
    if compact in mapping:
        return mapping[compact]
    try:
        numeric = float(raw)
        if numeric <= 125:
            return 1.25
        if numeric <= 250:
            return 2.0
        if numeric <= 500:
            return 3.0
        if numeric <= 1000:
            return 4.0
    except Exception:
        pass
    return 2.0


def _empirical_components(matches: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "tourney_name", "surface", "tourney_date", "w_svpt", "l_svpt",
        "w_ace", "l_ace", "w_1stWon", "w_2ndWon", "l_1stWon", "l_2ndWon",
    }
    if not needed.issubset(matches.columns):
        return pd.DataFrame()
    work = matches.copy()
    work["season"] = pd.to_numeric(work["tourney_date"], errors="coerce") // 10000
    numeric = [
        "w_svpt", "l_svpt", "w_ace", "l_ace", "w_1stWon", "w_2ndWon",
        "l_1stWon", "l_2ndWon",
    ]
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["total_svpt"] = work["w_svpt"] + work["l_svpt"]
    work["aces"] = work["w_ace"] + work["l_ace"]
    work["service_points_won"] = (
        work["w_1stWon"] + work["w_2ndWon"] + work["l_1stWon"] + work["l_2ndWon"]
    )
    work["ace_rate"] = work["aces"] / work["total_svpt"]
    work["service_win_rate"] = work["service_points_won"] / work["total_svpt"]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["season", "tourney_name", "surface", "ace_rate", "service_win_rate"]
    )
    return work[
        (work["total_svpt"] > 0)
        & work["surface"].astype(str).str.title().isin(["Hard", "Clay", "Grass"])
    ]


def build_empirical_surface_speed_table(matches: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Build a tournament-season condition proxy on a meaningful 1.0-neutral scale.

    Unlike the previous implementation, this intentionally does not compress all
    tournaments into roughly 0.9-1.1. Robust within-surface/year z-scores are
    transformed exponentially and clipped only at 0.55/1.55.
    """
    work = _empirical_components(matches)
    cols = [
        "season", "tournament", "tournament_key", "surface", "surface_speed",
        "matches_used", "ace_rate", "service_win_rate",
    ]
    if work.empty:
        out = pd.DataFrame(columns=cols)
        out.to_csv(output_path, index=False)
        return out

    grouped = work.groupby(["season", "tourney_name", "surface"], as_index=False).agg(
        ace_rate=("ace_rate", "mean"),
        service_win_rate=("service_win_rate", "mean"),
        matches_used=("ace_rate", "size"),
    )
    grouped = grouped[grouped["matches_used"] >= 3].copy()

    parts = []
    for _, group in grouped.groupby(["season", "surface"], sort=False):
        group = group.copy()
        for col in ("ace_rate", "service_win_rate"):
            median = float(group[col].median())
            mad = float((group[col] - median).abs().median())
            scale = max(mad * 1.4826, 0.004 if col == "ace_rate" else 0.008)
            group[f"{col}_z"] = ((group[col] - median) / scale).clip(-3.0, 3.0)
        condition_z = 0.68 * group["ace_rate_z"] + 0.32 * group["service_win_rate_z"]
        group["surface_speed"] = np.exp(0.18 * condition_z).clip(0.55, 1.55)
        parts.append(group)

    grouped = pd.concat(parts, ignore_index=True)
    grouped["tournament"] = grouped["tourney_name"].astype(str)
    grouped["tournament_key"] = grouped["tournament"].map(canonical_tournament)
    out = grouped[cols].sort_values(["season", "surface", "tournament"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def load_surface_speeds(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "tournament_key" not in frame.columns and "tournament" in frame.columns:
        frame["tournament_key"] = frame["tournament"].map(canonical_tournament)
    if "season" not in frame.columns:
        if "event_date" in frame.columns:
            frame["season"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.year
        else:
            frame["season"] = np.nan
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["surface_speed"] = pd.to_numeric(frame.get("surface_speed"), errors="coerce")
    if "surface" in frame.columns:
        frame["surface"] = frame["surface"].astype(str).str.title().replace({"I.Hard": "Hard"})
    return frame.dropna(subset=["season", "surface_speed"])


def lookup_surface_speed(
    speeds: pd.DataFrame,
    tournament: str,
    surface: str,
    season: int,
    *,
    prior_only: bool = True,
) -> tuple[float, float]:
    """Return (speed, missing flag).

    Training calls use ``prior_only=True`` so a tournament's current/future
    matches cannot inform its own historical feature. Prediction can combine
    this prior with a current-tournament empirical estimate separately.
    """
    if speeds.empty:
        return 1.0, 1.0
    cutoff = speeds["season"] < int(season) if prior_only else speeds["season"] <= int(season)
    eligible = speeds[cutoff].copy()
    if eligible.empty:
        return 1.0, 1.0
    key = canonical_tournament(tournament)
    surf = str(surface).title()
    exact = eligible[
        (eligible["tournament_key"] == key)
        & (eligible["surface"].astype(str).str.title() == surf)
    ]
    if not exact.empty:
        row = exact.sort_values("season").iloc[-1]
        return float(row["surface_speed"]), 0.0
    fallback = eligible[eligible["surface"].astype(str).str.title() == surf]
    if not fallback.empty:
        # Use a neutral surface median, not a random tournament from a different venue.
        latest_year = fallback["season"].max()
        return float(fallback[fallback["season"] == latest_year]["surface_speed"].median()), 1.0
    return 1.0, 1.0


def shrink_live_speed(prior_speed: float, live_speed: float | None, matches_used: float, k: float = 10.0) -> tuple[float, float]:
    if live_speed is None or not np.isfinite(live_speed) or matches_used <= 0:
        return float(prior_speed), 0.0
    weight = float(matches_used) / (float(matches_used) + float(k))
    combined = (1.0 - weight) * float(prior_speed) + weight * float(live_speed)
    return float(np.clip(combined, 0.55, 1.55)), weight


def prediction_surface_speed(
    prior_speeds: pd.DataFrame,
    live_speeds: pd.DataFrame,
    tournament: str,
    surface: str,
    season: int,
) -> tuple[float, dict]:
    prior, prior_missing = lookup_surface_speed(
        prior_speeds, tournament, surface, season, prior_only=True
    )
    live_value = None
    live_matches = 0.0
    if not live_speeds.empty:
        key = canonical_tournament(tournament)
        exact = live_speeds[
            (pd.to_numeric(live_speeds["season"], errors="coerce") == int(season))
            & (live_speeds["tournament_key"].astype(str) == key)
            & (live_speeds["surface"].astype(str).str.title() == str(surface).title())
        ]
        if not exact.empty:
            row = exact.sort_values("matches_used").iloc[-1]
            live_value = float(row["surface_speed"])
            live_matches = float(row.get("matches_used", 0) or 0)
    combined, live_weight = shrink_live_speed(prior, live_value, live_matches)
    return combined, {
        "prior_speed": float(prior),
        "prior_missing": bool(prior_missing),
        "live_speed": live_value,
        "live_matches": int(live_matches),
        "live_weight": float(live_weight),
    }

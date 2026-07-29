from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


def canonical_tournament(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\b(atp|wta)\b", " ", text, flags=re.I)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return " ".join(text.split())


def encode_tournament_level(value) -> float:
    """Encode TennisMyLife/Jeff-Sackmann level labels on a distinct 1-5 scale."""
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


def build_surface_speed_table(matches: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Estimate tournament court speed from serve outcomes.

    A value of 1.00 is neutral for that surface. Values above 1 are faster.
    The table is tournament-season specific and prediction code only uses prior
    seasons, preventing a future match from informing its own feature.
    """
    needed = {
        "tourney_name", "surface", "tourney_date", "w_svpt", "l_svpt",
        "w_ace", "l_ace", "w_1stWon", "w_2ndWon", "l_1stWon", "l_2ndWon",
    }
    if not needed.issubset(matches.columns):
        frame = pd.DataFrame(columns=[
            "season", "tournament", "tournament_key", "surface",
            "surface_speed", "matches_used",
        ])
        frame.to_csv(output_path, index=False)
        return frame

    work = matches.copy()
    work["season"] = pd.to_numeric(work["tourney_date"], errors="coerce") // 10000
    numeric = [
        "w_svpt", "l_svpt", "w_ace", "l_ace", "w_1stWon", "w_2ndWon",
        "l_1stWon", "l_2ndWon",
    ]
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["total_svpt"] = work["w_svpt"] + work["l_svpt"]
    work["ace_rate"] = (work["w_ace"] + work["l_ace"]) / work["total_svpt"]
    work["service_win_rate"] = (
        work["w_1stWon"] + work["w_2ndWon"] + work["l_1stWon"] + work["l_2ndWon"]
    ) / work["total_svpt"]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["season", "tourney_name", "surface", "ace_rate", "service_win_rate"]
    )
    work = work[(work["total_svpt"] > 0) & work["surface"].isin(["Hard", "Clay", "Grass"])]

    grouped = work.groupby(["season", "tourney_name", "surface"], as_index=False).agg(
        ace_rate=("ace_rate", "mean"),
        service_win_rate=("service_win_rate", "mean"),
        matches_used=("ace_rate", "size"),
    )
    grouped = grouped[grouped["matches_used"] >= 5].copy()
    if grouped.empty:
        grouped.to_csv(output_path, index=False)
        return grouped

    # Standardize within each surface and season, then compress to a stable scale.
    def standardized(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        for col in ["ace_rate", "service_win_rate"]:
            median = group[col].median()
            mad = (group[col] - median).abs().median()
            scale = max(float(mad) * 1.4826, 1e-4)
            group[f"{col}_z"] = ((group[col] - median) / scale).clip(-3, 3)
        score = 0.65 * group["ace_rate_z"] + 0.35 * group["service_win_rate_z"]
        group["surface_speed"] = (1.0 + 0.08 * score).clip(0.75, 1.25)
        return group

    grouped = pd.concat(
        [standardized(part) for _, part in grouped.groupby(["season", "surface"], sort=False)],
        ignore_index=True,
    )
    grouped["tournament"] = grouped["tourney_name"].astype(str)
    grouped["tournament_key"] = grouped["tournament"].map(canonical_tournament)
    output = grouped[[
        "season", "tournament", "tournament_key", "surface",
        "surface_speed", "matches_used", "ace_rate", "service_win_rate",
    ]].sort_values(["season", "surface", "tournament"])
    output.to_csv(output_path, index=False)
    return output


def load_surface_speeds(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "tournament_key" not in frame and "tournament" in frame:
        frame["tournament_key"] = frame["tournament"].map(canonical_tournament)
    frame["season"] = pd.to_numeric(frame.get("season"), errors="coerce")
    frame["surface_speed"] = pd.to_numeric(frame.get("surface_speed"), errors="coerce")
    return frame.dropna(subset=["season", "surface_speed"])


def lookup_surface_speed(
    speeds: pd.DataFrame,
    tournament: str,
    surface: str,
    season: int,
) -> tuple[float, float]:
    """Return (speed, missing flag), using only seasons before the match season."""
    if speeds.empty:
        return 1.0, 1.0
    eligible = speeds[speeds["season"] < int(season)]
    key = canonical_tournament(tournament)
    exact = eligible[
        (eligible["tournament_key"] == key)
        & (eligible["surface"].astype(str).str.title() == str(surface).title())
    ]
    if not exact.empty:
        row = exact.sort_values("season").iloc[-1]
        return float(row["surface_speed"]), 0.0
    fallback = eligible[
        eligible["surface"].astype(str).str.title() == str(surface).title()
    ]
    if not fallback.empty:
        latest = fallback["season"].max()
        return float(fallback[fallback["season"] == latest]["surface_speed"].median()), 1.0
    return 1.0, 1.0

"""Data-quality rules shared by the daily updater, the trainer and the live board.

Why this module exists (v0.4):

* The TennisMyLife archive records ``tourney_date`` as the tournament START date,
  while Matchstat records the MATCH date.  The old cross-provider key required the
  dates to be identical, so hundreds of 2026 matches were stored twice.  Every
  duplicate moved Elo, rolling form and H2H twice.
* Matchstat's player-history endpoint also returns ITF and Challenger matches.  The
  historical archive (2000-2025) is tour-level only, so mixing those rows into 2026
  inflated recent form for players who drop down a tier and made the current
  player state inconsistent with what the model was trained on.
* Matchstat labels most tour events ``"A"`` (encoded as ATP 250), so ATP 500s such
  as Barcelona, Rotterdam or Queen's were treated as 250s.

All functions are pure and idempotent: running them twice returns the same frame.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from .tournament_features import canonical_tournament

LOWER_TIER_NAME = re.compile(
    r"(challenger|\bitf\b|futures?\b|^\s*[mw]\s?-?(?:15|25|35|50|75|100)\b|\butr\b|junior)",
    flags=re.I,
)

# Recent category changes that a "most common level" rule gets wrong.
LEVEL_OVERRIDES = {
    "dallas": "500",
    "doha": "500",
    "winston salem": "250",
    "queens club": "500",
    "halle": "500",
}

TEAM_EVENT_NAME = re.compile(r"(davis cup|laver cup|united cup|atp cup|hopman)", flags=re.I)

STAT_COLUMNS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points", "minutes",
]


def norm_name(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return " ".join(text.split())


def name_tokens_key(value) -> str:
    """Primary loose player key: first-name initial + final surname token."""
    tokens = norm_name(value).split()
    if not tokens:
        return ""
    return f"{tokens[0][:1]}|{tokens[-1]}"


def name_keys(value) -> set[str]:
    """All loose keys (initial + any later token) so that provider variants such as
    ``Jaume Antoni Munar Clar`` and ``Jaume Munar`` share a key. Used together with
    the opponent and a date window, false matches are practically impossible."""
    tokens = norm_name(value).split()
    if not tokens:
        return set()
    if len(tokens) == 1:
        return {tokens[0]}
    return {f"{tokens[0][:1]}|{tok}" for tok in tokens[1:] if len(tok) > 1}


def is_lower_tier(name, level=None) -> bool:
    lv = str(level or "").strip().upper()
    if lv in {"C", "CH", "S"}:
        return True
    return bool(LOWER_TIER_NAME.search(str(name or "")))


def _event_candidates(name: str) -> list[str]:
    raw = str(name or "")
    parts = [raw] + [p.strip() for p in raw.split(" - ") if p.strip()]
    out = []
    for p in parts:
        key = canonical_tournament(p)
        if key and key not in out:
            out.append(key)
    return out


def build_level_map(frame: pd.DataFrame) -> dict[str, str]:
    """canonical tournament -> latest level code from the curated archive."""
    if frame.empty or "tourney_level" not in frame.columns:
        return dict(LEVEL_OVERRIDES)
    src = frame
    if "data_source" in frame.columns:
        src = frame[frame["data_source"].astype(str) != "Matchstat"]
    src = src.dropna(subset=["tourney_name", "tourney_level"])
    src = src[~src["tourney_level"].astype(str).str.upper().isin(["C", "D"])]
    src = src.sort_values("tourney_date")
    out: dict[str, str] = {}
    for name, level in zip(src["tourney_name"].astype(str), src["tourney_level"].astype(str)):
        out[canonical_tournament(name)] = level
    out.update(LEVEL_OVERRIDES)
    return out


def resolve_level(name: str, provider_level, level_map: dict[str, str]) -> str:
    """Best tour-level code for a tournament name."""
    lv = str(provider_level or "").strip().upper()
    if TEAM_EVENT_NAME.search(str(name or "")):
        return "D" if "davis" in str(name).lower() else "A"
    if lv in {"G", "M", "F", "O", "D"}:
        return lv
    for key in _event_candidates(name):
        if key in level_map:
            return str(level_map[key])
    return str(provider_level or "A")


def _stat_count(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in STAT_COLUMNS if c in frame.columns]
    return frame[cols].notna().sum(axis=1) if cols else pd.Series(0, index=frame.index)


def dedupe_cross_provider(frame: pd.DataFrame, window_days: int = 14) -> tuple[pd.DataFrame, int]:
    """Drop the same match reported by two providers with different dates/names.

    Two rows are the same match when the (winner, loser) loose keys match and the
    dates are within ``window_days``.  The row with more statistics is kept; missing
    fields on the kept row are filled from the dropped twin.
    """
    if frame.empty or not {"winner_name", "loser_name", "tourney_date"}.issubset(frame.columns):
        return frame, 0
    df = frame.copy()
    df["_date"] = pd.to_datetime(pd.to_numeric(df["tourney_date"], errors="coerce").astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    df["_wk"] = df["winner_name"].map(name_tokens_key)
    df["_lk"] = df["loser_name"].map(name_tokens_key)
    df["_stats"] = _stat_count(df)
    df["_src"] = df.get("data_source", pd.Series("", index=df.index)).astype(str)
    df["_tour"] = df.get("tourney_name", pd.Series("", index=df.index)).map(canonical_tournament)
    df["_round"] = df.get("round", pd.Series("", index=df.index)).astype(str)

    drop: set = set()
    fill_from: dict = {}
    # Candidate pairs: rows sharing any (winner key, loser key) combination.
    buckets: dict = {}
    for idx, wn, ln in zip(df.index, df["winner_name"], df["loser_name"]):
        for wk in name_keys(wn):
            for lk in name_keys(ln):
                buckets.setdefault((wk, lk), []).append(idx)
    pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        members = sorted(set(members), key=lambda i: df.at[i, "_date"] if pd.notna(df.at[i, "_date"]) else pd.Timestamp.min)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                da, db = df.at[a, "_date"], df.at[b, "_date"]
                if pd.isna(da) or pd.isna(db):
                    continue
                if (db - da).days > window_days:
                    break
                pairs.add((a, b) if a < b else (b, a))
    for a, b in sorted(pairs):
        if a in drop or b in drop:
            continue
        if abs((df.at[b, "_date"] - df.at[a, "_date"]).days) > window_days:
            continue
        if df.at[a, "_src"] == df.at[b, "_src"]:
            # Same provider: a genuine rematch (qualifying then main draw) is
            # legal, so only drop exact duplicates of event + round.
            if df.at[a, "_tour"] != df.at[b, "_tour"] or df.at[a, "_round"] != df.at[b, "_round"]:
                continue
        keep, lose = (a, b) if df.at[a, "_stats"] >= df.at[b, "_stats"] else (b, a)
        drop.add(lose)
        fill_from.setdefault(keep, []).append(lose)
    out = frame.copy()
    for keep, losers in fill_from.items():
        for lose in losers:
            missing = out.loc[keep].isna()
            if missing.any():
                cols = missing[missing].index
                out.loc[keep, cols] = frame.loc[lose, cols].values
    out = out.drop(index=list(drop))
    return out, len(drop)


def clean_master(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply every data-quality rule. Safe to call repeatedly."""
    report = {"input_rows": int(len(frame))}
    if frame.empty:
        return frame, report
    df = frame.copy()
    lower = df.apply(lambda r: is_lower_tier(r.get("tourney_name"), r.get("tourney_level")), axis=1)
    report["lower_tier_rows_removed"] = int(lower.sum())
    df = df[~lower]
    # The 2000-2025 archive contains main-draw matches only. Matchstat adds
    # qualifying (Q1-Q3); keep the training/live state consistent with history.
    if "round" in df.columns:
        qual = df["round"].astype(str).str.strip().str.match(r"^Q\d?$", case=False)
        report["qualifying_rows_removed"] = int(qual.sum())
        df = df[~qual]

    level_map = build_level_map(df)
    before = df["tourney_level"].astype(str).copy() if "tourney_level" in df.columns else None
    if "tourney_level" in df.columns:
        is_ms = df.get("data_source", pd.Series("", index=df.index)).astype(str).eq("Matchstat")
        df.loc[is_ms, "tourney_level"] = [
            resolve_level(n, lv, level_map)
            for n, lv in zip(df.loc[is_ms, "tourney_name"], df.loc[is_ms, "tourney_level"])
        ]
        report["levels_corrected"] = int((before != df["tourney_level"].astype(str)).sum())

    df, removed = dedupe_cross_provider(df)
    report["cross_provider_duplicates_removed"] = int(removed)
    report["output_rows"] = int(len(df))
    return df, report

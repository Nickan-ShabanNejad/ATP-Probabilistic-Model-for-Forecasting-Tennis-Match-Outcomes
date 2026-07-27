from __future__ import annotations
import re
import unicodedata
from pathlib import Path
import pandas as pd

ALIASES = {
    "roland garros": "french open",
    "nd garros": "french open",
    "us open": "us open",
    "u s open": "us open",
    "canada masters": "canadian masters",
    "canadian open": "canadian masters",
    "montreal": "canadian masters",
    "toronto": "canadian masters",
    "indian wells masters": "indian wells",
    "miami masters": "miami",
    "cincinnati masters": "cincinnati",
    "shanghai masters": "shanghai",
    "paris masters": "paris",
    "madrid masters": "madrid",
    "rome masters": "rome",
    "monte carlo masters": "monte carlo",
    "s hertogenbosch": "s hertogenbosch",
    "s-hertogenbosch": "s hertogenbosch",
    "queens club": "queens club",
    "queen s club": "queens club",
    "tour finals": "atp finals",
}

def canonical_tournament(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)

def load_surface_speeds(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"season", "tournament", "surface_speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Surface-speed file is missing columns: {sorted(missing)}")
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["surface_speed"] = pd.to_numeric(df["surface_speed"], errors="coerce")
    df["tournament_key"] = df["tournament"].map(canonical_tournament)
    return df.dropna(subset=["surface_speed"])

def add_surface_speed_feature(
    matches: pd.DataFrame,
    speeds: pd.DataFrame,
    *,
    tournament_col: str = "tourney_name",
    date_col: str = "tourney_date",
    surface_col: str = "surface",
) -> pd.DataFrame:
    """
    Adds a leakage-safe tournament-speed feature.

    For a match played in season Y, it uses the latest available rating from
    a season strictly before Y. It never uses the completed Y tournament's
    final rating to predict matches within that same tournament.
    """
    out = matches.copy()
    out[date_col] = pd.to_datetime(out[date_col].astype(str), errors="coerce")
    out["match_season"] = out[date_col].dt.year
    out["tournament_key"] = out[tournament_col].map(canonical_tournament)

    left = out.reset_index(names="_original_index").sort_values(
        ["tournament_key", "match_season"]
    )
    right = speeds[["tournament_key", "season", "surface_speed"]].sort_values(
        ["tournament_key", "season"]
    )

    # merge_asof with allow_exact_matches=False enforces previous-season only.
    merged = pd.merge_asof(
        left,
        right,
        left_on="match_season",
        right_on="season",
        by="tournament_key",
        direction="backward",
        allow_exact_matches=False,
    )

    # Fallback for new/renamed events: prior-season median for that surface,
    # then neutral tour average 1.00.
    historical = speeds.copy()
    if "surface" in historical.columns and surface_col in merged.columns:
        historical["surface_norm"] = historical["surface"].astype(str).str.title()
        med = (
            historical.groupby(["season", "surface_norm"], as_index=False)["surface_speed"]
            .median()
            .sort_values(["surface_norm", "season"])
        )
        missing = merged["surface_speed"].isna()
        if missing.any():
            fb_left = merged.loc[missing, ["_original_index","match_season",surface_col]].copy()
            fb_left["surface_norm"] = fb_left[surface_col].astype(str).str.title()
            fb_left = fb_left.sort_values(["surface_norm","match_season"])
            fb = pd.merge_asof(
                fb_left,
                med,
                left_on="match_season",
                right_on="season",
                by="surface_norm",
                direction="backward",
                allow_exact_matches=False,
            )
            merged.loc[missing, "surface_speed"] = fb.set_index("_original_index")[
                "surface_speed"
            ].reindex(merged.loc[missing, "_original_index"]).to_numpy()

    merged["surface_speed"] = merged["surface_speed"].fillna(1.0)
    merged["surface_speed_missing"] = merged["season"].isna().astype("int8")
    return (
        merged.sort_values("_original_index")
        .set_index("_original_index")
        .drop(columns=["tournament_key"], errors="ignore")
    )

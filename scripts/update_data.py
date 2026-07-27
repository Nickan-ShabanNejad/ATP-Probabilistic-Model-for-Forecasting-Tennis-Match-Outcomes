
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GENERATED = ROOT / "data" / "generated"
RAW.mkdir(parents=True, exist_ok=True)
GENERATED.mkdir(parents=True, exist_ok=True)

JEFF_ATP = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
MCP = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"

MATCH_REQUIRED = {"tourney_date", "winner_id", "loser_id", "surface"}
MATCH_KEY = ["tourney_date", "tourney_name", "round", "winner_id", "loser_id"]


def download(url: str, destination: Path, minimum_bytes: int = 50) -> dict:
    temp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        response = requests.get(
            url,
            timeout=120,
            headers={"User-Agent": "ATP-Model-GitHub-Action/2.0"},
        )
        response.raise_for_status()
        if len(response.content) < minimum_bytes:
            raise RuntimeError(f"download only contained {len(response.content)} bytes")
        temp.write_bytes(response.content)
        temp.replace(destination)
        return {
            "ok": True,
            "url": url,
            "bytes": len(response.content),
            "sha256": hashlib.sha256(response.content).hexdigest(),
        }
    except Exception as exc:
        temp.unlink(missing_ok=True)
        return {"ok": False, "url": url, "error": str(exc)}


def read_match_file(path: Path, source: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if not MATCH_REQUIRED.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["data_source"] = source
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df["winner_id"] = df["winner_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["loser_id"] = df["loser_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df = df.dropna(subset=["tourney_date", "winner_id", "loser_id", "surface"])
    df = df[(df["winner_id"] != "nan") & (df["loser_id"] != "nan")]
    return df


def completeness_score(df: pd.DataFrame) -> pd.Series:
    useful = [
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
        "winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points",
    ]
    cols = [c for c in useful if c in df.columns]
    return df[cols].notna().sum(axis=1) if cols else pd.Series(0, index=df.index)


def deduplicate_matches(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No valid match source was downloaded.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["_complete"] = completeness_score(merged)
    # TML wins ties because it is intended as the freshness/correction layer.
    merged["_source_priority"] = merged["data_source"].map(
        {"TennisMyLife": 2, "Jeff Sackmann": 1}
    ).fillna(0)

    key = [c for c in MATCH_KEY if c in merged.columns]
    merged = (
        merged.sort_values(key + ["_complete", "_source_priority"])
        .drop_duplicates(key, keep="last")
        .drop(columns=["_complete", "_source_priority"])
        .sort_values(["tourney_date", "tourney_name", "match_num"], na_position="last")
    )
    return merged


def refresh_match_data() -> dict:
    current_year = datetime.now(timezone.utc).year
    all_frames: list[pd.DataFrame] = []
    status: dict[str, dict] = {}

    for year in range(2000, current_year + 1):
        jeff_path = RAW / f"jeff_atp_matches_{year}.csv"
        tml_path = RAW / f"tml_atp_matches_{year}.csv"

        jeff_result = download(f"{JEFF_ATP}/atp_matches_{year}.csv", jeff_path)
        # TML is most important for recent seasons, but attempting all years allows corrections.
        tml_result = download(f"{TML}/{year}.csv", tml_path)

        jeff = read_match_file(jeff_path, "Jeff Sackmann") if jeff_path.exists() else pd.DataFrame()
        tml = read_match_file(tml_path, "TennisMyLife") if tml_path.exists() else pd.DataFrame()

        if not jeff.empty:
            all_frames.append(jeff)
        if not tml.empty:
            all_frames.append(tml)

        latest_values = []
        for frame in (jeff, tml):
            if not frame.empty:
                latest_values.append(int(frame["tourney_date"].max()))

        status[str(year)] = {
            "jeff_sackmann": {**jeff_result, "rows": len(jeff)},
            "tennis_my_life": {**tml_result, "rows": len(tml)},
            "latest_tourney_date": str(max(latest_values)) if latest_values else None,
        }

    master = deduplicate_matches(all_frames)
    master_path = GENERATED / "master_matches.csv.gz"
    master.to_csv(master_path, index=False, compression="gzip")

    source_counts = master["data_source"].value_counts().to_dict()
    latest = str(int(master["tourney_date"].max()))
    return {
        "years": status,
        "master_rows": len(master),
        "latest_tourney_date": latest,
        "selected_rows_by_source": source_counts,
        "output": str(master_path.relative_to(ROOT)),
    }


def refresh_reference_data() -> dict:
    results = {}

    players_path = RAW / "atp_players.csv"
    results["players"] = download(f"{JEFF_ATP}/atp_players.csv", players_path)

    rankings_raw = RAW / "atp_rankings_current.csv"
    ranking_result = download(f"{JEFF_ATP}/atp_rankings_current.csv", rankings_raw)
    if not ranking_result["ok"]:
        ranking_result = download(f"{JEFF_ATP}/atp_rankings_20s.csv", rankings_raw)

    ranking_meta = {**ranking_result}
    if rankings_raw.exists():
        rankings = pd.read_csv(rankings_raw, low_memory=False)
        needed = {"ranking_date", "ranking", "player_id"}
        if needed.issubset(rankings.columns):
            rankings["ranking_date"] = pd.to_numeric(rankings["ranking_date"], errors="coerce")
            rankings["ranking"] = pd.to_numeric(rankings["ranking"], errors="coerce")
            rankings["player_id"] = (
                rankings["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
            )
            rankings = rankings.dropna(subset=["ranking_date", "ranking", "player_id"])
            latest = int(rankings["ranking_date"].max())
            cols = ["ranking_date", "ranking", "player_id"]
            if "ranking_points" in rankings.columns:
                cols.append("ranking_points")
            current = (
                rankings[rankings["ranking_date"] == latest]
                .sort_values("ranking")
                .drop_duplicates("player_id")
                [cols]
            )
            current.to_csv(GENERATED / "current_rankings.csv", index=False)
            ranking_meta.update({"ranking_date": str(latest), "players": len(current)})
    results["rankings"] = ranking_meta
    return results


def refresh_charting_data() -> dict:
    files = {
        "matches": "charting-m-matches.csv",
        "overview": "charting-m-stats-Overview.csv",
        "net_points": "charting-m-stats-NetPoints.csv",
    }
    output = {}
    for label, filename in files.items():
        path = RAW / filename
        result = download(f"{MCP}/{filename}", path)
        if path.exists():
            try:
                result["rows"] = len(pd.read_csv(path, low_memory=False))
            except Exception as exc:
                result["parse_error"] = str(exc)
        output[label] = result
    return output


def main() -> None:
    freshness = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matches": refresh_match_data(),
        "reference": refresh_reference_data(),
        "match_charting_project": refresh_charting_data(),
    }

    (GENERATED / "freshness.json").write_text(
        json.dumps(freshness, indent=2), encoding="utf-8"
    )

    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "train_model.py")])
    print(json.dumps(freshness, indent=2))


if __name__ == "__main__":
    main()

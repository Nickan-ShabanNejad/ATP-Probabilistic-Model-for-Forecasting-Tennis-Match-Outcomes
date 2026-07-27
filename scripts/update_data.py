
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GENERATED = ROOT / "data" / "generated"
RAW.mkdir(parents=True, exist_ok=True)
GENERATED.mkdir(parents=True, exist_ok=True)

TML_API = "https://stats.tennismylife.org/api/data-files"


def get_tml_urls():
    r = requests.get(TML_API, timeout=60, headers={"User-Agent":"ATP-Model-GitHub-Action/3.0"})
    r.raise_for_status()
    return {f["name"]: f["url"] for f in r.json()["files"]}
MCP = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"

# Jeff Sackmann's ATP match repository currently returns 404 publicly.
# Keep an optional URL override so it can be restored without code changes.
JEFF_ATP = os.getenv("JEFF_ATP_BASE_URL", "").rstrip("/")
RANKINGS_CSV_URL = os.getenv("ATP_RANKINGS_CSV_URL", "").strip()
ATP_RANKINGS_URL = "https://www.atptour.com/en/rankings/singles"

MATCH_REQUIRED = {"tourney_date", "winner_id", "loser_id", "surface"}
MATCH_KEY = ["tourney_date", "tourney_name", "round", "winner_id", "loser_id"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def download(url: str, destination: Path, minimum_bytes: int = 50) -> dict:
    temp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        response = requests.get(
            url,
            timeout=120,
            headers={"User-Agent": "ATP-Model-GitHub-Action/3.0"},
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if len(response.content) < minimum_bytes:
            raise RuntimeError(f"download only contained {len(response.content)} bytes")
        if "text/html" in content_type.lower():
            raise RuntimeError("source returned HTML instead of CSV data")
        temp.write_bytes(response.content)
        temp.replace(destination)
        return {
            "ok": True,
            "url": url,
            "bytes": len(response.content),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "downloaded_at_utc": utc_now(),
        }
    except Exception as exc:
        temp.unlink(missing_ok=True)
        return {"ok": False, "url": url, "error": str(exc), "downloaded_at_utc": utc_now()}


def read_match_file(path: Path, source: str) -> tuple[pd.DataFrame, dict]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return pd.DataFrame(), {"valid": False, "error": f"CSV parse failed: {exc}"}

    missing = sorted(MATCH_REQUIRED - set(df.columns))
    if missing:
        return pd.DataFrame(), {"valid": False, "error": f"missing columns: {missing}"}

    df = df.copy()
    df["data_source"] = source
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df["winner_id"] = df["winner_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["loser_id"] = df["loser_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df = df.dropna(subset=["tourney_date", "winner_id", "loser_id", "surface"])
    df = df[(df["winner_id"] != "nan") & (df["loser_id"] != "nan")]

    if df.empty:
        return df, {"valid": False, "error": "no valid rows after normalization"}

    return df, {
        "valid": True,
        "rows": int(len(df)),
        "latest_tourney_date": str(int(df["tourney_date"].max())),
        "columns": int(len(df.columns)),
    }


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


def deduplicate_matches(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No valid match source was downloaded.")

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw_counts = {str(k): int(v) for k, v in raw["data_source"].value_counts().items()}
    raw["_complete"] = completeness_score(raw)
    raw["_source_priority"] = raw["data_source"].map(
        {"TennisMyLife": 2, "Jeff Sackmann": 1}
    ).fillna(0)

    key = [c for c in MATCH_KEY if c in raw.columns]
    master = (
        raw.sort_values(key + ["_complete", "_source_priority"])
        .drop_duplicates(key, keep="last")
        .drop(columns=["_complete", "_source_priority"])
        .sort_values(["tourney_date", "tourney_name", "match_num"], na_position="last")
    )
    kept_counts = {str(k): int(v) for k, v in master["data_source"].value_counts().items()}
    return master, {
        "raw_rows_by_source": raw_counts,
        "selected_rows_by_source": kept_counts,
        "duplicates_removed": int(len(raw) - len(master)),
    }


def refresh_match_data() -> dict:
    current_year = datetime.now(timezone.utc).year
    all_frames: list[pd.DataFrame] = []
    years: dict[str, dict] = {}

    source_summary = {
        "TennisMyLife": {"available": True, "successful_years": 0, "failed_years": 0},
        "Jeff Sackmann": {
            "available": bool(JEFF_ATP),
            "successful_years": 0,
            "failed_years": 0,
            "note": (
                "Disabled because the public repository currently returns 404. "
                "Set JEFF_ATP_BASE_URL to re-enable it."
                if not JEFF_ATP else "Enabled through JEFF_ATP_BASE_URL."
            ),
        },
    }

    tml_urls = get_tml_urls()

    for year in range(2000, current_year + 1):
        year_status: dict[str, dict] = {}

        tml_path = RAW / f"tml_atp_matches_{year}.csv"
        filename = f"{year}.csv"
        if filename in tml_urls:
            tml_download = download(tml_urls[filename], tml_path)
        else:
            tml_download = {"ok": False, "error": f"{filename} not found"}
        tml, tml_validation = read_match_file(tml_path, "TennisMyLife") if tml_path.exists() else (
            pd.DataFrame(), {"valid": False, "error": "file unavailable"}
        )
        if not tml.empty:
            all_frames.append(tml)
            source_summary["TennisMyLife"]["successful_years"] += 1
        else:
            source_summary["TennisMyLife"]["failed_years"] += 1
        year_status["tennis_my_life"] = {**tml_download, **tml_validation}

        if JEFF_ATP:
            jeff_path = RAW / f"jeff_atp_matches_{year}.csv"
            jeff_download = download(f"{JEFF_ATP}/atp_matches_{year}.csv", jeff_path)
            jeff, jeff_validation = read_match_file(
                jeff_path, "Jeff Sackmann"
            ) if jeff_path.exists() else (
                pd.DataFrame(), {"valid": False, "error": "file unavailable"}
            )
            if not jeff.empty:
                all_frames.append(jeff)
                source_summary["Jeff Sackmann"]["successful_years"] += 1
            else:
                source_summary["Jeff Sackmann"]["failed_years"] += 1
            year_status["jeff_sackmann"] = {**jeff_download, **jeff_validation}
        else:
            year_status["jeff_sackmann"] = {
                "ok": False,
                "skipped": True,
                "reason": "JEFF_ATP_BASE_URL is not configured.",
                "rows": 0,
            }

        dates = [
            x.get("latest_tourney_date")
            for x in year_status.values()
            if x.get("latest_tourney_date")
        ]
        year_status["latest_tourney_date"] = max(dates) if dates else None
        years[str(year)] = year_status

    master, merge_meta = deduplicate_matches(all_frames)
    master_path = GENERATED / "master_matches.csv.gz"
    master.to_csv(master_path, index=False, compression="gzip")

    latest = str(int(master["tourney_date"].max()))
    latest_dt = datetime.strptime(latest, "%Y%m%d").replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - latest_dt).days

    recent_year = master[master["tourney_date"] >= int(f"{current_year}0101")]
    return {
        "years": years,
        "sources": source_summary,
        "master_rows": int(len(master)),
        "current_year_rows": int(len(recent_year)),
        "latest_tourney_date": latest,
        "age_days": int(age_days),
        "stale": bool(age_days > 7),
        "freshness_threshold_days": 7,
        **merge_meta,
        "output": str(master_path.relative_to(ROOT)),
    }



def normalize_player_name(value: str) -> str:
    import re
    import unicodedata
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return " ".join(text.split())


def refresh_official_atp_rankings(master_path: Path) -> dict:
    """Read the official ATP singles ranking table and map names to match-data IDs."""
    try:
        response = requests.get(
            ATP_RANKINGS_URL,
            params={"rankRange": "0-5000", "region": "all"},
            timeout=120,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; ATP-Model-Rankings/1.0; "
                    "+https://github.com/)"
                )
            },
        )
        response.raise_for_status()
        tables = pd.read_html(response.text)
        if not tables:
            raise RuntimeError("official ATP page contained no HTML tables")

        ranking_table = None
        for table in tables:
            compact = {
                str(col).strip().lower().replace(" ", ""): col for col in table.columns
            }
            rank_col = next((v for k, v in compact.items() if k in {"rank", "ranking"}), None)
            player_col = next((v for k, v in compact.items() if "player" in k), None)
            points_col = next((v for k, v in compact.items() if "point" in k), None)
            if rank_col is not None and player_col is not None:
                ranking_table = table.copy()
                break

        if ranking_table is None:
            raise RuntimeError("could not identify Rank and Player columns")

        compact = {
            str(col).strip().lower().replace(" ", ""): col
            for col in ranking_table.columns
        }
        rank_col = next(v for k, v in compact.items() if k in {"rank", "ranking"})
        player_col = next(v for k, v in compact.items() if "player" in k)
        points_col = next((v for k, v in compact.items() if "point" in k), None)

        official = pd.DataFrame(
            {
                "ranking": pd.to_numeric(
                    ranking_table[rank_col].astype(str).str.extract(r"(\d+)")[0],
                    errors="coerce",
                ),
                "player": ranking_table[player_col].astype(str).str.strip(),
            }
        )
        if points_col is not None:
            official["ranking_points"] = pd.to_numeric(
                ranking_table[points_col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.extract(r"(\d+)")[0],
                errors="coerce",
            )

        official = official.dropna(subset=["ranking", "player"])
        official = official[official["player"].str.len() > 2]
        official["ranking"] = official["ranking"].astype(int)
        official["normalized_name"] = official["player"].map(normalize_player_name)
        official = official.drop_duplicates("normalized_name").sort_values("ranking")

        master = pd.read_csv(
            master_path,
            usecols=lambda c: c in {
                "winner_id", "winner_name", "loser_id", "loser_name", "tourney_date"
            },
            low_memory=False,
        )
        player_parts = []
        for side in ("winner", "loser"):
            part = master[
                [f"{side}_id", f"{side}_name", "tourney_date"]
            ].copy()
            part.columns = ["player_id", "match_name", "tourney_date"]
            player_parts.append(part)
        players = pd.concat(player_parts, ignore_index=True).dropna()
        players["player_id"] = (
            players["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        )
        players["normalized_name"] = players["match_name"].map(normalize_player_name)
        players = (
            players.sort_values("tourney_date")
            .drop_duplicates("normalized_name", keep="last")
        )

        current = official.merge(
            players[["player_id", "normalized_name"]],
            on="normalized_name",
            how="left",
        )
        current["ranking_date"] = datetime.now(timezone.utc).strftime("%Y%m%d")
        columns = [
            "ranking_date", "ranking", "player_id", "player", "ranking_points",
            "normalized_name"
        ]
        for col in columns:
            if col not in current:
                current[col] = None
        current[columns].to_csv(GENERATED / "current_rankings.csv", index=False)

        matched = int(current["player_id"].notna().sum())
        if len(current) < 50:
            raise RuntimeError(f"only {len(current)} ranking rows parsed")
        return {
            "ok": True,
            "method": "official ATP singles rankings page",
            "url": ATP_RANKINGS_URL,
            "ranking_date": str(current["ranking_date"].iloc[0]),
            "players": int(len(current)),
            "mapped_to_player_id": matched,
            "mapping_rate": float(matched / len(current)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "method": "official ATP singles rankings page",
            "url": ATP_RANKINGS_URL,
            "error": str(exc),
        }

def build_rankings_from_latest_matches(master_path: Path) -> dict:
    """Fallback only: latest observed match rankings are not live rankings."""
    master = pd.read_csv(master_path, low_memory=False)
    records = []

    for side in ("winner", "loser"):
        required = {f"{side}_id", f"{side}_rank", "tourney_date"}
        if not required.issubset(master.columns):
            continue
        cols = ["tourney_date", f"{side}_id", f"{side}_rank"]
        if f"{side}_rank_points" in master.columns:
            cols.append(f"{side}_rank_points")
        part = master[cols].copy()
        part.columns = [
            "ranking_date",
            "player_id",
            "ranking",
            *(["ranking_points"] if len(cols) == 4 else []),
        ]
        records.append(part)

    if not records:
        return {"ok": False, "error": "match files do not include ranking fields"}

    ranks = pd.concat(records, ignore_index=True)
    ranks["ranking_date"] = pd.to_numeric(ranks["ranking_date"], errors="coerce")
    ranks["ranking"] = pd.to_numeric(ranks["ranking"], errors="coerce")
    ranks["player_id"] = ranks["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    ranks = ranks.dropna(subset=["ranking_date", "ranking", "player_id"])
    latest_by_player = (
        ranks.sort_values("ranking_date")
        .drop_duplicates("player_id", keep="last")
        .sort_values("ranking")
    )
    latest_by_player.to_csv(GENERATED / "current_rankings.csv", index=False)
    latest = str(int(latest_by_player["ranking_date"].max()))
    return {
        "ok": True,
        "method": "latest observed ranking in match data",
        "warning": "This is a fallback and may lag the official weekly ATP ranking.",
        "ranking_date": latest,
        "players": int(len(latest_by_player)),
    }


def refresh_rankings(master_path: Path) -> dict:
    official = refresh_official_atp_rankings(master_path)
    if official.get("ok"):
        return official

    destination = RAW / "current_rankings_external.csv"
    if RANKINGS_CSV_URL:
        result = download(RANKINGS_CSV_URL, destination)
        if result["ok"]:
            try:
                ranks = pd.read_csv(destination, low_memory=False)
                needed = {"ranking_date", "ranking", "player_id"}
                if not needed.issubset(ranks.columns):
                    raise RuntimeError(f"missing columns: {sorted(needed - set(ranks.columns))}")
                ranks["ranking_date"] = pd.to_numeric(ranks["ranking_date"], errors="coerce")
                ranks["ranking"] = pd.to_numeric(ranks["ranking"], errors="coerce")
                ranks["player_id"] = ranks["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
                ranks = ranks.dropna(subset=["ranking_date", "ranking", "player_id"])
                latest = int(ranks["ranking_date"].max())
                current = (
                    ranks[ranks["ranking_date"] == latest]
                    .sort_values("ranking")
                    .drop_duplicates("player_id")
                )
                current.to_csv(GENERATED / "current_rankings.csv", index=False)
                return {
                    **result,
                    "method": "ATP_RANKINGS_CSV_URL",
                    "ranking_date": str(latest),
                    "players": int(len(current)),
                }
            except Exception as exc:
                result["parse_error"] = str(exc)

    fallback = build_rankings_from_latest_matches(master_path)
    fallback["external_url_configured"] = bool(RANKINGS_CSV_URL)
    fallback["official_atp_attempt"] = official
    return fallback


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
                frame = pd.read_csv(path, low_memory=False)
                result["rows"] = int(len(frame))
                result["valid"] = len(frame) > 0
            except Exception as exc:
                result["valid"] = False
                result["parse_error"] = str(exc)
        output[label] = result
    output["available"] = all(item.get("ok") and item.get("valid") for item in output.values())
    return output


def append_source_history(freshness: dict) -> None:
    path = GENERATED / "source_history.csv"
    row = {
        "updated_at_utc": freshness["updated_at_utc"],
        "latest_match_date": freshness["matches"]["latest_tourney_date"],
        "match_age_days": freshness["matches"]["age_days"],
        "master_rows": freshness["matches"]["master_rows"],
        "current_year_rows": freshness["matches"]["current_year_rows"],
        "duplicates_removed": freshness["matches"]["duplicates_removed"],
        "tml_rows_selected": freshness["matches"]["selected_rows_by_source"].get("TennisMyLife", 0),
        "jeff_rows_selected": freshness["matches"]["selected_rows_by_source"].get("Jeff Sackmann", 0),
        "charting_available": freshness["match_charting_project"].get("available", False),
        "ranking_date": freshness["reference"]["rankings"].get("ranking_date"),
        "ranking_method": freshness["reference"]["rankings"].get("method"),
    }
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True).tail(365)
    new.to_csv(path, index=False)


def main() -> None:
    matches = refresh_match_data()
    master_path = ROOT / matches["output"]

    freshness = {
        "updated_at_utc": utc_now(),
        "pipeline_version": "3.0",
        "matches": matches,
        "reference": {"rankings": refresh_rankings(master_path)},
        "match_charting_project": refresh_charting_data(),
    }

    (GENERATED / "freshness.json").write_text(
        json.dumps(freshness, indent=2), encoding="utf-8"
    )
    append_source_history(freshness)

    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "update_surface_speed.py"), "--start-year", "2012"])
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "train_model.py")])
    print(json.dumps(freshness, indent=2))


if __name__ == "__main__":
    main()



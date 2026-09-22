
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, subprocess, sys
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
JEFF = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"

def fetch(url, path):
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        if len(r.content) < 100:
            return False
        path.write_bytes(r.content)
        print("downloaded", url)
        return True
    except Exception as exc:
        print("failed", url, exc)
        return False

def valid_matches(path, year):
    try:
        df = pd.read_csv(path, low_memory=False)
        needed = {"tourney_date","winner_id","loser_id","surface"}
        if not needed.issubset(df.columns):
            return False, None
        d = pd.to_numeric(df.tourney_date, errors="coerce").dropna()
        latest = str(int(d.max()))
        return latest.startswith(str(year)), latest
    except Exception:
        return False, None

def refresh_matches():
    result = {}
    for year in range(2020, datetime.now(timezone.utc).year + 1):
        target = RAW / f"{year}.csv"
        source, latest = None, None
        if fetch(f"{JEFF}/atp_matches_{year}.csv", target):
            ok, latest = valid_matches(target, year)
            if ok: source = "Jeff Sackmann"
        if source is None and fetch(f"{TML}/{year}.csv", target):
            ok, latest = valid_matches(target, year)
            if ok: source = "TennisMyLife fallback"
        if source is None and target.exists():
            ok, latest = valid_matches(target, year)
            if ok: source = "Existing cached file"
        result[str(year)] = {"source": source or "Unavailable", "latest_tourney_date": latest}
    return result

def refresh_rankings():
    raw = RAW / "atp_rankings.csv"
    out = ROOT / "data" / "current_rankings.csv"
    chosen = None
    for url in (f"{JEFF}/atp_rankings_current.csv", f"{JEFF}/atp_rankings_20s.csv"):
        if fetch(url, raw):
            try:
                cols = set(pd.read_csv(raw, nrows=3).columns)
                if {"ranking_date","ranking","player_id"}.issubset(cols):
                    chosen = url
                    break
            except Exception:
                pass
    if chosen is None:
        return {"source": None, "ranking_date": None}
    df = pd.read_csv(raw, low_memory=False)
    df["ranking_date"] = pd.to_numeric(df["ranking_date"], errors="coerce")
    df["ranking"] = pd.to_numeric(df["ranking"], errors="coerce")
    df["player_id"] = df["player_id"].astype(str)
    df = df.dropna(subset=["ranking_date","ranking","player_id"])
    latest = int(df["ranking_date"].max())
    cols = ["ranking_date","ranking","player_id"]
    if "ranking_points" in df.columns: cols.append("ranking_points")
    cur = df[df.ranking_date == latest].sort_values("ranking").drop_duplicates("player_id")[cols]
    cur.to_csv(out, index=False)
    return {"source": chosen, "ranking_date": str(latest), "players": len(cur)}

def main():
    matches = refresh_matches()
    rankings = refresh_rankings()
    subprocess.check_call([sys.executable, str(ROOT/"src"/"train_model.py")])
    meta = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matches": matches,
        "rankings": rankings,
    }
    (ROOT/"data"/"freshness.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()

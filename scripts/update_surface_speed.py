from __future__ import annotations
import argparse
import io
from pathlib import Path
import pandas as pd
import requests

URL = "https://www.tennisabstract.com/cgi-bin/surface-speed.cgi?year={year}"
HEADERS = {"User-Agent": "ATP-model research updater/1.0"}

def fetch_year(year: int) -> pd.DataFrame:
    url = URL.format(year=year)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    table = next(
        t for t in tables
        if {"Date", "Tournament", "Surface", "Ace%", "Surface Speed"}.issubset(t.columns)
    )
    table = table.rename(columns={
        "Date": "event_date", "Tournament": "tournament",
        "Surface": "surface", "Ace%": "ace_pct",
        "Surface Speed": "surface_speed",
    })
    table["season"] = year
    table["event_date"] = pd.to_datetime(table["event_date"], errors="coerce").dt.date
    table["ace_pct"] = (
        table["ace_pct"].astype(str).str.replace("%", "", regex=False)
    )
    table["ace_pct"] = pd.to_numeric(table["ace_pct"], errors="coerce")
    table["surface_speed"] = pd.to_numeric(table["surface_speed"], errors="coerce")
    table["source_url"] = url
    return table[
        ["season","event_date","tournament","surface","ace_pct",
         "surface_speed","source_url"]
    ].dropna(subset=["tournament","surface_speed"])

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2012)
    p.add_argument("--end-year", type=int, default=pd.Timestamp.today().year)
    p.add_argument("--output", default="data/tournament_surface_speed.csv")
    args = p.parse_args()

    frames, failures = [], []
    for year in range(args.start_year, args.end_year + 1):
        try:
            frames.append(fetch_year(year))
            print(f"Downloaded {year}")
        except Exception as exc:
            failures.append((year, str(exc)))
            print(f"WARNING {year}: {exc}")

    if not frames:
        raise RuntimeError("No yearly tables were downloaded.")

    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["season","tournament"], keep="last")
        .sort_values(["season","event_date","tournament"])
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {len(out):,} rows to {path}")
    if failures:
        print("Failed years:", failures)

if __name__ == "__main__":
    main()

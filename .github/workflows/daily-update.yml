name: Daily ATP multi-source refresh

on:
  workflow_dispatch:
  schedule:
    - cron: "17 9 * * *"

permissions:
  contents: write

concurrency:
  group: atp-daily-update
  cancel-in-progress: false

jobs:
  update:
    runs-on: ubuntu-latest
    timeout-minutes: 45

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Download, merge, and retrain
        run: python scripts/update_data.py

      - name: Run smoke tests on refreshed model
        run: pytest -q

      - name: Verify deployable artifacts
        run: |
          test -s data/generated/player_state.csv.gz
          test -s data/generated/freshness.json
          test -s data/generated/metrics.json
          test -s model/model.joblib

          python - <<'PY'
          import json
          from pathlib import Path
          import pandas as pd

          root = Path(".")

          state = pd.read_csv(
              root / "data/generated/player_state.csv.gz"
          )

          assert len(state) > 100
          assert state["overall_elo"].notna().all()

          freshness = json.loads(
              (root / "data/generated/freshness.json").read_text()
          )

          assert freshness["matches"]["master_rows"] > 1000

          print("Player-state rows:", len(state))
          print(
              "Latest match:",
              freshness["matches"]["latest_tourney_date"],
          )
          print(
              "Selected source rows:",
              freshness["matches"]["selected_rows_by_source"],
          )
          print(
              "Ranking date:",
              freshness.get("reference", {})
              .get("rankings", {})
              .get("ranking_date", "unavailable"),
          )
          PY

      - name: Commit refreshed model and outputs
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

          git add \
            data/generated/player_state.csv.gz \
            data/generated/freshness.json \
            data/generated/metrics.json \
            model/model.joblib

          if [ -f data/generated/current_rankings.csv ]; then
            git add data/generated/current_rankings.csv
          fi

          git diff --cached --quiet || \
            git commit -m "Daily ATP multi-source model refresh"

          git push

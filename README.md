
# ATP Professional Multi-Source Prediction Model

This repository uses three complementary public tennis datasets:

1. **Jeff Sackmann tennis_atp** — historical ATP matches, players and rankings.
2. **TennisMyLife TML-Database** — recent/corrected ATP match rows and freshness layer.
3. **Jeff Sackmann Match Charting Project** — charted tactical and point-level aggregates.

## What the daily job does

`python scripts/update_data.py`:

- Downloads Jeff Sackmann and TennisMyLife yearly ATP match files.
- Normalizes player IDs and schemas.
- Deduplicates matching rows.
- Keeps the more statistically complete row; TennisMyLife wins exact completeness ties.
- Downloads a separate current ranking table.
- Downloads ATP player reference data.
- Downloads Match Charting overview, match metadata and net-point tables.
- Builds a compressed master match history.
- Recalculates:
  - Overall Elo
  - Hard, clay and grass Elo
  - Serve and return ratings
  - Last-5 and last-10 form
  - Surface form
  - Opponent strength
  - Performance versus Elo expectation
  - Rest and recent workload
  - Recent Elo change
  - Charted serve/return, winner/error and net profiles
- Trains the chronological logistic-regression model.
- Evaluates accuracy, log loss and Brier score.
- Generates current player state and source freshness metadata.

## GitHub installation

Upload every file and folder in this package to the root of your existing repository.
When GitHub asks whether to replace files, replace them.

Then open:

`Actions → Daily ATP multi-source refresh → Run workflow`

The initial run can take longer because it downloads the historical archive. The generated
raw/master match files are intentionally not committed by the workflow; only the compact model,
current state, current rankings, metrics and freshness report are committed.

## Generated outputs

- `data/generated/master_matches.csv.gz`
- `data/generated/current_rankings.csv`
- `data/generated/player_state.csv.gz`
- `data/generated/metrics.json`
- `data/generated/freshness.json`
- `model/model.joblib`

## Important limitations

The public repositories are not guaranteed to update immediately after every completed match.
The app displays freshness dates so stale inputs are visible. Match Charting coverage is selective,
so tactical features include availability and match-count indicators rather than pretending every
player has equal coverage.

This project estimates probabilities; it does not guarantee betting outcomes.


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


## Production upgrade

The pipeline now includes:

- Validation of downloaded content and required columns.
- Explicit source availability and contribution counts.
- Data-age thresholds and stale-data warnings.
- A `Data Health` Streamlit page.
- Daily source and model-performance history.
- GitHub quality gates that fail on suspiciously perfect model metrics.
- A corrected train/test construction that keeps match context unchanged when player order is flipped.

### Important source status

The historical `JeffSackmann/tennis_atp` repository currently returns HTTP 404.
The pipeline no longer performs dozens of known-failing requests. It uses TennisMyLife for
match results and the Match Charting Project for tactical data. Set the repository secret
`JEFF_ATP_BASE_URL` if a working mirror or restored endpoint becomes available.

For a proper live ranking CSV, set the optional repository secret:

`ATP_RANKINGS_CSV_URL`

The CSV must contain:

- `ranking_date`
- `ranking`
- `player_id`
- optionally `ranking_points`

Without that secret, ranking defaults are derived from the latest observed ranking in the match
history and are clearly labeled as a fallback in the app.


## Model, CLV, and live-ranking upgrade

### Model selection
Every refresh trains both logistic regression and histogram gradient boosting. The
chronological holdout winner is selected by lowest log loss, with Brier score as the
tie-breaker. The final selected model is then retrained using all available match rows.

### Closing-line value
The tracking database now stores both closing prices and calculates:

- Price CLV: opening odds / closing odds - 1
- Probability CLV: closing no-vig probability - opening no-vig probability
- Percentage of tracked bets that beat the closing price

Existing databases are migrated automatically when the app starts.

### Live ATP rankings
The updater first attempts to parse the official ATP singles rankings page. It maps official
player names to the ATP IDs found in the match database. If that fails, it uses the configured
ATP_RANKINGS_CSV_URL; if neither succeeds, it falls back to each player's latest observed
ranking and displays a warning.

## Tournament speed and head-to-head features

The production model now includes leakage-safe tournament court speed and head-to-head features.

- `court_speed` uses the latest available edition strictly before the prediction season.
- `h2h_edge` is a Beta-shrunk overall head-to-head advantage.
- `h2h_surface_edge` is the same calculation on the selected surface.
- `log_h2h_matches` lets the model distinguish a 1-match record from a deep rivalry.
- `data/generated/head_to_head.csv.gz` is rebuilt during training.
- `data/tournament_surface_speed.csv` is refreshed by `scripts/update_surface_speed.py`.

The Streamlit app includes a tournament selector and displays the speed rating and H2H record used for each prediction.

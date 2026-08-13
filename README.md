# ATP Professional Probability & Value Model — v4 Matchstat

This version rebuilds the production model around **chronological, leakage-safe player state** and uses Matchstat as the fresh-data layer while retaining the historical match archive already in the repository.

## One-time GitHub setup

Before running the workflow, rotate the RapidAPI key if it has ever been exposed and save the replacement only as a GitHub Actions secret:

`Settings → Secrets and variables → Actions → New repository secret`

Name it exactly:

`MATCHSTAT_API_KEY`

Do not paste the key into source code, screenshots, issues, or commits. `ATP_RANKINGS_CSV_URL` remains optional because Matchstat rankings are now the preferred ranking source.

Then run:

`Actions → Daily ATP production refresh → Run workflow`

The daily workflow fails early if `MATCHSTAT_API_KEY` is missing, so the deployed model cannot silently fall back to stale data.

## What changed in v4

### 1. Matchstat fresh-data layer

The updater now pulls current ATP singles rankings and current-season completed matches for the top players. It paginates Matchstat responses, deduplicates the same match returned through multiple player histories, maps Matchstat player IDs back to the historical IDs already used by the Elo system, and preserves Matchstat-only players with stable `ms:<id>` identifiers.

Matchstat is preferred over TennisMyLife when both providers contain the same match because it supplies richer current match statistics. Missing advanced values remain missing; they are never converted to zero.

### 2. Recent player form

The trained feature vector now contains rolling and exponentially weighted information available **before each match**:

- win rates over the last 3, 5, and 10 matches
- surface-specific last-10 form
- service-points-won over the last 1, 3, 5, and 10 matches
- return-points-won over the last 1, 3, 5, and 10 matches
- first-serve %, first-serve points won, and second-serve points won
- ace and double-fault rates
- total-point share
- break-point save and conversion rates
- opponent-strength-adjusted recent performance
- recent Elo change, rest days, and match workload

### 3. H2H is inside the model

The old manual post-prediction H2H probability adjustment is removed. H2H is now part of the trained feature vector using only prior meetings:

- sample-size-shrunk overall H2H edge
- sample-size-shrunk surface H2H edge
- H2H service-points-won differential
- H2H second-serve differential
- H2H break-point-conversion differential
- log sample-size features

`data/generated/head_to_head.csv.gz` is rebuilt every training run.

### 4. Court speed is rebuilt

The old compressed court-speed scale is replaced by a wider neutral-at-1.00 scale. Training uses only historical information available before the match. Prediction combines:

- a historical tournament-speed prior
- a live current-edition estimate from matches already completed at that event
- sample-size shrinkage toward the prior when the current event has little data

Court speed also interacts with player style: surface Elo, serve quality, return quality, ace rate, second-serve performance, and recent point share. The external Tennis Abstract file and the live empirical table are stored separately so one no longer overwrites the other.

### 5. Tournament/context features

Tournament level is now genuinely distinct:

- Challenger = 1
- ATP 250 = 2
- ATP 500 = 3
- Masters 1000 = 4
- ATP Finals = 4.5
- Grand Slam = 5

The model also receives best-of-3/best-of-5, indoor context, rest/workload, and level interactions instead of treating tournament category as an almost inert display field.

### 6. Optional advanced statistics

When Matchstat populates them, the pipeline incorporates winners, unforced errors, net success, and first-serve speed. Coverage is explicitly tracked, so matches without those fields do not become fake zero-error or zero-winner performances.

### 7. Market/odds safety

The Streamlit app still accepts the current decimal prices you want to bet and reports no-vig market probability, edge, EV, fair odds, and uncapped quarter Kelly.

`src/atp_model/odds.py` also contains leakage-safe helpers for historical Matchstat odds: a quote is eligible for a pre-match backtest only when its timestamp is **strictly before** the scheduled match start. If a trustworthy final pre-match quote is unavailable, the explicit opening quote can be used as a safe fallback. Live/end prices must never be treated as closing pre-match prices.

## Daily pipeline

`python scripts/update_data.py` now:

1. refreshes/caches the historical public match archive when available;
2. pulls Matchstat rankings and current-season player histories;
3. deduplicates providers with Matchstat priority;
4. writes the compressed `master_matches.csv.gz` used for training;
5. refreshes current rankings;
6. refreshes external court-speed data when available;
7. trains the v4 chronological model;
8. rebuilds H2H, empirical court speed, player state, metrics, and histories;
9. runs automated tests before GitHub commits deployable artifacts.

The workflow also commits `master_matches.csv.gz`, fixing the previous state where the deployed model and deployed master dataset could describe different training snapshots.

## Generated production artifacts

- `data/generated/master_matches.csv.gz`
- `data/generated/matchstat_current_rankings.csv`
- `data/generated/current_rankings.csv`
- `data/generated/player_state.csv.gz`
- `data/generated/head_to_head.csv.gz`
- `data/generated/tournament_surface_speed_empirical.csv`
- `data/tournament_surface_speed_external.csv`
- `data/generated/metrics.json`
- `data/generated/freshness.json`
- `data/generated/source_history.csv`
- `data/generated/model_history.csv`
- `model/model.joblib`

## Validation

Training compares logistic regression with histogram gradient boosting on a chronological holdout and selects the lower log loss (Brier score tie-breaker). Reported diagnostics include accuracy, log loss, Brier score, ROC AUC, and 10-bin expected calibration error.

All match features are captured before that match updates Elo, rolling statistics, H2H, or live event speed. Retirements/walkovers are excluded from the model target and state updates.

The app also computes data age dynamically when it loads. If the scheduled workflow stops running, the stale-data warning continues to age instead of remaining frozen at the last successful refresh.

This project estimates probabilities and betting value; it does not guarantee profitable outcomes.

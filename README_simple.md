# ATP Professional Probability & Value Model

This project is a pre-match ATP tennis model that estimates each player's probability of winning and compares that probability with the betting market to identify potential value.

## What the model uses

The model combines several types of information available before a match:

- overall Elo and surface-specific Elo
- current ATP ranking and ranking points
- recent form over the last 1, 3, 5, and 10 matches
- serve and return performance
- first-serve percentage and first/second-serve points won
- ace and double-fault rates
- break-point performance and total-point share
- opponent-strength-adjusted recent performance
- head-to-head history, including surface-specific H2H
- tournament level and match format
- rest, workload, and recent Elo movement
- tournament court speed and how that speed interacts with each player's style
- winners, unforced errors, net performance, and serve speed when those statistics are available

All historical features are built chronologically so the model only uses information that would have been known before the match being predicted.

## What the model produces

For a selected matchup, the app estimates:

- win probability for each player
- fair decimal odds
- bookmaker no-vig market probability
- model edge versus the market
- expected value (EV)
- uncapped quarter-Kelly stake recommendation

The model can use current Pinnacle or other bookmaker odds entered in the app to compare its own probability with the market.

## Data updates

The data pipeline refreshes ATP match results, rankings, player statistics, recent form, head-to-head records, Elo ratings, and court-speed information. Matchstat is used as the main fresh-data source, while the historical match archive is retained for long-term training.

## Model training

The model is trained and evaluated chronologically to reduce look-ahead bias. Candidate models are compared using probability-focused metrics such as log loss and Brier score, along with accuracy, ROC AUC, and calibration.

## Bet tracking

The app includes a tracking section for saved predictions and bets. It can track results, profit/loss, ROI, bankroll, model edge, closing-line value, and performance by factors such as surface and edge size.

This model is designed to estimate probabilities and betting value. It does not guarantee profitable outcomes.

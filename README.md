
# ATP Professional Probability & Value Model

A production-style ATP match modelling application with automatic data refresh, a calibrated probability workflow, betting-value calculations, prediction tracking and optional live odds ingestion.

## What is included

- Overall and surface-specific dynamic Elo
- Exponentially weighted serve and return ratings
- Last-5, last-10 and surface recent form
- Opponent-quality and performance-versus-expectation measures
- Rest and recent match-load variables
- Chronological model validation
- Pinnacle/no-vig comparison
- Expected value, fair odds and fractional Kelly reference
- SQLite by default; PostgreSQL through `DATABASE_URL`
- Prediction, ROI and closing-line tracking
- Optional current odds feed through The Odds API
- Docker and Docker Compose
- Daily GitHub Actions updater
- Automated smoke tests

## Immediate use

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows activation:

```text
.venv\Scripts\activate
```

The app works immediately using the bundled bootstrap model and player state.

## Automatic updating

The workflow at `.github/workflows/daily-update.yml` downloads available ATP files, retrains chronologically, validates output files and commits the refreshed state.

In GitHub:

1. Settings → Actions → General
2. Select **Read and write permissions**
3. Actions → **Daily ATP data and model update**
4. Run workflow

## Streamlit Community Cloud

Deploy `app.py` from the repository. Optional secrets:

```toml
DATABASE_URL = "postgresql+psycopg://..."
ODDS_API_KEY = "..."
ODDS_BOOKMAKER = "pinnacle"
```

Without `DATABASE_URL`, the app uses SQLite. Without `ODDS_API_KEY`, odds are entered manually.

## PostgreSQL

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## Data and odds caveats

The updater tries the live TML ATP repository and falls back to Jeff Sackmann's ATP repository. Publication can be delayed. The dashboard displays the latest included match date.

Pinnacle closed general public access to its own API in 2025. The optional integration therefore uses an external odds provider and only works when the provider and your plan expose Pinnacle tennis prices.

Unforced-error and shot-charting fields are not used by the core model because public Match Charting Project coverage is incomplete and non-random. They should be treated as an optional secondary layer, not required inputs.

This is probabilistic analysis, not a guarantee of profit.

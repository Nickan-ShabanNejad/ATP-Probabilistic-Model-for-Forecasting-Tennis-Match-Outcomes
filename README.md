[README.md](https://github.com/user-attachments/files/32691415/README.md)
# ATP Value Board — v0.4

Fair odds for every ATP 250 / 500 / Masters 1000 / ATP Finals / Grand Slam match, compared with Pinnacle, with capped fractional-Kelly stakes.

```
streamlit run app.py           # needs MATCHSTAT_API_KEY (+ PINNODDS_API_KEY) in secrets
ATP_DEMO=1 streamlit run app.py   # offline demo with SYNTHETIC prices (clearly badged)
python scripts/train_model.py     # retrain + market benchmark (~8 min)
pytest -q                         # 37 tests
```

## How good is the model? (walk-forward, 2018–2025, 18,600 matches)

Each season is predicted by a model trained only on earlier seasons, then compared with tennis-data.co.uk **Pinnacle closing odds** for the same matches.

| | Log loss (lower = better) | Accuracy |
|---|---|---|
| Model (v0.4) | 0.6055 | 66.1% |
| Pinnacle closing price | **0.5856** | **68.0%** |

| Betting rule, replayed at Pinnacle closing prices | Bets | Share of matches | Claimed EV | **Actual ROI** |
|---|---|---|---|---|
| v0.3 default: model only, EV ≥ 2% & edge ≥ 2% | 13,370 | 72% | +21.8% | **−5.7% ± 1.1%** |
| Model only, EV ≥ 5% & edge ≥ 5% | 9,103 | 49% | +27.8% | **−4.3% ± 1.3%** |
| v0.4 default: market-anchored, EV ≥ 2% | 2 | ~0% | +3.7% | n/a (too few) |

What this means: the market is more accurate than the model, and when the model disagrees with Pinnacle's closing price, Pinnacle is usually right. The model's measured weight on top of the market is **0.00** (unconstrained estimate −0.05 ± 0.04). Most of the old "value bets" were model error, not market error.

The full table, a season-by-season chart and the backtest update automatically on **Model & data health** after every retrain.

## How bets are decided now

* **Market-anchored (default).** The fair probability is `sigmoid(a·logit(model) + b·logit(Pinnacle no-vig))`, with `a, b` fitted walk-forward on historical closing odds (`a` is never allowed below 0). Currently `a = 0`, `b = 1.056`: that's Pinnacle's price with its favourite–longshot bias removed. A bet is called only when a price beats that fair value by the minimum EV. Against Pinnacle itself this rarely happens. The intended use is **line shopping**: open a match, type your sportsbook's price into **Check your price**, and the app says whether it's value and how much to stake.
* **Model only (experimental).** The raw model is used, as in v0.3. It's kept for research and CLV tracking and shows a warning with its backtest.
* **Stakes.** Fractional Kelly (default ¼), **capped at 2% of bankroll per bet** (adjustable). v0.3 had no cap.

## What changed in v0.4

### Bugs fixed
1. **Empty board.** The board filtered by calendar date in Toronto time. Asian events (Beijing, Tokyo, Shanghai…) are played overnight, so tomorrow's matches were hidden and today's had already started. The default is now **Next 48 hours**. When nothing shows, the board says why in plain English (started, other day, Challenger, unknown event…).
2. **Duplicate matches.** TennisMyLife stores the tournament *start* date and Matchstat the *match* date, so the cross-provider key never matched: **668 matches in 2026 were counted twice** (Elo, form and H2H moved twice).
3. **ITF / Challenger / qualifying rows in 2026 only** (7,454 rows). The 2000–2025 archive has none, so 2026 form and ratings were not comparable with what the model was trained on. They're removed now.
4. **Wrong tournament levels.** Matchstat labels most events "A", so Barcelona, Rotterdam, Queen's, Dubai and others were treated as ATP 250s (1,178 rows). Levels now come from the curated archive, with overrides for recent category changes.
5. **"US Men's Clay Court Championship – Houston" was treated as the US Open.** It got Grand Slam context and US Open court speed. Also fixed: Paris Masters being hijacked by Roland Garros, and Next Gen Finals / Davis Cup Finals being merged with the ATP Finals.
6. **Fake long-shot value.** Probabilities were clamped to 5–95%, so a player the model rated at 2% was priced as 5% and a 30.0 price looked like +50% EV. The clamp is now 1–99%.
7. **Slow board.** Every prediction made 16 separate gradient-boosting calls: about 25 s per match, so a 12-match slate took about 5 minutes on every 30-second refresh. Calls are batched now: **0.17 s per match (~150× faster)**.
8. **Sponsor names didn't resolve.** "China Open", "Japan Open", "Erste Bank Open", "Rolex Paris Masters", "Nitto ATP Finals", "Open 13 Provence" and similar names now map to the right event.
9. **Grand Slam sets model.** It's only slightly better than a coin flip (holdout log loss 0.678 vs 0.693), so it's now shrunk halfway to the market price and its stakes are capped.

### Model
* Data-quality layer (`src/atp_model/data_quality.py`) used by the updater, the trainer and the live board.
* New ratings: experience-dependent K-factor Elo (K = 150/(n+5)^0.3), margin-of-victory weighting by game share, a Slam multiplier, and a blended overall/surface rating with a best-of-5 interaction.
* Symmetric training (each match seen in both player orders), stronger regularisation, and model selection on a chronological holdout.
* Walk-forward market benchmark, market blend and betting backtest, all written to `metrics.json` (`market_benchmark`, `market_blend`, `backtests`, `backtest_by_year`).
* Current-season check: on 1,866 matches in 2026 against the market-average close, log loss went from 0.6206 (v0.3 pipeline) to 0.6169 (v0.4). Most of that comes from the data cleaning.

### App
* New dark theme, sidebar settings, KPI tiles, a **Recommended bets** card row, and matches grouped by tournament with win-probability bars, Pinnacle vs fair odds and a verdict chip.
* **Check your price** line-shopping calculator on every match.
* **Model & data health** opens with the honest model-vs-market scoreboard.
* Friendly page names (`app.py` is now a small router; the board lives in `board.py`).
* `ATP_DEMO=1` runs the whole app offline with synthetic, clearly badged prices.

### Pipeline
* `data/odds/tennis_data_atp.csv.gz`: closing odds 2010–2026 from tennis-data.co.uk. The daily job refreshes the current season on a best-effort basis, and the odds are used only for evaluation, never as features. Check tennis-data's terms before any commercial use.
* The workflow asserts pipeline version `6.0-v04-clean-data-dynelo-market-blend` and commits the odds file.

## Secrets

```toml
MATCHSTAT_API_KEY = "…"   # required: schedule, results, rankings
PINNODDS_API_KEY  = "…"   # recommended: direct Pinnacle prices
SUPABASE_URL = "…"        # optional: shared tracking
SUPABASE_KEY = "…"
```

GitHub Actions also needs `MATCHSTAT_API_KEY` (and the Supabase / PinnOdds keys for the settlement job).

## Honest limits

* Closing prices are the hardest benchmark. Earlier prices and soft sportsbooks are beatable more often, but we have no historical data to prove that. The live proof is **closing-line value (CLV)** on the Tracking page: log every bet and check whether the price you took beats Pinnacle's close.
* This tool estimates probabilities. It does not guarantee profit.

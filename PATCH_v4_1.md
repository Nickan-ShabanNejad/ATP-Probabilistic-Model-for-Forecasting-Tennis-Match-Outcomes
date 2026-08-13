# v4.1 patch — H2H smoke test + production tracking

This patch is intended to replace the first v4 package.

## H2H safety fix

- Directional H2H features are neutral until there are at least 4 prior meetings.
- A 1-0, 2-0, or 3-0 record is still displayed in the app, but it is not used as predictive edge.
- With 4+ meetings, the record and matchup-stat differences enter the model with sample-size shrinkage.
- Training and prediction-time H2H logic use the same threshold.
- The failing `test_small_h2h_samples_are_not_used_as_predictive_edge` case is covered.

## Tracking page fix / upgrade

- Implemented `get_predictions()` and `settle_prediction()` that the Tracking page requires.
- Added backward compatibility for old JSONL tracking rows.
- Added realized P/L, ROI, record, average edge, CLV, bankroll, peak/drawdown calculations.
- Added performance breakdowns by surface and model-edge bucket.
- Added CSV backup and restore/merge.
- Added optional starting bankroll.
- Tracking rows are kept in `data/tracking/` and ignored by Git so private betting history is not committed.

## Other fixes

- Fixed the Value Board page repo-root path.
- Removed the scikit-learn feature-name warning in prediction smoke tests by passing numpy arrays to the fitted pipeline.
- Local test suite: 9 passed.

After uploading this patch, rerun the Daily ATP production refresh so the model is retrained with the same H2H threshold used at prediction time.

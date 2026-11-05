# Retraining design

## Goal
Refresh the churn model monthly (or on a drift alert) without weakening evaluation integrity.

## Timeline per snapshot
`... history | features: before cutoff | cutoff = as_of - 90d | label: no order in (cutoff, as_of]`
- **Label latency:** retraining on date T uses `as_of = T`, so the newest labelled snapshot has cutoff T-90.
- **Scoring today** uses cutoff = T with no label. Features are relative to the cutoff, so both are consistent.
- `build_customer_dataset` refuses an `as_of` beyond the data: otherwise a late data feed would silently label everyone as churned.

## Pipeline (src/pipeline/run_pipeline.py)
clean -> features -> split -> challenger CV -> report. Dry-run by default. Every run writes only to
`pipeline_runs/<run_id>/`. Cache keys include file hashes and the config. Failed data checks stop the run and skip downstream steps.

## Champion / challenger
The champion is the pre-registered model in `app_artifacts/`. A challenger is the same recipe (family + hyperparameters)
refit on the new snapshot. Changing the family or tuning is a new decision and needs a new pre-registration file.

## Evaluation
- Cross-validation on train only, inside the pipeline.
- The random holdout of one snapshot cannot be reused after retraining, because the same customers reappear with new features.
  Proper evaluation is time-based: train on older snapshots, evaluate on the newest one.
- Live check: store predictions at scoring time and compare them with the real outcome 90 days later.

## Human gates (not automated)
1. Model selection / hyperparameter search. 2. Holdout evaluation. 3. Promotion to `app_artifacts/`
(then `export_artifacts`, which records hashes and library versions). Rollback = revert the artifact commit.

## Failure handling
Retries only for transient failures (e.g. tracking server). Data-check failures fail fast, with no retry.

## Known limits
The dataset is static (2009-2011), so scheduled runs are simulated with `--as-of` backfills. Cost assumptions are placeholders.
Same customers appear in several snapshots, so splits across snapshots must be time-ordered, never random.
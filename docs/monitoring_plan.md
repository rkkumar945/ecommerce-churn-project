# Monitoring plan

## What can be monitored, and when
| Signal | Needs labels? | Delay | Use |
|---|---|---|---|
| Data quality (nulls, ranges, schema) | No | none | Pipeline checks (`src/pipeline/checks.py`) |
| Feature drift (PSI, KS/chi-square) | No | none | Early warning |
| Score drift + alert rate | No | none | Shows what the campaign list will look like |
| Model performance (AUC, precision@k) | Yes | 90 days | Truth, but late: computed on predictions made 90+ days ago |

## Rules
- PSI > 0.10 = watch, > 0.25 = alert (rules of thumb). p-values alone are not used: with thousands of rows they flag tiny shifts.
- Policy: ALERT only if score PSI > 0.25 or >= 2 key features (top by SHAP) are in alert. `Tenure` is excluded: it drifts by construction as the cutoff moves.
- Sanity: an A/A split of the same snapshot must stay quiet (covered by tests).

## On an alert
1. Check upstream first: schema, null spikes, a changed data feed.
2. Check score drift and alert rate: is the campaign list size changing?
3. Run the retraining pipeline (challenger CV). Nothing is auto-promoted.
4. When 90-day labels arrive, compare realised precision@k with the expected value.

## Limits
- Static 2009-2011 dataset: drift is simulated with `--as-of` snapshots, not production traffic.
- Seasonality (Q4) looks like drift if snapshots from different seasons are compared; year-over-year comparisons are safer.
- Many features are correlated, so the count of drifted features overstates the number of independent changes.
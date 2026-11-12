# Final project review (Day 100)

## Outcome
The project delivers an end-to-end customer churn ranking workflow, from transaction cleaning and a leakage-aware target through model evaluation, monitoring, and a live app. The held-out test ROC-AUC is 0.818 (95% bootstrap CI 0.791 to 0.843). The ranking is more useful than the alert flag alone: at the selected 0.07 threshold the model flags nearly every customer.

Primary evaluation: `reports/model_comparison_report.md` and `reports/final_model_card.json`.

## Scorecard
| Dimension | Assessment | Evidence |
|---|---|---|
| Data quality and contracts | Cleaning removes guest checkouts, cancellations, and duplicates; pipeline checks protect inputs. | `src/cleaning/clean_data.py`, `src/pipeline/checks.py` |
| SQL and exploration | SQL analysis and EDA are documented in the project notebooks. | `notebooks/month1_sql/18_month1_sql_summary.ipynb`, `notebooks/month3_statistics_eda/63_eda_report_final.ipynb` |
| Modelling correctness | Features precede the cutoff, and the forward-looking target is tested for leakage. | `src/features/split.py`, `tests/test_build_features.py` |
| Evaluation | Model selection uses cross-validation; the held-out results include uncertainty and budget-oriented ranking metrics. | `reports/model_comparison_report.md`, `reports/final_model_choice_day84.md` |
| MLOps | The repository includes a retraining pipeline, MLflow tracking, and drift monitoring. | `src/pipeline/run_pipeline.py`, `src/tracking/mlflow_utils.py`, `src/monitoring/drift.py` |
| Deployment and communication | A live app is supported by deployment guidance and a documented live check. | `docs/deployment.md`, `reports/live_check_day96.md` |

## Strengths
- The target predicts no order in the 90 days after the cutoff, using only pre-cutoff transaction features.
- The final model choice is recorded separately from held-out evaluation.
- The README distinguishes descriptive EDA results from forward-looking model results and explains the operating-point limitation.
- Reproducibility and operational topics have dedicated design documentation.

## Limitations and next steps
- The available history is from one UK-based retailer over a limited period; performance should not be assumed to transfer to another retailer or later market conditions.
- The test set is small relative to the transaction history because the modelling unit is a customer. Keep uncertainty intervals alongside point estimates.
- The 0.07 alert threshold has very high recall but low selectivity. Treat the output primarily as a ranking and choose a contact budget with stakeholders before using alerts operationally.
- Validate the data feed, drift thresholds, and retraining process on future data before relying on automated decisions.

## Resume summary
- Built a customer churn ranking pipeline using SQL, pandas, and gradient-boosted trees, with a forward-looking 90-day target and leakage checks.
- Compared model families using cross-validation and reported held-out ROC-AUC with bootstrap confidence intervals.
- Added experiment tracking, a retraining pipeline, drift monitoring, and a Streamlit app with customer-level explanations.

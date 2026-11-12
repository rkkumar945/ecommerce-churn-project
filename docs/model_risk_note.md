# Model risk note

## Intended use
This model ranks existing customers by the likelihood of not ordering in the 90 days after a cutoff. It is a decision-support tool for prioritizing retention outreach, not a causal estimate of campaign impact and not a substitute for human review. See `README.md` and `reports/final_model_card.json`.

## Key risks and controls
| Risk | Why it matters | Current control / recommended action |
|---|---|---|
| Temporal leakage | Features or splits that use information after the prediction cutoff can make offline results misleading. | Features are built before the cutoff and the split/feature behavior is tested; rerun those checks whenever the target or pipeline changes. See `src/features/split.py` and `tests/test_build_features.py`. |
| Limited representativeness | The data covers one UK retailer and a historical period, so customer behavior and performance may differ elsewhere or over time. | Treat results as project-specific; validate on a later time period and on the intended deployment population before use. |
| Alert threshold and capacity | The selected 0.07 threshold flags nearly all customers, so the binary alert is not a practical contact list by itself. | Use the ranking with an explicit outreach budget; evaluate precision-at-k and campaign outcomes rather than interpreting the alert as a guaranteed churn event. See `reports/model_comparison_report.md`. |
| Drift and stale models | Changes in customers, products, or transaction processing can make scores less reliable. | Monitor input and score distributions, investigate alerts, and retrain only through a reviewed process. See `src/monitoring/drift.py` and `docs/monitoring_plan.md`. |
| Incorrect or incomplete data | Missing identifiers, cancellations, or schema changes can distort features and predictions. | Preserve the cleaning and pipeline data-contract checks, and stop the run when required checks fail. See `src/cleaning/clean_data.py` and `src/pipeline/checks.py`. |
| Human and business impact | Outreach decisions can be inefficient or unfair if scores are treated as certainty or used without outcome monitoring. | Keep a human decision-maker, document the contact policy, and compare outcomes across relevant customer groups before expanding use. |

## Monitoring and review
Review discrimination and calibration where labels permit, ranking performance at the actual contact budget, data quality, drift alerts, and campaign outcomes. Reassess the model and its threshold when the customer population, data feed, or retention policy changes. Operational guidance is in `docs/monitoring_plan.md` and `docs/deployment.md`.
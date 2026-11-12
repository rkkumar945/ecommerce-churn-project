# Final project review (Day 100)

## Master test result
Total <<FILL>>/100 (A <<FILL>>, B <<FILL>>, C <<FILL>>, D <<FILL>>, E <<FILL>>, F <<FILL>>). Questions I peeked on: <<FILL>>.
Details: `reports/master_test_day100.md`

## Scorecard (0 to 5)
| Dimension | Score | Evidence | Weakness I still see |
|---|---|---|---|
| Data cleaning and contracts | <<FILL>> | `src/cleaning/clean_data.py`, `src/pipeline/checks.py` | <<FILL>> |
| SQL | <<FILL>> | `notebooks/month1_sql/18_month1_sql_summary.ipynb` | <<FILL>> |
| Statistics and EDA | <<FILL>> | `notebooks/month3_statistics_eda/63_eda_report_final.ipynb` | <<FILL>> |
| Modelling correctness (leakage, CV) | <<FILL>> | `tests/test_build_features.py`, `src/features/split.py` | <<FILL>> |
| Evaluation honesty | <<FILL>> | `reports/model_comparison_report.md`, `reports/final_model_choice_day84.md` | <<FILL>> |
| MLOps (tracking, pipeline, monitoring) | <<FILL>> | `src/tracking/mlflow_utils.py`, `src/pipeline/run_pipeline.py`, `src/monitoring/drift.py` | <<FILL>> |
| Deployment and app | <<FILL>> | `docs/deployment.md`, `reports/live_check_day96.md` | <<FILL>> |
| Documentation | <<FILL>> | `README.md`, `docs/retraining_design.md` | <<FILL>> |
| Communication | <<FILL>> | `docs/interview_prep.md` | <<FILL>> |

## Skills, Day 1 to Day 100
Level: 0 = never used, 1 = used with docs open, 2 = built it independently, 3 = could teach it.
| Skill | Day 1 | Day 100 |
|---|---|---|
| SQL (joins, windows, CTEs) | <<FILL>> | <<FILL>> |
| pandas and NumPy | <<FILL>> | <<FILL>> |
| Statistics (tests, effect sizes) | <<FILL>> | <<FILL>> |
| scikit-learn, XGBoost, evaluation | <<FILL>> | <<FILL>> |
| SHAP | <<FILL>> | <<FILL>> |
| MLflow, pipelines, drift monitoring | <<FILL>> | <<FILL>> |
| Streamlit and deployment | <<FILL>> | <<FILL>> |
| Git and testing | <<FILL>> | <<FILL>> |
| Docker, FastAPI, LLMs (not started) | <<FILL>> | <<FILL>> |

## Retrospective
- Three mistakes and what each taught me: <<FILL>>
- Three decisions I would repeat: <<FILL>>
- Three things I would change: <<FILL>>

## Resume bullets (every number must come from `scripts/readme_facts.py`)
- Built an end-to-end customer churn system on 1.07M retail transactions (SQL, pandas, XGBoost, Streamlit), deployed as a live app with SHAP explanations.
- Designed a leakage-free target (features before a cutoff, label = no order in the next 90 days) and enforced it with invariance unit tests; test ROC-AUC <<FILL>> (95% CI <<FILL>>).
- Compared four model families with nested cross-validation and recorded the model choice in a commit before scoring the held-out set; reported bootstrap intervals and precision@k tied to a retention budget.
- Added MLflow tracking, a cached retraining pipeline with data contracts, PSI drift monitoring and hash-checked model artifacts, covered by <<FILL>> automated tests.
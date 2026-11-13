# Master test (executed 2026-11-13)

## Run context
Python 3.14.6 from `.venv-1`. Package versions observed in that environment: pytest 9.1.1, scikit-learn 1.9.1, xgboost 3.4.1, numpy 2.5.4, pandas 3.1.1, joblib 1.6.0, imbalanced-learn 0.14.2 and shap 0.54.1.

Sources: `.venv-1\Scripts\python.exe --version`; package versions queried with Python `importlib.metadata`.

## Executed checklist
| Check | Method | Result (PASS / FAIL / NOT RUN + reason) |
|---|---|---|
| a. Test suite | `.venv-1\Scripts\python.exe -m pytest -q` | PASS — 112 passed, 0 skipped. |
| b. Repository audit | `.venv-1\Scripts\python.exe scripts\repo_audit.py` | PASS — exit code 0; 0 FAIL, 7 WARN and 4 INFO. |
| c. Shipped model hash | SHA-256 of `app_artifacts/final_model.joblib` compared with `app_artifacts/model_manifest.json` | PASS — both are `bb3ed7b66de970c2c93a9a36be3de3d9ad8347cf3d83810c803450d4eaacdb8b`. |
| d. Model-card AUC | Exact numeric comparison of `reports/final_model_card.json` and `app_artifacts/model_manifest.json` `test_auc` | PASS — both are `0.8178241196144009`. |
| e1. README ROC-AUC, interval and PR-AUC | Compared README Results with `reports/model_comparison_report.md` | PASS — 0.818, 95% CI 0.791 to 0.843, and PR-AUC 0.838 agree. |
| e2. README train/held-out counts | Compared README Results with available report artifacts | NOT RUN — the report confirms 1,057 held-out customers, but no report artifact confirms the 4,226 training count. |
| e3. README top-10% precision, lift and reach | Searched the pre-existing evaluation artifacts: `reports/model_comparison_report.md`, `reports/final_model_card.json` and the Day 84 CSVs | NOT RUN — these values are not recorded in those evaluation artifacts, so they cannot be independently matched here. |
| e4. README nested-CV table | Compared rounded README scores with `reports/nested_cv_day84.csv` | PASS — the four README values match the CSV rounded to three decimals. |
| f. Placeholder scan across reports and docs | `rg` scan for common template markers and the requested placeholder phrases | FAIL — matching unfinished template markers remain in `reports/live_check_day96.md`; that file is outside the permitted edit scope. |
| Live app | No live browser check was performed | NOT RUN — NOT VERIFIED here; see `reports/live_check_day96.md`. |

Sources: README.md, Results, Key findings and Limitations; reports/model_comparison_report.md; reports/nested_cv_day84.csv; reports/final_model_card.json; app_artifacts/model_manifest.json; command results from 2026-11-13.

## Live app
NOT VERIFIED here. The previously recorded live-check notes are in [reports/live_check_day96.md](live_check_day96.md).

Sources: reports/live_check_day96.md.
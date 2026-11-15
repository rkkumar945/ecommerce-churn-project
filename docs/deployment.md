# Deployment

- **Live app:** https://ecommerce-churn-project.streamlit.app/
- **Platform:** Streamlit Community Cloud (free tier, no SLA)
- **Source:** GitHub `main`, entrypoint `src/app/app.py`, dependencies in `src/app/requirements.txt`
- **Python:** 3.14 (same minor version as the model build, see `app_artifacts/model_manifest.json`)
- **Deployed commit:** see the About tab of the app (Build details) | **Date:** see the About tab of the app (Build details)

## How updates work
Every push to `main` redeploys the app. Dependency changes trigger a full rebuild. Run `pytest` and the local
app before pushing. `app_artifacts/` is integrity-checked (sha256) and compared with the running library versions in the About tab.

## Release / promotion steps
1. Train and select the model in notebook 86 (`notebooks/month5_advanced_mlops/86_model_comparison_report.ipynb`).
2. The notebook saves the selected model to `notebooks/model/final_model.joblib`.
3. From the repository root, promote that exact file into the app artifacts:
   `python -m src.app.export_artifacts --model-path notebooks/model/final_model.joblib`.
   Without `--model-path`, the exporter checks `models/final_model.joblib` first and
   `notebooks/model/final_model.joblib` second.
4. Run `pytest -q` and verify the app locally before releasing.

## Behaviour to expect
- The app sleeps after about 12 hours without traffic; the first visitor wakes it (can take a minute).
- First customer view is slower because SHAP is imported lazily.
- Free-tier resource limits apply and can change without notice.

## Rollback
`git revert <commit>` and push; Cloud redeploys the previous state.

## Verification
See `reports/live_check_day96.md`.
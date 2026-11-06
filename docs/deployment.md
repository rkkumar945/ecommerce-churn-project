# Deployment

- **Live app:** https://ecommerce-churn-project.streamlit.app/
- **Platform:** Streamlit Community Cloud (free tier, no SLA)
- **Source:** GitHub `main`, entrypoint `src/app/app.py`, dependencies in `src/app/requirements.txt`
- **Python:** <x.y> (same minor version as the model build, see `app_artifacts/model_manifest.json`)
- **Deployed commit:** <hash> | **Date:** <date>

## How updates work
Every push to `main` redeploys the app. Dependency changes trigger a full rebuild. Run `pytest` and the local
app before pushing. `app_artifacts/` is integrity-checked (sha256) and compared with the running library versions in the About tab.

## Behaviour to expect
- The app sleeps after about 12 hours without traffic; the first visitor wakes it (can take a minute).
- First customer view is slower because SHAP is imported lazily.
- Free-tier resource limits apply and can change without notice.

## Rollback
`git revert <commit>` and push; Cloud redeploys the previous state.

## Verification
See `reports/live_check_day96.md`.
# Live check (Day 96)
URL: https://ecommerce-churn-project.streamlit.app/ | commit: <git rev-parse --short HEAD> | Python (Advanced settings): <x.y>

| # | Check | Expected | Result |
|---|---|---|---|
| 1 | URL logged-out window mein khulta hai | app load | Pass |
| 2 | Cold load time (sleep ke baad ya pehli visit) | note karo | Pass |
| 3 | Sidebar: model, AUC, threshold, build date | dikhte hain | Pass |
| 4 | Koi yellow "Library version mismatch" nahi | none | Pass |
| 5 | About > Build details: built_with vs running_on | python minor, scikit-learn, xgboost, numpy same | Pass |
| 6 | 3 noted IDs: score aur Risk rank local ke barabar (2 decimals) | identical | Pass |
| 7 | "Why this score?" chart + text aate hain | no error | Pass |
| 8 | What-if slider: impossible scenario par error message | error shown |Pass |
| 9 | Top at-risk tab: budget slider, "Add main reason" (SHAP), CSV download | works |Pass |
| 10 | Mobile (phone) par layout padhne layak | readable |Pass |

Cloud logs (Manage app): koi traceback / warning? ___
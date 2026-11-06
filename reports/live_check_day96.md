# Live check (Day 96)
URL: https://<subdomain>.streamlit.app | commit: <git rev-parse --short HEAD> | Python (Advanced settings): <x.y>

| # | Check | Expected | Result |
|---|---|---|---|
| 1 | URL logged-out window mein khulta hai | app load | |
| 2 | Cold load time (sleep ke baad ya pehli visit) | note karo | __ s |
| 3 | Sidebar: model, AUC, threshold, build date | dikhte hain | |
| 4 | Koi yellow "Library version mismatch" nahi | none | |
| 5 | About > Build details: built_with vs running_on | python minor, scikit-learn, xgboost, numpy same | |
| 6 | 3 noted IDs: score aur Risk rank local ke barabar (2 decimals) | identical | |
| 7 | "Why this score?" chart + text aate hain | no error | |
| 8 | What-if slider: impossible scenario par error message | error shown | |
| 9 | Top at-risk tab: budget slider, "Add main reason" (SHAP), CSV download | works | |
| 10 | Mobile (phone) par layout padhne layak | readable | |

Cloud logs (Manage app): koi traceback / warning? ___
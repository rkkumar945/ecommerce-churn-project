# ROLE

You are a senior ML engineer and Python code reviewer working inside my VS Code workspace. This repository is an E-Commerce Customer Churn project (XGBoost model + Streamlit app + feature pipeline + notebooks). An independent audit found real bugs and documentation/validation gaps. Your job is to fix them carefully, add tests, and report honestly. Do NOT overclaim.

# CONTEXT (what the audit found)

The saved XGBoost model is real and reproducible:
- Test ROC-AUC = 0.8178241196 (matches manifest exactly), average precision = 0.837692.
- 5,283 customers total, 4,226 train, 1,057 test (stratified random split, seed 42). 21 input features. Test positives (actual non-returners) = 598, negatives = 459, base rate = 56.5752%.
- Top 10% targeting: 106 customers, 98 actual non-returners, precision 92.45%, lift 1.634x, recall 16.39%.
- At legacy threshold 0.07: accuracy 59.98%, recall 99.83%, flags 1,019 of 1,057 customers (96.4%). Majority baseline accuracy is 56.58%. At threshold 0.50: accuracy 75.69%.

IMPORTANT interpretation rules (must be respected everywhere in code, UI text, README and comments):
- AUC 0.818 is NOT "81.8% accuracy".
- Top-10% precision 92.45% is NOT overall model accuracy.
- Model scores are NOT calibrated probabilities (mean score 0.5195 vs observed rate 0.5658). Never present a score of 0.70 as "70% churn chance".
- Predicting churn is not the same as proving that contacting a customer prevents churn.

The project is a useful historical churn-ranking prototype. It is NOT yet production-ready.

# HARD RULES (do not break)

1. Do NOT modify, retrain, overwrite or re-save the committed model artifact or its manifest/hash. Do not change model hyperparameters.
2. Do NOT tune thresholds, features or hyperparameters using the current test split. It has already been exposed.
3. Do NOT invent data, metrics, or results. If something cannot be verified because raw data is missing, say so and add documentation/TODO instead of faking it.
4. Do NOT delete existing tests to make things pass. Do NOT weaken assertions.
5. Make small, reviewable changes. Before editing any file, read it fully. After each fix group, run the tests.
6. Keep all existing public function names/signatures working unless a change is needed; if you change one, update every caller and test.
7. Show me a short plan first, then implement fix by fix. Explain each change in simple language (Hinglish is fine).

# TASKS (in priority order)

## TASK 1 (P1): One shared contact-list population across all views

Files: `src/app/app.py` (around lines 88-95, 262-267, 404-444), `src/app/service.py` (`add_scores`).

Problem: Customer lookup computes contact-list membership using ranks over ALL 5,283 customers, while the default top-list tab ranks only the 1,057 test customers. At a 10% budget, 7 of the 106 customers in the top-list are NOT in the global contact cutoff used by lookup (examples: customer IDs 13942, 13604, 14279).

Required fix:
- Define ONE explicit "active population" (decide it clearly, document it in code and show it in the UI, e.g. "Population: all 5,283 customers" or "Population: test set only (1,057)").
- Compute the selected customer-ID set ONCE (a single function, e.g. `select_contact_ids(df, budget_pct)`), and reuse it in: customer lookup, top-list tab, random-selection comparison, CSV export, and any summary numbers.
- If historical evaluation (test set) and operational list intentionally use different populations, give them different labels in the UI and never present them as the same list.

## TASK 2 (P1): Deterministic tie handling and exact budget

File: `src/app/service.py` (`add_scores` uses `rank(method="min")`), and app code comparing `top_share` with `k/n` while the table uses `sort_values(...).head(k)`.

Problem: Ties break the exact contact budget. Reproduced: 1% budget -> expected k=53 but rank-based membership selects 55; 25% budget -> expected 1,321 but selects 1,326. Other budgets that mismatch (budget%: expected k vs rank count): 4%: 211 vs 212, 7%: 370 vs 376, 12%: 634 vs 636, 15%: 792 vs 793, 22%: 1162 vs 1163, 26%: 1374 vs 1384, 28%: 1479 vs 1488.

Required fix:
- Use one deterministic ordering: sort by score descending, then customer ID ascending.
- Selection is always exactly `head(k)` of that ordering, where `k = round/floor(n * budget)` (pick one rule, document it, use it everywhere).
- Remove `rank(method="min")` based membership.
- If someone wants "include all tied customers", make that an explicit, labeled, documented option, not the default.

Tests to add:
- For every slider budget value used by the app (1..100 or whatever the slider offers): the selected ID set has exactly k elements.
- Cross-view equality test: lookup membership set == top-list set == CSV export set (same IDs).
- Tie-boundary regression test with a small synthetic dataframe containing tied scores at the cutoff.

## TASK 3 (P1/P2): What-if validator must reject impossible scenarios

File: `src/app/service.py` (`whatif_problems`, `apply_whatif`).

Reproduced problems:
- Customer 12347: `Recency = -10` is accepted with no error and `apply_whatif` returns -10. (UI slider blocks negatives, but the reusable service API does not.)
- Customer 12461: Tenure = 64 days, Frequency = 2. Setting `Orders_last90 = 1` and `Orders_prev90 = 1` is accepted, but a 64-day-old customer cannot have an order in the 90-to-180-day-ago window.
- Changing Recency while keeping average/last gaps fixed can break transaction-level consistency; features like OverdueRatio, OrderTrend, recent spend are recomputed but no feasible purchase history is reconstructed.

Required fix:
- Validate in the service layer (not only the UI): non-negative values, integer day/count inputs where appropriate, Recency <= Tenure, order-window constraints against customer age (the prev-90 window only exists if Tenure > 90, etc.), Orders_last90 + Orders_prev90 <= Frequency (total-order allocation), and single-order customer consistency (e.g. Frequency == 1 must have consistent gap features).
- Return clear, human-readable error messages from `whatif_problems`.
- In the UI, label the what-if tool as "partial-feature sensitivity analysis", NOT as an achievable causal intervention. Add a visible note: "Changing a score here does not prove that contacting a customer will prevent churn."
- Optional (only if simple and clean): generate a feasible synthetic transaction history and rebuild features from it. If this is too large, skip it and keep the clear label above.

Tests to add: negative Recency rejected; customer 12461 impossible window scenario rejected; Orders_last90 + Orders_prev90 > Frequency rejected; valid scenarios still accepted.

## TASK 4 (P1 documentation): Stop misleading metric wording

Search the whole repo (README, app text, reports, notebook markdown, docstrings, comments) for any wording that:
- equates AUC with accuracy,
- presents top-k precision (92.45%) as overall accuracy,
- calls scores "probabilities" or "% churn chance" without a calibration caveat,
- claims "impossible inputs fully rejected" (only true after TASK 3 is done and tested),
- says the test set was "scored once" (it is scored more than once computationally; repeated deterministic scoring is not leakage, but the wording is inaccurate — reword to describe what actually happened and note that earlier Month-4 notebooks already looked at the test split),
- calls a standard-deviation margin a "one-SE rule" in notebook 85 while the source uses parameter tuples (make wording consistent with what the code really does).

Fix the wording. Also add a short "Honest limitations" section to the README listing: random customer split at one cutoff (no out-of-time validation), uncalibrated scores, assumed (not measured) business utility, missing raw data, and not production-validated.

## TASK 5 (P1): Threshold and business-utility honesty

- The legacy threshold 0.07 flags 96.4% of customers and gives only a modest accuracy gain over the majority baseline (59.98% vs 56.58%). Add a clear note in the app/README. Do NOT change the default threshold using the current test set.
- The utility formula `40*TP - 5*FP - 50*FN` gives 21,720 at threshold 0.07 vs 21,625 for contacting everyone (advantage only 95 assumed units). Label these everywhere as "assumed units, not measured profit or retention uplift".
- Add a note: model ranking and treatment responsiveness are different; an actual non-returner is not automatically persuadable. Proper measurement needs a randomized campaign holdout (document this as a future step).

## TASK 6 (P2): Calibration support (development data only)

- Add a script/module (e.g. `src/evaluation/calibration.py`) that can fit sigmoid or isotonic calibration using ONLY training/validation predictions (e.g. cross-validated out-of-fold predictions on the training set), and evaluates reliability curve, Brier score and log loss.
- Do NOT fit calibration on the test set. Do NOT overwrite the committed model; save any calibrator as a separate, clearly named artifact.
- Until a calibrated version is validated on a fresh held-out period, keep the UI wording as "risk score / ranking score", not "probability".
- Optionally compare `scale_pos_weight=0.765984` vs `1.0` using training CV only.

## TASK 7 (P2): Cleaning and feature contract cleanup

Files: `src/cleaning/clean_data.py`, `src/features/build_features.py`, `src/pipeline/checks.py`.

- `clean_transactions(exclude_non_product=True)` uses a long non-product code list, but `build_customer_dataset(exclude_non_product=True)` removes only POST. Make them use ONE shared definition.
- Non-product and non-positive-price removal default to False, and non-cancellation negative quantity is not rejected. Decide an explicit documented policy; add validation checks in `checks.py`. IMPORTANT: changing cleaning policy changes customer eligibility, labels and features, so do NOT silently change exported tables or the committed model's inputs. If a policy change is needed, make it opt-in and document that retraining/re-evaluation is required.
- Spend features reflect purchase activity, not audited net revenue after returns. Document this in the feature docs.
- `as_of` currently allows up to one day beyond the latest timestamp (a synthetic as_of 23 hours after the last event is accepted). Enforce a documented inclusive/exclusive boundary and add an explicit data-completeness watermark parameter instead of assuming max observed event == feed completeness.
- Document country/region and window-boundary rules.

Add tests for each rule above.

## TASK 8 (P2): Reproducibility

- The test `test_committed_artifacts_load_and_env_matches` fails only because the audit environment differs from the manifest (Python 3.13.14 vs manifest 3.14.6; different scikit-learn/numpy/joblib). Do NOT "fix" this by loosening the test blindly. Instead: add a pinned environment file (`requirements.lock` / `environment.yml` / `pyproject` pins) matching the manifest, and make the test message clearly say "environment mismatch, not a prediction bug". Keep the strict check available.
- Retraining in a different environment gave AUC 0.819952 and changed some scores (max abs difference 0.068204) and some threshold-0.07 labels. Document: "saved-artifact inference is exactly reproducible; retraining is only reproducible in the pinned environment."
- Provide a single documented command (e.g. `make reproduce` or `python -m src.pipeline.run`) from raw data to export, with data/code hashes. If raw data is not in the repo, document the expected raw-data location and format; do NOT fabricate data.
- Add CI (e.g. GitHub Actions) that runs `PYTHONPATH=. pytest -q` in the pinned environment.
- The exporter depends on notebook-generated files; document or remove that dependency.

## TASK 9: Things you CANNOT fix by code (document only, do not fake)

Create `docs/VALIDATION_TODO.md` listing these open items clearly:
1. Raw transactions (about 1,067,371 rows) and intermediate CSVs are missing, so cleaning -> features -> labels cannot be independently reconciled. Provide a reproducible, versioned data acquisition route.
2. No out-of-time validation. Plan: freeze the current model and protocol, get a new untouched future cohort with mature 90-day outcomes, do rolling-origin evaluation, and decide/document existing-customer vs new-customer evaluation with proper group handling for repeated customers.
3. Calibration must be evaluated on a fresh held-out period.
4. Business impact must be measured via a randomized campaign holdout, with realistic contact cost/margin and confidence intervals.
5. Git history/pre-registration chronology cannot be authenticated from a ZIP; keep the git history in the real repo.
6. Live deployed website was not audited (security/deployment review pending).

# ACCEPTANCE CHECKLIST (verify before you say "done")

- [ ] Lookup, top-list, random selection and CSV export all use the same selected-ID set; test proves equality.
- [ ] For every slider budget, selected count == exactly k; ties resolved by (score desc, customer ID asc).
- [ ] Negative Recency, impossible tenure/window scenarios and over-allocated order counts are rejected with clear messages.
- [ ] No text in repo/app equates AUC with accuracy or top-k precision with overall accuracy; scores are not called probabilities.
- [ ] README has an "Honest limitations" section; `docs/VALIDATION_TODO.md` exists.
- [ ] Model artifact and manifest hash are unchanged (verify the SHA256 before and after).
- [ ] No threshold/hyperparameter tuning was done on the current test split.
- [ ] `PYTHONPATH=. pytest -q` run before and after; report counts (baseline was 137 passed, 1 failed (env mismatch), 4 skipped). Every new test passes.

# HOW TO WORK AND REPORT

1. First, read the repo structure and the files named above. Give me a short plan (which files you'll touch, in what order).
2. Implement Task 1 -> 2 -> 3 first (the real bugs), run tests, then Tasks 4-8, then Task 9 docs.
3. After each task, tell me in simple Hinglish: what was wrong, what you changed, which tests prove it.
4. At the end, give a final summary table: Task | Status (done / partially done / documented only) | Files changed | Tests added.
5. If you are unsure, or a fix would change model inputs/labels/artifacts, STOP and ask me instead of guessing.
6. Never claim the project is "production-ready" or "fully validated". The honest final status after these fixes is: "historical churn-ranking prototype with known bugs fixed and remaining validation steps documented."
